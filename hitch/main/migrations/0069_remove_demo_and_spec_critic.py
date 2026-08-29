import json
import os
import re
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.db import migrations, models
from django.utils import timezone

from hitch.main.repos import same_repo_or_worktree

REMOVED_WORKFLOW_KINDS = ("demo_deployment", "spec_critic")
REMOVED_AGENT_KINDS = (
    "demo",
    "spec_critic",
    "spec_critic_requirements",
    "spec_critic_risks",
    "spec_critic_tests",
    "spec_critic_synthesizer",
)
ACTIVE_INSTANCE_STATUSES = ("starting", "running")
DIRECT_TREE_STOP_TIMEOUT_SECONDS = 5.0
DIRECT_TREE_STOP_POLL_SECONDS = 0.02
PODMAN_TIMEOUT_SECONDS = 30
HITCH_MANAGED_LABEL = "io.hitch.managed"
HITCH_MANAGED_LABEL_VALUE = "demo"
HITCH_SESSION_LABEL = "io.hitch.session"
HITCH_TOKEN_LABEL = "io.hitch.demo_token"
HITCH_NAME_LABEL = "io.hitch.container_name"
_CONTAINER_TARGET_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_WORKER_UNIT_RE = re.compile(r"hitch-codex-worker-(?:[a-f0-9]{12}-)?(?P<instance_id>\d+)\.(?:service|scope)")


def _removed_instance_query(codex_instance_model, workflow_ids):
    query = models.Q(agent_kind__in=REMOVED_AGENT_KINDS)
    if workflow_ids:
        query |= models.Q(
            workflow_id__in=workflow_ids,
            purpose__in=("system_agent", "system_feedback"),
        )
    return codex_instance_model.objects.filter(query)


def _worker_cmdline_parts(instance) -> list[bytes]:
    if instance.pid <= 0:
        return []
    try:
        cmdline = (Path("/proc") / str(instance.pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return cmdline.split(b"\0")


def _cmdline_identifies_worker(instance, parts: list[bytes]) -> bool:
    if b"codex_worker" not in parts:
        return False
    try:
        argument_index = parts.index(b"--instance-id")
    except ValueError:
        return False
    return argument_index + 1 < len(parts) and parts[argument_index + 1] == str(instance.pk).encode()


def _worker_has_open_events_file(
    instance,
    *,
    proc_root: Path = Path("/proc"),
    pid: int | None = None,
) -> bool:
    """Whether the process owns this row's durable, deployment-local event log."""
    if not instance.events_path or not Path(instance.events_path).is_file():
        return False
    return _worker_has_open_file(
        instance,
        instance.events_path,
        proc_root=proc_root,
        pid=pid,
    )


def _worker_has_open_database_file(
    instance,
    *,
    proc_root: Path = Path("/proc"),
    pid: int | None = None,
) -> bool:
    """Whether the process is connected to this migration's SQLite database."""
    database_name = settings.DATABASES["default"]["NAME"]
    if not database_name or str(database_name) == ":memory:":
        return False
    database_path = Path(database_name)
    if not database_path.is_absolute():
        database_path = Path(settings.BASE_DIR) / database_path
    return _worker_has_open_file(
        instance,
        database_path,
        proc_root=proc_root,
        pid=pid,
    )


def _worker_has_open_file(
    instance,
    target: str | os.PathLike[str],
    *,
    proc_root: Path,
    pid: int | None = None,
) -> bool:
    worker_pid = instance.pid if pid is None else pid
    if worker_pid <= 0 or not target:
        return False
    fd_dir = proc_root / str(worker_pid) / "fd"
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return False
    for entry in entries:
        try:
            if os.path.samefile(entry, target):
                return True
        except OSError:
            continue
    return False


def _worker_cmdline_matches(instance) -> bool:
    parts = _worker_cmdline_parts(instance)
    if not _cmdline_identifies_worker(instance, parts):
        return False
    # Checkout and instance id are not deployment-local: two Hitch state roots
    # can share both. Require a resource owned by this row/database even when
    # argv points at the current release.
    return _worker_has_open_events_file(instance) or _worker_has_open_database_file(instance)


def _worker_scope_from_cgroup(instance) -> str:
    try:
        cgroup = (Path("/proc") / str(instance.pid) / "cgroup").read_text(errors="replace")
    except OSError:
        return ""
    for match in _WORKER_UNIT_RE.finditer(cgroup):
        if match.group("instance_id") == str(instance.pk):
            return match.group(0)
    return ""


def _worker_process_is_verified(instance) -> bool:
    if _worker_cmdline_matches(instance):
        return True
    if _cmdline_identifies_worker(instance, _worker_cmdline_parts(instance)):
        raise RuntimeError(f"cannot verify ownership of removed-feature worker {instance.pk}")
    return False


def _systemd_scope_is_missing(systemctl: str, scope_unit: str) -> bool:
    result = subprocess.run(
        [
            systemctl,
            "--user",
            "show",
            "--property=LoadState",
            "--value",
            scope_unit,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    return result.returncode == 0 and result.stdout.strip() == b"not-found"


def _scope_worker_ownership(
    instance,
    scope_unit: str,
    *,
    proc_root: Path = Path("/proc"),
) -> str:
    """Classify the live workers in a persisted scope using local resources."""
    if not proc_root.exists():
        return "empty"
    target = scope_unit.encode()
    live_workers = 0
    matching_workers = 0
    owned_workers = 0
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"codex_worker" not in cmdline.split(b"\0"):
            continue
        try:
            cgroup = (entry / "cgroup").read_bytes()
        except OSError:
            continue
        if target in cgroup:
            live_workers += 1
            pid = int(entry.name)
            if _cmdline_identifies_worker(instance, cmdline.split(b"\0")):
                matching_workers += 1
                if _worker_has_open_events_file(
                    instance,
                    proc_root=proc_root,
                    pid=pid,
                ) or _worker_has_open_database_file(
                    instance,
                    proc_root=proc_root,
                    pid=pid,
                ):
                    owned_workers += 1
    if owned_workers and owned_workers == live_workers:
        return "owned"
    if matching_workers:
        return "ambiguous"
    if live_workers:
        return "foreign"
    return "empty"


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    ppid: int
    start_time: int
    state: str


def _process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> _ProcessIdentity | None:
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
    except OSError:
        return None
    rparen = stat.rfind(")")
    if rparen == -1:
        return None
    fields = stat[rparen + 1 :].split()
    if len(fields) < 20:
        return None
    try:
        return _ProcessIdentity(
            pid=pid,
            ppid=int(fields[1]),
            start_time=int(fields[19]),
            state=fields[0],
        )
    except ValueError:
        return None


def _descendant_process_identities(
    root_pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[_ProcessIdentity, ...]:
    if not proc_root.exists():
        return ()
    identities: dict[int, _ProcessIdentity] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        identity = _process_identity(int(entry.name), proc_root=proc_root)
        if identity is not None:
            identities[identity.pid] = identity
    descendants: dict[int, _ProcessIdentity] = {}
    parents = {root_pid}
    while parents:
        children = {
            pid: identity for pid, identity in identities.items() if identity.ppid in parents and pid not in descendants
        }
        if not children:
            break
        descendants.update(children)
        parents = set(children)
    return tuple(descendants.values())


def _same_process(
    identity: _ProcessIdentity,
    *,
    proc_root: Path = Path("/proc"),
) -> bool:
    current = _process_identity(identity.pid, proc_root=proc_root)
    return current is not None and current.start_time == identity.start_time


def _signal_process(
    identity: _ProcessIdentity,
    sig: signal.Signals,
    *,
    proc_root: Path = Path("/proc"),
) -> None:
    if not _same_process(identity, proc_root=proc_root):
        return
    try:
        os.kill(identity.pid, sig)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise RuntimeError(f"failed to stop removed-feature descendant {identity.pid}") from exc


def _wait_for_processes_to_stop(
    identities: tuple[_ProcessIdentity, ...],
    *,
    proc_root: Path = Path("/proc"),
) -> None:
    deadline = time.monotonic() + DIRECT_TREE_STOP_TIMEOUT_SECONDS
    while True:
        live = []
        for identity in identities:
            current = _process_identity(identity.pid, proc_root=proc_root)
            if (
                current is not None
                and current.start_time == identity.start_time
                and current.state not in ("Z", "X", "x")
            ):
                live.append(identity.pid)
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("removed-feature processes remain after SIGKILL: " + ", ".join(str(pid) for pid in live))
        time.sleep(DIRECT_TREE_STOP_POLL_SECONDS)


def _force_stop_direct_worker(instance) -> None:
    if not _worker_cmdline_matches(instance):
        return
    try:
        if os.getsid(instance.pid) != instance.pid:
            raise RuntimeError(f"cannot safely force-stop removed-feature worker {instance.pk}")
    except ProcessLookupError:
        return
    root = _process_identity(instance.pid)
    if root is None or not _worker_cmdline_matches(instance) or not _same_process(root):
        return

    descendants = {identity.pid: identity for identity in _descendant_process_identities(instance.pid)}
    group_stopped = False
    try:
        os.killpg(instance.pid, signal.SIGSTOP)
        group_stopped = True
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise RuntimeError(f"failed to freeze removed-feature worker {instance.pk}") from exc

    try:
        stable_passes = 0
        deadline = time.monotonic() + DIRECT_TREE_STOP_TIMEOUT_SECONDS
        while group_stopped and stable_passes < 2:
            current = {identity.pid: identity for identity in _descendant_process_identities(instance.pid)}
            tree_changed = any(
                pid not in descendants or descendants[pid].start_time != identity.start_time
                for pid, identity in current.items()
            )
            descendants.update(current)
            for identity in current.values():
                _signal_process(identity, signal.SIGSTOP)
            stable_passes = stable_passes + 1 if not tree_changed else 0
            if time.monotonic() >= deadline:
                raise RuntimeError(f"could not freeze removed-feature worker tree {instance.pk}")
            time.sleep(DIRECT_TREE_STOP_POLL_SECONDS)

        for identity in descendants.values():
            _signal_process(identity, signal.SIGKILL)
        if group_stopped and _same_process(root):
            os.killpg(instance.pid, signal.SIGKILL)
        elif _same_process(root):
            _signal_process(root, signal.SIGKILL)
        _wait_for_processes_to_stop((root, *descendants.values()))
    except Exception:
        if group_stopped and _same_process(root):
            with suppress(OSError):
                os.killpg(instance.pid, signal.SIGCONT)
        for identity in descendants.values():
            with suppress(RuntimeError):
                _signal_process(identity, signal.SIGCONT)
        raise


def _force_stop_worker(instance) -> None:
    scope_unit = instance.systemd_scope_unit or _worker_scope_from_cgroup(instance)
    if scope_unit:
        # The row PID may not be published yet. Inspect the worker inside the
        # scope, and only trust row-specific event/database resources.
        scope_ownership = _scope_worker_ownership(instance, scope_unit)
        if scope_ownership == "foreign":
            return
        if scope_ownership == "ambiguous":
            raise RuntimeError(f"cannot verify ownership of removed-feature worker {instance.pk}")
        systemctl = shutil.which("systemctl")
        if systemctl is None:
            raise RuntimeError(f"cannot stop removed-feature worker {instance.pk}: systemctl missing")
        result = subprocess.run(
            [
                systemctl,
                "--user",
                "kill",
                "--kill-whom=all",
                "--signal=SIGKILL",
                scope_unit,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        if result.returncode != 0 and not _systemd_scope_is_missing(systemctl, scope_unit):
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"failed to stop removed-feature worker {instance.pk}: {detail}")
        return
    _force_stop_direct_worker(instance)


def _stop_worker_process(instance) -> None:
    """Stop without running callbacks that can route more removed work."""
    if not _worker_process_is_verified(instance):
        if instance.systemd_scope_unit:
            _force_stop_worker(instance)
        return
    if not instance.systemd_scope_unit:
        instance.systemd_scope_unit = _worker_scope_from_cgroup(instance)
    _force_stop_worker(instance)


def _retire_removed_instance(codex_instance_model, instance, *, now) -> None:
    while True:
        current = codex_instance_model.objects.filter(pk=instance.pk).first()
        if current is None:
            return
        if current.status not in ACTIVE_INSTANCE_STATUSES:
            # Terminal state is persisted before finish callbacks run, so a
            # matching process can still launch follow-up work.
            if current.pid > 0 and _cmdline_identifies_worker(
                current,
                _worker_cmdline_parts(current),
            ):
                _stop_worker_process(current)
            return
        if current.pid <= 0:
            if current.systemd_scope_unit:
                _force_stop_worker(current)
            updated = codex_instance_model.objects.filter(
                pk=current.pk,
                status__in=ACTIVE_INSTANCE_STATUSES,
                pid=current.pid,
            ).update(
                status="failed",
                ended_at=now,
                error="stopped because its feature was removed",
            )
            if updated:
                return
            continue
        _stop_worker_process(current)
        codex_instance_model.objects.filter(
            pk=current.pk,
            status__in=ACTIVE_INSTANCE_STATUSES,
        ).update(
            status="failed",
            ended_at=now,
            error="stopped because its feature was removed",
        )
        return


def _stop_removed_feature_workers(
    codex_instance_model,
    approval_request_model,
    user_input_request_model,
    *,
    workflow_ids,
) -> list[int]:
    instance_ids: list[int] = []
    retired_ids: set[int] = set()
    now = timezone.now()
    # A child may commit just before its parent is killed. Retire to a fixed
    # point before deleting the workflows that identify those children.
    while True:
        instances = list(
            _removed_instance_query(codex_instance_model, workflow_ids).exclude(pk__in=retired_ids).order_by("pk")
        )
        if not instances:
            break
        for instance in instances:
            _retire_removed_instance(codex_instance_model, instance, now=now)
            retired_ids.add(instance.pk)
            instance_ids.append(instance.pk)
    if instance_ids:
        approval_request_model.objects.filter(
            instance_id__in=instance_ids,
            decision="",
        ).update(decision="cancel", decided_at=now)
        user_input_request_model.objects.filter(
            instance_id__in=instance_ids,
            response__isnull=True,
        ).update(response={"answers": {}}, responded_at=now)
    return instance_ids


def _preserve_in_flight_requests(
    codex_instance_model,
    project_model,
    proposed_session_model,
    session_metadata_model,
    workflows,
) -> None:
    projects = list(project_model.objects.all())
    for workflow in workflows.filter(
        kind="spec_critic",
        status__in=("running", "blocked"),
    ):
        if (
            workflow.status == "blocked"
            and codex_instance_model.objects.filter(
                workflow_id=workflow.pk,
                purpose="system_feedback",
                status="completed",
            ).exists()
        ):
            continue
        state = workflow.state if isinstance(workflow.state, dict) else {}
        original_prompt = state.get("original_prompt")
        if not isinstance(original_prompt, str) or not original_prompt.strip():
            raise RuntimeError(f"cannot preserve request for Spec Critic workflow {workflow.pk}")
        user_message_index = state.get("next_user_message_index")
        implementation_exists = (
            isinstance(user_message_index, int)
            and not isinstance(user_message_index, bool)
            and codex_instance_model.objects.filter(
                thread_id=workflow.main_thread_id,
                purpose="user",
                user_message_index=user_message_index,
                started_at__gte=workflow.created_at,
                prompt=original_prompt,
            ).exists()
        ) or codex_instance_model.objects.filter(
            workflow_id=workflow.pk,
            purpose="user",
        ).exists()
        if implementation_exists:
            continue
        source_session, _created = session_metadata_model.objects.get_or_create(
            thread_id=workflow.main_thread_id,
            defaults={"cwd": workflow.cwd},
        )
        if not source_session.cwd.strip() and workflow.cwd.strip():
            source_session.cwd = workflow.cwd
            source_session.save(update_fields=["cwd"])
        project_id = source_session.project_id
        if project_id is None and not source_session.project_cleared:
            inferred_project = next(
                (
                    project
                    for project in projects
                    if same_repo_or_worktree(
                        source_session.cwd,
                        project.repo_path,
                        project.git_common_dir,
                    )
                ),
                None,
            )
            project_id = inferred_project.pk if inferred_project is not None else None
        proposed_session_model.objects.create(
            project_id=project_id,
            source_workflow_id=workflow.pk,
            source_session_id=source_session.pk,
            title="Request not started during upgrade",
            inbox_kind="proposal",
            summary=(
                "Hitch was upgraded while this request was waiting for removed "
                "preflight analysis, so implementation did not start. Select Do it "
                "to continue the original request in its session."
            ),
            prompt=original_prompt,
            outcome_metadata={
                "automation_status": "failed",
                "automation_error": "preflight feature removed during upgrade",
                "source_thread_id": workflow.main_thread_id,
                "resume_source_session": True,
                "auto_pr_enabled": state.get("auto_pr_enabled") is True,
                "auto_qa_enabled": state.get("auto_qa_enabled") is True,
                "auto_merge_to_local_branch": (state.get("auto_merge_to_local_branch") is True),
                "auto_merge_branch": (
                    state.get("auto_merge_branch") if isinstance(state.get("auto_merge_branch"), str) else ""
                ),
            },
        )


def _podman(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["podman", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=PODMAN_TIMEOUT_SECONDS,
    )


@dataclass(frozen=True)
class _DemoContainerRegistration:
    thread_id: str
    container_id: str
    container_name: str
    runtime: str
    registration_token: str


def _valid_container_target(target: str) -> str:
    if not _CONTAINER_TARGET_RE.fullmatch(target) or target.startswith("-"):
        raise RuntimeError("cannot safely identify a registered demo container")
    return target


def _container_missing_error(exc: subprocess.CalledProcessError) -> bool:
    detail = f"{exc.stderr or ''}\n{exc.stdout or ''}".lower()
    return any(
        marker in detail
        for marker in (
            "does not exist",
            "no such container",
            "no such object",
            "not found",
            "no container with name or id",
        )
    )


def _container_inspect(target: str) -> dict[str, object] | None:
    try:
        result = _podman(["inspect", target])
    except subprocess.CalledProcessError as exc:
        if _container_missing_error(exc):
            return None
        raise RuntimeError(f"could not inspect registered demo container {target}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not inspect registered demo container {target}") from exc
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"could not inspect registered demo container {target}") from exc
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        raise RuntimeError(f"could not inspect registered demo container {target}")
    return payload


def _container_labels(item: dict[str, object]) -> dict[str, str]:
    config = item.get("Config")
    labels: object = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        labels = item.get("Labels")
    if not isinstance(labels, dict):
        return {}
    return {str(key): str(value) for key, value in labels.items()}


def _discovered_container_targets(
    registration: _DemoContainerRegistration,
) -> set[str]:
    if not registration.registration_token:
        return set()
    result = _podman(
        [
            "ps",
            "-a",
            "--filter",
            f"label={HITCH_MANAGED_LABEL}={HITCH_MANAGED_LABEL_VALUE}",
            "--filter",
            f"label={HITCH_SESSION_LABEL}={registration.thread_id}",
            "--filter",
            f"label={HITCH_TOKEN_LABEL}={registration.registration_token}",
            "--format",
            "{{.ID}}",
        ]
    )
    return {_valid_container_target(line.strip()) for line in result.stdout.splitlines() if line.strip()}


def _registered_container_targets(
    registration: _DemoContainerRegistration,
) -> set[str]:
    # Successful pre-label rows persisted Podman's immutable returned ID. Their
    # generated name is not an independent ownership proof.
    explicit_targets = (
        (registration.container_id, registration.container_name)
        if registration.registration_token or not registration.container_id
        else (registration.container_id,)
    )
    targets = {_valid_container_target(target) for target in explicit_targets if target}
    targets.update(_discovered_container_targets(registration))
    return targets


def _matches_tokenless_legacy_container_id(
    registration: _DemoContainerRegistration,
    *,
    target: str,
    inspected: dict[str, object],
) -> bool:
    if registration.registration_token or target != registration.container_id:
        return False
    inspected_id = inspected.get("Id") or inspected.get("ID") or inspected.get("id")
    return isinstance(inspected_id, str) and inspected_id == registration.container_id


def _container_belongs_to_registration(
    registration: _DemoContainerRegistration,
    *,
    target: str,
    inspected: dict[str, object],
) -> bool:
    if _matches_tokenless_legacy_container_id(
        registration,
        target=target,
        inspected=inspected,
    ):
        return True
    labels = _container_labels(inspected)
    if labels.get(HITCH_MANAGED_LABEL) != HITCH_MANAGED_LABEL_VALUE:
        return False
    if labels.get(HITCH_SESSION_LABEL) != registration.thread_id:
        return False
    if registration.registration_token and labels.get(HITCH_TOKEN_LABEL) != registration.registration_token:
        return False
    return not (target == registration.container_name and labels.get(HITCH_NAME_LABEL) != registration.container_name)


def _remove_registered_container(registration: _DemoContainerRegistration, target: str) -> None:
    inspected = _container_inspect(target)
    if inspected is None:
        return
    if not _container_belongs_to_registration(registration, target=target, inspected=inspected):
        raise RuntimeError(f"refusing to remove demo container {target}: registration does not own it")
    try:
        _podman(["rm", "-f", target])
    except subprocess.CalledProcessError as exc:
        if not _container_missing_error(exc):
            raise RuntimeError(f"could not remove demo container {target}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not remove demo container {target}") from exc
    if _container_inspect(target) is not None:
        raise RuntimeError(f"demo container {target} remains after removal")


def _cleanup_removed_feature_containers(
    registrations: list[_DemoContainerRegistration],
) -> None:
    for registration in registrations:
        has_cleanup_identity = bool(
            registration.container_id or registration.container_name or registration.registration_token
        )
        if not has_cleanup_identity:
            continue
        if registration.runtime not in ("", "podman"):
            raise RuntimeError(f"cannot remove registered demo runtime {registration.runtime}")
        for target in sorted(_registered_container_targets(registration)):
            _remove_registered_container(registration, target)
        remaining = _discovered_container_targets(registration)
        if remaining:
            raise RuntimeError("registered demo containers remain: " + ", ".join(sorted(remaining)))


def retire_removed_features(apps, schema_editor):
    ApprovalRequest = apps.get_model("main", "ApprovalRequest")
    CodexInstance = apps.get_model("main", "CodexInstance")
    Project = apps.get_model("main", "Project")
    ProposedSession = apps.get_model("main", "ProposedSession")
    SessionDemo = apps.get_model("main", "SessionDemo")
    SessionMetadata = apps.get_model("main", "SessionMetadata")
    SystemWorkflow = apps.get_model("main", "SystemWorkflow")
    UserInputRequest = apps.get_model("main", "UserInputRequest")

    workflows = SystemWorkflow.objects.filter(kind__in=REMOVED_WORKFLOW_KINDS)
    workflow_ids = list(workflows.values_list("pk", flat=True))
    demo_workflow_ids = list(workflows.filter(kind="demo_deployment").values_list("pk", flat=True))
    if demo_workflow_ids:
        # Older rows may predate reliable agent-kind tagging. Normalize the
        # retired system turns so their prompts remain durable redaction keys.
        CodexInstance.objects.filter(
            workflow_id__in=demo_workflow_ids,
            purpose__in=("system_agent", "system_feedback"),
        ).update(agent_kind="demo")
    _stop_removed_feature_workers(
        CodexInstance,
        ApprovalRequest,
        UserInputRequest,
        workflow_ids=workflow_ids,
    )
    registrations = [
        _DemoContainerRegistration(*values)
        for values in SessionDemo.objects.values_list(
            "thread_id",
            "container_id",
            "container_name",
            "runtime",
            "registration_token",
        )
    ]
    _cleanup_removed_feature_containers(registrations)
    _preserve_in_flight_requests(
        CodexInstance,
        Project,
        ProposedSession,
        SessionMetadata,
        workflows,
    )
    if workflow_ids:
        CodexInstance.objects.filter(workflow_id__in=workflow_ids).update(workflow_id=None)
        workflows.delete()


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0068_alter_usersettings_reasoning_effort"),
    ]

    operations = [
        migrations.RunPython(retire_removed_features, migrations.RunPython.noop),
        migrations.RemoveField(
            model_name="usersettings",
            name="spec_critic_enabled",
        ),
        migrations.DeleteModel(
            name="SessionDemo",
        ),
    ]
