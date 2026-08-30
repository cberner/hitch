from django.db import migrations, models
from django.utils import timezone

_PR_STATE_KEYS = frozenset(
    {
        "auto_pull_result",
        "hitch_pr_handoff",
        "last_pr_watch",
        "last_pr_watch_turn_index",
        "pr_gates",
        "pr_handoff",
        "pr_stage_refresh",
    }
)
_RETIRED_WORKER_ERROR = "PR/QA wrapper retired during upgrade"


def migrate_pr_state_and_retire_wrappers(apps, schema_editor):
    SessionPullRequest = apps.get_model("main", "SessionPullRequest")
    SystemWorkflow = apps.get_model("main", "SystemWorkflow")
    WorkflowSteeringMessage = apps.get_model("main", "WorkflowSteeringMessage")
    CodexInstance = apps.get_model("main", "CodexInstance")
    SystemAgentRun = apps.get_model("main", "SystemAgentRun")
    ApprovalRequest = apps.get_model("main", "ApprovalRequest")
    UserInputRequest = apps.get_model("main", "UserInputRequest")
    database = schema_editor.connection.alias
    now = timezone.now()

    seen_threads = set()
    workflows = (
        SystemWorkflow.objects.using(database)
        .filter(kind="pr_qa")
        .order_by("main_thread_id", "-updated_at", "-pk")
    )
    for workflow in workflows.iterator():
        if workflow.main_thread_id in seen_threads:
            continue
        state = workflow.state if isinstance(workflow.state, dict) else {}
        handoff = state.get("pr_handoff")
        if not _has_pr_identity(handoff):
            continue
        seen_threads.add(workflow.main_thread_id)
        copied_state = {
            key: value for key, value in state.items() if key in _PR_STATE_KEYS
        }
        record, _created = SessionPullRequest.objects.using(database).update_or_create(
            thread_id=workflow.main_thread_id,
            defaults={"cwd": workflow.cwd, "state": copied_state},
        )
        SessionPullRequest.objects.using(database).filter(pk=record.pk).update(
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
        )

    nonterminal = SystemWorkflow.objects.using(database).filter(
        kind="pr_qa",
        status__in=("running", "blocked"),
    )
    for workflow in nonterminal.iterator():
        state = workflow.state if isinstance(workflow.state, dict) else {}
        pending = list(
            WorkflowSteeringMessage.objects.using(database)
            .filter(workflow_id=workflow.pk)
            .order_by("created_at", "pk")
            .values_list("prompt", flat=True)
        )
        state = {**state, "pr_qa_wrapper_retired": True}
        if pending:
            state["retired_steering_messages"] = pending
        review_only = (
            state.get("open_pr_on_lgtm", True) is not True
            and state.get("review_guidance") is True
        )
        if review_only:
            agent_kind = "review_guidance"
        elif _has_pr_identity(state.get("pr_handoff")):
            agent_kind = "pr_watch"
        else:
            agent_kind = "pr_publish"
        CodexInstance.objects.using(database).filter(
            workflow_id=workflow.pk,
            purpose="user",
            status__in=("starting", "running"),
        ).update(
            workflow_id=None,
            agent_kind=agent_kind,
            workflow_routing_started_at=None,
        )
        retired_instances = CodexInstance.objects.using(database).filter(
            workflow_id=workflow.pk,
            purpose__in=("system_agent", "system_feedback"),
            status__in=("starting", "running"),
        )
        retired_instance_ids = list(retired_instances.values_list("pk", flat=True))
        if retired_instance_ids:
            retired_instances.update(
                workflow_id=None,
                workflow_routing_started_at=None,
                status="failed",
                ended_at=now,
                interrupt_requested_at=now,
                error=_RETIRED_WORKER_ERROR,
            )
            ApprovalRequest.objects.using(database).filter(
                instance_id__in=retired_instance_ids,
                decision="",
            ).update(decision="cancel", decided_at=now)
            UserInputRequest.objects.using(database).filter(
                instance_id__in=retired_instance_ids,
                response__isnull=True,
            ).update(response={"answers": {}}, responded_at=now)
        SystemAgentRun.objects.using(database).filter(
            workflow_id=workflow.pk,
            status__in=("starting", "running"),
        ).update(status="failed", error=_RETIRED_WORKER_ERROR)
        workflow.state = state
        workflow.status = "completed"
        workflow.step = "pr_wrapper_retired"
        workflow.save(
            update_fields=["status", "step", "state", "updated_at"],
            using=database,
        )


def _has_pr_identity(value):
    if not isinstance(value, dict):
        return False
    url = value.get("url")
    if isinstance(url, str) and url.strip():
        return True
    repo = value.get("repository_full_name")
    number = value.get("pr_number")
    return (
        isinstance(repo, str)
        and bool(repo.strip())
        and isinstance(number, int)
        and not isinstance(number, bool)
        and number > 0
    )


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0071_remove_hitch_pr_publication_claim"),
    ]

    operations = [
        migrations.CreateModel(
            name="SessionPullRequest",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("thread_id", models.CharField(max_length=128, unique=True)),
                ("cwd", models.CharField(max_length=4096)),
                ("state", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.RunPython(
            migrate_pr_state_and_retire_wrappers,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.DeleteModel(name="WorkflowSteeringMessage"),
    ]
