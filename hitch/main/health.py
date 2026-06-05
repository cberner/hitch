"""Health metrics for the Hitch health dashboard.

Gathers leak signals (Codex app-servers, managed worktree disk, stuck worker
rows) and backlog signals (workflows, PR monitors, pending human handoffs) into
one report so a glance tells whether something is piling up. Every metric is
collected defensively: a failure in one collector degrades that row to
"unavailable" rather than breaking the whole page.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from django.conf import settings
from django.db.models import Q, QuerySet
from django.utils import timezone

from hitch.main import codex_pool, disk_cleanup, host_probes, system_agents
from hitch.main.context_processors import server_git_hash
from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    ProposedSession,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.worktrees import discover_managed_worktrees

logger = logging.getLogger(__name__)

SEVERITY_OK = "ok"
SEVERITY_WARN = "warn"
SEVERITY_DANGER = "danger"
SEVERITY_UNKNOWN = "unknown"

_SEVERITY_RANK = {
    SEVERITY_OK: 0,
    SEVERITY_UNKNOWN: 1,
    SEVERITY_WARN: 2,
    SEVERITY_DANGER: 3,
}
_SEVERITY_LABEL = {
    SEVERITY_OK: "OK",
    SEVERITY_WARN: "Attention",
    SEVERITY_DANGER: "Problem",
    SEVERITY_UNKNOWN: "Unknown",
}

# A running/starting turn legitimately owns ~1 app-server, plus a small warm
# pool, so only flag a meaningful surplus as a likely leak.
_APP_SERVER_LEAK_WARN = 5
_APP_SERVER_LEAK_DANGER = 10
# A turn still "running" after this long is almost certainly a leaked row whose
# worker process is gone.
_STUCK_TURN_AGE = timedelta(hours=6)
_STALE_BLOCKED_AGE = timedelta(days=7)
_RECENT_FAILURE_AGE = timedelta(hours=24)
# Bound how often the (proc-scanning, disk-walking) report is rebuilt under load.
_REPORT_CACHE_TTL = timedelta(seconds=15)
_report_cache_lock = threading.Lock()
_report_cache: tuple[datetime, HealthReport] | None = None
# Runserver fd / CLOSE_WAIT alarm floors. Both are trend signals (healthy idle is
# ~80 fds, 0 CLOSE_WAIT); these floors only catch a runaway snapshot.
_RUNSERVER_FD_WARN = 800
_CLOSE_WAIT_WARN = 50

# Blocked-workflow failure-mode buckets, matched in order against the stored
# error text. ``benign`` buckets (normal user action / external limits) never
# raise severity even when fresh. Derived from the failure-mode census in the
# health investigation handoff.
_BLOCKED_BUCKETS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("gh pr create failures", ("gh pr create", "pr create --fill"), False),
    ("Worker exited before completion", ("worker process exited",), False),
    (
        "State-DB lock / transport closed",
        ("transportclosederror", "database is locked", "sqlite state runtime", "state db locked"),
        False,
    ),
    ("CodexInstance NOT NULL IntegrityError", ("not null constraint failed: main_codexinstance",), False),
    ("Invalid JSON schema", ("invalid_json_schema",), False),
    ("Stopped by user", ("stopped by user",), True),
    ("Rate limited (429 / usage)", ("429", "too many requests", "usage limit"), True),
    ("Unsupported workflow kind", ("no longer supported",), True),
)


@dataclass(frozen=True)
class HealthMetric:
    key: str
    label: str
    value: str
    severity: str = SEVERITY_OK
    detail: str = ""


@dataclass(frozen=True)
class HealthSection:
    title: str
    metrics: list[HealthMetric] = field(default_factory=list)


@dataclass(frozen=True)
class HealthReport:
    generated_at: str
    server_git_hash: str
    overall_severity: str
    sections: list[HealthSection]

    @property
    def overall_label(self) -> str:
        return _SEVERITY_LABEL.get(self.overall_severity, "Unknown")

    @property
    def headline_metric(self) -> HealthMetric | None:
        """The worst-severity metric driving the overall status, or ``None`` when OK.

        Surfaces the single row that pushed the dashboard off "OK" so callers
        need not scan every section. Ties are broken by section order, then by
        metric order within a section (strict ``>`` keeps the first-seen winner).
        """
        if self.overall_severity == SEVERITY_OK:
            return None
        worst: HealthMetric | None = None
        for section in self.sections:
            for metric in section.metrics:
                if worst is None or _SEVERITY_RANK.get(metric.severity, 0) > _SEVERITY_RANK.get(
                    worst.severity, 0
                ):
                    worst = metric
        return worst

    def copy_text(self) -> str:
        """Plain-text summary built for pasting into a chat with the assistant."""
        lines = [
            "Hitch health report",
            f"Generated: {self.generated_at}",
            f"Overall: {self.overall_label.upper()}",
        ]
        headline = self.headline_metric
        if headline is not None:
            lines.append(f"Headline: {headline.label} — {headline.value}")
        if self.server_git_hash:
            lines.append(f"Server git hash: {self.server_git_hash}")
        for section in self.sections:
            lines.append("")
            lines.append(f"[{section.title}]")
            for metric in section.metrics:
                flag = "" if metric.severity == SEVERITY_OK else f"  <{metric.severity}>"
                row = f"- {metric.label}: {metric.value}{flag}"
                lines.append(row)
                if metric.detail:
                    lines.append(f"    {metric.detail}")
        return "\n".join(lines)


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _safe_metric(key: str, label: str, collector: Callable[[], HealthMetric]) -> HealthMetric:
    try:
        return collector()
    except Exception:
        logger.exception("failed to collect health metric %s", key)
        return HealthMetric(
            key=key,
            label=label,
            value="unavailable",
            severity=SEVERITY_UNKNOWN,
            detail="Failed to read this metric; see server logs.",
        )


def _count_metric(
    key: str,
    label: str,
    counter: Callable[[], int],
    *,
    warn_at: int | None = None,
    danger_at: int | None = None,
    detail: str = "",
) -> HealthMetric:
    def collect() -> HealthMetric:
        count = counter()
        severity = SEVERITY_OK
        if danger_at is not None and count >= danger_at:
            severity = SEVERITY_DANGER
        elif warn_at is not None and count >= warn_at:
            severity = SEVERITY_WARN
        return HealthMetric(key=key, label=label, value=str(count), severity=severity, detail=detail)

    return _safe_metric(key, label, collect)


def _active_turn_count() -> int:
    return CodexInstance.objects.filter(
        status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING)
    ).count()


def _app_server_metric() -> HealthMetric:
    def collect() -> HealthMetric:
        running = codex_pool.count_running_codex_app_servers()
        active = _active_turn_count()
        surplus = max(0, running - active)
        severity = SEVERITY_OK
        if surplus >= _APP_SERVER_LEAK_DANGER:
            severity = SEVERITY_DANGER
        elif surplus >= _APP_SERVER_LEAK_WARN:
            severity = SEVERITY_WARN
        detail = (
            f"{active} active turn(s); {surplus} beyond active turns. "
            "A persistent surplus indicates leaked app-servers contending on "
            "the CODEX_HOME state-DB lock — use Nuke Codex instances."
        )
        return HealthMetric(
            key="app_servers",
            label="Codex app-servers running",
            value=str(running),
            severity=severity,
            detail=detail,
        )

    return _safe_metric("app_servers", "Codex app-servers running", collect)


def _hitch_disk_metric() -> HealthMetric:
    def collect() -> HealthMetric:
        usage = disk_cleanup.hitch_home_disk_usage()
        if usage is None:
            return HealthMetric(
                key="hitch_disk",
                label="~/.hitch disk usage",
                value="unavailable",
                severity=SEVERITY_UNKNOWN,
            )
        severity = SEVERITY_DANGER if usage.over_limit else SEVERITY_OK
        value = f"{_human_bytes(usage.used_bytes)} ({usage.percent_of_disk:.1f}% of disk)"
        detail = f"Cleanup ceiling {_human_bytes(usage.limit_bytes)}. " + (
            "Over the ceiling — finished-session cleanup will start deleting eligible worktrees."
            if usage.over_limit
            else "Under the cleanup ceiling."
        )
        return HealthMetric(
            key="hitch_disk",
            label="~/.hitch disk usage",
            value=value,
            severity=severity,
            detail=detail,
        )

    return _safe_metric("hitch_disk", "~/.hitch disk usage", collect)


def _worktree_metric() -> HealthMetric:
    def collect() -> HealthMetric:
        # Count only (a shallow stat of the worktree roots). The on-disk size of
        # worktrees is the dominant part of ~/.hitch, already reported by the
        # disk metric, so summing each tree here would just re-walk it.
        worktrees = discover_managed_worktrees()
        return HealthMetric(
            key="worktrees",
            label="Managed worktrees",
            value=str(len(worktrees)),
            severity=SEVERITY_OK,
            detail="Live git worktrees under ~/.hitch/worktrees; their size is in "
            "the ~/.hitch disk figure below.",
        )

    return _safe_metric("worktrees", "Managed worktrees", collect)


def _stuck_turn_count() -> int:
    cutoff = timezone.now() - _STUCK_TURN_AGE
    return CodexInstance.objects.filter(
        status__in=(CodexInstance.STATUS_STARTING, CodexInstance.STATUS_RUNNING),
        started_at__lt=cutoff,
    ).count()


def _stale_blocked_count() -> int:
    cutoff = timezone.now() - _STALE_BLOCKED_AGE
    return len(system_agents.archive_stale_blocked_workflows(older_than=cutoff, apply=False))


def _leak_section() -> HealthSection:
    return HealthSection(
        title="Leaks",
        metrics=[
            _app_server_metric(),
            _count_metric(
                "active_turns",
                "Active turns",
                _active_turn_count,
                detail="Worker rows in starting/running state.",
            ),
            _count_metric(
                "stuck_turns",
                "Stuck turns (>6h)",
                _stuck_turn_count,
                warn_at=1,
                detail="Running rows older than 6h — likely a dead worker that never reached a terminal state.",
            ),
            _worktree_metric(),
            _hitch_disk_metric(),
        ],
    )


def _backlog_section() -> HealthSection:
    return HealthSection(
        title="Backlogs",
        metrics=[
            _count_metric(
                "running_workflows",
                "Running workflows",
                lambda: SystemWorkflow.objects.filter(status=SystemWorkflow.STATUS_RUNNING).count(),
            ),
            _count_metric(
                "blocked_workflows",
                "Blocked workflows",
                lambda: SystemWorkflow.objects.filter(status=SystemWorkflow.STATUS_BLOCKED).count(),
                warn_at=1,
                detail="Workflows halted on an error, awaiting attention.",
            ),
            _count_metric(
                "stale_blocked_workflows",
                "Stale blocked PR-QA (>7d)",
                _stale_blocked_count,
                warn_at=1,
                detail="Blocked PR-QA workflows older than 7 days. Clear with "
                "the archive_stale_blocked_workflows command.",
            ),
            _count_metric(
                "pr_monitors",
                "PR monitors active",
                lambda: SystemWorkflow.objects.filter(
                    kind=SystemWorkflow.KIND_PR_QA,
                    status=SystemWorkflow.STATUS_RUNNING,
                    step=system_agents.STEP_PR_MONITORING,
                ).count(),
            ),
            _count_metric(
                "pending_approvals",
                "Pending approvals",
                lambda: ApprovalRequest.objects.filter(decision=ApprovalRequest.DECISION_PENDING).count(),
                warn_at=1,
                detail="Approval handoffs waiting on a human decision.",
            ),
            _count_metric(
                "pending_inputs",
                "Pending input requests",
                lambda: UserInputRequest.objects.filter(response__isnull=True).count(),
                warn_at=1,
                detail="Plan-mode input prompts waiting on a human answer.",
            ),
            _count_metric(
                "pending_proposals",
                "Pending proposed sessions",
                lambda: ProposedSession.objects.filter(outcome_status=ProposedSession.OUTCOME_UNSET).count(),
                detail="Auto-proposed sessions not yet accepted/rejected.",
            ),
        ],
    )


def _worker_scope_section() -> HealthSection:
    """Leaked-grandchild detector and concurrent worker-scope count (problems 1 & 3)."""
    title = "Worker scopes (leaks)"
    try:
        probe = host_probes.probe_worker_scopes()
    except Exception:
        logger.exception("failed to probe worker scopes")
        return HealthSection(
            title,
            [
                HealthMetric(
                    "leaked_scopes",
                    "Leaked worker scopes",
                    "unavailable",
                    SEVERITY_UNKNOWN,
                    "Failed to probe /proc; see server logs.",
                )
            ],
        )

    metrics: list[HealthMetric] = []
    leaked = probe.leaked
    metrics.append(
        HealthMetric(
            key="leaked_scopes",
            label="Leaked worker scopes",
            value=str(len(leaked)),
            severity=SEVERITY_DANGER if leaked else SEVERITY_OK,
            detail=(
                f"{_human_bytes(probe.total_leaked_rss_bytes)} held by orphaned grandchildren of "
                "dead workers. Kill with: "
                "systemctl --user kill --kill-whom=all --signal=SIGKILL <scope>."
                if leaked
                else "No terminal worker scope still holds live processes."
            ),
        )
    )
    for scope in leaked:
        commands = scope.foreign_commands or [proc.comm for proc in scope.processes]
        rendered = ", ".join(command[:120] for command in commands if command) or "unknown"
        metrics.append(
            HealthMetric(
                key=f"leaked_scope_{scope.instance_id}",
                label=scope.scope_unit,
                value=_human_bytes(scope.total_rss_bytes),
                severity=SEVERITY_DANGER,
                detail=f"instance {scope.instance_id} status={scope.db_status}; "
                f"{len(scope.processes)} live pid(s): {rendered}",
            )
        )

    nproc = host_probes.cpu_count() or 1
    active = probe.active_count
    active_severity = SEVERITY_OK
    if active > nproc:
        active_severity = SEVERITY_DANGER
    elif active > max(1, nproc // 2):
        active_severity = SEVERITY_WARN
    metrics.append(
        HealthMetric(
            key="active_scopes",
            label="Active worker scopes",
            value=str(active),
            severity=active_severity,
            detail=f"{nproc} CPU core(s); concurrent worker scopes contend for CPU.",
        )
    )
    return HealthSection(title, metrics)


def _load_metric() -> HealthMetric:
    def collect() -> HealthMetric:
        average = host_probes.load_average()
        if average is None:
            return HealthMetric(
                "load_avg", "Load average", "unavailable", SEVERITY_UNKNOWN
            )
        nproc = host_probes.cpu_count() or 1
        one, five, fifteen = average
        severity = SEVERITY_OK
        if one > nproc:
            severity = SEVERITY_DANGER
        elif one > nproc * 0.7:
            severity = SEVERITY_WARN
        return HealthMetric(
            key="load_avg",
            label="Load average (1/5/15m)",
            value=f"{one:.2f} / {five:.2f} / {fifteen:.2f}",
            severity=severity,
            detail=f"{nproc} CPU core(s); alarm when 1-min load exceeds the core count.",
        )

    return _safe_metric("load_avg", "Load average", collect)


def _fd_metric() -> HealthMetric:
    def collect() -> HealthMetric:
        count = host_probes.runserver_fd_count()
        if count is None:
            return HealthMetric(
                "runserver_fds", "Runserver open fds", "unavailable", SEVERITY_UNKNOWN
            )
        return HealthMetric(
            key="runserver_fds",
            label="Runserver open fds",
            value=str(count),
            severity=SEVERITY_WARN if count >= _RUNSERVER_FD_WARN else SEVERITY_OK,
            detail="Healthy idle is ~80; a steady climb points to a leaked streaming socket.",
        )

    return _safe_metric("runserver_fds", "Runserver open fds", collect)


def _close_wait_metric() -> HealthMetric:
    def collect() -> HealthMetric:
        count = host_probes.runserver_close_wait_count()
        if count is None:
            return HealthMetric(
                "close_wait",
                "Runserver CLOSE_WAIT sockets",
                "unavailable",
                SEVERITY_UNKNOWN,
            )
        return HealthMetric(
            key="close_wait",
            label="Runserver CLOSE_WAIT sockets",
            value=str(count),
            severity=SEVERITY_WARN if count >= _CLOSE_WAIT_WARN else SEVERITY_OK,
            detail="Aborted-client sockets not yet cleaned up; each holds an fd.",
        )

    return _safe_metric("close_wait", "Runserver CLOSE_WAIT sockets", collect)


def _host_section() -> HealthSection:
    return HealthSection(
        title="Host / CPU",
        metrics=[_load_metric(), _fd_metric(), _close_wait_metric()],
    )


def _blocked_bucket_section() -> HealthSection:
    title = "Blocked workflow buckets"
    try:
        metrics = _blocked_bucket_metrics()
    except Exception:
        logger.exception("failed to build blocked-workflow buckets")
        return HealthSection(
            title,
            [HealthMetric("blocked_buckets", title, "unavailable", SEVERITY_UNKNOWN)],
        )
    if not metrics:
        metrics = [
            HealthMetric("blocked_buckets_none", "Blocked workflows", "0", SEVERITY_OK)
        ]
    return HealthSection(title, metrics)


def _classify_blocked_error(error: str) -> tuple[str, bool]:
    lowered = error.lower()
    for label, needles, benign in _BLOCKED_BUCKETS:
        if any(needle in lowered for needle in needles):
            return label, benign
    return "Other", False


def _blocked_bucket_metrics() -> list[HealthMetric]:
    cutoff = timezone.now() - _RECENT_FAILURE_AGE
    counts: dict[str, int] = {}
    fresh: dict[str, int] = {}
    last_seen: dict[str, datetime] = {}
    benign_by_label: dict[str, bool] = {}
    for row in SystemWorkflow.objects.filter(
        status=SystemWorkflow.STATUS_BLOCKED
    ).values("state", "updated_at"):
        state = row["state"] if isinstance(row["state"], dict) else {}
        label, benign = _classify_blocked_error(str(state.get("error", "")))
        benign_by_label[label] = benign
        counts[label] = counts.get(label, 0) + 1
        updated = row["updated_at"]
        if updated is not None:
            if label not in last_seen or updated > last_seen[label]:
                last_seen[label] = updated
            if updated >= cutoff:
                fresh[label] = fresh.get(label, 0) + 1

    ordered_labels = [label for label, _, _ in _BLOCKED_BUCKETS] + ["Other"]
    metrics: list[HealthMetric] = []
    for label in ordered_labels:
        count = counts.get(label, 0)
        if count == 0:
            continue
        new_24h = fresh.get(label, 0)
        benign = benign_by_label.get(label, False)
        severity = SEVERITY_WARN if (new_24h > 0 and not benign) else SEVERITY_OK
        seen = last_seen.get(label)
        seen_text = seen.date().isoformat() if seen is not None else "unknown"
        metrics.append(
            HealthMetric(
                key=f"blocked_bucket_{_slug(label)}",
                label=label,
                value=str(count),
                severity=severity,
                detail=f"last {seen_text}; +{new_24h} in 24h",
            )
        )
    return metrics


def _slug(label: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in label.lower())


def _recent_failed_queryset() -> QuerySet[CodexInstance]:
    cutoff = timezone.now() - _RECENT_FAILURE_AGE
    return CodexInstance.objects.filter(status=CodexInstance.STATUS_FAILED).filter(
        Q(ended_at__gte=cutoff) | Q(ended_at__isnull=True, started_at__gte=cutoff)
    )


def _recent_failed_count(*needles: str) -> int:
    queryset = _recent_failed_queryset()
    if needles:
        matcher = Q()
        for needle in needles:
            matcher |= Q(error__icontains=needle)
        queryset = queryset.filter(matcher)
    return queryset.count()


def _recent_failure_section() -> HealthSection:
    return HealthSection(
        title="Recent Codex failures (24h)",
        metrics=[
            _count_metric(
                "failed_24h",
                "Failed turns",
                _recent_failed_count,
                detail="Worker rows that reached a failed state in the last 24h.",
            ),
            _count_metric(
                "worker_exited_24h",
                "Worker exited before completion",
                lambda: _recent_failed_count("worker process exited"),
                warn_at=1,
                detail="Downstream of leaked grandchildren (problem 1).",
            ),
            _count_metric(
                "db_lock_24h",
                "State-DB lock / transport closed",
                lambda: _recent_failed_count("TransportClosedError", "database is locked"),
                warn_at=1,
                detail="Codex state-DB lock contention (problem 2).",
            ),
        ],
    )


def _build_health_report() -> HealthReport:
    sections = [
        _worker_scope_section(),
        _host_section(),
        _leak_section(),
        _blocked_bucket_section(),
        _backlog_section(),
        _recent_failure_section(),
    ]
    overall = SEVERITY_OK
    for section in sections:
        for metric in section.metrics:
            if _SEVERITY_RANK.get(metric.severity, 0) > _SEVERITY_RANK[overall]:
                overall = metric.severity
    return HealthReport(
        generated_at=timezone.now().isoformat(timespec="seconds"),
        server_git_hash=server_git_hash(),
        overall_severity=overall,
        sections=sections,
    )


def collect_health_report() -> HealthReport:
    """Build the dashboard report, cached for a short window.

    A single build scans ``/proc`` twice and walks ``~/.hitch`` for its disk
    figure, so repeated or concurrent dashboard loads must not each repeat that
    work. The lock is held across the build so concurrent requests collapse onto
    one computation rather than piling up parallel walks; the result is reused
    for ``_REPORT_CACHE_TTL``. Bypassed under tests so each sees fresh DB state.
    """
    if getattr(settings, "TESTING", False):
        return _build_health_report()
    global _report_cache
    with _report_cache_lock:
        cached = _report_cache
        if cached is not None and timezone.now() - cached[0] < _REPORT_CACHE_TTL:
            return cached[1]
        report = _build_health_report()
        _report_cache = (timezone.now(), report)
        return report
