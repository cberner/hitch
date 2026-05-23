"""Database models for the main hitch app.

CodexInstance tracks a detached ``codex app-server`` worker subprocess that
runs exactly one turn for a Codex thread and then exits. The row outlives the
Django process that spawned it so the running turn can be reconciled (status,
events file location) after a server restart.

ApprovalRequest and UserInputRequest are the cross-process handoffs for
interactive browser prompts: the worker creates a pending row when codex's
app-server asks the client for a decision or structured user input, the
Django view records the user's pick on a POST, and the worker's polling loop
picks the answer up and responds to the JSON-RPC request. The rows outlive
both sides so the SSE stream and the request handler can race freely without
losing the answer.
"""

from __future__ import annotations

from typing import ClassVar, override

from django.conf import settings
from django.db import models


class Project(models.Model):
    """A git repository grouping for visible Codex sessions."""

    AUTO_PR_FOLLOW_GLOBAL: ClassVar[str] = "follow_global"
    AUTO_PR_ON: ClassVar[str] = "on"
    AUTO_PR_OFF: ClassVar[str] = "off"
    AUTO_PR_CHOICES: ClassVar[tuple[tuple[str, str], ...]] = (
        (AUTO_PR_FOLLOW_GLOBAL, "Follow global"),
        (AUTO_PR_ON, "On"),
        (AUTO_PR_OFF, "Off"),
    )

    name = models.CharField(max_length=200)
    repo_path = models.CharField(max_length=4096, unique=True)
    git_common_dir = models.CharField(max_length=4096, blank=True, default="")
    auto_pr_mode = models.CharField(
        max_length=16,
        choices=AUTO_PR_CHOICES,
        default=AUTO_PR_FOLLOW_GLOBAL,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name", "repo_path"]

    @override
    def __str__(self) -> str:
        return self.name


class StandingOrder(models.Model):
    """A project-scoped recurring goal that can propose Codex sessions."""

    AMBITION_INCREMENTAL = "incremental"
    AMBITION_MEDIUM = "medium"
    AMBITION_HIGH = "high"
    AMBITION_YOLO = "yolo"
    AMBITION_CHOICES: ClassVar[tuple[tuple[str, str], ...]] = (
        (AMBITION_INCREMENTAL, "Incremental"),
        (AMBITION_MEDIUM, "Medium"),
        (AMBITION_HIGH, "High"),
        (AMBITION_YOLO, "YOLO"),
    )

    CONFIDENCE_MEDIUM = "medium"
    CONFIDENCE_HIGH = "high"
    CONFIDENCE_VERY_HIGH = "very_high"
    CONFIDENCE_CHOICES: ClassVar[tuple[tuple[str, str], ...]] = (
        (CONFIDENCE_MEDIUM, "Medium"),
        (CONFIDENCE_HIGH, "High"),
        (CONFIDENCE_VERY_HIGH, "Very high"),
    )

    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="standing_orders",
    )
    title = models.CharField(max_length=200)
    goal = models.TextField()
    ambition = models.CharField(
        max_length=32,
        choices=AMBITION_CHOICES,
        default=AMBITION_INCREMENTAL,
    )
    confidence_threshold = models.CharField(
        max_length=32,
        choices=CONFIDENCE_CHOICES,
        default=CONFIDENCE_HIGH,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [
            models.Index(fields=["project", "created_at"]),
        ]

    @override
    def __str__(self) -> str:
        return self.title


class ProposedSession(models.Model):
    """A standing-order session proposal awaiting user acceptance."""

    INBOX_KIND_PROPOSAL = "proposal"
    INBOX_KIND_NOTICE = "notice"

    OUTCOME_UNSET = ""
    OUTCOME_ACCEPTED = "accepted"
    OUTCOME_REJECTED = "rejected"
    OUTCOME_DISMISSED = "dismissed"

    INBOX_KIND_CHOICES: ClassVar[tuple[tuple[str, str], ...]] = (
        (INBOX_KIND_PROPOSAL, "Proposal"),
        (INBOX_KIND_NOTICE, "Notice"),
    )

    OUTCOME_CHOICES: ClassVar[tuple[tuple[str, str], ...]] = (
        (OUTCOME_UNSET, "Not set"),
        (OUTCOME_ACCEPTED, "Accepted"),
        (OUTCOME_REJECTED, "Rejected"),
        (OUTCOME_DISMISSED, "Dismissed"),
    )

    standing_order = models.ForeignKey(
        StandingOrder,
        on_delete=models.CASCADE,
        related_name="proposed_sessions",
    )
    source_workflow = models.ForeignKey(
        "SystemWorkflow",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="proposed_sessions",
    )
    title = models.CharField(max_length=200)
    inbox_kind = models.CharField(
        max_length=16,
        choices=INBOX_KIND_CHOICES,
        default=INBOX_KIND_PROPOSAL,
    )
    summary = models.TextField(blank=True, default="")
    confidence = models.CharField(
        max_length=32,
        choices=StandingOrder.CONFIDENCE_CHOICES,
        default=StandingOrder.CONFIDENCE_MEDIUM,
    )
    relevant_files = models.JSONField(default=list, blank=True)
    candidate_session = models.ForeignKey(
        "SessionMetadata",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="standing_order_candidate_proposals",
    )
    judge_session = models.ForeignKey(
        "SessionMetadata",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="standing_order_judge_proposals",
    )
    accepted_session = models.ForeignKey(
        "SessionMetadata",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="accepted_standing_order_proposals",
    )
    outcome_status = models.CharField(
        max_length=32,
        choices=OUTCOME_CHOICES,
        blank=True,
        default=OUTCOME_UNSET,
    )
    outcome_notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [
            models.Index(fields=["standing_order", "created_at"]),
            models.Index(fields=["outcome_status", "created_at"]),
        ]

    @override
    def __str__(self) -> str:
        return self.title


class SessionMetadata(models.Model):
    """Local metadata for a Codex thread that Hitch does not own on disk."""

    thread_id = models.CharField(max_length=128, unique=True)
    cwd = models.CharField(max_length=4096, blank=True, default="")
    project = models.ForeignKey(
        Project,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="sessions",
    )
    project_cleared = models.BooleanField(default=False)
    auto_pr_enabled = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["project", "-updated_at"]),
        ]

    @override
    def __str__(self) -> str:
        return f"SessionMetadata(thread_id={self.thread_id}, project={self.project_id})"


class SessionDemo(models.Model):
    """Active web demo target for a Codex session."""

    STATUS_REQUESTED = "requested"
    STATUS_PREPARING = "preparing"
    STATUS_ACTIVE = "active"
    STATUS_STOPPED = "stopped"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = (
        (STATUS_REQUESTED, "requested"),
        (STATUS_PREPARING, "preparing"),
        (STATUS_ACTIVE, "active"),
        (STATUS_STOPPED, "stopped"),
        (STATUS_FAILED, "failed"),
    )

    thread_id = models.CharField(max_length=128, unique=True)
    host = models.CharField(max_length=255, default="127.0.0.1")
    port = models.PositiveIntegerField()
    container_id = models.CharField(max_length=128, blank=True, default="")
    container_name = models.CharField(max_length=128, blank=True, default="")
    runtime = models.CharField(max_length=32, default="podman")
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_ACTIVE)
    last_error = models.TextField(blank=True, default="")
    generation = models.PositiveIntegerField(default=0)
    registration_token = models.CharField(max_length=128, blank=True, default="")
    logs = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["thread_id", "status"],
                name="sessiondemo_thread_status_idx",
            ),
            models.Index(fields=["status"], name="sessiondemo_status_idx"),
        ]

    @override
    def __str__(self) -> str:
        return (
            f"SessionDemo(thread_id={self.thread_id}, "
            f"target={self.host}:{self.port}, status={self.status})"
        )


class CodexInstance(models.Model):
    """One row per spawned Codex worker subprocess.

    A worker is detached from the Django parent (``start_new_session=True``) so
    it survives a server restart. We identify a worker by ``pid`` plus
    ``started_at`` — PIDs alone are recycled, but the pair is effectively
    unique on the host for the lifetime of the worker.
    """

    STATUS_STARTING = "starting"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = (
        (STATUS_STARTING, "starting"),
        (STATUS_RUNNING, "running"),
        (STATUS_COMPLETED, "completed"),
        (STATUS_FAILED, "failed"),
    )
    PURPOSE_USER = "user"
    PURPOSE_SYSTEM_AGENT = "system_agent"
    PURPOSE_SYSTEM_FEEDBACK = "system_feedback"

    PURPOSE_CHOICES = (
        (PURPOSE_USER, "user"),
        (PURPOSE_SYSTEM_AGENT, "system agent"),
        (PURPOSE_SYSTEM_FEEDBACK, "system feedback"),
    )

    pid = models.IntegerField()
    thread_id = models.CharField(max_length=128, db_index=True)
    cwd = models.CharField(max_length=4096)
    # Stored on the row instead of passed as a CLI argument so prompts
    # beginning with a dash can't be reinterpreted as argparse options and
    # so a worker can be re-spawned from this row alone if needed.
    prompt = models.TextField(blank=True, default="")
    # Thread-scoped developer instructions from the settings dialog. Each
    # detached worker re-supplies them on resume so the turn does not rely on
    # app-server in-memory state from the request process that created the
    # thread.
    developer_instructions = models.TextField(blank=True, default="")
    base_instructions = models.TextField(blank=True, default="")
    enable_memories = models.BooleanField(default=False)
    model = models.CharField(max_length=256, blank=True, default="")
    reasoning_effort = models.CharField(max_length=32, blank=True, default="")
    sandbox_policy = models.CharField(max_length=32, blank=True, default="")
    approval_mode = models.CharField(max_length=32, blank=True, default="")
    plan_mode = models.BooleanField(default=False)
    auto_pr_enabled = models.BooleanField(default=False)
    auto_pr_triggered_at = models.DateTimeField(null=True, blank=True)
    events_path = models.CharField(max_length=4096)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_STARTING)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    # Set the first time a Stop click delivers SIGTERM to the worker so a
    # second click on the still-active row knows to escalate from "polite
    # SDK interrupt" to SIGKILL. Null means no Stop has been issued yet.
    interrupt_requested_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    purpose = models.CharField(
        max_length=32, choices=PURPOSE_CHOICES, default=PURPOSE_USER
    )
    workflow_id = models.BigIntegerField(null=True, blank=True, db_index=True)
    agent_kind = models.CharField(max_length=64, blank=True, default="")
    display_author = models.CharField(max_length=128, blank=True, default="")
    output_schema = models.JSONField(default=None, blank=True, null=True)
    user_message_index = models.PositiveIntegerField(null=True, blank=True, db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["thread_id", "-started_at"]),
            models.Index(fields=["status"]),
            models.Index(fields=["purpose"]),
        ]

    @override
    def __str__(self) -> str:
        return f"CodexInstance(pid={self.pid}, thread_id={self.thread_id}, status={self.status})"


class SystemWorkflow(models.Model):
    """Durable state for Hitch-managed system-agent workflows."""

    KIND_PR_QA = "pr_qa"
    KIND_STANDING_ORDER_RUN = "standing_order_run"

    STATUS_RUNNING = "running"
    STATUS_BLOCKED = "blocked"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_MAX_ITERATIONS_REACHED = "max_iterations_reached"

    STATUS_CHOICES = (
        (STATUS_RUNNING, "running"),
        (STATUS_BLOCKED, "blocked"),
        (STATUS_COMPLETED, "completed"),
        (STATUS_FAILED, "failed"),
        (STATUS_MAX_ITERATIONS_REACHED, "max iterations reached"),
    )

    kind = models.CharField(max_length=64)
    main_thread_id = models.CharField(max_length=128, db_index=True)
    cwd = models.CharField(max_length=4096)
    status = models.CharField(max_length=64, choices=STATUS_CHOICES, default=STATUS_RUNNING)
    step = models.CharField(max_length=64, blank=True, default="")
    iteration = models.PositiveIntegerField(default=0)
    max_iterations = models.PositiveIntegerField(default=3)
    state = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["kind", "main_thread_id", "status"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["kind", "main_thread_id"],
                condition=models.Q(status="running"),
                name="uniq_running_system_workflow",
            )
        ]

    @override
    def __str__(self) -> str:
        return (
            f"SystemWorkflow(kind={self.kind}, main_thread_id={self.main_thread_id}, "
            f"status={self.status})"
        )


class SystemAgentRun(models.Model):
    """One hidden Hitch system-agent turn inside a workflow."""

    STATUS_STARTING = "starting"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = (
        (STATUS_STARTING, "starting"),
        (STATUS_RUNNING, "running"),
        (STATUS_COMPLETED, "completed"),
        (STATUS_FAILED, "failed"),
    )

    workflow = models.ForeignKey(
        SystemWorkflow, on_delete=models.CASCADE, related_name="agent_runs"
    )
    agent_kind = models.CharField(max_length=64)
    thread_id = models.CharField(max_length=128, db_index=True)
    instance = models.ForeignKey(
        CodexInstance,
        on_delete=models.CASCADE,
        related_name="system_agent_runs",
    )
    status = models.CharField(max_length=64, choices=STATUS_CHOICES, default=STATUS_STARTING)
    input = models.JSONField(default=dict, blank=True)
    output = models.JSONField(default=dict, blank=True)
    raw_output = models.TextField(blank=True, default="")
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["workflow", "-created_at"]),
            models.Index(fields=["agent_kind", "status"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["instance"],
                name="uniq_system_agent_run_instance",
            )
        ]

    @override
    def __str__(self) -> str:
        return (
            f"SystemAgentRun(agent_kind={self.agent_kind}, thread_id={self.thread_id}, "
            f"status={self.status})"
        )


class ApprovalRequest(models.Model):
    """One row per ``item/{commandExecution,fileChange}/requestApproval``.

    The worker subprocess receives the JSON-RPC request on its SDK reader
    thread, creates this row in the ``pending`` state, and then blocks
    polling the row until the Django view records a decision via
    ``POST /approval/<id>/``. Storing the request in the database (rather
    than in the worker's memory) is what makes the cross-process race
    safe: the SSE stream surfaces the row id to the browser, the user
    POSTs a decision, and the worker's polling loop wakes up and answers
    the SDK call with the recorded value.

    ``decision`` carries the app-server wire string — ``accept`` /
    ``decline`` / ``cancel`` — rather than a UI label, so the worker can
    splat it straight into the JSON-RPC reply without an extra lookup.
    """

    DECISION_PENDING = ""
    DECISION_ACCEPT = "accept"
    DECISION_DECLINE = "decline"
    DECISION_CANCEL = "cancel"
    # Backwards-compatible names for call sites that still talk in UI-ish terms.
    DECISION_APPROVED = DECISION_ACCEPT
    DECISION_DENIED = DECISION_DECLINE
    DECISION_ABORT = DECISION_CANCEL
    LEGACY_DECISION_ALIASES: ClassVar[dict[str, str]] = {
        "approved": DECISION_ACCEPT,
        "denied": DECISION_DECLINE,
        "abort": DECISION_CANCEL,
    }

    DECISION_CHOICES = (
        (DECISION_PENDING, "pending"),
        (DECISION_ACCEPT, "accept"),
        (DECISION_DECLINE, "decline"),
        (DECISION_CANCEL, "cancel"),
    )

    instance = models.ForeignKey(
        CodexInstance, on_delete=models.CASCADE, related_name="approvals"
    )
    # The full JSON-RPC method name (``item/commandExecution/requestApproval``
    # or ``item/fileChange/requestApproval``) — kept as the raw wire string
    # so a future SDK extension that adds a third approval kind doesn't have
    # to teach us a new enum, just opt in by name.
    method = models.CharField(max_length=128)
    params = models.JSONField(default=dict, blank=True)
    decision = models.CharField(
        max_length=32, choices=DECISION_CHOICES, default=DECISION_PENDING, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    decided_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["instance", "-created_at"]),
        ]

    @override
    def __str__(self) -> str:
        decision = self.decision or "pending"
        return (
            f"ApprovalRequest(pk={self.pk}, instance={self.instance_id}, "
            f"method={self.method}, decision={decision})"
        )

    @classmethod
    def normalize_decision(cls, decision: str) -> str:
        return cls.LEGACY_DECISION_ALIASES.get(decision, decision)


class UserInputRequest(models.Model):
    """One row per app-server ``request_user_input`` prompt.

    Plan-mode turns can ask the client to collect structured answers from the
    human before continuing. The SDK delivers that as a synchronous
    server-to-client JSON-RPC request, so the worker needs the same durable
    browser handoff pattern used by approvals: write a pending row, emit an
    SSE event, block until the browser records a JSON response, then return it
    to app-server.
    """

    instance = models.ForeignKey(
        CodexInstance, on_delete=models.CASCADE, related_name="input_requests"
    )
    method = models.CharField(max_length=128)
    params = models.JSONField(default=dict, blank=True)
    response = models.JSONField(default=None, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    responded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["instance", "-created_at"]),
        ]

    @override
    def __str__(self) -> str:
        state = "answered" if self.response is not None else "pending"
        return (
            f"UserInputRequest(pk={self.pk}, instance={self.instance_id}, "
            f"method={self.method}, state={state})"
        )


class ArchivedSessionTokenUsage(models.Model):
    """Cached token usage for archived Codex sessions."""

    thread_id = models.CharField(max_length=128, unique=True)
    rollout_path = models.CharField(max_length=4096, blank=True, default="")
    rollout_mtime_ns = models.PositiveBigIntegerField(default=0)
    input_tokens = models.PositiveBigIntegerField(default=0)
    cached_input_tokens = models.PositiveBigIntegerField(default=0)
    output_tokens = models.PositiveBigIntegerField(default=0)
    total_tokens = models.PositiveBigIntegerField(default=0)
    context_tokens = models.PositiveBigIntegerField(default=0)
    model_context_window = models.PositiveBigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @override
    def __str__(self) -> str:
        return (
            f"ArchivedSessionTokenUsage(thread_id={self.thread_id}, "
            f"total_tokens={self.total_tokens})"
        )


class UserSettings(models.Model):
    """Per-account mirror of the settings that guests keep in signed cookies."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="hitch_settings"
    )
    model = models.CharField(max_length=256, blank=True, default="")
    reasoning_effort = models.CharField(max_length=32, blank=True, default="")
    sandbox_policy = models.CharField(max_length=32, blank=True, default="")
    approval_mode = models.CharField(max_length=32, blank=True, default="auto_review")
    coding_agent = models.CharField(max_length=32, blank=True, default="")
    extra_system_prompt = models.TextField(blank=True, default="")
    use_worktrees = models.BooleanField(default=False)
    auto_pr_enabled = models.BooleanField(default=False)
    show_archived_sessions = models.BooleanField(default=False)
    last_selected_repo = models.CharField(max_length=4096, blank=True, default="")
    selected_project = models.ForeignKey(
        Project,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="selected_by_settings",
    )
    enable_memories = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    @override
    def __str__(self) -> str:
        return f"UserSettings(user={self.user_id})"
