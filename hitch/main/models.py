"""Database models for the main hitch app.

CodexInstance tracks a detached ``codex app-server`` worker subprocess that
runs exactly one turn for a Codex thread and then exits. The row outlives the
Django process that spawned it so the running turn can be reconciled (status,
events file location) after a server restart.

ApprovalRequest is the cross-process handoff for interactive
``commandExecution`` / ``fileChange`` approvals: the worker creates a
pending row when codex's app-server escalates an action, the Django view
records the user's pick on a POST, and the worker's polling loop picks
the decision up and answers the JSON-RPC request. The row outlives both
sides so the SSE stream and the request handler can race freely without
losing the answer.
"""

from __future__ import annotations

from typing import override

from django.db import models


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
    events_path = models.CharField(max_length=4096)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_STARTING)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    # Set the first time a Stop click delivers SIGTERM to the worker so a
    # second click on the still-active row knows to escalate from "polite
    # SDK interrupt" to SIGKILL. Null means no Stop has been issued yet.
    interrupt_requested_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["thread_id", "-started_at"]),
            models.Index(fields=["status"]),
        ]

    @override
    def __str__(self) -> str:
        return f"CodexInstance(pid={self.pid}, thread_id={self.thread_id}, status={self.status})"


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

    ``decision`` carries the wire string codex's ``ReviewDecision`` enum
    expects — ``approved`` / ``denied`` / ``abort`` — rather than a UI
    label, so the worker can splat it straight into the JSON-RPC reply
    without an extra lookup.
    """

    DECISION_PENDING = ""
    DECISION_APPROVED = "approved"
    DECISION_DENIED = "denied"
    DECISION_ABORT = "abort"

    DECISION_CHOICES = (
        (DECISION_PENDING, "pending"),
        (DECISION_APPROVED, "approved"),
        (DECISION_DENIED, "denied"),
        (DECISION_ABORT, "abort"),
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
