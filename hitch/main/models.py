"""Database models for the main hitch app.

CodexInstance tracks a detached ``codex app-server`` worker subprocess that
runs exactly one turn for a Codex thread and then exits. The row outlives the
Django process that spawned it so the running turn can be reconciled (status,
events file location) after a server restart.
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
    events_path = models.CharField(max_length=4096)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default=STATUS_STARTING)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["thread_id", "-started_at"]),
            models.Index(fields=["status"]),
        ]

    @override
    def __str__(self) -> str:
        return f"CodexInstance(pid={self.pid}, thread_id={self.thread_id}, status={self.status})"
