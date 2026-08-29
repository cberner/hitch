import json
import tempfile
from pathlib import Path
from typing import Any

from django.test import SimpleTestCase

from hitch.main.runtime import codex_events


def _event(
    method: str,
    payload: dict[str, object],
    *,
    event_seq: int | None = None,
    recorded_at: int | None = None,
) -> str:
    event: dict[str, object] = {"method": method, "payload": payload}
    if event_seq is not None:
        event["eventSeq"] = event_seq
    if recorded_at is not None:
        event["recordedAt"] = recorded_at
    return json.dumps(event)


class PruneDiffEventsTests(SimpleTestCase):
    def test_removes_diff_events_and_preserves_other_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = Path(raw) / "events.jsonl"
            events_path.write_text(
                "\n".join(
                    [
                        _event("item/started", {"id": "item-1"}),
                        _event(
                            codex_events.TURN_DIFF_UPDATED_METHOD,
                            {"diff": "large diff"},
                        ),
                        _event("turn/completed", {"status": "completed"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            events_path.chmod(0o600)

            freed = codex_events.prune_diff_events(events_path)
            compacted = events_path.read_text(encoding="utf-8")
            compacted_mode = events_path.stat().st_mode & 0o777
            temporary_files = list(Path(raw).glob("*.compact"))

        self.assertGreater(freed, 0)
        self.assertNotIn("turn/diff/updated", compacted)
        self.assertNotIn("large diff", compacted)
        self.assertIn("item/started", compacted)
        self.assertIn("turn/completed", compacted)
        self.assertEqual(compacted_mode, 0o600)
        self.assertEqual(temporary_files, [])

    def test_noop_keeps_log_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = Path(raw) / "events.jsonl"
            original = _event("turn/completed", {"status": "completed"}) + "\n"
            events_path.write_text(original, encoding="utf-8")

            freed = codex_events.prune_diff_events(events_path)

            self.assertEqual(freed, 0)
            self.assertEqual(events_path.read_text(encoding="utf-8"), original)


class LatestGoalFromEventPathsTests(SimpleTestCase):
    def test_applies_updates_and_clears_in_event_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            first = Path(raw) / "first.jsonl"
            second = Path(raw) / "second.jsonl"
            first.write_text(
                "\n".join(
                    [
                        _event(
                            codex_events.GOAL_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "goal": {"objective": "Initial cleanup"},
                            },
                        ),
                        _event(codex_events.GOAL_CLEARED_METHOD, {"threadId": "thread-1"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                _event(
                    codex_events.GOAL_UPDATED_METHOD,
                    {
                        "threadId": "thread-1",
                        "goal": {"objective": "Ship the status strip"},
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            goal = codex_events.latest_goal_from_event_paths([first, second], thread_id="thread-1")

        self.assertEqual(goal, "Ship the status strip")

    def test_prefers_recorded_time_across_overlapping_workers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            older_worker = Path(raw) / "older-worker.jsonl"
            newer_worker = Path(raw) / "newer-worker.jsonl"
            older_worker.write_text(
                _event(
                    codex_events.GOAL_CLEARED_METHOD,
                    {"threadId": "thread-1"},
                    recorded_at=30,
                )
                + "\n",
                encoding="utf-8",
            )
            newer_worker.write_text(
                _event(
                    codex_events.GOAL_UPDATED_METHOD,
                    {
                        "threadId": "thread-1",
                        "goal": {"objective": "Stale newer worker objective"},
                    },
                    recorded_at=20,
                )
                + "\n",
                encoding="utf-8",
            )

            goal = codex_events.latest_goal_from_event_paths([older_worker, newer_worker], thread_id="thread-1")

        self.assertIsNone(goal)

    def test_uses_event_sequence_when_recorded_time_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            older_worker = Path(raw) / "older-worker.jsonl"
            newer_worker = Path(raw) / "newer-worker.jsonl"
            older_worker.write_text(
                _event(
                    codex_events.GOAL_CLEARED_METHOD,
                    {"threadId": "thread-1"},
                    event_seq=30,
                )
                + "\n",
                encoding="utf-8",
            )
            newer_worker.write_text(
                _event(
                    codex_events.GOAL_UPDATED_METHOD,
                    {
                        "threadId": "thread-1",
                        "goal": {"objective": "Stale newer worker objective"},
                    },
                    event_seq=20,
                )
                + "\n",
                encoding="utf-8",
            )

            goal = codex_events.latest_goal_from_event_paths([older_worker, newer_worker], thread_id="thread-1")

        self.assertIsNone(goal)

    def test_latest_goal_tokens_uses_latest_goal_update(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            codex_events.GOAL_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "goal": {
                                    "objective": "Initial review",
                                    "tokensUsed": 10,
                                },
                            },
                            event_seq=1,
                        ),
                        _event(
                            codex_events.GOAL_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "goal": {
                                    "objective": "Current review",
                                    "tokens_used": 2500,
                                },
                            },
                            event_seq=2,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            tokens = codex_events.latest_goal_tokens_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertEqual(tokens, 2500)

    def test_latest_goal_tokens_for_instance_handles_missing_instance(self) -> None:
        self.assertIsNone(codex_events.latest_goal_tokens_for_instance(None))


    def test_accepts_recorded_at_alias_and_ignores_bad_order_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "method": codex_events.GOAL_UPDATED_METHOD,
                                "payload": {
                                    "thread_id": "thread-1",
                                    "goal": {"objective": "Alias objective"},
                                },
                                "recorded_at": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "method": codex_events.GOAL_CLEARED_METHOD,
                                "payload": {"threadId": "thread-1"},
                                "recordedAt": True,
                                "eventSeq": True,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            goal = codex_events.latest_goal_from_event_paths([path], thread_id="thread-1")

        self.assertEqual(goal, "Alias objective")

    def test_ignores_other_threads_and_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        "{not json",
                        _event(
                            codex_events.GOAL_UPDATED_METHOD,
                            {
                                "threadId": "other-thread",
                                "goal": {"objective": "Wrong thread"},
                            },
                        ),
                        _event("item/started", {"threadId": "thread-1"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            goal = codex_events.latest_goal_from_event_paths([path], thread_id="thread-1")

        self.assertIsNone(goal)

    def test_ignores_malformed_goal_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(codex_events.GOAL_UPDATED_METHOD, {"threadId": "thread-1"}),
                        _event(
                            codex_events.GOAL_UPDATED_METHOD,
                            {"threadId": "thread-1", "goal": []},
                        ),
                        _event(
                            codex_events.GOAL_UPDATED_METHOD,
                            {"threadId": "thread-1", "goal": {"objective": 123}},
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            goal = codex_events.latest_goal_from_event_paths([path], thread_id="thread-1")

        self.assertIsNone(goal)

    def test_missing_file_and_empty_objective_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                _event(
                    codex_events.GOAL_UPDATED_METHOD,
                    {
                        "thread_id": "thread-1",
                        "goal": {"objective": "   "},
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            goal = codex_events.latest_goal_from_event_paths([Path(raw) / "missing.jsonl", path], thread_id="thread-1")

        self.assertIsNone(goal)


class LatestTaskPlanFromEventPathsTests(SimpleTestCase):
    def test_returns_latest_visible_task_plan_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "explanation": "Older plan",
                                "plan": [{"step": "Inspect", "status": "pending"}],
                            },
                            event_seq=1,
                        ),
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "explanation": "Current plan",
                                "plan": [
                                    {"step": "Inspect", "status": "completed"},
                                    {"step": "Patch", "status": "in_progress"},
                                    {"step": "Verify", "status": "unexpected"},
                                    {"step": "Handle malformed status", "status": []},
                                ],
                            },
                            event_seq=2,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_task_plan_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.explanation, "Current plan")
        self.assertEqual(snapshot.order, (0, 2, 2))
        self.assertEqual(
            [(step.step, step.status) for step in snapshot.steps],
            [
                ("Inspect", "completed"),
                ("Patch", "inProgress"),
                ("Verify", "pending"),
                ("Handle malformed status", "pending"),
            ],
        )

    def test_empty_latest_task_plan_clears_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "plan": [{"step": "Inspect", "status": "pending"}],
                            },
                            event_seq=1,
                        ),
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {"threadId": "thread-1", "plan": []},
                            event_seq=2,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_task_plan_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.explanation, "")
        self.assertEqual(snapshot.steps, ())
        self.assertEqual(snapshot.order, (0, 2, 2))

    def test_uses_fallback_order_for_unsequenced_task_plan_logs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "plan": [{"step": "Stale fallback task"}],
                            },
                        ),
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {
                                "threadId": "thread-1",
                                "plan": [{"step": "Latest fallback task"}],
                            },
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_task_plan_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.steps[0].step, "Latest fallback task")
        self.assertEqual(snapshot.order, (0, 0, 2))

    def test_prefers_recorded_time_across_task_plan_logs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            older_worker = Path(raw) / "older-worker.jsonl"
            newer_worker = Path(raw) / "newer-worker.jsonl"
            older_worker.write_text(
                _event(
                    codex_events.TASK_PLAN_UPDATED_METHOD,
                    {
                        "threadId": "thread-1",
                        "plan": [{"step": "Actually latest", "status": "in_progress"}],
                    },
                    recorded_at=30,
                )
                + "\n",
                encoding="utf-8",
            )
            newer_worker.write_text(
                _event(
                    codex_events.TASK_PLAN_UPDATED_METHOD,
                    {
                        "threadId": "thread-1",
                        "plan": [{"step": "Stale worker", "status": "pending"}],
                    },
                    recorded_at=20,
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_task_plan_from_event_paths(
                [older_worker, newer_worker],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.steps[0].step, "Actually latest")
        self.assertEqual(snapshot.order, (30, 0, 1))

    def test_ignores_other_threads_and_malformed_task_plan_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        "{not json",
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {
                                "threadId": "other-thread",
                                "plan": [{"step": "Wrong thread", "status": "pending"}],
                            },
                        ),
                        _event(
                            codex_events.TASK_PLAN_UPDATED_METHOD,
                            {"threadId": "thread-1", "plan": [{"step": 7}]},
                        ),
                        json.dumps(
                            {
                                "method": codex_events.TASK_PLAN_UPDATED_METHOD,
                                "payload": [],
                            }
                        ),
                        _event("item/started", {"threadId": "thread-1"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_task_plan_from_event_paths(
                [path],
                thread_id="thread-1",
            )

            self.assertIsNone(snapshot)


class PrSnapshotFromObservationTurnsTests(SimpleTestCase):
    def test_non_pr_github_calls_do_not_establish_pr_identity(self) -> None:
        snapshot = codex_events.pr_snapshot_from_observation_turns(
            [
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_pr",
                            "arguments": {
                                "repo_full_name": "cberner/hitch",
                                "pr_number": 93,
                            },
                            "result": {
                                "structuredContent": {
                                    "url": ("https://github.com/cberner/hitch/pull/93"),
                                    "state": "closed",
                                }
                            },
                        },
                    ),
                )
            ]
        )

        self.assertIsNone(snapshot)

    def test_non_pr_github_calls_update_current_pr_only(self) -> None:
        current_url = "https://github.com/cberner/hitch/pull/94"
        unrelated_url = "https://github.com/cberner/hitch/pull/93"

        snapshot = codex_events.pr_snapshot_from_observation_turns(
            [
                codex_events.PrObservationTurn(
                    is_pr_prompt=True,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_create_pull_request",
                            "result": {
                                "structuredContent": {
                                    "url": current_url,
                                    "state": "open",
                                }
                            },
                        },
                    ),
                ),
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_pr",
                            "arguments": {
                                "repo_full_name": "cberner/hitch",
                                "pr_number": 93,
                            },
                            "result": {
                                "structuredContent": {
                                    "url": unrelated_url,
                                    "state": "closed",
                                }
                            },
                        },
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_pr",
                            "arguments": {
                                "repo_full_name": "cberner/hitch",
                                "pr_number": 94,
                            },
                            "result": {
                                "structuredContent": {
                                    "url": current_url,
                                    "state": "closed",
                                    "merged": False,
                                }
                            },
                        },
                    ),
                ),
            ]
        )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["url"], current_url)
        self.assertEqual(snapshot["state"], "closed")

    def test_completed_non_pr_lifecycle_turn_clears_previous_pr_identity(self) -> None:
        result = codex_events.pr_observation_result_from_turns(
            [
                codex_events.PrObservationTurn(
                    is_pr_prompt=True,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_pr",
                            "result": {
                                "structuredContent": {
                                    "url": ("https://github.com/cberner/hitch/pull/94"),
                                    "state": "closed",
                                }
                            },
                        },
                    ),
                ),
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(),
                    has_lifecycle_activity=True,
                ),
            ]
        )

        self.assertIsNone(result.snapshot)
        self.assertTrue(result.superseded_by_lifecycle)

    def test_irrelevant_non_pr_mcp_call_does_not_prevent_lifecycle_clear(self) -> None:
        result = codex_events.pr_observation_result_from_turns(
            [
                codex_events.PrObservationTurn(
                    is_pr_prompt=True,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_pr",
                            "result": {
                                "structuredContent": {
                                    "url": ("https://github.com/cberner/hitch/pull/94"),
                                    "state": "closed",
                                }
                            },
                        },
                    ),
                ),
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "linear",
                            "tool": "create_issue",
                            "result": {"structuredContent": {"identifier": "ENG-123"}},
                        },
                    ),
                    has_lifecycle_activity=True,
                ),
            ]
        )

        self.assertIsNone(result.snapshot)
        self.assertTrue(result.superseded_by_lifecycle)

    def test_unrelated_non_pr_ci_check_does_not_prevent_lifecycle_clear(self) -> None:
        result = codex_events.pr_observation_result_from_turns(
            [
                codex_events.PrObservationTurn(
                    is_pr_prompt=True,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_pr",
                            "result": {
                                "structuredContent": {
                                    "url": ("https://github.com/cberner/hitch/pull/94"),
                                    "state": "open",
                                    "head_sha": "abc123",
                                }
                            },
                        },
                    ),
                ),
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_get_commit_combined_status",
                            "arguments": {
                                "repo_full_name": "cberner/hitch",
                                "commit_sha": "unrelated",
                            },
                            "result": {"structuredContent": {"statuses": [{"state": "success"}]}},
                        },
                    ),
                    has_lifecycle_activity=True,
                ),
            ]
        )

        self.assertIsNone(result.snapshot)
        self.assertTrue(result.superseded_by_lifecycle)

    def test_non_pr_ci_check_for_current_pr_keeps_pr_epoch(self) -> None:
        result = codex_events.pr_observation_result_from_turns(
            [
                codex_events.PrObservationTurn(
                    is_pr_prompt=True,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_pr",
                            "result": {
                                "structuredContent": {
                                    "url": ("https://github.com/cberner/hitch/pull/94"),
                                    "state": "open",
                                    "head_sha": "abc123",
                                }
                            },
                        },
                    ),
                ),
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_get_commit_combined_status",
                            "arguments": {
                                "repo_full_name": "cberner/hitch",
                                "commit_sha": "abc123",
                            },
                            "result": {"structuredContent": {"statuses": [{"state": "success"}]}},
                        },
                    ),
                    has_lifecycle_activity=True,
                ),
            ]
        )

        self.assertIsNotNone(result.snapshot)
        assert result.snapshot is not None
        self.assertEqual(result.snapshot["ci_status"], "success")
        self.assertEqual(result.snapshot["latest_commit_sha"], "abc123")
        self.assertFalse(result.superseded_by_lifecycle)

    def _pr_open_turn(self) -> "codex_events.PrObservationTurn":
        return codex_events.PrObservationTurn(
            is_pr_prompt=True,
            is_completed=True,
            items=(
                {
                    "type": "mcpToolCall",
                    "server": "codex_apps",
                    "tool": "github_fetch_pr",
                    "result": {
                        "structuredContent": {
                            "url": "https://github.com/cberner/hitch/pull/94",
                            "state": "open",
                            "head_sha": "abc123",
                        }
                    },
                },
            ),
        )

    @staticmethod
    def _fetch_run_jobs_item(run_id: int) -> dict[str, object]:
        return {
            "type": "mcpToolCall",
            "server": "codex_apps",
            "tool": "github_fetch_workflow_run_jobs",
            "arguments": {"run_id": run_id},
            "result": {
                "structuredContent": {
                    "jobs": [
                        {
                            "name": "test-suite",
                            "status": "completed",
                            "conclusion": "failure",
                        }
                    ]
                }
            },
        }

    def test_fetch_workflow_run_jobs_for_current_pr_keeps_pr_epoch(self) -> None:
        # ``fetch_workflow_run_jobs`` carries only a ``run_id`` -- no PR
        # identity and no commit SHA. A follow-up turn that drills into a run
        # already seen for the current PR (its id captured from a commit-
        # correlated ``fetch_commit_workflow_runs``) must stay attributed to
        # that PR; treating it as unrelated work would wipe the live PR epoch
        # and drop the PR-follow-up workflow (and the CI failure) from view.
        result = codex_events.pr_observation_result_from_turns(
            [
                self._pr_open_turn(),
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(
                        {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_fetch_commit_workflow_runs",
                            "arguments": {
                                "repo_full_name": "cberner/hitch",
                                "commit_sha": "abc123",
                            },
                            "result": {
                                "structuredContent": {
                                    "workflow_runs": [
                                        {
                                            "id": 42,
                                            "status": "completed",
                                            "conclusion": "failure",
                                        }
                                    ]
                                }
                            },
                        },
                        self._fetch_run_jobs_item(42),
                    ),
                    has_lifecycle_activity=True,
                ),
            ]
        )

        self.assertIsNotNone(result.snapshot)
        assert result.snapshot is not None
        self.assertEqual(result.snapshot["pr_number"], 94)
        self.assertEqual(result.snapshot["ci_status"], "failure")
        self.assertEqual(result.snapshot["failing_jobs"], ["test-suite"])
        self.assertFalse(result.superseded_by_lifecycle)

    def test_fetch_workflow_run_jobs_for_unknown_run_supersedes_epoch(self) -> None:
        # A job check for a ``run_id`` never seen among the current PR's runs
        # carries no correlation signal, so it must NOT be attributed to the
        # current PR -- it is unrelated work that supersedes the epoch.
        result = codex_events.pr_observation_result_from_turns(
            [
                self._pr_open_turn(),
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    items=(self._fetch_run_jobs_item(999),),
                    has_lifecycle_activity=True,
                ),
            ]
        )

        self.assertIsNone(result.snapshot)
        self.assertTrue(result.superseded_by_lifecycle)

    def test_run_ids_are_cleared_when_pr_head_advances(self) -> None:
        # After a new push advances the head, run ids captured for the old head
        # must not keep correlating: a later drill into the superseded run must
        # not overwrite the new head's CI with the old commit's failure.
        def runs_item(commit_sha: str, run_id: int, conclusion: str) -> dict[str, object]:
            return {
                "type": "mcpToolCall",
                "server": "codex_apps",
                "tool": "github_fetch_commit_workflow_runs",
                "arguments": {"commit_sha": commit_sha},
                "result": {
                    "structuredContent": {
                        "workflow_runs": [
                            {
                                "id": run_id,
                                "status": "completed",
                                "conclusion": conclusion,
                            }
                        ]
                    }
                },
            }

        def head_item(head_sha: str) -> dict[str, object]:
            return {
                "type": "mcpToolCall",
                "server": "codex_apps",
                "tool": "github_fetch_pr",
                "result": {
                    "structuredContent": {
                        "url": "https://github.com/cberner/hitch/pull/94",
                        "state": "open",
                        "head_sha": head_sha,
                    }
                },
            }

        turns = [
            codex_events.PrObservationTurn(is_pr_prompt=True, is_completed=True, items=(head_item("head1"),)),
            codex_events.PrObservationTurn(
                is_pr_prompt=False,
                is_completed=True,
                has_lifecycle_activity=True,
                items=(
                    runs_item("head1", 42, "failure"),
                    self._fetch_run_jobs_item(42),
                ),
            ),
            # A push advances the head and its CI is green.
            codex_events.PrObservationTurn(
                is_pr_prompt=False,
                is_completed=True,
                has_lifecycle_activity=True,
                items=(head_item("head2"), runs_item("head2", 77, "success")),
            ),
        ]

        result = codex_events.pr_observation_result_from_turns(turns)
        self.assertIsNotNone(result.snapshot)
        assert result.snapshot is not None
        self.assertEqual(result.snapshot["head_sha"], "head2")
        self.assertEqual(result.snapshot["ci_status"], "success")
        self.assertEqual(result.snapshot["workflow_run_ids"], [77])

        # A later turn drilling into the superseded run 42 no longer correlates.
        stale_result = codex_events.pr_observation_result_from_turns(
            [
                *turns,
                codex_events.PrObservationTurn(
                    is_pr_prompt=False,
                    is_completed=True,
                    has_lifecycle_activity=True,
                    items=(self._fetch_run_jobs_item(42),),
                ),
            ]
        )
        self.assertIsNone(stale_result.snapshot)
        self.assertTrue(stale_result.superseded_by_lifecycle)


class LatestPrSnapshotFromEventPathsTests(SimpleTestCase):
    def test_recovers_latest_pr_handoff_from_github_mcp_results(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "arguments": {"repository_full_name": "cberner/hitch"},
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/169"),
                                            "number": 169,
                                            "state": "open",
                                            "merged": False,
                                            "mergeable": True,
                                            "head": "feature",
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_list_pull_request_review_threads",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 169,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "review_threads": [
                                                {
                                                    "id": "resolved",
                                                    "is_resolved": True,
                                                    "path": "a.py",
                                                },
                                                {
                                                    "id": "open",
                                                    "is_resolved": False,
                                                    "is_outdated": False,
                                                    "path": "b.py",
                                                    "line": 12,
                                                    "comments": [],
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_commit_workflow_runs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "commit_sha": "abc123",
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "workflow_runs": [
                                                {
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                }
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=30,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["url"], "https://github.com/cberner/hitch/pull/169")
        self.assertEqual(snapshot["repository_full_name"], "cberner/hitch")
        self.assertEqual(snapshot["pr_number"], 169)
        self.assertEqual(snapshot["head_sha"], "abc123")
        self.assertEqual(snapshot["unresolved_thread_count"], 1)
        self.assertEqual(snapshot["ci_status"], "success")
        self.assertEqual(snapshot["latest_commit_sha"], "abc123")

    def test_latest_pr_identity_replaces_stale_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/168"),
                                            "number": 168,
                                            "state": "open",
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/169"),
                                            "number": 169,
                                            "state": "open",
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["pr_number"], 169)
        self.assertEqual(snapshot["url"], "https://github.com/cberner/hitch/pull/169")

    def test_pr_identity_from_followup_tool_replaces_stale_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/168"),
                                            "number": 168,
                                            "head": "stale-branch",
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_pr_comments",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 169,
                                    },
                                    "result": {"structuredContent": {"comments": [{"body": "new PR feedback"}]}},
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["repository_full_name"], "cberner/hitch")
        self.assertEqual(snapshot["pr_number"], 169)
        self.assertEqual(snapshot["comment_count"], 1)
        self.assertNotIn("url", snapshot)
        self.assertNotIn("head", snapshot)

    def test_review_signal_prefers_requested_changes_over_approval_and_reactions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_list_pull_request_reviews",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 169,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "reviews": [
                                                {
                                                    "author": {"login": "approver"},
                                                    "state": "APPROVED",
                                                },
                                                {
                                                    "author": {"login": "blocker"},
                                                    "state": "CHANGES_REQUESTED",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_get_pr_reactions",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 169,
                                    },
                                    "result": {"structuredContent": {"reactions": [{"content": "+1"}]}},
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["review_count"], 2)
        self.assertEqual(snapshot["reaction_count"], 1)
        self.assertEqual(snapshot["review_signal"], "changes_requested")

    def test_review_signal_uses_each_reviewers_latest_review(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                _event(
                    "item/completed",
                    {
                        "threadId": "thread-1",
                        "item": {
                            "type": "mcpToolCall",
                            "server": "codex_apps",
                            "tool": "github_list_pull_request_reviews",
                            "arguments": {
                                "repo_full_name": "cberner/hitch",
                                "pr_number": 169,
                            },
                            "result": {
                                "structuredContent": {
                                    "reviews": [
                                        {
                                            "author": {"login": "reviewer"},
                                            "state": "APPROVED",
                                            "submitted_at": "2026-08-11T12:05:00Z",
                                        },
                                        {
                                            "author": {"login": "reviewer"},
                                            "state": "CHANGES_REQUESTED",
                                            "submitted_at": "2026-08-11T12:00:00Z",
                                        },
                                    ]
                                }
                            },
                        },
                    },
                    recorded_at=10,
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths([path], thread_id="thread-1")

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["review_count"], 1)
        self.assertEqual(snapshot["review_signal"], "approved")

    def test_comment_does_not_supersede_reviewers_change_request(self) -> None:
        target: dict[str, Any] = {}

        codex_events._copy_review_fields(
            target,
            {
                "reviews": [
                    {
                        "author": {"login": "reviewer"},
                        "state": "CHANGES_REQUESTED",
                        "submitted_at": "2026-08-11T12:00:00Z",
                    },
                    {
                        "author": {"login": "reviewer"},
                        "state": "COMMENTED",
                        "submitted_at": "2026-08-11T12:05:00Z",
                    },
                ]
            },
        )

        self.assertEqual(target["review_count"], 1)
        self.assertEqual(target["review_signal"], "changes_requested")

    def test_sparse_reviews_use_decisions_and_input_order(self) -> None:
        target: dict[str, Any] = {}

        codex_events._copy_review_fields(
            target,
            {
                "reviews": [
                    "malformed",
                    {"author": {"login": "reviewer"}, "state": "COMMENTED"},
                    {"author": {"login": "reviewer"}, "state": "APPROVED"},
                    {
                        "author": {"login": "reviewer"},
                        "state": "CHANGES_REQUESTED",
                    },
                ]
            },
        )

        self.assertEqual(target["review_count"], 2)
        self.assertEqual(target["review_signal"], "changes_requested")

    def test_dismissed_review_does_not_supersede_reviewers_approval(self) -> None:
        target: dict[str, Any] = {}

        codex_events._copy_review_fields(
            target,
            {
                "reviews": [
                    {
                        "author": {"login": "reviewer"},
                        "state": "APPROVED",
                        "submitted_at": "2026-08-11T12:00:00Z",
                    },
                    {
                        "author": {"login": "reviewer"},
                        "state": "DISMISSED",
                        "submitted_at": "2026-08-11T12:05:00Z",
                    },
                ]
            },
        )

        self.assertEqual(target["review_count"], 1)
        self.assertEqual(target["review_signal"], "approved")

    def test_dismissed_review_clears_reviewers_change_request(self) -> None:
        reviews = [
            {
                "author": {"login": "reviewer"},
                "state": "CHANGES_REQUESTED",
                "submitted_at": "2026-08-11T12:00:00Z",
            },
            {
                "author": {"login": "reviewer"},
                "state": "DISMISSED",
                "submitted_at": "2026-08-11T12:05:00Z",
            },
        ]

        for ordered_reviews in (reviews, list(reversed(reviews))):
            with self.subTest(ordered_reviews=ordered_reviews):
                target: dict[str, Any] = {}
                codex_events._copy_review_fields(
                    target,
                    {"reviews": ordered_reviews},
                )

                self.assertEqual(target["review_count"], 1)
                self.assertEqual(target["review_signal"], "commented")

    def test_review_signal_groups_rest_reviews_by_user_login(self) -> None:
        target: dict[str, Any] = {}

        codex_events._copy_review_fields(
            target,
            {
                "reviews": [
                    {
                        "user": {"login": "reviewer"},
                        "state": "CHANGES_REQUESTED",
                        "submitted_at": "2026-08-11T12:00:00Z",
                    },
                    {
                        "user": {"login": "reviewer"},
                        "state": "APPROVED",
                        "submitted_at": "2026-08-11T12:05:00Z",
                    },
                ]
            },
        )

        self.assertEqual(target["review_count"], 1)
        self.assertEqual(target["review_signal"], "approved")

    def test_ci_status_reports_failure_when_some_workflow_runs_still_pending(
        self,
    ) -> None:
        # A PR with both a completed-failure workflow and one still in progress
        # must be surfaced as ``failure`` regardless of list order, matching
        # the precedence used by the combined-status and per-job paths. The
        # snapshot drives the PR follow-up agent's behaviour; under-reporting
        # the failure as ``pending`` keeps the user uninformed while a CI
        # break sits unaddressed.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/169"),
                                            "number": 169,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_commit_workflow_runs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "commit_sha": "abc123",
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "workflow_runs": [
                                                {
                                                    "status": "in_progress",
                                                    "conclusion": "",
                                                },
                                                {
                                                    "status": "completed",
                                                    "conclusion": "failure",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "failure")

    def test_ci_status_from_jobs_is_unknown_when_every_entry_is_malformed(
        self,
    ) -> None:
        # ``fetch_workflow_run_jobs`` results that round-trip through MCP can
        # arrive with a ``jobs`` list whose entries are non-dict (e.g. the
        # remote returned strings rather than the structured job payload the
        # snapshot reader expects). The reader filters those out one by one,
        # but the prior implementation then returned ``ci_status="success"``
        # for a list with zero observed completed jobs -- the snapshot
        # falsely claims the PR's CI is green, the follow-up agent treats
        # the CI gate as passing, and the user is led to ship a PR whose
        # jobs were never actually evaluated.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/170"),
                                            "number": 170,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 42,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": ["lint", "build", "deploy"],
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "unknown")

    def test_pr_snapshot_clears_stale_review_thread_list_on_clean_re_observation(
        self,
    ) -> None:
        # A PR turn that observes one unresolved review thread, resolves it,
        # and then re-checks the threads must end with the snapshot reflecting
        # the second observation -- the same head SHA, zero open threads, an
        # empty ``unresolved_threads`` list. Before the fix the merger
        # treated the second update's ``unresolved_threads: []`` as "no
        # information" and skipped it, so the snapshot kept the stale thread
        # from the first observation alongside ``unresolved_thread_count=0``.
        # That inconsistent state propagates into ``workflow.state.pr_handoff``
        # and is rendered verbatim into the next PR follow-up turn's
        # prompt via ``_format_pr_handoff``; the persisted-handoff text the
        # agent reads then contradicts itself, which can mislead the monitor
        # into reporting the stale thread back as still-unresolved and stall
        # the Review gate on a thread that GitHub already shows resolved.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/171"),
                                            "number": 171,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_list_pull_request_review_threads",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 171,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "review_threads": [
                                                {
                                                    "id": "thread-A",
                                                    "is_resolved": False,
                                                    "is_outdated": False,
                                                    "path": "x.py",
                                                    "line": 12,
                                                }
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_list_pull_request_review_threads",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 171,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "review_threads": [
                                                {
                                                    "id": "thread-A",
                                                    "is_resolved": True,
                                                    "is_outdated": False,
                                                    "path": "x.py",
                                                    "line": 12,
                                                }
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["unresolved_thread_count"], 0)
        self.assertEqual(snapshot.get("unresolved_threads", []), [])

    def test_pr_snapshot_clears_stale_failing_jobs_on_clean_re_observation(
        self,
    ) -> None:
        # A PR turn that observes one failing CI job, the user pushes a fix,
        # and the agent re-checks the same workflow's jobs must end with
        # the snapshot reflecting the second observation -- ``ci_status``
        # ``success`` AND ``failing_jobs`` cleared. Before the fix
        # ``_copy_ci_fields`` only wrote ``failing_jobs`` when the list was
        # non-empty, so the clean second observation produced an update
        # without the key. ``_merge_pr_snapshot_update`` therefore kept the
        # first observation's stale failing list alongside ``ci_status:
        # "success"``, and ``gh_observations._ci_gate`` -- which short-circuits
        # to BLOCKED whenever ``failing_jobs`` has any items, regardless of
        # ``ci_status`` -- then surfaced the PR as "Failing CI jobs were
        # observed" to the PR follow-up agent. The follow-up workflow looped
        # feedback rounds trying to "fix" CI that was already green, burning
        # iterations until ``max_iterations`` was reached. Identical shape to
        # the ``unresolved_threads`` bug 48b0840 fixed at the merge layer,
        # but here the stale list never even reached the merge guard.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/172"),
                                            "number": 172,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 42,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "lint",
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                                {
                                                    "name": "test-suite",
                                                    "status": "completed",
                                                    "conclusion": "failure",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 43,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "lint",
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                                {
                                                    "name": "test-suite",
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "success")
        self.assertEqual(snapshot.get("failing_jobs", []), [])
        self.assertEqual(snapshot.get("pending_jobs", []), [])

    def test_pr_snapshot_clears_stale_failing_jobs_across_ci_tools(
        self,
    ) -> None:
        # A PR turn that observes a failing job via ``fetch_workflow_run_jobs``
        # and then re-checks the same commit via ``fetch_commit_workflow_runs``
        # -- which speaks for the same workflow-run universe and so can
        # authoritatively report a clean state -- must end with the snapshot
        # reflecting the second observation: ``ci_status`` ``success`` AND
        # ``failing_jobs`` cleared. Before the fix only
        # ``fetch_workflow_run_jobs`` wrote ``failing_jobs`` / ``pending_jobs``
        # (commit 1c14f01 fixed the same-tool re-observation case), so the
        # cross-tool re-observation produced an update without those keys.
        # ``_merge_pr_snapshot_update`` therefore kept the first observation's
        # stale failing list alongside ``ci_status: "success"``, and
        # ``gh_observations._ci_gate`` -- which short-circuits to BLOCKED
        # whenever ``failing_jobs`` has any items, regardless of ``ci_status``
        # -- then surfaced the PR as "Failing CI jobs were observed" to the
        # PR follow-up agent. The follow-up workflow looped feedback rounds
        # trying to "fix" CI that was already green, burning iterations until
        # ``max_iterations`` was reached. The cross-tool clear path triggers
        # in practice whenever a CI auto-retry flips a run to success without
        # a repo push, so the head-SHA-changed reset never runs.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/200"),
                                            "number": 200,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 42,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "lint",
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                                {
                                                    "name": "test-suite",
                                                    "status": "completed",
                                                    "conclusion": "failure",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_commit_workflow_runs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "commit_sha": "abc123",
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "workflow_runs": [
                                                {
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "success")
        self.assertEqual(snapshot.get("failing_jobs", []), [])
        self.assertEqual(snapshot.get("pending_jobs", []), [])

    def test_pr_snapshot_preserves_workflow_run_failing_jobs_across_combined_status(
        self,
    ) -> None:
        # ``get_commit_combined_status`` covers GitHub's commit Statuses API
        # (external CI integrations registered via the Statuses API), not the
        # workflow runs / check runs that populate ``failing_jobs`` via
        # ``fetch_workflow_run_jobs``. A success from combined-status therefore
        # proves nothing about whether the workflow-run job recovered, and
        # clearing the per-job list on that signal would degrade the
        # actionable "Failing CI jobs were observed" BLOCKED gate to a
        # bogus "CI is passing" PASSED gate -- the follow-up agent would
        # then ship a PR whose workflow-run job is still red. Only
        # ``fetch_commit_workflow_runs`` (which observes the same workflow-
        # run universe) and ``fetch_workflow_run_jobs`` (which enumerates the
        # jobs themselves) may clear the per-job lists; combined-status
        # writes ``ci_status`` only.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/202"),
                                            "number": 202,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 42,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "test-suite",
                                                    "status": "completed",
                                                    "conclusion": "failure",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_get_commit_combined_status",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "commit_sha": "abc123",
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "statuses": [
                                                {"state": "success"},
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.get("failing_jobs", []), ["test-suite"])

    def test_pr_snapshot_preserves_failing_jobs_on_unknown_ci_observation(
        self,
    ) -> None:
        # Even for ``fetch_commit_workflow_runs`` (the rollup tool that DOES
        # speak for the workflow-run universe and so legitimately clears
        # ``failing_jobs`` on a definitive ``success``), the cross-tool clear
        # must only fire on success. An empty ``workflow_runs`` list yields
        # ``ci_status="unknown"`` (``_ci_status_from_runs([])``), which proves
        # nothing about whether the previously-observed failing job actually
        # recovered. Overwriting ``failing_jobs``/``pending_jobs`` on that
        # signal would degrade the actionable "Failing CI jobs were observed"
        # gate to a non-actionable pending/waiting state -- the follow-up
        # agent then stops driving toward a CI fix even though nothing has
        # refuted the failure. ``failure`` / ``pending`` results from the
        # rollup tool likewise don't enumerate which jobs are bad, so the
        # prior per-job list remains the most specific signal we have.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/201"),
                                            "number": 201,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 42,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "test-suite",
                                                    "status": "completed",
                                                    "conclusion": "failure",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_commit_workflow_runs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "commit_sha": "abc123",
                                    },
                                    "result": {"structuredContent": {"workflow_runs": []}},
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "unknown")
        self.assertEqual(snapshot.get("failing_jobs", []), ["test-suite"])

    def test_pr_snapshot_clears_stale_failing_jobs_when_workflow_runs_pending(
        self,
    ) -> None:
        # When a failed workflow job is re-run on the same commit, GitHub
        # resets that workflow run to queued/in-progress on the same
        # ``run_id``. ``fetch_commit_workflow_runs`` therefore observes
        # ``[{"status": "in_progress"}]`` (no completed-failure runs), and
        # ``_ci_status_from_runs`` returns ``"pending"`` -- the rerun
        # supersedes the previously-observed failure even though the new
        # observation is not yet a definitive clean state. Without clearing
        # the prior per-job ``failing_jobs`` list, ``_ci_gate`` keeps short-
        # circuiting to BLOCKED on "Failing CI jobs were observed" instead
        # of returning the "CI is still running" pending gate that the
        # ``pending`` ci_status would otherwise drive, so the follow-up
        # agent keeps pushing fixes for a job that is already being
        # re-evaluated. The head-SHA-changed reset in
        # ``_merge_pr_handoff_dicts`` does not help here because no new
        # commit was pushed -- the same commit is being re-tested. Clear
        # at the source so a rerun-pending observation drops the stale
        # failing list. ``pending_jobs`` stays put because workflow-runs
        # observations don't enumerate jobs.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/203"),
                                            "number": 203,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 42,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "test-suite",
                                                    "status": "completed",
                                                    "conclusion": "failure",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_commit_workflow_runs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "commit_sha": "abc123",
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "workflow_runs": [
                                                {
                                                    "status": "in_progress",
                                                    "conclusion": "",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "pending")
        self.assertEqual(snapshot.get("failing_jobs", []), [])

    def test_pr_snapshot_clears_stale_pending_jobs_on_clean_re_observation(
        self,
    ) -> None:
        # Mirror of the failing-jobs case for ``pending_jobs``: a job that
        # was queued/in-progress on the first observation and completed
        # successfully on the second must not leak its name into the
        # persisted ``pending_jobs`` list. ``system_agents`` does not
        # surface pending jobs as a hard block (no "pending list short-
        # circuit" sibling to ``_pr_list_has_items(failing_jobs)``), but
        # ``_ci_feedback_details`` does render pending job names into the
        # feedback prompt sent to the next PR follow-up turn, so a stale
        # name there asks the next agent to chase a job that already
        # completed.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/173"),
                                            "number": 173,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 50,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "lint",
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                                {
                                                    "name": "deploy",
                                                    "status": "in_progress",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_fetch_workflow_run_jobs",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "run_id": 50,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "jobs": [
                                                {
                                                    "name": "lint",
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                                {
                                                    "name": "deploy",
                                                    "status": "completed",
                                                    "conclusion": "success",
                                                },
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["ci_status"], "success")
        self.assertEqual(snapshot.get("pending_jobs", []), [])
        self.assertEqual(snapshot.get("failing_jobs", []), [])

    def test_pr_snapshot_clears_stale_review_signal_on_clean_re_observation(
        self,
    ) -> None:
        # A PR turn that observes a CHANGES_REQUESTED review and then re-checks
        # the reviews after the reviewer dropped their verdict (the tool now
        # returns no actionable reviews) must end with the snapshot reflecting
        # the second observation -- ``review_count`` zero AND ``review_signal``
        # cleared. Before the fix ``_copy_review_fields`` only wrote
        # ``review_signal`` when the new list still produced a state, so the
        # clean second observation emitted an update without the key.
        # ``_merge_pr_snapshot_update`` therefore kept the stale
        # ``"changes_requested"`` from the first observation alongside
        # ``review_count=0``, and ``gh_observations._review_gate`` -- which
        # short-circuits to BLOCKED whenever ``review_signal`` is
        # ``"changes_requested"`` regardless of ``review_count`` -- then
        # surfaced the PR as "A reviewer requested changes." to the PR
        # follow-up agent. The follow-up workflow looped feedback rounds
        # trying to "address" feedback that the PR no longer carries, burning
        # iterations until ``max_iterations`` was reached. Same shape as the
        # ``unresolved_threads`` bug 48b0840 fixed at the merge layer, but
        # for the review-state signal that drives the Review gate verdict.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/174"),
                                            "number": 174,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_list_pull_request_reviews",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 174,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "reviews": [
                                                {"state": "CHANGES_REQUESTED"},
                                            ]
                                        }
                                    },
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_list_pull_request_reviews",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 174,
                                    },
                                    "result": {"structuredContent": {"reviews": []}},
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["review_count"], 0)
        # The clear is recorded as an explicit ``""`` (rather than popping the
        # key) so the cross-worker handoff merge in
        # ``pr_handoff._merge_pr_handoff_dicts`` can drop the stale
        # persisted verdict from an earlier monitor/feedback run.
        self.assertEqual(snapshot["review_signal"], "")

    def test_pr_snapshot_review_clear_preserves_reaction_thumbs_up(
        self,
    ) -> None:
        # When a +1 reaction observation already recorded ``thumbs_up`` and a
        # follow-up reviews observation yields no actionable signal, the
        # snapshot must keep the reaction-derived approval rather than
        # stomping it with a reviews-driven clear -- the reviews tool only
        # speaks for review-derived signals (changes_requested / approved /
        # commented), not for the reactions-driven thumbs_up signal.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {
                                            "url": ("https://github.com/cberner/hitch/pull/175"),
                                            "number": 175,
                                            "head_sha": "abc123",
                                        }
                                    },
                                },
                            },
                            recorded_at=5,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_get_pr_reactions",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 175,
                                    },
                                    "result": {"structuredContent": {"reactions": [{"content": "+1"}]}},
                                },
                            },
                            recorded_at=10,
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_list_pull_request_reviews",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 175,
                                    },
                                    "result": {"structuredContent": {"reviews": []}},
                                },
                            },
                            recorded_at=20,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot["review_count"], 0)
        self.assertEqual(snapshot["reaction_count"], 1)
        self.assertEqual(snapshot["review_signal"], "thumbs_up")

    def test_pr_snapshot_ignores_other_threads_and_non_github_tools(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            path.write_text(
                "\n".join(
                    [
                        "{not json",
                        _event(
                            "item/completed",
                            {
                                "threadId": "other-thread",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "codex_apps",
                                    "tool": "github_create_pull_request",
                                    "result": {
                                        "structuredContent": {"url": ("https://github.com/cberner/hitch/pull/169")}
                                    },
                                },
                            },
                        ),
                        _event(
                            "item/completed",
                            {
                                "threadId": "thread-1",
                                "item": {
                                    "type": "mcpToolCall",
                                    "server": "linear",
                                    "tool": "create_issue",
                                    "result": {"structuredContent": {"number": 169}},
                                },
                            },
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            snapshot = codex_events.latest_pr_snapshot_from_event_paths(
                [path],
                thread_id="thread-1",
            )

        self.assertIsNone(snapshot)


class FinalizePrSnapshotTests(SimpleTestCase):
    def test_accepts_repo_and_numeric_pr_number(self) -> None:
        snapshot = codex_events._finalize_pr_snapshot({"repository_full_name": "cberner/hitch", "pr_number": 7})
        self.assertIsNotNone(snapshot)

    def test_rejects_bool_pr_number(self) -> None:
        # bool is an int subclass; the identity guard must reject it so
        # _finalize_pr_snapshot agrees with _pr_snapshot_has_identity.
        snapshot = {"repository_full_name": "cberner/hitch", "pr_number": True}
        self.assertIsNone(codex_events._finalize_pr_snapshot(snapshot))
        self.assertFalse(codex_events._pr_snapshot_has_identity(snapshot))
