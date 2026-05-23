import json
import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from hitch.main import codex_events


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

            goal = codex_events.latest_goal_from_event_paths(
                [first, second], thread_id="thread-1"
            )

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

            goal = codex_events.latest_goal_from_event_paths(
                [older_worker, newer_worker], thread_id="thread-1"
            )

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

            goal = codex_events.latest_goal_from_event_paths(
                [older_worker, newer_worker], thread_id="thread-1"
            )

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

            goal = codex_events.latest_goal_from_event_paths(
                [Path(raw) / "missing.jsonl", path], thread_id="thread-1"
            )

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
                                    "arguments": {
                                        "repository_full_name": "cberner/hitch"
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "url": (
                                                "https://github.com/cberner/hitch/pull/169"
                                            ),
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
                                            "url": (
                                                "https://github.com/cberner/hitch/pull/168"
                                            ),
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
                                            "url": (
                                                "https://github.com/cberner/hitch/pull/169"
                                            ),
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
                                            "url": (
                                                "https://github.com/cberner/hitch/pull/168"
                                            ),
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
                                    "result": {
                                        "structuredContent": {
                                            "comments": [{"body": "new PR feedback"}]
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
                                                {"state": "APPROVED"},
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
                                    "tool": "github_get_pr_reactions",
                                    "arguments": {
                                        "repo_full_name": "cberner/hitch",
                                        "pr_number": 169,
                                    },
                                    "result": {
                                        "structuredContent": {
                                            "reactions": [{"content": "+1"}]
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
        self.assertEqual(snapshot["review_count"], 2)
        self.assertEqual(snapshot["reaction_count"], 1)
        self.assertEqual(snapshot["review_signal"], "changes_requested")

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
                                        "structuredContent": {
                                            "url": (
                                                "https://github.com/cberner/hitch/pull/169"
                                            )
                                        }
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
