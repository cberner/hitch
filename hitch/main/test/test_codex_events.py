import json
import tempfile
from pathlib import Path

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
    def test_noop_keeps_log_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            events_path = Path(raw) / "events.jsonl"
            original = _event("turn/completed", {"status": "completed"}) + "\n"
            events_path.write_text(original, encoding="utf-8")

            freed = codex_events.prune_diff_events(events_path)

            self.assertEqual(freed, 0)
            self.assertEqual(events_path.read_text(encoding="utf-8"), original)


class LatestGoalFromEventPathsTests(SimpleTestCase):
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

    def test_latest_goal_tokens_for_instance_handles_missing_instance(self) -> None:
        self.assertIsNone(codex_events.latest_goal_tokens_for_instance(None))

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
