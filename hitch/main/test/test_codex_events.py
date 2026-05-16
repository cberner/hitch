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
