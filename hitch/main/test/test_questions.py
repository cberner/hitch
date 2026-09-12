import json
import queue
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, override
from unittest.mock import MagicMock, patch

from django.template.loader import render_to_string
from django.test import SimpleTestCase, TestCase
from openai_codex.client import CodexClient, CodexConfig, _resolve_codex_bin
from openai_codex.errors import InvalidRequestError, TransportClosedError

from hitch.main.management.commands import codex_worker
from hitch.main.models import CodexInstance, UserInputRequest
from hitch.main.runtime import rollout
from hitch.main.runtime.question_transport import (
    QuestionClient,
    QuestionCodex,
    _Question,
    async_question_params,
)
from hitch.main.sessions.entry_render import collapse_flat_entries, render_entries
from hitch.main.sessions.session_stage_refresh import _thread_ids_awaiting_input


@contextmanager
def question_transport() -> Iterator[tuple[QuestionClient, Any, Any, Any, Any]]:
    incoming: queue.Queue[Any] = queue.Queue()
    outgoing: queue.Queue[Any] = queue.Queue()
    notifications: queue.Queue[Any] = queue.Queue()
    responses: queue.Queue[Any] = queue.Queue()
    client = QuestionClient()

    def receive() -> Any:
        message = incoming.get(timeout=5)
        if isinstance(message, BaseException):
            raise message
        return message

    with (
        patch.object(CodexClient, "_read_message", side_effect=receive),
        patch.object(client, "_write_message", side_effect=outgoing.put),
        patch.object(client._router, "route_notification", side_effect=notifications.put),
        patch.object(client._router, "route_response", side_effect=responses.put),
    ):
        reader = threading.Thread(target=client._reader_loop, daemon=True)
        reader.start()
        try:
            yield client, incoming, outgoing, notifications, responses
        finally:
            client.close()
            incoming.put(TransportClosedError("test transport closed"))
            reader.join(timeout=2)
            assert not reader.is_alive()


def question(
    request_id: int, *, blocking: bool = False, turn_id: str = "turn", dynamic_tool: bool = False,
) -> dict[str, Any]:
    return {
        "id": request_id, "method": "item/tool/call" if dynamic_tool else "item/tool/requestUserInput",
        "params": {
            "threadId": "thread", "turnId": turn_id, "itemId": str(request_id),
            "isBlocking": blocking,
            "questions": [{"id": "choice", "question": f"Question {request_id}?"}],
        },
    }


def async_question() -> dict[str, Any]:
    # The async message shape emitted by the reported session's Codex runtime.
    return {"method": "item/completed", "params": {
        "threadId": "thread", "turnId": "turn", "item": {
            "type": "agentMessage", "id": "async-choice", "delivery": "async", "phase": "final_answer",
            "text": "Which scope?\n- Keep the smaller scope\n- Keep historical goal support",
            "questions": [
                {"title": "Which scope?", "options": ["Keep the smaller scope", "Keep historical goal support"]},
                {"title": "Any other constraints?"},
            ],
        },
    }}


class QuestionRuntimeTests(SimpleTestCase):
    def test_bundled_runtime_supports_nonblocking_input_and_transport_startup(self) -> None:
        # Resolve the installed bundle directly: a newer executable on PATH must
        # not mask a dependency pin that cannot serve nonblocking questions.
        with TemporaryDirectory() as directory:
            subprocess.run(
                [str(_resolve_codex_bin(CodexConfig())), "app-server", "generate-json-schema",
                 "--experimental", "--out", directory],
                check=True, capture_output=True, text=True, timeout=30,
            )
            schema = json.loads((Path(directory) / "ToolRequestUserInputParams.json").read_text())
            self.assertEqual(schema["properties"]["isBlocking"], {"type": "boolean"})
            self.assertIn("isBlocking", schema["required"])
            with QuestionCodex(CodexConfig(env={
                "CODEX_HOME": directory, "CODEX_SQLITE_HOME": directory,
            })) as codex:
                self.assertIsInstance(codex._client, QuestionClient)


class QuestionTransportTests(SimpleTestCase):
    def test_close_waits_for_started_tool_cleanup(self) -> None:
        with question_transport() as (client, incoming, outgoing, _events, _responses):
            entered = threading.Event()
            cancelled = threading.Event()
            release = threading.Event()
            closed = threading.Event()

            def handler(method: str, params: Any) -> dict[str, Any]:
                entered.set()
                while not client.question_cancelled():
                    cancelled.wait(0.01)
                cancelled.set()
                release.wait(5)
                return {"success": True}

            def close() -> None:
                client.close()
                closed.set()

            client._approval_handler = handler
            incoming.put(question(1, dynamic_tool=True))
            self.assertTrue(entered.wait(2))
            closer = threading.Thread(target=close, daemon=True)
            closer.start()
            try:
                self.assertTrue(cancelled.wait(2))
                self.assertFalse(closed.wait(2.1))
            finally:
                release.set()
                closer.join(timeout=2)
            self.assertTrue(closed.is_set())
            self.assertTrue(outgoing.empty())

    def test_tool_delivery_requires_a_successful_transport_write(self) -> None:
        for outcome in ("sent", "cancelled", "write_failed", "callback_failed"):
            with (
                self.subTest(outcome=outcome),
                question_transport() as (client, incoming, outgoing, events, _responses),
            ):
                entered = threading.Event()
                release = threading.Event()
                delivered = MagicMock()
                if outcome == "callback_failed":
                    delivered.side_effect = RuntimeError("database is locked")

                def handler(
                    method: str, params: Any, entered: threading.Event = entered,
                    release: threading.Event = release, delivered: MagicMock = delivered,
                ) -> dict[str, Any]:
                    client.on_response_sent(delivered)
                    entered.set()
                    release.wait(2)
                    return {"success": True}

                client._approval_handler = handler
                incoming.put(question(1, dynamic_tool=True))
                self.assertTrue(entered.wait(2))
                with client._questions_lock:
                    worker = client._questions[1].worker
                assert worker is not None
                if outcome == "cancelled":
                    incoming.put({"method": "turn/completed", "params": {
                        "threadId": "thread", "turn": {"id": "turn"},
                    }})
                    events.get(timeout=2)

                def write(message: Any, delivered: MagicMock = delivered, outcome: str = outcome) -> None:
                    delivered.assert_not_called()
                    if outcome == "write_failed":
                        raise TransportClosedError("write failed")
                    outgoing.put(message)

                with (
                    patch.object(client, "_write_message", side_effect=write),
                    patch("hitch.main.runtime.question_transport.logger.exception") as log_error,
                ):
                    release.set()
                    worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                if outcome in ("sent", "callback_failed"):
                    delivered.assert_called_once()
                    self.assertEqual(outgoing.get(timeout=2)["result"], {"success": True})
                else:
                    delivered.assert_not_called()
                    self.assertTrue(outgoing.empty())
                if outcome == "callback_failed":
                    log_error.assert_called_once()
                    self.assertFalse(client.question_cancelled())
                    client._approval_handler = lambda *_args: {"success": True}
                    incoming.put(question(2, dynamic_tool=True))
                    self.assertEqual(outgoing.get(timeout=2)["id"], 2)

    def test_async_message_answers_steer_once_and_leave_notifications_live(self) -> None:
        with question_transport() as (client, incoming, outgoing, notifications, _responses):
            entered: queue.Queue[Any] = queue.Queue()
            answered = threading.Event()
            steers: queue.Queue[Any] = queue.Queue()

            def handler(method: str, params: Any) -> dict[str, Any]:
                entered.put((method, params))
                self.assertTrue(answered.wait(2))
                return {"answers": {"0": {"answers": ["Keep historical goal support"]},
                                    "1": {"answers": ["First line\nSecond line"]}}}

            client._approval_handler = handler
            with patch.object(client, "turn_steer", side_effect=lambda *args: steers.put(args)):
                event = async_question()
                incoming.put(event)
                method, params = entered.get(timeout=2)
                self.assertEqual(method, "item/tool/requestUserInput")
                self.assertFalse(params["isBlocking"])
                self.assertEqual(params["questions"][0]["options"][0], {"label": "Keep the smaller scope"})
                self.assertEqual(notifications.get(timeout=2).method, "item/completed")
                self.assertTrue(outgoing.empty())
                answered.set()
                self.assertEqual(steers.get(timeout=2), (
                    "thread", "turn", "Answers to your questions:\n\nWhich scope?\nKeep historical goal support"
                    "\n\nAny other constraints?\nFirst line\nSecond line",
                ))
                incoming.put(event)
                notifications.get(timeout=2)
                self.assertTrue(entered.empty())
                self.assertTrue(outgoing.empty())

    def test_async_skip_and_cancellation_do_not_send_input_or_rpc_answers(self) -> None:
        for cancel in (False, True):
            with self.subTest(cancel=cancel), question_transport() as (
                client, incoming, outgoing, notifications, _responses,
            ):
                entered = threading.Event()
                finished = threading.Event()

                def handler(
                    _method: str, _params: Any, *, cancel: bool = cancel,
                    entered: threading.Event = entered, finished: threading.Event = finished,
                ) -> dict[str, Any]:
                    entered.set()
                    while cancel and not client.question_cancelled():
                        threading.Event().wait(0.01)
                    finished.set()
                    return {"answers": {}}

                client._approval_handler = handler
                with patch.object(client, "turn_steer") as steer:
                    incoming.put(async_question())
                    self.assertTrue(entered.wait(2))
                    notifications.get(timeout=2)
                    if cancel:
                        incoming.put({"method": "turn/completed", "params": {
                            "threadId": "thread", "turn": {"id": "turn", "status": "completed", "items": []},
                        }})
                        notifications.get(timeout=2)
                    self.assertTrue(finished.wait(2))
                    client.close()
                    steer.assert_not_called()
                    self.assertTrue(outgoing.empty())

    def test_only_structured_completed_async_questions_are_normalized(self) -> None:
        change: dict[str, Any]
        for change in ({"delivery": "sync"}, {"questions": []}, {"questions": [{"title": 1}]},
                       {"questions": [{"title": "Question?", "options": [1]}]}):
            event = async_question()
            event["params"]["item"].update(change)
            self.assertIsNone(async_question_params(event))
        event = async_question()
        event["method"] = "item/started"
        self.assertIsNone(async_question_params(event))

    def test_tools_and_questions_leave_progress_and_rpc_responses_live(self) -> None:
        with question_transport() as (client, incoming, outgoing, notifications, responses):
            entered: queue.Queue[str] = queue.Queue()
            answers = {"1": threading.Event(), "2": threading.Event()}

            def handler(method: str, params: Any) -> dict[str, Any]:
                key = params["itemId"]
                entered.put(key)
                while not answers[key].wait(0.01):
                    if client.question_cancelled():
                        return {"answers": {}}
                return {"answers": {"choice": {"answers": [key]}}}

            client._approval_handler = handler
            incoming.put(question(1, dynamic_tool=True))
            incoming.put(question(2, blocking=True))
            self.assertEqual({entered.get(timeout=2), entered.get(timeout=2)}, {"1", "2"})
            incoming.put({"method": "item/agentMessage/delta", "params": {
                "threadId": "thread", "turnId": "turn", "itemId": "message", "delta": "Still working",
            }})
            self.assertEqual(notifications.get(timeout=2).method, "item/agentMessage/delta")
            incoming.put({"id": "steer", "result": {"turnId": "turn"}})
            self.assertEqual(responses.get(timeout=2)["id"], "steer")
            self.assertTrue(outgoing.empty())
            for key in ("2", "1"):
                answers[key].set()
                self.assertEqual(outgoing.get(timeout=2), {
                    "id": int(key), "result": {"answers": {"choice": {"answers": [key]}}},
                })

    def test_server_resolution_and_turn_end_cancel_only_matching_requests(self) -> None:
        for dynamic_tool in (False, True):
            with (
                self.subTest(dynamic_tool=dynamic_tool),
                question_transport() as (client, incoming, outgoing, notifications, _responses),
            ):
                entered: queue.Queue[str] = queue.Queue()
                cancelled: queue.Queue[str] = queue.Queue()

                def handler(
                    method: str, params: Any, entered: queue.Queue[str] = entered,
                    cancelled: queue.Queue[str] = cancelled,
                ) -> dict[str, Any]:
                    entered.put(params["itemId"])
                    while not client.question_cancelled():
                        threading.Event().wait(0.01)
                    cancelled.put(params["itemId"])
                    return {"answers": {}}

                client._approval_handler = handler
                incoming.put(question(1, dynamic_tool=dynamic_tool))
                incoming.put(question(2, turn_id="other-turn", dynamic_tool=dynamic_tool))
                self.assertEqual({entered.get(timeout=2), entered.get(timeout=2)}, {"1", "2"})
                incoming.put({"method": "serverRequest/resolved", "params": {"threadId": "other", "requestId": 1}})
                notifications.get(timeout=2)
                self.assertTrue(cancelled.empty())
                incoming.put({"method": "serverRequest/resolved", "params": {"threadId": "thread", "requestId": 1}})
                self.assertEqual(cancelled.get(timeout=2), "1")
                incoming.put({"method": "turn/completed", "params": {
                    "threadId": "other", "turn": {"id": "other-turn"},
                }})
                incoming.put({"method": "turn/completed", "params": {
                    "threadId": "thread", "turn": {"id": "other-turn"},
                }})
                self.assertEqual(cancelled.get(timeout=2), "2")
                self.assertTrue(outgoing.empty())

    def test_close_or_eof_releases_tool_handlers_without_sending_answers(self) -> None:
        for eof in (True, False):
            with self.subTest(eof=eof), question_transport() as (client, incoming, outgoing, _events, _responses):
                entered = threading.Event()
                cancelled = threading.Event()

                def handler(
                    method: str, params: Any,
                    entered: threading.Event = entered, cancelled: threading.Event = cancelled,
                ) -> dict[str, Any]:
                    entered.set()
                    while not client.question_cancelled():
                        cancelled.wait(0.01)
                    cancelled.set()
                    return {"answers": {}}

                client._approval_handler = handler
                incoming.put(question(1, dynamic_tool=True))
                self.assertTrue(entered.wait(2))
                if eof:
                    incoming.put(TransportClosedError("disconnected"))
                else:
                    client.close()
                self.assertTrue(cancelled.wait(2))
                self.assertTrue(outgoing.empty())


class QuestionHandoffTests(TestCase):
    @override
    def setUp(self) -> None:
        self.instance = CodexInstance.objects.create(
            pid=1, thread_id="thread", cwd="/repo", prompt="hi", events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
        )

    def test_nonblocking_questions_do_not_mark_running_session_as_waiting(self) -> None:
        for params, waiting in (({}, True), ({"isBlocking": True}, True), ({"isBlocking": False}, False)):
            with self.subTest(params=params):
                row = UserInputRequest.objects.create(instance=self.instance, params=params)
                self.assertEqual(_thread_ids_awaiting_input(["thread"]), {"thread"} if waiting else set())
                row.delete()

    def test_browser_answer_reaches_codex_in_wire_format(self) -> None:
        events: list[tuple[str, Any]] = []
        handler = codex_worker._make_approval_handler(
            instance=self.instance, write_event=lambda method, payload: events.append((method, payload)),
            approval_mode="approve_all",
        )

        def answer(_interval: float) -> None:
            row = UserInputRequest.objects.get(instance=self.instance)
            self.assertFalse(row.params["isBlocking"])
            self.assertEqual(self.client.post(f"/input/{row.pk}/", {
                "answers": json.dumps({
                    "choice": "First line\nSecond line", "multiple": ["UI", "CLI"],
                    "canonical": {"answers": ["Keep history"]}, "optional": [],
                }),
            }).status_code, 200)

        with patch("hitch.main.management.commands.codex_worker.time.sleep", side_effect=answer):
            result = handler("item/tool/requestUserInput", question(1)["params"])
        self.assertEqual(result, {"answers": {
            "choice": {"answers": ["First line\nSecond line"]},
            "multiple": {"answers": ["UI", "CLI"]},
            "canonical": {"answers": ["Keep history"]}, "optional": {"answers": []},
        }})
        self.assertEqual([method for method, _payload in events], ["input/requested", "input/resolved"])
        self.assertFalse(events[-1][1]["cancelled"])

    def test_async_message_uses_durable_question_and_browser_answer_handoff(self) -> None:
        for rejected in (False, True):
            with self.subTest(rejected=rejected):
                self._check_async_handoff(rejected)
                UserInputRequest.objects.all().delete()

    def _check_async_handoff(self, rejected: bool) -> None:
        events: list[tuple[str, Any]] = []
        client = QuestionClient()
        client._approval_handler = codex_worker._make_approval_handler(
            instance=self.instance, write_event=lambda method, payload: events.append((method, payload)),
            approval_mode="approve_all",
            question_cancelled=client.question_cancelled,
            on_async_answer_settled=client.on_async_answer_settled,
        )
        params = async_question_params(async_question())
        assert params is not None

        def answer(_interval: float) -> None:
            row = UserInputRequest.objects.get(instance=self.instance)
            self.assertFalse(_thread_ids_awaiting_input(["thread"]))
            self.assertEqual(self.client.post(f"/input/{row.pk}/", {
                "answers": json.dumps({"0": "Keep historical goal support"}),
            }).status_code, 200)

        def deliver(*_args: Any) -> None:
            self.assertEqual([method for method, _ in events], ["input/requested"])
            if rejected:
                raise InvalidRequestError(-32600, "Originating turn is no longer active")

        other = _Question("thread", "other-turn")
        client._questions["other"] = other
        with (
            patch("hitch.main.management.commands.codex_worker.time.sleep", side_effect=answer),
            patch.object(client, "turn_steer", side_effect=deliver) as steer,
            patch.object(client._router, "fail_all") as fail_all,
        ):
            client._answer_question("async:choice", {
                "method": "item/tool/requestUserInput", "params": params,
            }, _Question("thread", "turn"))
        steer.assert_called_once_with(
            "thread", "turn", "Answers to your questions:\n\nWhich scope?\nKeep historical goal support",
        )
        self.assertEqual([method for method, _ in events], ["input/requested", "input/resolved"])
        self.assertFalse(events[-1][1]["cancelled"])
        self.assertEqual("error" in events[-1][1], rejected)
        self.assertFalse(client.question_cancelled())
        self.assertFalse(other.cancelled.is_set())
        fail_all.assert_not_called()

    def test_cancelled_question_is_closed_and_rejects_late_answers(self) -> None:
        events: list[tuple[str, Any]] = []
        handler = codex_worker._make_approval_handler(
            instance=self.instance, write_event=lambda method, payload: events.append((method, payload)),
            approval_mode="auto_review", question_cancelled=lambda: True,
        )
        self.assertEqual(handler("item/tool/requestUserInput", question(1)["params"]), {"answers": {}})
        row = UserInputRequest.objects.get(instance=self.instance)
        self.assertEqual(row.response, {"answers": {}})
        self.assertTrue(events[-1][1]["cancelled"])
        self.assertEqual(self.client.post(f"/input/{row.pk}/", {"answers": '{"choice":"late"}'}).status_code, 409)


class QuestionBrowserTests(SimpleTestCase):
    def test_multiple_questions_preserve_drafts_and_independent_answers(self) -> None:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        html = """
            <script>
                window.EventSource = class {
                    constructor() { this.handlers = {}; window.stream = this; }
                    addEventListener(name, handler) { this.handlers[name] = handler; }
                    close() {}
                };
                window.posts = [];
                window.hitch = { postForm(url, body) {
                    window.posts.push({url, answers: JSON.parse(body.get('answers'))});
                    return Promise.resolve({ok: true});
                }};
                window.emit = (event) => window.stream.handlers.message({data: JSON.stringify(event)});
            </script>
            <main data-session-main data-stream-url="/events">
                <span data-live-status data-state="working"></span>
                <details data-turn-notice data-thread-id="thread" hidden>
                    <summary data-turn-notice-title></summary><div data-turn-notice-body></div>
                </details>
                <div data-live-root data-input-url-template="/input/0/"></div>
            </main>
            <div data-pending-question-bar hidden><button data-pending-questions></button></div>
            <textarea data-composer-input>Keep my draft</textarea>
        """ + render_to_string("_session_script.html")
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(headless=True)
            except PlaywrightError as exc:
                self.skipTest(f"playwright browser unavailable: {exc}")
            try:
                page = browser.new_page()
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.set_content(html)
                for request_id in (1, 2):
                    params: Any = {
                        "isBlocking": request_id == 1,
                        "questions": [{"id": "choice", "question": f"Question {request_id}?"}],
                    }
                    if request_id == 2:
                        event = async_question()
                        page.evaluate("window.emit", {"method": "item/started", "payload": event["params"]})
                        self.assertEqual(page.locator(".message.agent").count(), 1)
                        params = async_question_params(event)
                        assert params is not None
                    page.evaluate("window.emit", {
                        "method": "input/requested", "payload": {"id": request_id, "params": params},
                    })
                page.evaluate("window.emit", {"method": "item/completed", "payload": async_question()["params"]})
                self.assertEqual(page.locator(".message.agent").count(), 0)
                self.assertEqual(page.locator("[data-pending-questions]").inner_text(), "2 questions")
                first = page.locator('[data-input-request-id="1"]')
                second = page.locator('[data-input-request-id="2"]')
                self.assertIn("Answer to continue", first.inner_text())
                self.assertIn("Codex is continuing", second.inner_text())
                self.assertEqual(second.locator('[aria-pressed="true"]').inner_text(), "Keep the smaller scope")
                self.assertEqual(page.evaluate("window.posts"), [])
                first.locator("textarea").fill("Unsubmitted answer")
                second.get_by_role("button", name="Keep historical goal support", exact=True).click()
                second.locator("textarea").last.fill("First line\nSecond line")
                second.get_by_role("button", name="Submit").click()
                self.assertEqual(page.evaluate("window.posts"), [{
                    "url": "/input/2/",
                    "answers": {"0": "Keep historical goal support", "1": "First line\nSecond line"},
                }])
                page.evaluate("window.emit", {"method": "input/resolved", "payload": {
                    "id": 2, "response": {"answers": {"choice": "First line\nSecond line"}},
                }})
                # Replayed requests must not duplicate controls or discard drafts.
                page.evaluate("window.emit", {"method": "input/requested", "payload": {"id": 1}})
                self.assertEqual(page.locator("[data-input-request-id]").count(), 2)
                self.assertEqual(first.locator("textarea").input_value(), "Unsubmitted answer")
                self.assertEqual(page.locator("[data-composer-input]").input_value(), "Keep my draft")
                self.assertEqual(page.locator("[data-pending-questions]").inner_text(), "1 question")
                page.evaluate("window.emit", {"method": "input/resolved", "payload": {"id": 1, "cancelled": True}})
                self.assertIn("Question closed", first.inner_text())
                self.assertTrue(page.locator("[data-pending-question-bar]").is_hidden())

                item = async_question()["params"]["item"]
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "rollout.jsonl"
                    path.write_text(json.dumps({"type": "event_msg", "payload": {
                        "type": "item_completed", "item": {**item, "type": "AgentMessage",
                            "content": [{"type": "Text", "text": item["text"]}]},
                    }}) + "\n")
                    preview = rollout.session_history_page(path)
                    assert preview is not None
                    sdk_thread = SimpleNamespace(turns=[SimpleNamespace(
                        items=[SimpleNamespace(root=SimpleNamespace(**item))],
                    )])
                    histories = [
                        list(collapse_flat_entries(list(rollout.iter_entries(path)))),
                        list(collapse_flat_entries(list(preview.flat_entries))), list(render_entries(sdk_thread)),
                    ]
                for entries in histories:
                    persisted = render_to_string("_session_entries.html", {"entries": entries})
                    page.set_content(html.replace("<div data-live-root", persisted +
                                                  '<div data-hide-transcript="true" data-live-root'))
                    self.assertEqual(page.locator("[data-async-question-item-id]").count(), 1)
                    request: dict[str, Any] = {"method": "input/requested", "payload": {
                        "id": 3, "params": async_question_params(async_question()),
                    }}
                    page.evaluate("window.emit", request)
                    page.evaluate("window.emit", request)
                    self.assertEqual(page.locator("[data-async-question-item-id]").count(), 0)
                    self.assertEqual(page.locator("[data-input-request-id]").count(), 1)
                    page.evaluate("""html => {
                        const history = document.createElement('div');
                        history.innerHTML = html;
                        document.querySelector('[data-session-main]').prepend(history);
                        document.dispatchEvent(new CustomEvent('hitch:transcript-updated', {detail: {root: history}}));
                    }""", persisted)
                    self.assertEqual(page.locator("[data-async-question-item-id]").count(), 0)
                    # Preselection requires Submit; Skip discards the default.
                    card = page.locator('[data-input-request-id="3"]')
                    self.assertEqual(page.evaluate("window.posts"), [])
                    card.get_by_role("button", name="Skip", exact=True).click()
                    self.assertEqual(page.evaluate("window.posts"), [{"url": "/input/3/", "answers": {}}])
                    page.evaluate("window.emit", {"method": "input/resolved", "payload": {
                        "id": 3, "error": "Originating turn is no longer active",
                    }})
                    self.assertIn("Answer could not be delivered", card.inner_text())
                    self.assertNotIn("Answers submitted", card.inner_text())
                    self.assertEqual(page.locator("[data-composer-input]").input_value(), "Keep my draft")
                    request["payload"]["id"] = 4
                    page.evaluate("window.emit", request)
                    page.locator('[data-input-request-id="4"]').get_by_role("button", name="Submit").click()
                    self.assertEqual(page.evaluate("window.posts.at(-1)"), {
                        "url": "/input/4/", "answers": {"0": "Keep the smaller scope"},
                    })
                self.assertEqual(errors, [])
            finally:
                browser.close()
