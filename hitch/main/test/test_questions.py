import json
import queue
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, override
from unittest.mock import patch

from django.template.loader import render_to_string
from django.test import SimpleTestCase, TestCase
from openai_codex.client import CodexClient, CodexConfig, _resolve_codex_bin
from openai_codex.errors import TransportClosedError

from hitch.main.management.commands import codex_worker
from hitch.main.models import CodexInstance, UserInputRequest
from hitch.main.runtime.question_transport import QuestionClient, QuestionCodex
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


def question(request_id: int, *, blocking: bool = False, turn_id: str = "turn") -> dict[str, Any]:
    return {
        "id": request_id, "method": "item/tool/requestUserInput",
        "params": {
            "threadId": "thread", "turnId": turn_id, "itemId": str(request_id),
            "isBlocking": blocking,
            "questions": [{"id": "choice", "question": f"Question {request_id}?"}],
        },
    }


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
    def test_multiple_questions_leave_progress_and_rpc_responses_live(self) -> None:
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
            incoming.put(question(1, blocking=True))
            incoming.put(question(2))
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

    def test_server_resolution_and_turn_end_cancel_only_matching_questions(self) -> None:
        with question_transport() as (client, incoming, outgoing, notifications, _responses):
            entered: queue.Queue[str] = queue.Queue()
            cancelled: queue.Queue[str] = queue.Queue()

            def handler(method: str, params: Any) -> dict[str, Any]:
                entered.put(params["itemId"])
                while not client.question_cancelled():
                    threading.Event().wait(0.01)
                cancelled.put(params["itemId"])
                return {"answers": {}}

            client._approval_handler = handler
            incoming.put(question(1))
            incoming.put(question(2, turn_id="other-turn"))
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

    def test_close_or_eof_releases_question_handlers_without_sending_answers(self) -> None:
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
                incoming.put(question(1))
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
                    page.evaluate("window.emit", {
                        "method": "input/requested", "payload": {"id": request_id, "params": {
                            "isBlocking": request_id == 1,
                            "questions": [{"id": "choice", "question": f"Question {request_id}?"}],
                        }},
                    })
                self.assertEqual(page.locator("[data-pending-questions]").inner_text(), "2 questions")
                first = page.locator('[data-input-request-id="1"]')
                second = page.locator('[data-input-request-id="2"]')
                self.assertIn("Answer to continue", first.inner_text())
                self.assertIn("Codex is continuing", second.inner_text())
                first.locator("textarea").fill("Unsubmitted answer")
                second.locator("textarea").fill("First line\nSecond line")
                second.get_by_role("button", name="Submit").click()
                self.assertEqual(page.evaluate("window.posts"), [{
                    "url": "/input/2/", "answers": {"choice": "First line\nSecond line"},
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
                self.assertEqual(errors, [])
            finally:
                browser.close()
