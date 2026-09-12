"""Defer questions and dynamic tools without blocking the SDK's stdout reader."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import override

from openai_codex import Codex, CodexConfig
from openai_codex._initialize_metadata import validate_initialize_metadata
from openai_codex.client import CodexClient
from openai_codex.errors import JsonRpcError
from openai_codex.models import JsonValue

logger = logging.getLogger(__name__)


def async_question_params(message: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    if message.get("method") != "item/completed" or "id" in message:
        return None
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    item = params.get("item")
    if not isinstance(item, dict) or item.get("type") != "agentMessage" or item.get("delivery") != "async":
        return None
    if not all(isinstance(value, str) and value for value in (
        params.get("threadId"), params.get("turnId"), item.get("id"),
    )):
        return None
    questions = item.get("questions")
    if not isinstance(questions, list) or not questions:
        return None
    normalized: list[JsonValue] = []
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not isinstance(question.get("title"), str):
            return None
        options = question.get("options") or []
        if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
            return None
        normalized.append({
            "id": str(index), "question": question["title"],
            "options": [{"label": option} for option in options],
        })
    return {
        "threadId": params["threadId"], "turnId": params["turnId"], "itemId": item["id"],
        "delivery": "async", "isBlocking": False, "questions": normalized,
    }


def async_question_answer(params: dict[str, JsonValue], response: dict[str, JsonValue]) -> str:
    answers = response.get("answers")
    questions = params.get("questions")
    if not isinstance(answers, dict) or not isinstance(questions, list):
        return ""
    parts = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = question.get("id")
        if not isinstance(question_id, str):
            continue
        answer = answers.get(question_id)
        values = answer.get("answers") if isinstance(answer, dict) else None
        if isinstance(values, list) and values:
            parts.append(f"{question['question']}\n" + "\n".join(str(value) for value in values))
    return "Answers to your questions:\n\n" + "\n\n".join(parts) if parts else ""


def is_user_input_request(method: str) -> bool:
    return method in {"request_user_input", "requestUserInput"} or method.endswith(
        ("/requestUserInput", "/request_user_input")
    )


@dataclass
class _Question:
    thread_id: JsonValue
    turn_id: JsonValue
    cancelled: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None
    on_response_sent: list[Callable[[], None]] = field(default_factory=list)
    on_async_answer_settled: Callable[[str | None], None] | None = None


class QuestionClient(CodexClient):
    def __init__(self, config: CodexConfig | None = None) -> None:
        super().__init__(config=config)
        self._questions: dict[str | int, _Question] = {}
        self._questions_lock = threading.Lock()
        self._question_context = threading.local()
        self._questions_closed = threading.Event()
        self._async_items: set[tuple[str, str, str]] = set()

    def question_cancelled(self) -> bool:
        question = getattr(self._question_context, "question", None)
        return self._questions_closed.is_set() or (
            isinstance(question, _Question) and question.cancelled.is_set()
        )

    def on_response_sent(self, callback: Callable[[], None]) -> None:
        """Acknowledge tool evidence only after its response is written."""
        question = self._question_context.question
        question.on_response_sent.append(callback)

    def on_async_answer_settled(self, callback: Callable[[str | None], None]) -> None:
        question = self._question_context.question
        question.on_async_answer_settled = callback

    @override
    def _read_message(self) -> dict[str, JsonValue]:
        # Requests must leave the reader free to receive their cancellation,
        # turn completion, progress, and other RPC responses.
        while True:
            try:
                message = super()._read_message()
            except BaseException:
                self._questions_closed.set()
                raise
            method = message.get("method")
            request_id = message.get("id")
            if (
                isinstance(method, str)
                and (is_user_input_request(method) or method == "item/tool/call")
                and isinstance(request_id, str | int)
            ):
                self._defer_question(request_id, message)
                continue
            self._observe_resolution(message)
            params = async_question_params(message)
            if params is not None:
                key = (str(params["threadId"]), str(params["turnId"]), str(params["itemId"]))
                if key not in self._async_items:
                    self._async_items.add(key)
                    self._defer_question(f"async:{key[2]}", {
                        "method": "item/tool/requestUserInput", "params": params,
                    })
            return message

    def _defer_question(self, request_id: str | int, message: dict[str, JsonValue]) -> None:
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        question = _Question(params.get("threadId"), params.get("turnId"))
        worker = threading.Thread(
            target=self._answer_question, args=(request_id, message, question),
            name="hitch-question", daemon=True,
        )
        question.worker = worker
        with self._questions_lock:
            if self._questions_closed.is_set():
                return
            self._questions[request_id] = question
            worker.start()

    def _answer_question(
        self, request_id: str | int, message: dict[str, JsonValue], question: _Question,
    ) -> None:
        self._question_context.question = question
        delivery_error = None
        try:
            response = self._handle_server_request(message)
            if not self.question_cancelled():
                params = message.get("params")
                if isinstance(params, dict) and params.get("delivery") == "async":
                    # Async messages have no server request to resolve. Their
                    # answers are ordinary user input to the originating turn.
                    answer = async_question_answer(params, response)
                    if answer:
                        try:
                            self.turn_steer(str(question.thread_id), str(question.turn_id), answer)
                        except JsonRpcError as exc:
                            # A rejected answer does not invalidate other RPCs
                            # or questions sharing this transport.
                            delivery_error = str(exc)
                            return
                else:
                    self._write_message({"id": request_id, "result": response})
                for callback in question.on_response_sent:
                    try:
                        callback()
                    except Exception:
                        logger.exception("Failed to acknowledge a sent response; evidence remains available for retry")
        except BaseException as exc:
            if not self.question_cancelled():
                self._router.fail_all(exc)
                self._questions_closed.set()
        finally:
            try:
                if question.on_async_answer_settled is not None:
                    question.on_async_answer_settled(delivery_error)
            finally:
                with self._questions_lock:
                    self._questions.pop(request_id, None)
                del self._question_context.question

    def _observe_resolution(self, message: dict[str, JsonValue]) -> None:
        if "id" in message:
            return
        method = message.get("method")
        params = message.get("params")
        if not isinstance(params, dict):
            return
        with self._questions_lock:
            if method == "serverRequest/resolved":
                request_id = params.get("requestId")
                question = self._questions.get(request_id) if isinstance(request_id, str | int) else None
                if question is not None and question.thread_id == params.get("threadId"):
                    question.cancelled.set()
            elif method == "turn/completed":
                turn = params.get("turn")
                if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                    for question in self._questions.values():
                        if question.thread_id == params.get("threadId") and question.turn_id == turn["id"]:
                            question.cancelled.set()

    @override
    def close(self) -> None:
        self._questions_closed.set()
        super().close()
        with self._questions_lock:
            workers = [question.worker for question in self._questions.values()]
        for worker in workers:
            if worker is not None:
                # Cancellation ends polling, but an already-started mutation
                # such as Auto-pull must finish before the worker process exits.
                worker.join()


class QuestionCodex(Codex):
    """Use the normal SDK API with a question-aware worker transport."""

    def __init__(self, config: CodexConfig | None = None) -> None:
        # The pinned SDK doesn't accept a transport factory. Keep its startup
        # and metadata validation while substituting only the message reader.
        self._client = QuestionClient(config=config)
        try:
            self._client.start()
            self._init = validate_initialize_metadata(self._client.initialize())
        except Exception:
            self._client.close()
            raise
