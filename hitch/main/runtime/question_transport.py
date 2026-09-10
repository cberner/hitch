"""Defer question replies without blocking the pinned SDK's stdout reader."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import override

from openai_codex import Codex, CodexConfig
from openai_codex._initialize_metadata import validate_initialize_metadata
from openai_codex.client import CodexClient
from openai_codex.models import JsonValue


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


class QuestionClient(CodexClient):
    def __init__(self, config: CodexConfig | None = None) -> None:
        super().__init__(config=config)
        self._questions: dict[str | int, _Question] = {}
        self._questions_lock = threading.Lock()
        self._question_context = threading.local()
        self._questions_closed = threading.Event()

    def question_cancelled(self) -> bool:
        question = getattr(self._question_context, "question", None)
        return self._questions_closed.is_set() or (
            isinstance(question, _Question) and question.cancelled.is_set()
        )

    @override
    def _read_message(self) -> dict[str, JsonValue]:
        # Codex owns whether a question blocks generation. Both kinds must leave
        # the transport free to receive progress, further questions and Stop.
        while True:
            try:
                message = super()._read_message()
            except BaseException:
                self._questions_closed.set()
                raise
            method = message.get("method")
            request_id = message.get("id")
            if isinstance(method, str) and is_user_input_request(method) and isinstance(request_id, str | int):
                self._defer_question(request_id, message)
                continue
            self._observe_resolution(message)
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
            self._questions[request_id] = question
            worker.start()

    def _answer_question(
        self, request_id: str | int, message: dict[str, JsonValue], question: _Question,
    ) -> None:
        self._question_context.question = question
        try:
            response = self._handle_server_request(message)
            if not self.question_cancelled():
                self._write_message({"id": request_id, "result": response})
        except BaseException as exc:
            if not self.question_cancelled():
                self._router.fail_all(exc)
                self._questions_closed.set()
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
        deadline = time.monotonic() + 2
        for worker in workers:
            if worker is not None:
                worker.join(timeout=max(0, deadline - time.monotonic()))


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
