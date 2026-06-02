"""Detached worker that runs one Claude Code turn and streams events to disk.

This is the Claude-backend counterpart to ``codex_worker``. It is launched by
``codex_pool._launch_worker_process`` for ``CodexInstance`` rows whose
``backend`` is ``claude``. The lifecycle mirrors the Codex worker:

  1. Mark the row ``running`` and open its JSONL events file.
  2. Drive the local ``claude`` CLI through ``ClaudeSDKClient`` for one turn.
  3. Translate every SDK message into Codex-shaped events (see
     ``hitch.main.claude_translate``) and append them to the events file, so the
     existing SSE/render/parse stack works unchanged.
  4. Map the SDK ``ResultMessage`` onto the row's terminal status and persist
     the Claude session id for later resume.

Interactive approvals reuse the same ``ApprovalRequest`` row + ``approval/*``
event contract as the Codex worker, driven here by the SDK ``can_use_tool``
callback. Mid-turn steering and Stop reuse the sibling control file and SIGTERM
handling, adapted to the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import json
import logging
import os
import signal
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import IO, Any, override

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    ToolPermissionContext,
)
from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone

from hitch.main import claude_options, claude_translate
from hitch.main.claude_tools import build_hitch_mcp_server
from hitch.main.codex_pool import (
    cleanup_requested_input_images_for,
    control_path_for,
)
from hitch.main.management.commands.codex_worker import (
    _apply_worker_oom_score_adjust,
    _create_pending_approval,
    _create_pending_user_input,
    _notify_system_agents,
    _wait_for_decision,
    _wait_for_user_input_response,
    request_cancel,
)
from hitch.main.models import ApprovalRequest, CodexInstance

logger = logging.getLogger(__name__)

# Claude tool names that need an interactive approval, mapped onto the Codex
# approval method the browser already knows how to label. Read-only tools are
# auto-approved via ``allowed_tools`` and never reach ``can_use_tool``; any
# other tool that arrives here is allowed by default (matching Codex's
# auto-reviewer, which only escalates commands and file edits).
_COMMAND_APPROVAL_METHOD = "item/commandExecution/requestApproval"
_FILE_APPROVAL_METHOD = "item/fileChange/requestApproval"
# Tools that are neither a shell command nor a file edit (e.g. a mutating MCP
# tool pulled in from project/user settings) still must be gated -- the SDK's
# ``allowed_tools`` only auto-approves, it does not block anything else, so an
# ungated tool here would bypass Hitch's approval modes.
_TOOL_APPROVAL_METHOD = "item/tool/requestApproval"
_FILE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
# The SDK's ``AskUserQuestion`` tool (common in plan mode) reaches the same
# ``can_use_tool`` callback. Rather than a bare allow/deny, it is routed to the
# structured-input handoff (``UserInputRequest`` + the ``input/requested`` UI
# the Codex ``request_user_input`` tool already uses) so the user picks from the
# generated choices.
_ASK_USER_QUESTION_TOOL = "AskUserQuestion"
_ASK_USER_QUESTION_METHOD = "item/userInput/askUserQuestion"
_STEER_POLL_INTERVAL = 0.2
# Grace window at turn end: a steer can be queued (the row is still marked
# running until ``handle`` flips its status) between the loop's last drain and
# teardown. Give the control-file tailer a brief window to surface that steer
# before exiting so the prompt is not silently dropped.
_STEER_DRAIN_GRACE = 0.05


class Command(BaseCommand):
    help = "Run one Claude Code turn for an existing CodexInstance and stream events to disk."

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--instance-id", type=int, required=True)
        parser.add_argument("--reasoning-effort", type=str, default=None)
        parser.add_argument("--model", type=str, default=None)
        parser.add_argument("--sandbox-policy", type=str, default=None)
        parser.add_argument("--approval-mode", type=str, default=None)
        parser.add_argument("--web-search-mode", type=str, default=None)
        parser.add_argument("--plan-mode", action="store_true")

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        instance = CodexInstance.objects.get(pk=options["instance_id"])

        _apply_worker_oom_score_adjust()
        instance.pid = os.getpid()
        instance.status = CodexInstance.STATUS_RUNNING
        instance.save(update_fields=["pid", "status"])

        try:
            with open(instance.events_path, "a", buffering=1, encoding="utf-8") as events_file:
                runner = _TurnRunner(
                    instance=instance,
                    events_file=events_file,
                    model=options.get("model"),
                    reasoning_effort=options.get("reasoning_effort"),
                    sandbox_policy=options.get("sandbox_policy"),
                    approval_mode=options.get("approval_mode"),
                    web_search_mode=(
                        options.get("web_search_mode") or instance.web_search_mode or None
                    ),
                    plan_mode=options.get("plan_mode", False),
                )
                asyncio.run(runner.run())
        except Exception as exc:  # noqa: BLE001 - record any failure, then re-raise
            instance.status = CodexInstance.STATUS_FAILED
            instance.ended_at = timezone.now()
            instance.error = repr(exc)
            instance.save(update_fields=["status", "ended_at", "error"])
            _notify_system_agents(instance)
            cleanup_requested_input_images_for(instance)
            raise

        instance.ended_at = timezone.now()
        update_fields = ["status", "ended_at", "error"]
        if runner.session_id and runner.session_id != instance.claude_session_id:
            instance.claude_session_id = runner.session_id
            update_fields.append("claude_session_id")
        if runner.failed or not runner.saw_result:
            # A stream that closes without a ResultMessage is a truncated/aborted
            # turn, not a success -- mark it failed rather than silently completed.
            instance.status = CodexInstance.STATUS_FAILED
            instance.error = runner.error or "claude turn ended without a result"
        else:
            instance.status = CodexInstance.STATUS_COMPLETED
        instance.save(update_fields=update_fields)
        _notify_system_agents(instance)
        cleanup_requested_input_images_for(instance)


class _TurnRunner:
    """Owns the asyncio turn: streaming, approvals, steering, and interrupt."""

    def __init__(
        self,
        *,
        instance: CodexInstance,
        events_file: IO[str],
        model: str | None,
        reasoning_effort: str | None,
        sandbox_policy: str | None,
        approval_mode: str | None,
        web_search_mode: str | None,
        plan_mode: bool,
    ) -> None:
        self._instance = instance
        self._events_file = events_file
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._sandbox_policy = sandbox_policy
        self._approval_mode = approval_mode
        self._web_search_mode = web_search_mode
        self._plan_mode = plan_mode
        self._translator = claude_translate.EventTranslator()
        self._seq: Callable[[], int] = itertools.count(1).__next__
        self._client: ClaudeSDKClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._steer_wakeup: asyncio.Event | None = None
        # Count of steered follow-up queries issued mid-turn. Incremented in the
        # steer thread before the query is scheduled, drained on the loop thread
        # to decide whether more responses still need draining. The lock guards
        # the cross-thread read/modify/write.
        self._steer_pending = 0
        self._steer_lock = threading.Lock()
        self._cancelled = False
        self.session_id = instance.claude_session_id
        self.failed = False
        self.error = ""
        self.saw_result = False

    async def run(self) -> None:
        options = self._build_options()
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._steer_wakeup = asyncio.Event()
        self._install_signal_handlers(loop)
        async with ClaudeSDKClient(options=options) as client:
            self._client = client
            await client.query(self._turn_input())
            steer_task = asyncio.create_task(self._forward_steer_requests())
            try:
                # ``receive_response`` stops after one ResultMessage, but a steer
                # issued mid-turn schedules another ``query`` whose response then
                # needs its own drain. Loop until every issued query (the initial
                # one plus any steered follow-ups) has produced its result, so a
                # steered prompt's output is never dropped at context close.
                outstanding = 1
                while outstanding > 0:
                    async for message in client.receive_response():
                        self._capture_session_id(message)
                        for method, payload in self._translator.translate(message):
                            self._write_event(method, payload)
                        if isinstance(message, ResultMessage):
                            self._record_result(message)
                    outstanding = outstanding - 1 + self._take_steer_pending()
                    if outstanding == 0:
                        # Let the tailer catch a steer that landed just as this
                        # response drained, before we tear the turn down.
                        await asyncio.sleep(_STEER_DRAIN_GRACE)
                        outstanding += self._take_steer_pending()
            finally:
                steer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await steer_task

    def _take_steer_pending(self) -> int:
        """Return and reset the count of steered queries awaiting a drain."""
        with self._steer_lock:
            pending = self._steer_pending
            self._steer_pending = 0
        return pending

    # -- options & prompt --------------------------------------------------

    def _build_options(self) -> Any:
        instance = self._instance
        resume = instance.claude_session_id or None
        mcp_server = (
            build_hitch_mcp_server(cwd=instance.cwd, thread_id=instance.thread_id)
            if instance.purpose == CodexInstance.PURPOSE_USER
            else None
        )
        return claude_options.build_options(
            cwd=instance.cwd,
            model=self._model or instance.model or None,
            reasoning_effort=self._reasoning_effort or instance.reasoning_effort or None,
            sandbox_policy=self._sandbox_policy,
            approval_mode=self._approval_mode,
            web_search_mode=self._web_search_mode,
            plan_mode=self._plan_mode,
            base_instructions=instance.base_instructions or None,
            output_schema=instance.output_schema,
            resume_session_id=resume,
            session_id=None if resume else instance.thread_id,
            mcp_server=mcp_server,
            can_use_tool=self._can_use_tool,
        )

    def _turn_input(self) -> str | AsyncIterator[dict[str, Any]]:
        """Return the query input: a plain prompt, or a message stream w/ images.

        Per-turn developer guidance rides in front of the user prompt; the SDK
        has no separate per-turn system channel. When the row carries image
        attachments, the prompt becomes an Anthropic-style content list (text +
        base64 image blocks) delivered through the streaming-input form of
        ``ClaudeSDKClient.query``.
        """
        instance = self._instance
        prompt = instance.prompt
        if instance.developer_instructions:
            prompt = f"{instance.developer_instructions}\n\n{prompt}"
        return _build_query_input(prompt, instance.input_image_paths)

    # -- streaming bookkeeping --------------------------------------------

    def _capture_session_id(self, message: Any) -> None:
        candidate = ""
        if isinstance(message, SystemMessage):
            raw = message.data.get("session_id")
            candidate = raw if isinstance(raw, str) else ""
        elif isinstance(message, ResultMessage):
            candidate = message.session_id
        elif isinstance(message, AssistantMessage):
            # ``session_id`` exists on AssistantMessage in current SDKs; guard
            # with getattr so an older/newer SDK shape can't crash the turn.
            raw = getattr(message, "session_id", "")
            candidate = raw if isinstance(raw, str) else ""
        if candidate:
            self.session_id = candidate

    def _record_result(self, message: ResultMessage) -> None:
        self.saw_result = True
        if message.subtype != "success" or message.is_error:
            self.failed = True
            self.error = _result_error(message)

    def _write_event(self, method: str, payload: dict[str, Any]) -> None:
        event = {
            "method": method,
            "payload": payload,
            "recordedAt": time.time_ns() // 1_000,
            "eventSeq": self._seq(),
        }
        self._events_file.write(json.dumps(event) + "\n")

    # -- approvals ---------------------------------------------------------

    async def _can_use_tool(
        self, tool_name: str, tool_input: dict[str, Any], _context: ToolPermissionContext
    ) -> Any:
        # Read-only tools are auto-approved via ``allowed_tools`` and never
        # reach here, so anything that does is potentially mutating and must be
        # gated -- including unknown MCP tools from project/user settings.
        method = _approval_method(tool_name)
        if self._approval_mode == claude_options.APPROVAL_DENY_ALL:
            return claude_options.deny_result("Denied by Hitch approval policy.")
        # Hidden system-agent runs (QA/spec/autonomous) have no visible approval
        # UI, so a browser ``ApprovalRequest`` would just wait out the timeout
        # and be denied. Under ``auto_review`` -- the mode these runs are pinned
        # to -- auto-approve instead, matching how the Codex backend lets its
        # hidden reviewer runs proceed. Sandbox/read-only tool gating still
        # bounds what these runs can do.
        if (
            self._instance.purpose == CodexInstance.PURPOSE_SYSTEM_AGENT
            and self._approval_mode == claude_options.APPROVAL_AUTO_REVIEW
        ):
            # Auto-approve only the built-in mutating tools, and only when the
            # workflow runs under a write sandbox. ``readOnly`` already blocks
            # Bash/file via ``resolve_tool_lists``; a run with no sandbox would
            # otherwise auto-run *unsandboxed* commands/edits with no visible
            # approval, so a propose/review-only or prompt-injected hidden agent
            # could touch the host. Anything else (and project ``.claude`` MCP
            # tools, which also reach here) is denied.
            writes_allowed = self._sandbox_policy in (
                claude_options.SANDBOX_WORKSPACE_WRITE,
                claude_options.SANDBOX_DANGER_FULL_ACCESS,
            )
            if writes_allowed and (tool_name == "Bash" or tool_name in _FILE_TOOLS):
                return claude_options.allow_result()
            return claude_options.deny_result(
                "Hidden system-agent runs may only auto-run built-in Bash/file "
                "tools under a write sandbox."
            )
        if tool_name == _ASK_USER_QUESTION_TOOL:
            return await self._ask_user_question(tool_input)
        params = _approval_params(method, tool_name, tool_input)
        request_id = await asyncio.to_thread(
            _create_pending_approval,
            instance_id=self._instance.pk,
            method=method,
            params=params,
        )
        self._write_event(
            "approval/requested", {"id": request_id, "method": method, "params": params}
        )
        decision = await asyncio.to_thread(_wait_for_decision, request_id)
        self._write_event(
            "approval/resolved",
            {"id": request_id, "method": method, "decision": decision},
        )
        if _decision_allows(decision):
            return claude_options.allow_result()
        return claude_options.deny_result("Declined by the user.")

    async def _ask_user_question(self, tool_input: dict[str, Any]) -> Any:
        """Collect answers to a Claude ``AskUserQuestion`` via the input UI.

        Reuses the ``UserInputRequest`` handoff the Codex ``request_user_input``
        tool uses: render the generated questions/options, wait for the user's
        picks, then return them to the tool. ``AskUserQuestionInput`` carries an
        optional ``answers`` map "populated by the permission component", so the
        selections ride back through ``updated_input`` on an *allow* -- a deny
        would surface a valid answer to the model as a declined tool call.
        """
        questions = _ask_user_question_params(tool_input)
        if not questions:
            return claude_options.deny_result("No question to ask.")
        params = {"questions": questions}
        request_id = await asyncio.to_thread(
            _create_pending_user_input,
            instance_id=self._instance.pk,
            method=_ASK_USER_QUESTION_METHOD,
            params=params,
        )
        self._write_event(
            "input/requested",
            {"id": request_id, "method": _ASK_USER_QUESTION_METHOD, "params": params},
        )
        response = await asyncio.to_thread(_wait_for_user_input_response, request_id)
        self._write_event(
            "input/resolved",
            {"id": request_id, "method": _ASK_USER_QUESTION_METHOD, "response": response},
        )
        raw_answers = response.get("answers") if isinstance(response, dict) else None
        answers_map = (
            _ask_user_answers_map(tool_input, raw_answers)
            if isinstance(raw_answers, dict)
            else {}
        )
        if not answers_map:
            return claude_options.deny_result("The user did not answer the question.")
        return claude_options.allow_result(
            updated_input={**tool_input, "answers": answers_map}
        )

    # -- steering & interrupt ---------------------------------------------

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(signal.SIGTERM, self._on_sigterm)
        # ``steer_instance`` signals workers with SIGUSR1. The default
        # disposition terminates the process, so install a handler even though
        # the control-file drain polls -- without it a steer request would kill
        # an active Claude turn instead of delivering the prompt. The handler
        # just nudges the drain; the poll loop does the actual forwarding.
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(signal.SIGUSR1, self._on_steer_signal)

    def _on_steer_signal(self) -> None:
        if self._steer_wakeup is not None:
            self._steer_wakeup.set()

    def _on_sigterm(self) -> None:
        # A first Stop click sends SIGTERM: interrupt gracefully. A second click
        # escalates to SIGKILL (no handler), tearing the worker down.
        if self._cancelled or self._client is None:
            return
        self._cancelled = True
        # ``can_use_tool`` may be blocked in ``_wait_for_decision`` (a worker
        # thread) awaiting a browser approval. That wait polls the shared cancel
        # flag, so set it -- otherwise the approval sits until its timeout and the
        # interrupt below never runs because the turn is parked on the callback.
        request_cancel()
        asyncio.create_task(self._interrupt())

    async def _interrupt(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.interrupt()

    async def _forward_steer_requests(self) -> None:
        """Tail the control file and inject queued steer prompts into the turn.

        Woken immediately by SIGUSR1 (``steer_instance``) and on a short poll
        fallback for requests queued before the handler was installed.
        """
        control_path = control_path_for(self._instance)
        offset = 0
        while True:
            offset = await asyncio.to_thread(self._drain_once, control_path, offset)
            if self._steer_wakeup is not None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._steer_wakeup.wait(), timeout=_STEER_POLL_INTERVAL
                    )
                self._steer_wakeup.clear()
            else:
                await asyncio.sleep(_STEER_POLL_INTERVAL)

    def _drain_once(self, control_path: Path, offset: int) -> int:
        try:
            with control_path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
        except FileNotFoundError:
            return offset
        newline = chunk.rfind(b"\n")
        if newline < 0:
            return offset
        complete = chunk[: newline + 1]
        for raw in complete.splitlines():
            text, image_paths = _steer_request(raw)
            if (text or image_paths) and self._client is not None and self._loop is not None:
                query_input = _build_query_input(text, image_paths)
                # Register the follow-up before scheduling so the loop thread
                # can't finish draining the prior response and exit before it
                # learns another response is coming.
                with self._steer_lock:
                    self._steer_pending += 1
                asyncio.run_coroutine_threadsafe(
                    self._client.query(query_input), self._loop
                )
        return offset + len(complete)


def _approval_method(tool_name: str) -> str:
    if tool_name == "Bash":
        return _COMMAND_APPROVAL_METHOD
    if tool_name in _FILE_TOOLS:
        return _FILE_APPROVAL_METHOD
    return _TOOL_APPROVAL_METHOD


def _approval_params(
    method: str, tool_name: str, tool_input: dict[str, Any]
) -> dict[str, Any]:
    item: dict[str, Any]
    if method == _COMMAND_APPROVAL_METHOD:
        item = {
            "type": "commandExecution",
            "command": _str(tool_input.get("command")),
        }
    elif method == _FILE_APPROVAL_METHOD:
        path = ""
        for key in ("file_path", "path", "notebook_path"):
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                path = value
                break
        item = {"type": "fileChange", "changes": [{"path": path}]}
    else:
        # The browser renders the item JSON as the approval detail, so carry the
        # arguments (like the translated ``mcpToolCall``) -- otherwise the user
        # approves a mutating tool seeing only its name, not what it will do.
        item = {"type": "toolCall", "tool": tool_name, "arguments": tool_input}
    return {"item": item, "tool": tool_name}


def _ask_user_question_params(tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Map an ``AskUserQuestion`` tool input onto the ``input/requested`` schema.

    The browser input UI keys answers by a per-question ``id`` (which
    ``AskUserQuestion`` does not provide) and renders ``options`` as single
    select; a synthetic id is added per question and each option's
    label/description is carried through.
    """
    raw_questions = tool_input.get("questions")
    if not isinstance(raw_questions, list):
        return []
    questions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            continue
        text = _str(raw.get("question"))
        if not text:
            continue
        options: list[dict[str, Any]] = []
        for raw_option in raw.get("options") or []:
            if not isinstance(raw_option, dict):
                continue
            label = _str(raw_option.get("label"))
            if label:
                options.append(
                    {"label": label, "description": _str(raw_option.get("description"))}
                )
        questions.append(
            {
                "id": f"q{index}",
                "header": _str(raw.get("header")),
                "question": text,
                "options": options,
                # Clarifying questions need a real answer, so don't pre-select.
                "requires_explicit_choice": True,
            }
        )
    return questions


def _ask_user_answers_map(
    tool_input: dict[str, Any], raw_answers: dict[str, Any]
) -> dict[str, str]:
    """Build the ``AskUserQuestion`` ``answers`` map from the collected picks.

    ``raw_answers`` is keyed by the synthetic ``q<index>`` ids the input UI uses;
    the tool's ``answers`` map is keyed by the question itself. The exact key the
    CLI matches on is unspecified, so each answered question is recorded under
    both its ``question`` text and its ``header`` (extra keys are ignored).
    """
    raw_questions = tool_input.get("questions")
    if not isinstance(raw_questions, list):
        return {}
    answers: dict[str, str] = {}
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            continue
        value = _str(raw_answers.get(f"q{index}"))
        if not value:
            continue
        for key in (_str(raw.get("question")), _str(raw.get("header"))):
            if key:
                answers[key] = value
    return answers


_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _build_query_input(text: str, image_paths: Any) -> str | AsyncIterator[dict[str, Any]]:
    """Return query input for ``client.query``: a plain string, or a streamed
    user message with base64 image blocks when attachments are present.

    Shared by the initial turn and the steer control-file drain so steered
    image attachments reach the model too.
    """
    blocks = _image_content_blocks(image_paths)
    if not blocks:
        return text
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(blocks)

    async def _stream() -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "user",
            "message": {"role": "user", "content": content},
            "parent_tool_use_id": None,
        }

    return _stream()


def _image_content_blocks(paths: Any) -> list[dict[str, Any]]:
    """Build Anthropic base64 image blocks for the saved attachment paths.

    Unreadable files or unknown extensions are skipped rather than failing the
    whole turn -- a missing attachment should not lose the user's prompt.
    """
    if not isinstance(paths, list):
        return []
    blocks: list[dict[str, Any]] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = Path(raw)
        media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            logger.warning("failed to read image attachment %s", raw)
            continue
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(data).decode("ascii"),
                },
            }
        )
    return blocks


def _decision_allows(decision: Any) -> bool:
    if isinstance(decision, dict):
        return True
    return bool(decision == ApprovalRequest.DECISION_ACCEPT)


def _result_error(message: ResultMessage) -> str:
    if message.errors:
        return "; ".join(str(err) for err in message.errors)
    if message.result:
        return message.result
    return f"claude turn ended with status {message.subtype}"


def _steer_request(raw: bytes) -> tuple[str, list[str]]:
    """Parse one steer control line into ``(text, image_paths)``."""
    try:
        request = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", []
    if not isinstance(request, dict) or request.get("op") != "steer":
        return "", []
    raw_text = request.get("input")
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    raw_paths = request.get("inputImagePaths")
    image_paths = (
        [p for p in raw_paths if isinstance(p, str) and p.strip()]
        if isinstance(raw_paths, list)
        else []
    )
    return text, image_paths


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""
