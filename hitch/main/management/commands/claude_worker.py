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
    _notify_system_agents,
    _wait_for_decision,
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
_STEER_POLL_INTERVAL = 0.2


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
                async for message in client.receive_response():
                    self._capture_session_id(message)
                    for method, payload in self._translator.translate(message):
                        self._write_event(method, payload)
                    if isinstance(message, ResultMessage):
                        self._record_result(message)
            finally:
                steer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await steer_task

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
        item = {"type": "toolCall", "tool": tool_name}
    return {"item": item, "tool": tool_name}


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
