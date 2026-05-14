import shutil
import threading
from collections.abc import Iterator
from typing import Any

from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from openai_codex import AppServerConfig, Codex

from hitch.main.repos import discover_repos

# Friendly labels for non-message thread item types. Anything not in this map
# falls back to the raw type tag so we never silently drop an item from the UI.
_NON_MESSAGE_LABELS = {
    "commandExecution": "Command",
    "mcpToolCall": "MCP tool call",
    "dynamicToolCall": "Tool call",
    "fileChange": "File change",
    "webSearch": "Web search",
    "collabAgentToolCall": "Collab agent call",
    "imageView": "Image view",
    "imageGeneration": "Image generation",
    "reasoning": "Reasoning",
    "plan": "Plan",
    "hookPrompt": "Hook prompt",
    "enteredReviewMode": "Entered review mode",
    "exitedReviewMode": "Exited review mode",
    "contextCompaction": "Context compaction",
}


def index(request: HttpRequest) -> HttpResponse:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        sessions = codex.thread_list().data
    sessions = sorted(sessions, key=lambda s: s.updated_at, reverse=True)
    repos = [str(p) for p in discover_repos()]
    return render(
        request,
        "index.html",
        {"sessions": sessions, "repos": repos, "new_session_url": reverse("new_session")},
    )


def session(request: HttpRequest, session_id: str) -> HttpResponse:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        thread = codex._client.thread_read(session_id, include_turns=True).thread
    entries = list(_render_entries(thread))
    return render(request, "session.html", {"thread": thread, "entries": entries})


@require_http_methods(["POST"])
def new_session(request: HttpRequest) -> HttpResponse:
    prompt = request.POST.get("prompt", "").strip()
    cwd = request.POST.get("cwd", "").strip()
    if not prompt:
        return HttpResponseBadRequest("prompt is required")
    if not cwd:
        return HttpResponseBadRequest("cwd is required")
    # Restrict cwd to a discovered repo so an arbitrary path can't be injected
    # via the form post.
    allowed = {str(p) for p in discover_repos()}
    if cwd not in allowed:
        return HttpResponseBadRequest("cwd must be a discovered repository")

    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        thread = codex.thread_start(cwd=cwd)
        thread_id = thread.id
    # Run the initial turn from a background worker so the HTTP request can
    # redirect immediately. The thread is already persisted by thread_start,
    # so the worker just resumes it to issue the user prompt.
    _spawn_initial_turn(thread_id, prompt)
    return redirect("session", session_id=thread_id)


def _render_entries(thread: Any) -> Iterator[dict[str, Any]]:
    """Walk every turn's items in order, emitting one entry per item.

    Every thread item — user/agent messages and every tool-call variant — gets
    its own entry so the UI shows the full activity log instead of collapsing
    tool calls behind an aggregate count. Each entry carries the turn's
    started_at timestamp; per-item timestamps are not exposed by the SDK.
    """
    for turn in thread.turns:
        timestamp = getattr(turn, "started_at", None)
        for thread_item in turn.items:
            item = thread_item.root
            item_type = item.type
            if item_type == "userMessage":
                yield {
                    "kind": "user",
                    "text": _user_message_text(item),
                    "timestamp": timestamp,
                }
            elif item_type == "agentMessage":
                yield {
                    "kind": "agent",
                    "text": item.text,
                    "timestamp": timestamp,
                }
            else:
                yield {
                    "kind": "tool_call",
                    "type": item_type,
                    "label": _NON_MESSAGE_LABELS.get(item_type, item_type),
                    "detail": _tool_call_detail(item, item_type),
                    "status": _tool_call_status(item),
                    "timestamp": timestamp,
                }


def _user_message_text(item: Any) -> str:
    parts: list[str] = []
    for input_item in item.content:
        inner = input_item.root
        match inner.type:
            case "text":
                parts.append(inner.text)
            case "mention":
                parts.append(f"@{inner.name}")
            case "skill":
                parts.append(f"/{inner.name}")
            case "image":
                parts.append("[image]")
            case "localImage":
                parts.append(f"[image: {inner.path}]")
    return "\n".join(parts)


def _spawn_initial_turn(thread_id: str, prompt: str) -> None:
    threading.Thread(
        target=_run_initial_turn,
        args=(thread_id, prompt),
        daemon=True,
    ).start()


def _run_initial_turn(thread_id: str, prompt: str) -> None:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        thread = codex.thread_resume(thread_id)
        thread.run(prompt)


def _tool_call_detail(item: Any, item_type: str) -> str:
    """Return a short, human-readable description of a tool-call item.

    Returns an empty string for item types that do not carry useful inline
    detail; the label alone is enough to surface them in the UI.
    """
    match item_type:
        case "commandExecution":
            return getattr(item, "command", "") or ""
        case "mcpToolCall":
            return f"{item.server} / {item.tool}"
        case "dynamicToolCall":
            namespace = getattr(item, "namespace", None)
            return f"{namespace}::{item.tool}" if namespace else item.tool
        case "fileChange":
            paths = [str(change.path) for change in getattr(item, "changes", []) or []]
            if not paths:
                return ""
            if len(paths) == 1:
                return paths[0]
            return f"{paths[0]} (+{len(paths) - 1} more)"
        case "webSearch":
            return getattr(item, "query", "") or ""
        case "plan":
            text = getattr(item, "text", "") or ""
            return text.split("\n", 1)[0]
        case "imageView":
            return getattr(item, "path", "") or ""
        case "imageGeneration":
            return (
                getattr(item, "revised_prompt", None)
                or getattr(item, "saved_path", None)
                or ""
            )
        case "collabAgentToolCall":
            tool = getattr(item, "tool", None)
            tool_name = getattr(tool, "value", None) or (tool if isinstance(tool, str) else "")
            receivers = getattr(item, "receiver_thread_ids", None) or []
            if receivers:
                return f"{tool_name} → {receivers[0]}"
            return str(tool_name)
        case _:
            return ""


def _tool_call_status(item: Any) -> str | None:
    """Return a non-success status string (e.g. ``failed``) or None.

    Hides ``completed`` so the common case stays uncluttered; surfaces
    in-progress, failed, and declined so unusual outcomes are visible.
    """
    status = getattr(item, "status", None)
    if status is None:
        return None
    value = getattr(status, "value", status)
    if not isinstance(value, str) or value == "completed":
        return None
    return value
