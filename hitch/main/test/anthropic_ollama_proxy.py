"""Minimal Anthropic Messages API -> ollama (OpenAI API) translation proxy.

The ``claude`` CLI (driven by ``claude-agent-sdk``) speaks the Anthropic
Messages API and honours ``ANTHROPIC_BASE_URL``. Ollama only exposes an
OpenAI-compatible API. This in-process proxy bridges the two so an integration
test can run a real Claude Code turn against a local ``qwen`` model -- the
Claude-backend analog of the Codex ``test_sdk_runs_turn_via_ollama`` test.

It is intentionally minimal: it forwards a single (non-streaming) completion
request to ollama and re-emits the reply as an Anthropic streaming SSE
sequence. It does not implement tool use -- a trivial text turn is enough to
prove the wiring end to end.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, override

OLLAMA_CHAT_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "qwen2.5-coder:0.5b"


def anthropic_to_openai_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten an Anthropic Messages request into OpenAI chat messages."""
    messages: list[dict[str, str]] = []
    system_text = _blocks_text(payload.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        text = _blocks_text(message.get("content"))
        if text:
            messages.append({"role": message.get("role", "user"), "content": text})
    return messages


def anthropic_sse(text: str, model: str) -> bytes:
    """Render ``text`` as an Anthropic streaming Messages SSE response."""
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_proxy",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events
    )


def _blocks_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            block.get("text", "")
            for block in value
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(p for p in parts if p)
    return ""


def _call_ollama(messages: list[dict[str, str]]) -> str:
    body = json.dumps(
        {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    ).encode()
    request = urllib.request.Request(
        OLLAMA_CHAT_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        data = json.loads(response.read())
    return str(data["choices"][0]["message"]["content"])


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}
        # The CLI may preflight token counting; answer with a tiny JSON body
        # rather than an SSE stream.
        if self.path.rstrip("/").endswith("count_tokens"):
            self._json({"input_tokens": 1})
            return
        try:
            text = _call_ollama(anthropic_to_openai_messages(payload))
        except Exception as exc:  # noqa: BLE001 - fail the turn, don't mask it
            # Surface proxy/Ollama failures as an HTTP error rather than a normal
            # assistant message: otherwise the integration test sees a completed
            # turn with text and passes even though the leg it validates is broken.
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"proxy error: {exc}".encode())
            return
        model = payload.get("model", "claude")
        body = anthropic_sse(text, model if isinstance(model, str) else "claude")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._json({})

    def _json(self, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    @override
    def log_message(self, *_args: Any) -> None:
        # Silence the default stderr access logging in tests.
        return


class AnthropicOllamaProxy:
    """Context manager that serves the proxy on an ephemeral localhost port."""

    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def __enter__(self) -> AnthropicOllamaProxy:
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
