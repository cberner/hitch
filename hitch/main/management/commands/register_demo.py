from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.urls import reverse

from hitch.main import demo
from hitch.main.models import SessionDemo

FILE_TEXT_READ_BYTES = demo.MAX_LOG_CHARS * 4 + 4


class Command(BaseCommand):
    help = "Register a web demo container for a Hitch session."

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--session-id", required=True)
        parser.add_argument("--token", required=True)
        parser.add_argument(
            "--status",
            required=True,
            choices=[
                SessionDemo.STATUS_PREPARING,
                SessionDemo.STATUS_ACTIVE,
                SessionDemo.STATUS_FAILED,
            ],
        )
        parser.add_argument("--container-name", default="")
        parser.add_argument("--container-id", default="")
        parser.add_argument("--host", default="")
        parser.add_argument("--port", type=int)
        parser.add_argument("--runtime", default="")
        parser.add_argument("--logs", default="")
        parser.add_argument("--logs-file", default="")
        parser.add_argument("--error", default="")
        parser.add_argument("--error-file", default="")
        parser.add_argument("--json", action="store_true", dest="json_output")

    @override
    def handle(self, *args: Any, **options: Any) -> str | None:
        payload: dict[str, Any] = {
            "token": options["token"],
            "status": options["status"],
        }
        _add_optional(payload, "container_name", options["container_name"])
        _add_optional(payload, "container_id", options["container_id"])
        _add_optional(payload, "host", options["host"])
        _add_optional(payload, "runtime", options["runtime"])
        if options["port"] is not None:
            payload["port"] = options["port"]
        logs = _option_text_or_file(
            text=options["logs"],
            file_path=options["logs_file"],
            text_option="--logs",
            file_option="--logs-file",
        )
        error = _option_text_or_file(
            text=options["error"],
            file_path=options["error_file"],
            text_option="--error",
            file_option="--error-file",
        )
        _add_optional(payload, "logs", logs)
        _add_optional(payload, "error", error)
        try:
            session_demo = demo.register_demo_container(options["session_id"], payload)
        except demo.DemoError as exc:
            raise CommandError(str(exc)) from exc

        response = {
            "status": session_demo.status,
            "demo_url": reverse(
                "session_demo_proxy_root",
                kwargs={"session_id": session_demo.thread_id},
            ),
        }
        if options["json_output"]:
            return json.dumps(response, sort_keys=True)
        return (
            f"Registered demo as {session_demo.status}\n"
            f"Demo: {response['demo_url']}"
        )


def _add_optional(payload: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        payload[key] = value


def _option_text_or_file(
    *,
    text: str,
    file_path: str,
    text_option: str,
    file_option: str,
) -> str:
    if text and file_path:
        raise CommandError(f"use either {text_option} or {file_option}, not both")
    if not file_path:
        return text
    path = Path(file_path)
    try:
        file_size = path.stat().st_size
        start = max(file_size - FILE_TEXT_READ_BYTES, 0)
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(FILE_TEXT_READ_BYTES)
    except OSError as exc:
        raise CommandError(f"failed to read {file_option}: {exc}") from exc
    try:
        return _decode_bounded_utf8(data, truncated=start > 0)
    except UnicodeDecodeError as exc:
        raise CommandError(
            f"failed to read {file_option}: file is not valid UTF-8"
        ) from exc


def _decode_bounded_utf8(data: bytes, *, truncated: bool) -> str:
    if not data:
        return ""
    offset = _utf8_tail_start_offset(data) if truncated else 0
    return data[offset:].decode("utf-8")[-demo.MAX_LOG_CHARS:]


def _utf8_tail_start_offset(data: bytes) -> int:
    offset = 0
    while offset < min(4, len(data)) and 0x80 <= data[offset] <= 0xBF:
        offset += 1
    return offset
