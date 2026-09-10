"""Validate browser answers against the Codex user-input response protocol."""

from typing import Any


def wire_user_input_response(response: dict[str, Any]) -> dict[str, Any]:
    answers = {}
    for key, value in response["answers"].items():
        if isinstance(value, dict):
            if set(value) != {"answers"}:
                raise ValueError("invalid answer object")
            value = value["answers"]
            if not isinstance(value, list):
                raise ValueError("answer object requires a string list")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(answer, str) for answer in value):
            raise ValueError("answers must be strings or string lists")
        answers[key] = {"answers": value}
    return {"answers": answers}
