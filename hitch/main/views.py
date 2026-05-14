import shutil

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from openai_codex import AppServerConfig, Codex


def index(request: HttpRequest) -> HttpResponse:
    config = AppServerConfig(codex_bin=shutil.which("codex"))
    with Codex(config=config) as codex:
        sessions = codex.thread_list().data
    sessions = sorted(sessions, key=lambda s: s.updated_at, reverse=True)
    return render(request, "index.html", {"sessions": sessions})
