from __future__ import annotations

from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

from hitch.main import demo


class DemoProxyMiddleware:
    """Route *.demo.localhost requests to the matching session demo."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        session_id = demo.session_id_from_demo_host(request.get_host())
        if session_id is None:
            return self.get_response(request)
        path = request.path_info.lstrip("/")
        return demo.proxy_demo_request(request, session_id, path, path_prefix="/")
