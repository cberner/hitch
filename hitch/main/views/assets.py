"""Shared session assets, revalidated against their current rendered contents."""

import hashlib

from django.http import Http404, HttpRequest, HttpResponse
from django.template.loader import render_to_string
from django.utils.cache import get_conditional_response
from django.views.decorators.http import require_safe

_CONTENT_TYPES = {"session.js": "text/javascript", "session.css": "text/css"}


@require_safe
def session_asset(request: HttpRequest, asset: str) -> HttpResponse:
    content_type = _CONTENT_TYPES.get(asset)
    if content_type is None:
        raise Http404("asset not found")
    # No request context: these shared includes must never contain session data.
    body = render_to_string(f"assets/{asset}").encode()
    etag = '"' + hashlib.sha256(body).hexdigest() + '"'
    response = HttpResponse(body, content_type=content_type, headers={
        "ETag": etag, "Cache-Control": "no-cache",
    })
    return get_conditional_response(request, etag=etag, response=response) or response
