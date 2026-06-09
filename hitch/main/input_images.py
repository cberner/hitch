"""Enforce per-request limits on image attachments during multipart upload.

``_limit_input_image_uploads`` wraps a view so oversized or too-many image
attachments are rejected while the request body is still streaming, before the
view (or Codex) ever sees them. The streaming upload handler caps both the
per-image size and the attachment count; ``_input_image_request_size_error``
rejects an over-large request up front from the Content-Length header.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, override

from django.core.exceptions import SuspiciousOperation
from django.core.files.uploadedfile import UploadedFile
from django.core.files.uploadhandler import FileUploadHandler
from django.http import HttpRequest, HttpResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt, csrf_protect

_INPUT_IMAGE_FIELD = "input_images"
_INPUT_IMAGE_MAX_COUNT = 4
_INPUT_IMAGE_MAX_BYTES = 20 * 1024 * 1024
_INPUT_IMAGE_MAX_REQUEST_BYTES = (
    _INPUT_IMAGE_MAX_COUNT * _INPUT_IMAGE_MAX_BYTES + 1024 * 1024
)
_INPUT_IMAGE_ACCEPT = "image/png,image/jpeg,image/gif,image/webp"


class _InputImageLimitUploadHandler(FileUploadHandler):
    def __init__(self, request: HttpRequest | None = None) -> None:
        super().__init__(request)
        self._input_image_count = 0
        self._current_input_image_bytes = 0
        self._tracking_input_image = False

    @override
    def new_file(
        self,
        field_name: str,
        file_name: str,
        content_type: str,
        content_length: int | None,
        charset: str | None = None,
        content_type_extra: dict[str, bytes] | None = None,
    ) -> None:
        super().new_file(
            field_name,
            file_name,
            content_type,
            content_length,
            charset,
            content_type_extra,
        )
        self._tracking_input_image = field_name == _INPUT_IMAGE_FIELD
        self._current_input_image_bytes = 0
        if not self._tracking_input_image:
            return
        self._input_image_count += 1
        if self._input_image_count > _INPUT_IMAGE_MAX_COUNT:
            raise SuspiciousOperation(
                f"at most {_INPUT_IMAGE_MAX_COUNT} image attachments are allowed"
            )
        if content_length is not None and content_length > _INPUT_IMAGE_MAX_BYTES:
            raise SuspiciousOperation("image attachment is too large")

    @override
    def receive_data_chunk(self, raw_data: bytes, _start: int) -> bytes:
        if self._tracking_input_image:
            self._current_input_image_bytes += len(raw_data)
            if self._current_input_image_bytes > _INPUT_IMAGE_MAX_BYTES:
                raise SuspiciousOperation("image attachment is too large")
        return raw_data

    @override
    def file_complete(self, _file_size: int) -> UploadedFile | None:
        return None


def _limit_input_image_uploads(
    view_func: Callable[..., HttpResponse],
) -> Callable[..., HttpResponse]:
    protected_view = csrf_protect(view_func)

    @csrf_exempt
    @wraps(view_func)
    def wrapper(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        if error := _input_image_request_size_error(request):
            return HttpResponseBadRequest(error)
        content_type = (
            request.content_type or request.META.get("CONTENT_TYPE", "")
        ).lower()
        if request.method == "POST" and content_type.startswith("multipart/"):
            request.upload_handlers.insert(0, _InputImageLimitUploadHandler(request))
        try:
            return protected_view(request, *args, **kwargs)
        except SuspiciousOperation as exc:
            message = str(exc)
            if message.startswith(("image attachment", "at most ")):
                return HttpResponseBadRequest(message)
            raise

    return wrapper


def _input_image_request_size_error(request: HttpRequest) -> str | None:
    raw_content_length = request.META.get("CONTENT_LENGTH")
    if not raw_content_length:
        return None
    try:
        content_length = int(raw_content_length)
    except ValueError:
        return None
    if content_length > _INPUT_IMAGE_MAX_REQUEST_BYTES:
        return "image attachments are too large"
    return None
