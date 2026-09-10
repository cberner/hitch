"""Fresh account quota reads for quota-bounded coding sessions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

from django.test import TestCase
from openai_codex.errors import InvalidRequestError, MethodNotFoundError
from openai_codex.generated.v2_all import GetAccountRateLimitsResponse

from hitch.main.models import CodexInstance
from hitch.main.runtime.codex_tools import ToolContext, handle_dynamic_tool_call, registered_dynamic_tool_specs


def _quota_response(used: int, secondary: int | None = 60) -> GetAccountRateLimitsResponse:
    return GetAccountRateLimitsResponse.model_validate({
        "rateLimits": {
            "limitId": "codex",
            "planType": "pro",
            "primary": {"usedPercent": used, "windowDurationMins": 300, "resetsAt": 1800000000},
            "secondary": {"usedPercent": secondary, "windowDurationMins": 10080} if secondary is not None else None,
        },
    })


class CodexQuotaToolTests(TestCase):
    def _call(self, **context: Any) -> dict[str, Any]:
        return handle_dynamic_tool_call(
            {"namespace": "hitch", "tool": "get_codex_quota", "arguments": {}},
            ToolContext(cwd="/repo", thread_id="current-thread", **context),
        )

    def test_registration_and_role_authorization(self) -> None:
        spec = next(spec for spec in registered_dynamic_tool_specs() if spec["name"] == "get_codex_quota")
        self.assertEqual(spec["namespace"], "hitch")
        self.assertEqual(spec["inputSchema"], {"type": "object", "properties": {}, "additionalProperties": False})
        for kind in ("unrelated", "autonomous_goal_run", "autonomous_goal_reviewer"):
            with self.subTest(kind=kind), patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry") as run:
                context = {"purpose": CodexInstance.PURPOSE_SYSTEM_AGENT, "agent_kind": kind}
                self.assertNotIn("get_codex_quota", [spec["name"] for spec in registered_dynamic_tool_specs(**context)])
                self.assertFalse(self._call(**context)["success"])
                run.assert_not_called()

    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_each_call_fetches_live_quota_and_failure_never_returns_previous_value(self, run: MagicMock) -> None:
        codex = MagicMock()
        run.side_effect = lambda _factory, operation, **_kwargs: operation(codex)
        codex._client.request.side_effect = [
            _quota_response(20), _quota_response(35),
            InvalidRequestError(-32600, "authentication required"),
            MethodNotFoundError(-32601, "method not found"),
            TimeoutError("timed out"),
        ]
        now = datetime(2026, 9, 10, tzinfo=UTC)
        with (
            patch("hitch.main.caches._RATE_LIMITS_CACHE_VALUE", {"remaining_percent": 99}),
            patch("hitch.main.caches._RATE_LIMITS_CACHE_HAS_VALUE", True),
            patch("hitch.main.caches.rate_limit.claim", return_value=False) as claim,
            patch("hitch.main.runtime.codex_tools.timezone.now", return_value=now),
        ):
            for remaining in (80, 65):
                response = self._call(enable_memories=True, web_search_mode="live")
                self.assertTrue(response["success"])
                data = json.loads(response["contentItems"][0]["text"])
                self.assertEqual(data["fetched_at"], now.isoformat())
                quota = data["rate_limits"]
                self.assertEqual(quota["limit_id"], "codex")
                self.assertEqual(quota["plan_type"], "pro")
                self.assertEqual(quota["primary"], {
                    "remaining_percent": remaining, "used_percent": 100 - remaining,
                    "window_duration_mins": 300, "resets_at": 1800000000,
                })
                self.assertEqual(quota["secondary"]["remaining_percent"], 40)
                self.assertEqual(quota["secondary"]["window_duration_mins"], 10080)
                self.assertIsNone(quota["secondary"]["resets_at"])
            for _ in range(3):
                response = self._call()
                self.assertFalse(response["success"])
                self.assertNotIn("remaining_percent", response["contentItems"][0]["text"])
            claim.assert_not_called()
        self.assertEqual(codex._client.request.call_count, 5)
        codex._client.request.assert_called_with(
            "account/rateLimits/read", None, response_model=GetAccountRateLimitsResponse,
        )
        self.assertEqual(run.call_args_list[0].kwargs, {"enable_memories": True, "web_search_mode": "live"})

    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_missing_windows_and_percentage_boundaries(self, run: MagicMock) -> None:
        for used, remaining in ((0, 100), (100, 0), (110, 0), (-10, 100)):
            with self.subTest(used=used):
                run.return_value = _quota_response(used, secondary=None)
                response = self._call()
                self.assertTrue(response["success"])
                quota = json.loads(response["contentItems"][0]["text"])["rate_limits"]
                self.assertEqual(quota["primary"]["remaining_percent"], remaining)
                self.assertIsNone(quota["secondary"])
        run.return_value = GetAccountRateLimitsResponse.model_validate({
            "rateLimits": {"secondary": {"usedPercent": 75}},
        })
        response = self._call()
        self.assertTrue(response["success"])
        quota = json.loads(response["contentItems"][0]["text"])["rate_limits"]
        self.assertIsNone(quota["primary"])
        self.assertEqual(quota["secondary"]["remaining_percent"], 25)
        self.assertIsNone(quota["secondary"]["window_duration_mins"])
        run.return_value = GetAccountRateLimitsResponse.model_validate({"rateLimits": {}})
        response = self._call()
        self.assertFalse(response["success"])
        self.assertIn("remaining quota is unknown", response["contentItems"][0]["text"])

    @patch("hitch.main.runtime.app_server_pool.run_borrowed_op_with_retry")
    def test_rejects_arguments_without_fetching(self, run: MagicMock) -> None:
        response = handle_dynamic_tool_call(
            {"tool": "get_codex_quota", "arguments": {"cached": True}},
            ToolContext(cwd="/repo", thread_id="current-thread"),
        )
        self.assertFalse(response["success"])
        run.assert_not_called()
