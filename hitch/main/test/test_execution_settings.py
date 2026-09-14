from django.test import SimpleTestCase

from hitch.main.models import SessionMetadata
from hitch.main.sessions.execution_settings import (
    PreviousTurnApproval,
    RequestApproval,
    resolve_approval,
)


class ApprovalResolutionTests(SimpleTestCase):
    def test_explicit_overrides_win_for_requests_and_background_turns(self) -> None:
        for mode in ("deny_all", "prompt_user", "auto_review", "approve_all"):
            metadata = SessionMetadata(
                approval_mode=mode, approval_snapshot_mode="approve_all", approval_snapshot_instance_id=7,
            )
            for defaults in (RequestApproval("deny_all"), PreviousTurnApproval(7, "deny_all")):
                with self.subTest(mode=mode, defaults=defaults):
                    resolved = resolve_approval(metadata, defaults)
                    self.assertEqual((resolved.mode, resolved.source), (mode, "session_override"))

    def test_reset_snapshot_applies_only_to_its_previous_turn(self) -> None:
        metadata = SessionMetadata(approval_snapshot_mode="deny_all", approval_snapshot_instance_id=7)
        for defaults, mode, source in (
            (RequestApproval("approve_all"), "approve_all", "request"),
            (PreviousTurnApproval(7, "approve_all"), "deny_all", "session_snapshot"),
            (PreviousTurnApproval(8, "auto_review"), "auto_review", "previous_turn"),
        ):
            with self.subTest(defaults=defaults):
                resolved = resolve_approval(metadata, defaults)
                self.assertEqual((resolved.mode, resolved.source), (mode, source))

    def test_missing_or_invalid_session_settings_fall_back_to_the_context(self) -> None:
        for metadata in (
            None,
            SessionMetadata(approval_snapshot_mode="deny_all"),
            SessionMetadata(approval_mode="invalid", approval_snapshot_mode="invalid", approval_snapshot_instance_id=7),
        ):
            for defaults, source in (
                (RequestApproval("prompt_user"), "request"),
                (PreviousTurnApproval(7, "prompt_user"), "previous_turn"),
            ):
                with self.subTest(metadata=metadata, defaults=defaults):
                    resolved = resolve_approval(metadata, defaults)
                    self.assertEqual((resolved.mode, resolved.source), ("prompt_user", source))
        for defaults in (RequestApproval(""), PreviousTurnApproval(7, "invalid")):
            resolved = resolve_approval(None, defaults)
            self.assertEqual((resolved.mode, resolved.source), ("auto_review", "default"))
