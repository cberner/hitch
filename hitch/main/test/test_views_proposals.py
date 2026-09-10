"""Inbox page and proposal endpoint tests."""

import html
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import (
    TestCase,
)
from django.urls import reverse

from hitch.main.models import (
    ProposedSession,
    SessionMetadata,
)
from hitch.main.test.support import (
    _cookie_value,
    _make_project,
    _seed_cookies,
    _setup_codex,
)
from hitch.main.test.views_helpers import (
    _SELECTED_PROJECT_COOKIE,
    _SHOW_NO_PROJECT_SESSIONS_COOKIE,
    _VISIBLE_SESSION_PROJECTS_COOKIE,
)


class InboxViewTests(TestCase):
    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_page_lists_proposals_for_selected_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            prompt=(
                "Go ahead and implement this proposed session.\n\n"
                "Implementation guidance:\n"
                "Add focused rollout parser tests before changing behavior."
            ),
            confidence=ProposedSession.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/rollout.py"],
            outcome_metadata={
                "auto_pr_enabled": True,
                "auto_qa_enabled": False,
            },
        )
        ProposedSession.objects.create(
            project=other_project,
            title="Other proposal",
            summary="Should not render.",
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(f'href="{reverse("inbox")}" aria-current="page"', nav_html)
        self.assertIn(
            'class="primary-nav-badge" aria-label="1 inbox message">1</span>',
            nav_html,
        )
        self.assertContains(response, "data-visible-projects-open")
        self.assertContains(response, "Visible projects")
        main_start = body.index("<main>")
        self.assertLess(body.index('aria-label="Inbox actions"'), main_start)
        self.assertLess(body.index("data-visible-projects-open"), main_start)
        self.assertContains(
            response,
            '<dialog class="new-session" data-visible-projects-dialog',
            html=False,
        )
        self.assertContains(response, "Add parser coverage")
        self.assertContains(response, "This adds focused parser coverage.")
        self.assertContains(response, "hitch/main/rollout.py")
        self.assertContains(response, "data-proposed-session-do")
        self.assertContains(response, f'data-proposed-session-id="{proposal.pk}"')
        self.assertContains(response, f'data-proposed-session-project="{project.pk}"')
        start_modal_title = '<h2 id="do-session-title" tabindex="-1" autofocus>Continue proposed session</h2>'
        self.assertContains(response, start_modal_title)
        self.assertContains(response, "if (doHeading) doHeading.focus();")
        self.assertNotContains(response, "doPrompt.focus()")
        self.assertContains(
            response,
            'if (doForm) doForm.addEventListener("submit", () => hideDialog(doDialog));',
        )
        self.assertContains(response, 'data-proposed-session-auto-pr="true"')
        self.assertContains(response, 'data-proposed-session-auto-qa="false"')
        self.assertContains(
            response,
            'data-proposed-session-prompt="Go ahead and implement this proposed session.',
        )
        self.assertNotContains(response, "auto goals")
        self.assertContains(response, f'aria-label="Actions for {proposal.title}"')
        self.assertContains(
            response,
            f'action="{reverse("update_proposed_session_outcome", args=[proposal.pk])}"',
        )
        proposal_header_start = body.index('<div class="proposal-header">')
        proposal_actions_start = body.index('<div class="proposal-actions">', proposal_header_start)
        proposal_menu_start = body.index('<div class="proposal-menu"', proposal_header_start)
        self.assertLess(proposal_menu_start, proposal_actions_start)
        self.assertContains(response, f'value="{ProposedSession.OUTCOME_DISMISSED}"')
        self.assertContains(response, 'name="proposed_session"')
        self.assertNotContains(response, "Other proposal")


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_recovers_stale_proposal_start_claim(self, mock_codex: MagicMock, _mock_discover: MagicMock) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        stale_claimed_at = datetime.now(UTC) - ProposedSession.ACCEPTED_SESSION_START_CLAIM_TTL - timedelta(seconds=1)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "resolved_by": "user",
                "accepted_thread_id": "",
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (stale_claimed_at.isoformat()),
            },
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Add parser coverage")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)
        self.assertNotIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )
        self.assertNotIn("resolved_by", proposal.outcome_metadata)


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_keeps_active_proposal_start_claim_hidden(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "accepted_thread_id": "",
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (datetime.now(UTC).isoformat()),
            },
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Add parser coverage")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertIn(
            ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY,
            proposal.outcome_metadata,
        )


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_visible_projects_filter_messages(self, mock_codex: MagicMock, _mock_discover: MagicMock) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _setup_codex(mock_codex)
        ProposedSession.objects.create(
            project=project,
            title="Matching proposal",
            summary="Should not render.",
        )
        ProposedSession.objects.create(
            project=other_project,
            title="Other proposal",
            summary="Should render.",
        )
        ProposedSession.objects.create(
            title="No repo notice",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            summary="No project attached.",
        )

        response = self.client.post(
            reverse("update_visible_session_projects"),
            data={
                "visible_project": [str(other_project.pk)],
                "show_no_project_sessions": "true",
                "next": reverse("inbox"),
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("inbox"))
        self.assertEqual(
            _cookie_value(response, _VISIBLE_SESSION_PROJECTS_COOKIE),
            f"[{other_project.pk}]",
        )
        self.assertEqual(
            _cookie_value(response, _SHOW_NO_PROJECT_SESSIONS_COOKIE),
            "true",
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Visible projects")
        self.assertContains(response, "Other proposal")
        self.assertContains(response, "No repo notice")
        self.assertContains(response, "No repo -")
        self.assertNotContains(response, "Matching proposal")


    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_update_outcome_rejects_proposal_hidden_by_visible_project_filter(self, mock_cleanup: MagicMock) -> None:
        visible_project = _make_project()
        hidden_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(
            self.client,
            **{
                _SELECTED_PROJECT_COOKIE: str(visible_project.pk),
                _VISIBLE_SESSION_PROJECTS_COOKIE: f"[{visible_project.pk}]",
            },
        )
        proposal = ProposedSession.objects.create(
            project=hidden_project,
            title="Add docs coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {
                "outcome_status": ProposedSession.OUTCOME_REJECTED,
                "reason": "Not useful enough.",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"proposed session is required")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        mock_cleanup.assert_not_called()


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_prefills_prompt_and_project_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex)
        prompt = "Debug and fix the user's issue from session UID thread-1.\n\nUser issue: "

        response = self.client.get(reverse("new_session"), {"prompt": prompt, "project": str(project.pk)})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, html.escape(prompt))
        self.assertContains(response, f'value="{project.pk}" selected')


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/other")])
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_rejects_unavailable_project_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex)

        response = self.client.get(reverse("new_session"), {"prompt": "debug this", "project": str(project.pk)})

        self.assertEqual(response.status_code, 404)


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_prefills_bare_repo_cwd_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex)

        response = self.client.get(reverse("new_session"), {"prompt": "debug this", "cwd": "/repo"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "debug this")
        self.assertContains(response, 'value="__bare_repo__" selected')
        self.assertContains(response, '<option value="/repo" selected>')
        self.assertNotContains(response, f'value="{project.pk}" selected')


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/other")])
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_rejects_unavailable_bare_repo_cwd(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        _setup_codex(mock_codex)

        response = self.client.get(reverse("new_session"), {"cwd": "/repo"})

        self.assertEqual(response.status_code, 404)


    @patch("hitch.main.repos.discover_repos", return_value=[Path("/other")])
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_rejects_proposed_session_for_unavailable_repo(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            prompt="Add focused rollout parser tests before changing behavior.",
        )

        response = self.client.get(f"{reverse('new_session')}?proposed_session={proposal.pk}")

        self.assertEqual(response.status_code, 404)
        mock_codex.assert_not_called()


    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_reject_proposed_session_requires_reason(self, mock_cleanup: MagicMock) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_REJECTED},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"reason is required")
        mock_cleanup.assert_not_called()


    def test_outcome_endpoint_cannot_accept_without_starting_session(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_ACCEPTED},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"proposal must be started before acceptance")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)
        self.assertIsNone(proposal.accepted_session)


    def test_notice_rejects_non_dismissed_outcome(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        notice = ProposedSession.objects.create(
            project=project,
            title="No proposal from Improve tests",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[notice.pk]),
            {"outcome_status": ProposedSession.OUTCOME_ACCEPTED},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"outcome status is invalid")


    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_update_outcome_rejects_already_resolved_proposal(self, mock_cleanup: MagicMock) -> None:
        # A stale inbox tab can still post a dismiss/reject for an already
        # accepted legacy proposal. Re-deciding it must be refused so the
        # recorded outcome is not corrupted.
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=candidate,
        )
        for outcome in (
            ProposedSession.OUTCOME_DISMISSED,
            ProposedSession.OUTCOME_REJECTED,
        ):
            with self.subTest(outcome=outcome):
                response = self.client.post(
                    reverse("update_proposed_session_outcome", args=[proposal.pk]),
                    {"outcome_status": outcome, "reason": "Changed my mind."},
                )

                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.content, b"proposed session has already been resolved")
                proposal.refresh_from_db()
                self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
                self.assertEqual(proposal.accepted_session, candidate)
        # Resolving an already accepted proposal never removes its worktree.
        mock_cleanup.assert_not_called()


    def test_update_outcome_rejects_unset_target_status(self) -> None:
        # OUTCOME_UNSET is the pending inbox state, not a decision; the endpoint
        # must not let a request re-open a proposal by posting it.
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_UNSET},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"outcome status is invalid")
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_UNSET)



    def test_resolves_ordinary_proposals_and_notices(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        for kind, outcome, reason in (
            ("proposal", "rejected", "Already done"),
            ("proposal", "dismissed", ""),
            ("notice", "dismissed", ""),
        ):
            with self.subTest(kind=kind, outcome=outcome):
                proposal = ProposedSession.objects.create(project=project, title="Follow-up", inbox_kind=kind)
                response = self.client.post(
                    reverse("update_proposed_session_outcome", args=[proposal.pk]),
                    {"outcome_status": outcome, "reason": reason},
                )
                self.assertRedirects(response, reverse("inbox"), fetch_redirect_response=False)
                proposal.refresh_from_db()
                self.assertEqual(proposal.outcome_status, outcome)
                self.assertEqual(proposal.outcome_notes, reason)
                self.assertEqual(proposal.outcome_metadata["resolved_by"], "user")
