"""Autonomous-goal page and proposal endpoint tests."""


import html
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import override
from unittest.mock import MagicMock, patch

from django.test import (
    TestCase,
)
from django.urls import reverse
from django.utils import timezone

from hitch.main import views
from hitch.main.goals import autonomous_goal_prompts, autonomous_goal_proposal_stack, autonomous_goal_run_display
from hitch.main.models import (
    AutonomousGoal,
    CodexInstance,
    ProposedSession,
    SessionMetadata,
    SystemAgentRun,
    SystemWorkflow,
)
from hitch.main.runtime import codex_events
from hitch.main.test.support import (
    _cookie_value,
    _make_project,
    _seed_cookies,
    _setup_codex,
)
from hitch.main.test.views_helpers import (
    _SELECTED_PROJECT_COOKIE,
    _SHOW_NO_PROJECT_SESSIONS_COOKIE,
    _USE_WORKTREES_COOKIE,
    _VISIBLE_SESSION_PROJECTS_COOKIE,
)
from hitch.main.workflows import autonomous_goals, system_agents


class AutonomousGoalViewTests(TestCase):
    @override
    def setUp(self) -> None:
        super().setUp()
        self.quota_patcher = patch(
            "hitch.main.workflows.autonomous_goals._auto_proposals_paused_by_usage_quota_throttled",
            return_value=False,
        )
        self.mock_auto_proposals_paused_by_quota = self.quota_patcher.start()
        self.addCleanup(self.quota_patcher.stop)

    @patch("hitch.main.workflows.autonomous_goals.maybe_start_auto_proposal_workflows")
    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_get_pages_do_not_start_auto_proposals(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_scheduler: MagicMock,
    ) -> None:
        project = _make_project()
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)

        for route in ("index", "inbox", "autonomous_goals"):
            with self.subTest(route=route):
                response = self.client.get(reverse(route))
                self.assertEqual(response.status_code, 200)

        mock_scheduler.assert_not_called()

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_lists_goals_and_inbox_count_for_selected_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            web_search_mode=AutonomousGoal.WEB_SEARCH_LIVE,
            auto_merge_to_local_branch=True,
            auto_merge_branch="main",
            proposal_budget=25_000_000,
        )
        AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not render.",
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Deleted goal",
            goal="Should not render.",
            deleted_at=timezone.now(),
        )
        ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )
        ProposedSession.objects.create(
            project=other_project,
            title="Other proposal",
            summary="Should not count for selected project.",
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        nav_start = body.index('<nav class="primary-nav"')
        nav_end = body.index("</nav>", nav_start)
        nav_html = body[nav_start:nav_end]
        self.assertIn(
            f'href="{reverse("autonomous_goals")}" aria-current="page"', nav_html
        )
        self.assertIn(f'href="{reverse("inbox")}"', nav_html)
        self.assertIn(
            'class="primary-nav-badge" aria-label="1 inbox message">1</span>',
            nav_html,
        )
        self.assertIn(">auto goals</a>", nav_html)
        self.assertContains(response, "--accent-soft")
        self.assertContains(response, "--shadow-lg")
        self.assertContains(response, "[hidden] { display: none !important; }")
        self.assertContains(response, "Improve tests")
        self.assertContains(response, "Ambition")
        self.assertContains(response, "Ambition: High")
        self.assertContains(response, "Autonomy")
        self.assertContains(response, "Autonomy: Draft patch")
        self.assertContains(response, "Auto-QA: On")
        self.assertContains(
            response, 'value="draft_patch" data-auto-qa-supported="true"'
        )
        self.assertContains(
            response, 'value="draft_pr" data-auto-qa-supported="false"'
        )
        self.assertContains(
            response,
            'value="draft_pr" data-auto-qa-supported="false" data-auto-qa-required="true"',
        )
        self.assertContains(response, "Web search: Live")
        self.assertContains(response, "Proposal budget: 25M tokens")
        self.assertContains(response, "Auto-proposal: Off")
        self.assertContains(response, "Auto merge: main")
        self.assertContains(response, 'class="goal-menu" data-goal-menu')
        self.assertContains(response, 'role="menuitem">Run</button>')
        self.assertContains(
            response,
            f'action="{reverse("run_autonomous_goal", args=[goal.pk])}"',
        )
        self.assertContains(response, 'role="menuitem">Delete</button>')
        self.assertContains(
            response,
            f'action="{reverse("delete_autonomous_goal", args=[goal.pk])}"',
        )
        self.assertContains(
            response,
            f'data-edit-url="{reverse("edit_autonomous_goal", args=[goal.pk])}"',
        )
        self.assertContains(
            response, f'data-autonomy="{AutonomousGoal.AUTONOMY_DRAFT_PATCH}"'
        )
        self.assertContains(response, 'data-auto-qa="true"')
        self.assertContains(
            response, f'data-web-search-mode="{AutonomousGoal.WEB_SEARCH_LIVE}"'
        )
        self.assertContains(response, 'data-auto-proposal-enabled="false"')
        self.assertContains(response, 'data-proposal-budget="25"')
        self.assertContains(response, 'data-auto-merge-to-local-branch="true"')
        self.assertContains(response, 'data-auto-merge-branch="main"')
        self.assertContains(response, 'data-autonomous-goal-edit')
        self.assertNotContains(response, "Add parser coverage")
        self.assertNotContains(response, 'name="proposed_session"')
        self.assertNotContains(response, "Other goal")
        self.assertNotContains(response, "Deleted goal")

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_renders_goal_body_as_markdown(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        goal_text = "\n".join(
            [
                "# Parser audit",
                "",
                "- Add fixture coverage",
                "- Check [docs](https://example.com/parser)",
                "",
                "<script>alert(1)</script>",
            ]
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Improve parser",
            goal=goal_text,
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        body_start = body.index('<div class="goal-body markdown">')
        body_end = body.index('<div class="goal-meta">', body_start)
        goal_body_html = body[body_start:body_end]
        self.assertIn("<h1>Parser audit</h1>", goal_body_html)
        self.assertIn("<ul>", goal_body_html)
        self.assertIn("<li>Add fixture coverage</li>", goal_body_html)
        self.assertIn(
            '<a href="https://example.com/parser">docs</a>',
            goal_body_html,
        )
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", goal_body_html)
        self.assertNotIn("<script>", goal_body_html)
        self.assertIn(f'data-goal="{html.escape(goal_text, quote=True)}"', body)

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_shows_tappable_run_status_indicators(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        blocked_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve raptorq",
            goal="Investigate raptorq failures.",
        )
        running_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        running_no_tokens_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve parser",
            goal="Track persisted token usage.",
        )
        running_unrecorded_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve formatter",
            goal="Track unrecorded token usage.",
        )
        blocked_no_log_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve docs",
            goal="Investigate documentation failures.",
        )
        blocked_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(blocked_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step=system_agents.STEP_BLOCKED,
            state={
                "autonomous_goal_id": blocked_goal.pk,
                "error": "raptorq decoder exhausted repair symbols",
            },
        )
        blocked_instance = CodexInstance.objects.create(
            pid=0,
            thread_id="blocked-agent-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_FAILED,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=blocked_workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=blocked_workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=blocked_instance.thread_id,
            instance=blocked_instance,
            status=SystemAgentRun.STATUS_FAILED,
        )

        def make_goal_events_path(thread_id: str, tokens_used: int) -> str:
            with tempfile.NamedTemporaryFile(
                prefix="autonomous-goal-events-",
                suffix=".jsonl",
                mode="w",
                delete=False,
            ) as events:
                events.write(
                    json.dumps(
                        {
                            "method": codex_events.GOAL_UPDATED_METHOD,
                            "payload": {
                                "threadId": thread_id,
                                "goal": {
                                    "objective": "Autonomous goal",
                                    "tokensUsed": tokens_used,
                                },
                            },
                        }
                    )
                    + "\n"
                )
                path = events.name
            self.addCleanup(Path(path).unlink, missing_ok=True)
            return path

        running_events_path = make_goal_events_path("running-agent-thread", 950_000)
        unrecorded_events_path = make_goal_events_path(
            "unrecorded-agent-thread",
            250_000,
        )
        running_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(running_goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": running_goal.pk,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: 400_000,
                autonomous_goals._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY: {
                    "running-agent-thread": 100_000,
                },
            },
        )
        older_running_instance = CodexInstance.objects.create(
            pid=0,
            thread_id="older-running-agent-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=running_workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=running_workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=older_running_instance.thread_id,
            instance=older_running_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        running_instance = CodexInstance.objects.create(
            pid=0,
            thread_id="running-agent-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path=running_events_path,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=running_workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=running_workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=running_instance.thread_id,
            instance=running_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        no_tokens_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                running_no_tokens_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": running_no_tokens_goal.pk,
                autonomous_goal_prompts._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_USED_STATE_KEY: 700_000,
            },
        )
        no_tokens_instance = CodexInstance.objects.create(
            pid=0,
            thread_id="no-tokens-agent-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=no_tokens_workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=no_tokens_workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=no_tokens_instance.thread_id,
            instance=no_tokens_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        unrecorded_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                running_unrecorded_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={
                "autonomous_goal_id": running_unrecorded_goal.pk,
                autonomous_goals._AUTONOMOUS_GOAL_PROPOSAL_BUDGET_TOKEN_TOTALS_STATE_KEY: [
                    "not-a-dict"
                ],
            },
        )
        unrecorded_instance = CodexInstance.objects.create(
            pid=0,
            thread_id="unrecorded-agent-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path=unrecorded_events_path,
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=unrecorded_workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        SystemAgentRun.objects.create(
            workflow=unrecorded_workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=unrecorded_instance.thread_id,
            instance=unrecorded_instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                blocked_no_log_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_BLOCKED,
            step=system_agents.STEP_BLOCKED,
            state={
                "autonomous_goal_id": blocked_no_log_goal.pk,
                "error": "blocked before the run log was created",
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-run-status-dialog')
        self.assertContains(response, 'data-state="blocked"')
        self.assertContains(
            response, 'data-run-status-title="Autonomous goal is blocked"'
        )
        self.assertContains(response, "raptorq decoder exhausted repair symbols")
        self.assertContains(
            response,
            f'data-run-status-log-url="{reverse("autonomous_goal_run_log", args=[blocked_workflow.pk])}"',
        )
        self.assertContains(response, 'data-state="running"')
        self.assertContains(
            response, 'data-run-status-title="Autonomous goal is running"'
        )
        self.assertContains(response, "Tokens used: 1,250,000 tokens")
        self.assertContains(response, "Tokens used: 700,000 tokens")
        self.assertContains(response, "Tokens used: 250,000 tokens")
        self.assertContains(response, "This autonomous goal run is still working.")
        self.assertContains(response, "blocked before the run log was created")
        self.assertContains(
            response,
            f'data-run-status-log-url="{reverse("autonomous_goal_run_log", args=[running_workflow.pk])}"',
        )
        self.assertContains(response, 'data-run-status-log-url=""', count=1)
        self.assertNotContains(response, 'data-run-status-log-url="None"')

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_shows_not_running_run_status_reasons(
        self,
        mock_codex: MagicMock,
        mock_discover: MagicMock,
        mock_default_branch_commit_hash: MagicMock,
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Manual goal",
            goal="Run only when requested.",
        )
        pending_goal = AutonomousGoal.objects.create(
            project=project,
            title="Pending goal",
            goal="Wait on proposal review.",
            auto_proposal_enabled=True,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=pending_goal,
            title="Review me",
        )
        accepted_block_goal = AutonomousGoal.objects.create(
            project=project,
            title="Accepted session goal",
            goal="Wait for accepted work to finish.",
            auto_proposal_enabled=True,
        )
        accepted_session = SessionMetadata.objects.create(
            thread_id="accepted-session-thread",
            cwd="/repo",
            project=project,
            derived_stage="implementation",
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=accepted_block_goal,
            title="Accepted work",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=accepted_session,
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                accepted_block_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": accepted_block_goal.pk},
        )
        AutonomousGoal.objects.create(
            project=project,
            title="No change goal",
            goal="Wait for new commits.",
            auto_proposal_enabled=True,
            auto_proposal_last_no_proposal_sha="a" * 40,
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Advanced goal",
            goal="Ready after branch changes.",
            auto_proposal_enabled=True,
            auto_proposal_last_no_proposal_sha="b" * 40,
        )
        skipped_goal = AutonomousGoal.objects.create(
            project=project,
            title="Skipped goal",
            goal="Report no-op runs.",
            auto_proposal_enabled=True,
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                skipped_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_SKIPPED,
            state={
                "autonomous_goal_id": skipped_goal.pk,
                "candidate": {"message": "No useful docs proposal was found."},
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="manual"')
        self.assertContains(response, ">Manual</button>", html=False)
        self.assertContains(
            response,
            "Auto-proposal is off. Use Run to start this goal manually.",
        )
        self.assertContains(response, 'data-state="review"')
        self.assertContains(response, ">Review</button>", html=False)
        self.assertContains(
            response,
            "Not running because a proposal from this goal is waiting in the inbox.",
        )
        self.assertContains(response, ">Waiting</button>", html=False)
        self.assertContains(
            response,
            "Not running because an accepted session from this goal is not "
            "Done or archived yet.",
        )
        self.assertContains(response, 'data-state="waiting"')
        self.assertContains(response, ">No change</button>", html=False)
        self.assertContains(
            response,
            "It will try again after that branch changes.",
        )
        self.assertContains(response, ">Ready</button>", html=False)
        self.assertContains(
            response,
            "Auto-proposal is enabled. This goal will start when the scheduler runs "
            "and quota allows.",
        )
        self.assertContains(response, 'data-state="skipped"')
        self.assertContains(response, ">Skipped</button>", html=False)
        self.assertContains(response, "No useful docs proposal was found.")
        self.assertGreaterEqual(mock_default_branch_commit_hash.call_count, 2)

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_shows_quota_pause_instead_of_completed_run(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        self.mock_auto_proposals_paused_by_quota.return_value = True
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                autonomous_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "candidate": {"message": "The last run proposed useful work."},
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="quota"')
        self.assertContains(response, ">Quota</button>", html=False)
        self.assertContains(response, "remaining Codex quota is below")
        self.assertNotContains(response, ">Done</button>", html=False)
        self.assertNotContains(response, "The last run proposed useful work.")

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_treats_completed_auto_proposal_as_ready_not_done(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        auto_goal = AutonomousGoal.objects.create(
            project=project,
            title="Auto goal",
            goal="Keep proposing work.",
            auto_proposal_enabled=True,
        )
        manual_goal = AutonomousGoal.objects.create(
            project=project,
            title="Manual goal",
            goal="Run only on demand.",
            auto_proposal_enabled=False,
        )
        for goal in (auto_goal, manual_goal):
            SystemWorkflow.objects.create(
                kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
                main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
                cwd="/repo",
                status=SystemWorkflow.STATUS_COMPLETED,
                step=system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED,
                state={
                    "autonomous_goal_id": goal.pk,
                    "candidate": {"message": f"{goal.title} proposed useful work."},
                },
            )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, ">Ready</button>", html=False)
        self.assertContains(response, ">Manual</button>", html=False)
        self.assertNotContains(response, ">Done</button>", html=False)
        self.assertContains(
            response, "The last autonomous goal run created a proposal and stopped."
        )

    @patch(
        "hitch.main.workflows.autonomous_goals.default_branch_commit_hash",
        return_value="a" * 40,
    )
    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_treats_pending_stack_proposal_as_ready_to_continue(
        self,
        mock_codex: MagicMock,
        _mock_discover: MagicMock,
        _mock_default_branch_commit_hash: MagicMock,
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            auto_proposal_last_no_proposal_sha="a" * 40,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        source_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                autonomous_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED,
            state={"autonomous_goal_id": autonomous_goal.pk},
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            source_workflow=source_workflow,
            title="Add parser coverage",
            candidate_session=candidate,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="ready"')
        self.assertContains(response, ">Ready</button>", html=False)
        self.assertNotContains(response, ">Done</button>", html=False)
        self.assertNotContains(response, ">No change</button>", html=False)
        self.assertNotContains(
            response,
            "Not running because a proposal from this goal is waiting in the inbox.",
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_treats_budget_exhausted_stack_proposal_as_review(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
            proposal_budget=1000,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Add parser coverage",
            candidate_session=candidate,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
                "proposal_budget_tokens_used": 1000,
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="review"')
        self.assertContains(response, ">Review</button>", html=False)
        self.assertContains(
            response,
            "Not running because a proposal from this goal is waiting in the inbox.",
        )
        self.assertNotContains(response, ">Ready</button>", html=False)

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_treats_legacy_stopped_stack_proposal_as_ready(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        source_workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                autonomous_goal.pk
            ),
            cwd="/repo",
            status=SystemWorkflow.STATUS_COMPLETED,
            step=system_agents.STEP_AUTONOMOUS_GOAL_PROPOSED,
            state={
                "autonomous_goal_id": autonomous_goal.pk,
                "stacked_diff_stopped_reason": "judge_confidence_below_threshold",
            },
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            source_workflow=source_workflow,
            title="Add parser coverage",
            candidate_session=candidate,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="ready"')
        self.assertContains(response, ">Ready</button>", html=False)
        self.assertNotContains(
            response,
            "Not running because a proposal from this goal is waiting in the inbox.",
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_shows_manual_goal_with_pending_stack_proposal_as_review(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=False,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Add parser coverage",
            candidate_session=candidate,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="review"')
        self.assertContains(response, ">Review</button>", html=False)
        self.assertContains(
            response,
            "Not running because a proposal from this goal is waiting in the inbox.",
        )
        self.assertNotContains(
            response,
            "Auto-proposal is off. Use Run to start this goal manually.",
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_blocks_pending_proposal_without_stack_metadata(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Ordinary proposal",
            candidate_session=candidate,
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="review"')
        self.assertContains(response, ">Review</button>", html=False)
        self.assertContains(
            response,
            "Not running because a proposal from this goal is waiting in the inbox.",
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_blocks_stack_continuation_when_extra_pending_proposal_exists(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            stacked_diff_depth=3,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Older pending review",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Add parser coverage",
            candidate_session=candidate,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
            },
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="review"')
        self.assertContains(response, ">Review</button>", html=False)
        self.assertContains(
            response,
            "Not running because a proposal from this goal is waiting in the inbox.",
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_keeps_other_goal_ready_when_accepted_automation_is_in_flight(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Queued goal",
            goal="Wait while project automation is active.",
            auto_proposal_enabled=True,
        )
        blocker_goal = AutonomousGoal.objects.create(
            project=project,
            title="Accepted implementation",
            goal="Run an accepted autonomous implementation.",
        )
        implementation = SessionMetadata.objects.create(
            thread_id="implementation-thread",
            cwd="/repo",
            project=project,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=blocker_goal,
            title="Accepted autonomous proposal",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=implementation,
            outcome_metadata={
                "accepted_by": autonomous_goal_proposal_stack.AUTONOMOUS_GOAL_AUTONOMY_ACCEPTED_BY,
            },
        )
        CodexInstance.objects.create(
            pid=0,
            thread_id=implementation.thread_id,
            cwd="/repo",
            prompt="run accepted implementation",
            status=CodexInstance.STATUS_RUNNING,
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="ready"')
        self.assertContains(response, ">Ready</button>", html=False)
        self.assertContains(response, 'data-state="waiting"')
        self.assertContains(response, ">Waiting</button>", html=False)
        self.assertContains(
            response,
            "Not running because an accepted session from this goal is not "
            "Done or archived yet.",
        )
        self.assertNotContains(response, ">Queued</button>", html=False)

    @patch(
        "hitch.main.repos.discover_repos",
        return_value=[Path("/repo"), Path("/other")],
    )
    @patch("hitch.main.views.common.Codex")
    def test_page_queues_auto_goal_when_auto_proposal_runs_in_other_project(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Queued goal",
            goal="Wait while global automation is active.",
            auto_proposal_enabled=True,
        )
        running_goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Running goal",
            goal="This hidden run owns the global queue.",
            auto_proposal_enabled=True,
        )
        SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(
                running_goal.pk
            ),
            cwd="/other",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": running_goal.pk, "auto_proposal": True},
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-state="queued"')
        self.assertContains(response, ">Queued</button>", html=False)
        self.assertContains(
            response,
            "Not running because another auto-proposal run is active.",
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_edit_form_sync_preserves_auto_qa_choice_when_required(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
        )

        response = self.client.get(reverse("autonomous_goals"))
        self.assertEqual(response.status_code, 200)
        body = response.content.decode()
        self.assertIn(
            'autoQa.dataset.autoQaUserChecked = autoQa.checked ? "true" : "false";',
            body,
        )
        self.assertIn(
            'autoQa.checked = autoQa.dataset.autoQaUserChecked === "true";',
            body,
        )
        self.assertIn("delete editGoalAutoQa.dataset.autoQaUserChecked;", body)
        self.assertIn("autoQa.disabled = required || !supported;", body)

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_draft_pr_goal_shows_auto_qa_required_on_reopen(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PR,
            auto_qa_enabled=False,
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Auto-QA: Required")
        self.assertContains(response, 'data-autonomy="draft_pr"')
        self.assertContains(response, 'data-auto-qa="false"')
        self.assertContains(
            response,
            'value="draft_pr" data-auto-qa-supported="false" data-auto-qa-required="true"',
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_page_lists_proposals_for_selected_project(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_HIGH,
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        judge = SessionMetadata.objects.create(
            thread_id="judge-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            prompt=(
                "Go ahead and implement this proposed session.\n\n"
                "Autonomous goal objective:\n"
                "Find useful test coverage increments.\n\n"
                "Implementation guidance:\n"
                "Add focused rollout parser tests before changing behavior."
            ),
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            relevant_files=["hitch/main/rollout.py"],
            candidate_session=candidate,
            judge_session=judge,
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
        self.assertContains(response, 'data-visible-projects-open')
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
        self.assertContains(response, 'data-proposed-session-do')
        self.assertContains(response, f'data-proposed-session-id="{proposal.pk}"')
        self.assertContains(response, f'data-proposed-session-project="{project.pk}"')
        start_modal_title = (
            '<h2 id="do-session-title" tabindex="-1" autofocus>'
            "Continue proposed session</h2>"
        )
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
        self.assertContains(
            response, f'aria-label="Actions for {proposal.title}"'
        )
        self.assertContains(
            response,
            f'action="{reverse("update_proposed_session_outcome", args=[proposal.pk])}"',
        )
        proposal_header_start = body.index('<div class="proposal-header">')
        proposal_actions_start = body.index(
            '<div class="proposal-actions">', proposal_header_start
        )
        proposal_menu_start = body.index(
            '<div class="proposal-menu"', proposal_header_start
        )
        self.assertLess(proposal_menu_start, proposal_actions_start)
        self.assertContains(
            response, f'value="{ProposedSession.OUTCOME_DISMISSED}"'
        )
        self.assertContains(response, "Judge log")
        self.assertContains(response, 'name="proposed_session"')
        self.assertNotContains(response, "Other proposal")

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_page_shows_autonomous_goal_stack_number(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            stacked_diff_depth=5,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Expand parser coverage",
            summary="This builds on the first stack.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 2,
                "proposal_budget_tokens_used": 1250,
                "stacked_diff_continuation_stopped_reason": "candidate_no_proposal",
            },
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Improve tests - High ambition - High confidence - Stack 2 of 3",
        )
        self.assertContains(response, "Tokens used: 1,250 tokens")
        self.assertContains(response, "Stack stopped: no further proposal found")

    def test_proposed_session_stack_label_omits_invalid_iteration(self) -> None:
        proposed_session = ProposedSession(
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 4,
            }
        )

        self.assertEqual(
            autonomous_goal_run_display._proposed_session_stack_label(proposed_session),
            "",
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_page_omits_stack_label_without_stack_metadata(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        autonomous_goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_HIGH,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            stacked_diff_depth=5,
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=autonomous_goal,
            title="Ordinary proposal",
            summary="This is not a stack entry.",
            confidence=AutonomousGoal.CONFIDENCE_HIGH,
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            "Improve tests - High ambition - High confidence",
        )
        self.assertNotContains(response, "Stack 1 of 5")

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_inbox_recovers_stale_proposal_start_claim(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        stale_claimed_at = (
            datetime.now(UTC)
            - ProposedSession.ACCEPTED_SESSION_START_CLAIM_TTL
            - timedelta(seconds=1)
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "resolved_by": "user",
                "accepted_thread_id": "",
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (
                    stale_claimed_at.isoformat()
                ),
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
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (
                    datetime.now(UTC).isoformat()
                ),
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
    def test_inbox_visible_projects_filter_messages(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
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
    def test_reject_proposed_session_uses_visible_project_filter(
        self, mock_cleanup: MagicMock
    ) -> None:
        selected_project = _make_project()
        visible_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(
            self.client,
            **{
                _SELECTED_PROJECT_COOKIE: str(selected_project.pk),
                _VISIBLE_SESSION_PROJECTS_COOKIE: f"[{visible_project.pk}]",
            },
        )
        proposal = ProposedSession.objects.create(
            project=visible_project,
            title="Add docs coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {
                "outcome_status": ProposedSession.OUTCOME_REJECTED,
                "reason": "Not useful enough.",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("inbox"))
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertEqual(proposal.outcome_notes, "Not useful enough.")
        mock_cleanup.assert_not_called()

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_update_outcome_rejects_proposal_hidden_by_visible_project_filter(
        self, mock_cleanup: MagicMock
    ) -> None:
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
    def test_new_session_page_prefills_proposed_session(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            summary="This adds focused parser coverage.",
            prompt="Add focused rollout parser tests before changing behavior.",
        )

        response = self.client.get(
            f"{reverse('new_session')}?proposed_session={proposal.pk}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="proposed_session"')
        self.assertContains(response, f'value="{proposal.pk}"')
        self.assertContains(response, "Add focused rollout parser tests")
        self.assertContains(response, f'value="{project.pk}" selected')
        self.assertContains(response, f'href="{reverse("inbox")}"')

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_recovers_stale_proposal_start_claim(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex)
        stale_claimed_at = (
            datetime.now(UTC)
            - ProposedSession.ACCEPTED_SESSION_START_CLAIM_TTL
            - timedelta(seconds=1)
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            prompt="Add focused rollout parser tests before changing behavior.",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            outcome_metadata={
                "accepted_by": "user",
                "resolved_by": "user",
                "accepted_thread_id": "",
                ProposedSession.ACCEPTED_SESSION_START_CLAIMED_AT_METADATA_KEY: (
                    stale_claimed_at.isoformat()
                ),
            },
        )

        response = self.client.get(
            f"{reverse('new_session')}?proposed_session={proposal.pk}"
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, f'value="{proposal.pk}"')
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
    def test_new_session_page_prefills_prompt_and_project_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex)
        prompt = (
            "Debug and fix the user's issue from session UID thread-1.\n\n"
            "User issue: "
        )

        response = self.client.get(
            reverse("new_session"), {"prompt": prompt, "project": str(project.pk)}
        )

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

        response = self.client.get(
            reverse("new_session"), {"prompt": "debug this", "project": str(project.pk)}
        )

        self.assertEqual(response.status_code, 404)

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_new_session_page_prefills_bare_repo_cwd_from_query(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _setup_codex(mock_codex)

        response = self.client.get(
            reverse("new_session"), {"prompt": "debug this", "cwd": "/repo"}
        )

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

        response = self.client.get(
            f"{reverse('new_session')}?proposed_session={proposal.pk}"
        )

        self.assertEqual(response.status_code, 404)
        mock_codex.assert_not_called()

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_lists_no_proposal_notice_with_dismiss(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        ProposedSession.objects.create(
            autonomous_goal=goal,
            title="No proposal from Improve tests",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            summary="No concrete test increment was worth proposing.",
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No proposal from Improve tests")
        self.assertContains(response, "From autonomous goal: Improve tests")
        self.assertContains(
            response, "No concrete test increment was worth proposing."
        )
        self.assertContains(response, "Dismiss")
        self.assertNotContains(response, 'data-proposed-session-id="')
        self.assertNotContains(response, 'data-reject-url="')

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_lists_agent_created_proposal(
        self, mock_codex: MagicMock, _mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add CLI proposal tests",
            summary="Cover the proposed session CLI.",
            prompt="Implement tests for the proposed session CLI.",
            relevant_files=["hitch/main/management/commands/propose_session.py"],
        )

        response = self.client.get(reverse("inbox"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Add CLI proposal tests")
        self.assertContains(response, "From coding agent")
        self.assertContains(response, 'data-proposed-session-do')
        self.assertContains(response, f'data-proposed-session-id="{proposal.pk}"')
        self.assertContains(response, "Implement tests for the proposed session CLI.")
        self.assertContains(response, f'data-proposed-session-project="{project.pk}"')

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_shows_create_form_inline_when_no_goals(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-autonomous-goal-create-form')
        self.assertContains(response, "data-autonomous-goal-auto-qa")
        self.assertContains(
            response,
            '<input type="checkbox" name="auto_qa" value="true" data-autonomous-goal-auto-qa disabled>',
            html=True,
        )
        self.assertContains(
            response, 'value="draft_patch" data-auto-qa-supported="true"'
        )
        self.assertContains(
            response, 'value="draft_pr" data-auto-qa-supported="false"'
        )
        self.assertContains(response, "Create autonomous goal")
        self.assertNotContains(response, "No autonomous goals yet.")
        self.assertNotContains(
            response,
            '<button type="button" role="menuitem" data-create-autonomous-goal-open>',
        )
        self.assertNotContains(
            response,
            '<dialog class="new-session" data-create-autonomous-goal-dialog',
        )

    @patch("hitch.main.repos.discover_repos", return_value=[Path("/repo")])
    @patch("hitch.main.views.common.Codex")
    def test_page_moves_create_form_to_header_dialog_when_goals_exist(
        self, mock_codex: MagicMock, mock_discover: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        _setup_codex(mock_codex)
        AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )

        response = self.client.get(reverse("autonomous_goals"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'class="page-menu" data-page-menu')
        self.assertContains(
            response,
            '<button type="button" role="menuitem" data-create-autonomous-goal-open>',
        )
        self.assertContains(response, 'role="menuitem">Run all</button>')
        self.assertContains(
            response,
            '<dialog class="new-session" data-create-autonomous-goal-dialog',
        )
        self.assertNotContains(response, '<p class="section-label">Create</p>')

    def test_create_autonomous_goal_for_selected_project(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        response = self.client.post(
            reverse("create_autonomous_goal"),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_YOLO,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PR,
                "auto_qa": "true",
                "auto_proposal": "true",
                "stacked_diff_depth": "100",
                "proposal_budget": "25",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
                "web_search_mode": AutonomousGoal.WEB_SEARCH_LIVE,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal = AutonomousGoal.objects.get()
        self.assertEqual(goal.project, project)
        self.assertEqual(goal.title, "Improve tests")
        self.assertEqual(goal.ambition, AutonomousGoal.AMBITION_YOLO)
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PR)
        self.assertFalse(goal.auto_qa_enabled)
        self.assertEqual(goal.stacked_diff_depth, 100)
        self.assertEqual(goal.proposal_budget, 25_000_000)
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_LIVE)
        self.assertTrue(goal.auto_proposal_enabled)
        self.assertEqual(
            goal.confidence_threshold,
            AutonomousGoal.CONFIDENCE_VERY_HIGH,
        )

    def test_create_autonomous_goal_stores_auto_merge_branch(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))

        with patch("hitch.main.views.common.local_branch_names", return_value=["main"]):
            response = self.client.post(
                reverse("create_autonomous_goal"),
                {
                    "title": "Improve tests",
                    "goal": "Find useful test coverage increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                    "auto_qa": "true",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "main",
                },
            )

        self.assertEqual(response.status_code, 302)
        goal = AutonomousGoal.objects.get()
        self.assertTrue(goal.auto_qa_enabled)
        self.assertTrue(goal.auto_merge_to_local_branch)
        self.assertEqual(goal.auto_merge_branch, "main")

    def test_edit_autonomous_goal_updates_selected_project_goal(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            auto_proposal_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_CACHED,
            proposal_budget=10_000_000,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "auto_qa": "true",
                "auto_proposal": "false",
                "stacked_diff_depth": "4",
                "proposal_budget": "30",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
                "web_search_mode": AutonomousGoal.WEB_SEARCH_DISABLED,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.title, "Improve docs")
        self.assertEqual(goal.goal, "Find useful docs increments.")
        self.assertEqual(goal.ambition, AutonomousGoal.AMBITION_HIGH)
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PATCH)
        self.assertTrue(goal.auto_qa_enabled)
        self.assertEqual(goal.stacked_diff_depth, 4)
        self.assertEqual(goal.proposal_budget, 30_000_000)
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_DISABLED)
        self.assertFalse(goal.auto_proposal_enabled)
        self.assertEqual(
            goal.confidence_threshold,
            AutonomousGoal.CONFIDENCE_VERY_HIGH,
        )

    def test_edit_autonomous_goal_clears_proposal_budget_when_blank(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            proposal_budget=10000,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_INCREMENTAL,
                "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                "proposal_budget": "",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertIsNone(goal.proposal_budget)

    def test_edit_autonomous_goal_can_reset_web_search_to_codex_default(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_LIVE,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_INCREMENTAL,
                "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                "web_search_mode": AutonomousGoal.WEB_SEARCH_DEFAULT,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_DEFAULT)

    def test_edit_autonomous_goal_updates_auto_merge_branch(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
        )

        with patch("hitch.main.views.common.local_branch_names", return_value=["release"]):
            response = self.client.post(
                reverse("edit_autonomous_goal", args=[goal.pk]),
                {
                    "title": "Improve tests",
                    "goal": "Find useful test coverage increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                    "auto_qa": "true",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "release",
                },
            )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertTrue(goal.auto_qa_enabled)
        self.assertTrue(goal.auto_merge_to_local_branch)
        self.assertEqual(goal.auto_merge_branch, "release")

    def test_edit_autonomous_goal_clears_auto_merge_when_unchecked(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "auto_qa": "true",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertTrue(goal.auto_qa_enabled)
        self.assertFalse(goal.auto_merge_to_local_branch)
        self.assertEqual(goal.auto_merge_branch, "")

    def test_edit_autonomous_goal_preserves_autonomy_when_omitted(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PR,
            auto_proposal_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            web_search_mode=AutonomousGoal.WEB_SEARCH_CACHED,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PR)
        self.assertEqual(goal.web_search_mode, AutonomousGoal.WEB_SEARCH_CACHED)
        self.assertTrue(goal.auto_proposal_enabled)

    def test_edit_autonomous_goal_clears_auto_proposal_no_proposal_sha(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
            auto_proposal_last_no_proposal_sha="a" * 40,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve tests",
                "goal": "Find useful test coverage increments.",
                "ambition": AutonomousGoal.AMBITION_INCREMENTAL,
                "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                "auto_proposal": "true",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertTrue(goal.auto_proposal_enabled)
        self.assertEqual(goal.auto_proposal_last_no_proposal_sha, "")

    def test_edit_autonomous_goal_preserves_auto_qa_when_omitted(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertEqual(goal.autonomy, AutonomousGoal.AUTONOMY_DRAFT_PATCH)
        self.assertTrue(goal.auto_qa_enabled)

    def test_edit_autonomous_goal_disables_auto_qa_when_false_is_explicit(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            ambition=AutonomousGoal.AMBITION_INCREMENTAL,
            autonomy=AutonomousGoal.AUTONOMY_DRAFT_PATCH,
            auto_qa_enabled=True,
            confidence_threshold=AutonomousGoal.CONFIDENCE_HIGH,
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Improve docs",
                "goal": "Find useful docs increments.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                "auto_qa": "false",
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertFalse(goal.auto_qa_enabled)

    def test_edit_autonomous_goal_is_scoped_to_selected_project(self) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not change.",
        )

        response = self.client.post(
            reverse("edit_autonomous_goal", args=[goal.pk]),
            {
                "title": "Changed",
                "goal": "Changed.",
                "ambition": AutonomousGoal.AMBITION_HIGH,
                "confidence_threshold": AutonomousGoal.CONFIDENCE_VERY_HIGH,
            },
        )

        self.assertEqual(response.status_code, 404)
        goal.refresh_from_db()
        self.assertEqual(goal.title, "Other goal")
        self.assertEqual(goal.goal, "Should not change.")

    def test_edit_autonomous_goal_rejects_invalid_posts(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )

        for data, message in (
            (
                {
                    "title": "",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "title is required",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "goal is required",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": "huge",
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "ambition is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": "self_driving",
                    "auto_proposal": "false",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "autonomy is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_qa": "yes",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "auto-QA setting is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "maybe",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "auto-proposal is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "stacked_diff_depth": "0",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "stacked diff depth is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                    "stacked_diff_depth": "101",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "stacked diff depth is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "stacked_diff_depth": "several",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "stacked diff depth is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "proposal_budget": "0",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "proposal budget is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "proposal_budget": "1e1000000",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "proposal budget is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "proposal_budget": "0.0000001",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "proposal budget is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "proposal_budget": "many",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "proposal budget is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "stacked_diff_depth": "2",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                },
                "stacked diff depth requires draft patch or draft PR",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "auto_proposal": "false",
                    "confidence_threshold": "absolute",
                },
                "confidence threshold is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "web_search_mode": "maybe",
                },
                "web search setting is invalid",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_PROPOSE_ONLY,
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "main",
                },
                "auto merge requires auto-QA",
            ),
            (
                {
                    "title": "Improve docs",
                    "goal": "Find useful docs increments.",
                    "ambition": AutonomousGoal.AMBITION_HIGH,
                    "autonomy": AutonomousGoal.AUTONOMY_DRAFT_PATCH,
                    "auto_qa": "true",
                    "confidence_threshold": AutonomousGoal.CONFIDENCE_HIGH,
                    "auto_merge_to_local_branch": "true",
                    "auto_merge_branch": "missing",
                },
                "auto merge branch is invalid",
            ),
        ):
            with self.subTest(message=message):
                response = self.client.post(
                    reverse("edit_autonomous_goal", args=[goal.pk]),
                    data,
                )

                self.assertContains(response, message, status_code=400)

        goal.refresh_from_db()
        self.assertEqual(goal.title, "Improve tests")
        self.assertEqual(goal.goal, "Find useful test coverage increments.")

    def test_delete_autonomous_goal_soft_deletes_selected_project_goal(self) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
            auto_proposal_enabled=True,
        )
        other_goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should stay.",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        other_goal.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)
        self.assertFalse(goal.auto_proposal_enabled)
        self.assertIsNone(other_goal.deleted_at)

    def test_delete_autonomous_goal_is_scoped_to_selected_project(self) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not delete.",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 404)
        goal.refresh_from_db()
        self.assertIsNone(goal.deleted_at)

    def test_delete_autonomous_goal_preserves_accepted_proposal(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            accepted_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        goal.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)
        proposal.refresh_from_db()
        self.assertEqual(proposal.autonomous_goal_id, goal.pk)
        self.assertEqual(
            system_agents.accepted_visible_system_thread_ids(),
            {"candidate-thread"},
        )

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_delete_autonomous_goal_dismisses_unresolved_proposal(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertEqual(
            proposal.outcome_notes,
            system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
        )
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_delete_autonomous_goal_cleans_hidden_stacked_proposal(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_DISMISSED,
            outcome_metadata={"stacked_diff_hidden_until_complete": True},
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertEqual(
            proposal.outcome_notes,
            system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
        )
        self.assertFalse(
            proposal.outcome_metadata["stacked_diff_hidden_until_complete"]
        )
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_delete_autonomous_goal_keeps_accepted_proposal_worktree(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            accepted_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            title="Add parser coverage",
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        mock_cleanup.assert_not_called()

    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_delete_autonomous_goal_reconciles_terminal_running_workflow(
        self, mock_interrupt: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_FAILED,
            error="worker process exited before callback",
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        mock_interrupt.assert_not_called()
        goal.refresh_from_db()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertIn("worker process exited before callback", run.error)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        proposal = ProposedSession.objects.get(source_workflow=workflow)
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertEqual(
            proposal.outcome_notes,
            system_agents.AUTONOMOUS_GOAL_DELETED_ERROR,
        )

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_delete_autonomous_goal_stops_running_workflow(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk, "session_cwd": "/repo-worktree"},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        mock_interrupt.return_value = instance

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        mock_interrupt.assert_called_once_with(
            instance.pk, expected_thread_id=instance.thread_id
        )
        mock_cleanup.assert_not_called()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        self.assertEqual(run.error, "Autonomous goal deleted by user")
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_BLOCKED)
        self.assertEqual(workflow.step, system_agents.STEP_BLOCKED)
        self.assertEqual(workflow.state["error"], "Autonomous goal deleted by user")
        goal.refresh_from_db()
        self.assertIsNotNone(goal.deleted_at)

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch("hitch.main.workflows.system_agents.codex_pool.interrupt_instance")
    def test_delete_autonomous_goal_cleans_worktree_when_interrupt_is_terminal(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk, "session_cwd": "/repo-worktree"},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )
        terminal_instance = instance
        terminal_instance.status = CodexInstance.STATUS_FAILED
        mock_interrupt.return_value = terminal_instance

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_FAILED)
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.workflows.autonomous_goals.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.workflows.system_agents.codex_pool.interrupt_instance",
        return_value=None,
    )
    def test_delete_autonomous_goal_keeps_goal_when_running_workflow_cannot_stop(
        self, mock_interrupt: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        workflow = SystemWorkflow.objects.create(
            kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            main_thread_id=autonomous_goals._autonomous_goal_main_thread_id(goal.pk),
            cwd="/repo",
            status=SystemWorkflow.STATUS_RUNNING,
            step=system_agents.STEP_AUTONOMOUS_GOAL_CANDIDATE_RUNNING,
            state={"autonomous_goal_id": goal.pk, "session_cwd": "/repo-worktree"},
        )
        instance = CodexInstance.objects.create(
            pid=0,
            thread_id="goal-thread",
            cwd="/repo",
            prompt="run autonomous goal",
            events_path="/dev/null",
            status=CodexInstance.STATUS_RUNNING,
            purpose=CodexInstance.PURPOSE_SYSTEM_AGENT,
            workflow_id=workflow.pk,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            agent_kind=system_agents.AUTONOMOUS_GOAL_AGENT_KIND,
            thread_id=instance.thread_id,
            instance=instance,
            status=SystemAgentRun.STATUS_RUNNING,
        )

        response = self.client.post(reverse("delete_autonomous_goal", args=[goal.pk]))

        self.assertContains(
            response,
            "autonomous goal run could not be stopped",
            status_code=400,
        )
        mock_interrupt.assert_called_once_with(
            instance.pk, expected_thread_id=instance.thread_id
        )
        mock_cleanup.assert_not_called()
        run.refresh_from_db()
        workflow.refresh_from_db()
        self.assertEqual(run.status, SystemAgentRun.STATUS_RUNNING)
        self.assertEqual(workflow.status, SystemWorkflow.STATUS_RUNNING)
        goal.refresh_from_db()
        self.assertIsNone(goal.deleted_at)

    @patch("hitch.main.workflows.autonomous_goals.start_autonomous_goal_workflow")
    def test_run_single_starts_selected_project_goal(
        self, mock_start: MagicMock
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not run.",
        )

        response = self.client.post(reverse("run_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_start.call_count, 1)
        self.assertEqual(mock_start.call_args.kwargs["autonomous_goal"], goal)
        self.assertTrue(mock_start.call_args.kwargs["use_worktrees"])

    @patch("hitch.main.workflows.autonomous_goals.start_autonomous_goal_workflow")
    def test_run_single_always_uses_worktrees(
        self, mock_start: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(
            self.client,
            hitch_selected_project_id=str(project.pk),
            **{_USE_WORKTREES_COOKIE: "true"},
        )
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )

        response = self.client.post(reverse("run_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        self.assertTrue(mock_start.call_args.kwargs["use_worktrees"])

    @patch("hitch.main.workflows.autonomous_goals.start_autonomous_goal_workflow")
    def test_run_single_skips_goal_blocked_by_accepted_session(
        self, mock_start: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        accepted = SessionMetadata.objects.create(
            thread_id="accepted-thread",
            cwd="/repo",
            project=project,
            derived_stage="implementation",
        )
        ProposedSession.objects.create(
            project=project,
            autonomous_goal=goal,
            title="Accepted proposal",
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=accepted,
        )

        response = self.client.post(reverse("run_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 302)
        mock_start.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals.start_autonomous_goal_workflow")
    def test_run_single_is_scoped_to_selected_project(
        self, mock_start: MagicMock
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not run.",
        )

        response = self.client.post(reverse("run_autonomous_goal", args=[goal.pk]))

        self.assertEqual(response.status_code, 404)
        mock_start.assert_not_called()

    @patch("hitch.main.workflows.autonomous_goals.start_autonomous_goal_workflow")
    def test_run_all_starts_each_selected_project_goal(
        self, mock_start: MagicMock
    ) -> None:
        project = _make_project()
        other_project = _make_project(name="Other", repo_path="/other")
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        first = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        second = AutonomousGoal.objects.create(
            project=project,
            title="Improve docs",
            goal="Find useful docs increments.",
        )
        AutonomousGoal.objects.create(
            project=project,
            title="Deleted goal",
            goal="Should not run.",
            deleted_at=timezone.now(),
        )
        AutonomousGoal.objects.create(
            project=other_project,
            title="Other goal",
            goal="Should not run.",
        )

        response = self.client.post(reverse("run_autonomous_goals"))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            [call.kwargs["autonomous_goal"] for call in mock_start.call_args_list],
            [first, second],
        )
        self.assertEqual(
            [call.kwargs["use_worktrees"] for call in mock_start.call_args_list],
            [True, True],
        )

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_reject_proposed_session_requires_reason(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_REJECTED},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.content, b"reason is required")
        mock_cleanup.assert_not_called()

    @patch("hitch.main.views.common.Codex")
    def test_accept_proposed_session_links_candidate_session(
        self, mock_codex: MagicMock
    ) -> None:
        codex = _setup_codex(mock_codex, models=[])
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_ACCEPTED},
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        candidate.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        self.assertEqual(proposal.accepted_session, candidate)
        self.assertEqual(candidate.codex_name, "Add parser coverage")
        self.assertEqual(candidate.codex_display_title, "Add parser coverage")
        codex._client.thread_set_name.assert_called_once_with(
            "candidate-thread", "Add parser coverage"
        )

    @patch(
        "hitch.main.views.common.goal_workflows."
        "stop_running_autonomous_goal_stack_after_proposal_resolution"
    )
    @patch("hitch.main.views.common.Codex")
    def test_accept_proposed_session_stops_background_stack(
        self, mock_codex: MagicMock, mock_stop_stack: MagicMock
    ) -> None:
        _setup_codex(mock_codex, models=[])
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_ACCEPTED},
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED)
        mock_stop_stack.assert_called_once_with(
            goal.pk,
            proposal.pk,
            ProposedSession.OUTCOME_ACCEPTED,
        )

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.views.common.goal_workflows."
        "stop_running_autonomous_goal_stack_after_proposal_resolution"
    )
    def test_resolving_visible_stack_proposal_stops_background_stack_before_cleanup(
        self, mock_stop_stack: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        mock_stop_stack.return_value = True
        calls: list[str] = []

        def record_stop(*_args: object) -> bool:
            calls.append("stop")
            return True

        def record_cleanup(*_args: object) -> None:
            calls.append("cleanup")

        for outcome_status in (
            ProposedSession.OUTCOME_REJECTED,
            ProposedSession.OUTCOME_DISMISSED,
        ):
            with self.subTest(outcome_status=outcome_status):
                calls.clear()
                mock_stop_stack.reset_mock()
                mock_cleanup.reset_mock()
                mock_stop_stack.side_effect = record_stop
                mock_cleanup.side_effect = record_cleanup
                candidate = SessionMetadata.objects.create(
                    thread_id=f"candidate-{outcome_status}",
                    cwd=f"/repo-worktree-{outcome_status}",
                    project=project,
                )
                proposal = ProposedSession.objects.create(
                    project=project,
                    autonomous_goal=goal,
                    candidate_session=candidate,
                    title="Add parser coverage",
                    outcome_metadata={
                        "stacked_diff_depth": 3,
                        "stacked_diff_iteration": 1,
                        "stacked_diff_hidden_until_complete": False,
                    },
                )
                data = {"outcome_status": outcome_status}
                if outcome_status == ProposedSession.OUTCOME_REJECTED:
                    data["reason"] = "Not the right direction."

                response = self.client.post(
                    reverse("update_proposed_session_outcome", args=[proposal.pk]),
                    data,
                )

                self.assertEqual(response.status_code, 302)
                proposal.refresh_from_db()
                self.assertEqual(proposal.outcome_status, outcome_status)
                self.assertEqual(proposal.outcome_metadata["resolved_by"], "user")
                mock_stop_stack.assert_called_once_with(
                    goal.pk,
                    proposal.pk,
                    outcome_status,
                )
                mock_cleanup.assert_called_once_with(candidate.cwd)
                self.assertEqual(calls, ["stop", "cleanup"])

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    @patch(
        "hitch.main.views.common.goal_workflows."
        "stop_running_autonomous_goal_stack_after_proposal_resolution",
        return_value=False,
    )
    def test_reject_visible_stack_proposal_keeps_worktree_when_stop_fails(
        self, mock_stop_stack: MagicMock, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            project=project,
            autonomous_goal=goal,
            candidate_session=candidate,
            title="Add parser coverage",
            outcome_metadata={
                "stacked_diff_depth": 3,
                "stacked_diff_iteration": 1,
                "stacked_diff_hidden_until_complete": False,
            },
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {
                "outcome_status": ProposedSession.OUTCOME_REJECTED,
                "reason": "Not the right direction.",
            },
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        mock_stop_stack.assert_called_once_with(
            goal.pk,
            proposal.pk,
            ProposedSession.OUTCOME_REJECTED,
        )
        mock_cleanup.assert_not_called()

    def test_dismiss_notice_updates_outcome(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        notice = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="No proposal from Improve tests",
            inbox_kind=ProposedSession.INBOX_KIND_NOTICE,
            summary="No concrete test increment was worth proposing.",
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[notice.pk]),
            {"outcome_status": ProposedSession.OUTCOME_DISMISSED},
        )

        self.assertEqual(response.status_code, 302)
        notice.refresh_from_db()
        self.assertEqual(notice.outcome_status, ProposedSession.OUTCOME_DISMISSED)

    def test_notice_rejects_non_dismissed_outcome(self) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        notice = ProposedSession.objects.create(
            autonomous_goal=goal,
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
    def test_dismiss_proposed_session_uses_distinct_outcome(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            candidate_session=SessionMetadata.objects.create(
                thread_id="candidate-thread",
                cwd="/repo-worktree",
                project=project,
            ),
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {"outcome_status": ProposedSession.OUTCOME_DISMISSED},
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_DISMISSED)
        self.assertNotEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertEqual(proposal.outcome_notes, "")
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_reject_proposed_session_cleans_candidate_worktree(
        self, mock_cleanup: MagicMock
    ) -> None:
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            candidate_session=candidate,
        )

        response = self.client.post(
            reverse("update_proposed_session_outcome", args=[proposal.pk]),
            {
                "outcome_status": ProposedSession.OUTCOME_REJECTED,
                "reason": "Not useful enough.",
            },
        )

        self.assertEqual(response.status_code, 302)
        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertEqual(proposal.outcome_notes, "Not useful enough.")
        mock_cleanup.assert_called_once_with("/repo-worktree")

    @patch("hitch.main.views.common.cleanup_managed_worktree_path")
    def test_update_outcome_rejects_already_resolved_proposal(
        self, mock_cleanup: MagicMock
    ) -> None:
        # A proposal accepted into its candidate session (accepted_session ==
        # candidate_session) un-hides that otherwise-hidden system thread, so the
        # user can see and work in it. A stale inbox tab can still post a
        # dismiss/reject for the same proposal; re-deciding it must be refused so
        # the recorded outcome is not corrupted and the live session stays
        # visible.
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        candidate = SessionMetadata.objects.create(
            thread_id="candidate-thread",
            cwd="/repo-worktree",
            project=project,
            is_hidden_system_session=True,
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
            title="Add parser coverage",
            candidate_session=candidate,
            outcome_status=ProposedSession.OUTCOME_ACCEPTED,
            accepted_session=candidate,
        )
        self.assertIn(
            "candidate-thread",
            system_agents.accepted_visible_system_thread_ids(),
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
                self.assertEqual(
                    response.content, b"proposed session has already been resolved"
                )
                proposal.refresh_from_db()
                self.assertEqual(
                    proposal.outcome_status, ProposedSession.OUTCOME_ACCEPTED
                )
                self.assertEqual(proposal.accepted_session, candidate)
        # The accepted session stayed visible and its worktree was never removed.
        self.assertIn(
            "candidate-thread",
            system_agents.accepted_visible_system_thread_ids(),
        )
        mock_cleanup.assert_not_called()

    def test_update_outcome_rejects_unset_target_status(self) -> None:
        # OUTCOME_UNSET is the pending inbox state, not a decision; the endpoint
        # must not let a request re-open a proposal by posting it.
        project = _make_project()
        _seed_cookies(self.client, hitch_selected_project_id=str(project.pk))
        goal = AutonomousGoal.objects.create(
            project=project,
            title="Improve tests",
            goal="Find useful test coverage increments.",
        )
        proposal = ProposedSession.objects.create(
            autonomous_goal=goal,
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

    def test_accept_helper_does_not_overwrite_resolved_proposal(self) -> None:
        # The accept path (new-session "Do it") and the inbox outcome endpoint
        # race on the same proposal. If the inbox endpoint already rejected it
        # -- which also cleans up the candidate worktree -- the accept helper
        # must leave that decision intact rather than flip it to accepted, which
        # would leave accepted_session pointing at a removed worktree. Exactly
        # one transition wins across both endpoints.
        project = _make_project()
        proposal = ProposedSession.objects.create(
            project=project,
            title="Add parser coverage",
            outcome_status=ProposedSession.OUTCOME_REJECTED,
            outcome_notes="Not useful enough.",
        )
        started = SessionMetadata.objects.create(
            thread_id="started-thread",
            cwd="/repo",
            project=project,
        )

        views._accept_proposed_session_for_session(proposal, started)

        proposal.refresh_from_db()
        self.assertEqual(proposal.outcome_status, ProposedSession.OUTCOME_REJECTED)
        self.assertIsNone(proposal.accepted_session)
        self.assertEqual(proposal.outcome_notes, "Not useful enough.")
