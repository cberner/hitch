"""Data-migration tests for the views-owned columns."""

import importlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import call, patch

from django.core.exceptions import FieldDoesNotExist
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import (
    SimpleTestCase,
    TransactionTestCase,
)
from django.utils import timezone


class ResetStaleStageCacheMigrationTests(TransactionTestCase):
    """The 0048 data migration heals transient stage rows persisted by the
    pre-fix write path, which the mtime-keyed read guard would otherwise serve
    indefinitely after the active owner exits without rewriting the rollout."""

    migrate_from = [("main", "0047_sessionmetadata_derived_stage")]
    migrate_to = [("main", "0048_reset_stale_stage_cache")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_reset_clears_persisted_stage_cache(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(self.migrate_from).apps
        SessionMetadata = old_apps.get_model("main", "SessionMetadata")
        SessionMetadata.objects.create(
            thread_id="stale-transient",
            derived_stage="implementation",
            derived_stage_source_mtime_ns=123,
        )
        SessionMetadata.objects.create(
            thread_id="already-empty",
            derived_stage="",
            derived_stage_source_mtime_ns=0,
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(self.migrate_to).apps
        SessionMetadata = new_apps.get_model("main", "SessionMetadata")
        stale = SessionMetadata.objects.get(thread_id="stale-transient")
        self.assertEqual(stale.derived_stage, "")
        self.assertEqual(stale.derived_stage_source_mtime_ns, 0)
        empty = SessionMetadata.objects.get(thread_id="already-empty")
        self.assertEqual(empty.derived_stage, "")
        self.assertEqual(empty.derived_stage_source_mtime_ns, 0)


class ApprovalModeLiveEditableMigrationTests(TransactionTestCase):
    migrate_from = [("main", "0060_remove_qa_panel_settings")]
    migrate_to = [("main", "0061_codexinstance_approval_mode_live_editable")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_keeps_pre_existing_workers_not_live_editable(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(self.migrate_from).apps
        CodexInstance = old_apps.get_model("main", "CodexInstance")
        rows = [
            ("active-prompt", "prompt_user", "running"),
            ("active-approve", "approve_all", "starting"),
            ("active-auto", "auto_review", "running"),
            ("done-prompt", "prompt_user", "completed"),
        ]
        for idx, (thread_id, approval_mode, status) in enumerate(rows, start=1):
            CodexInstance.objects.create(
                pid=idx,
                thread_id=thread_id,
                cwd="/repo",
                prompt="hi",
                events_path="/dev/null",
                status=status,
                approval_mode=approval_mode,
            )

        new_apps = self._migrate(self.migrate_to).loader.project_state(self.migrate_to).apps
        CodexInstance = new_apps.get_model("main", "CodexInstance")

        for thread_id, _approval_mode, _status in rows:
            self.assertFalse(CodexInstance.objects.get(thread_id=thread_id).approval_mode_live_editable)


class StagedCustomCodingAgentRemovalMigrationTests(TransactionTestCase):
    migrate_from = [("main", "0064_codexinstance_codex_error_info")]
    migrate_to = [("main", "0065_remove_custom_coding_agent_fields")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_keeps_legacy_worker_column_and_supports_rollback(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(self.migrate_from).apps
        legacy_codex_instance_model = old_apps.get_model("main", "CodexInstance")
        legacy_instance = legacy_codex_instance_model.objects.create(
            pid=1,
            thread_id="legacy-worker-before-migration",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            base_instructions="legacy prompt",
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(self.migrate_to).apps
        CodexInstance = new_apps.get_model("main", "CodexInstance")
        UserSettings = new_apps.get_model("main", "UserSettings")

        with self.assertRaises(FieldDoesNotExist):
            CodexInstance._meta.get_field("base_instructions")
        with self.assertRaises(FieldDoesNotExist):
            UserSettings._meta.get_field("coding_agent")

        instance = CodexInstance.objects.create(
            pid=2,
            thread_id="new-worker",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
        )
        legacy_codex_instance_model.objects.create(
            pid=3,
            thread_id="legacy-worker-after-migration",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            base_instructions="legacy follow-up prompt",
        )
        legacy_instance.refresh_from_db()
        self.assertEqual(legacy_instance.base_instructions, "legacy prompt")

        with connection.cursor() as cursor:
            codex_columns = {
                column.name: column
                for column in connection.introspection.get_table_description(cursor, CodexInstance._meta.db_table)
            }
            settings_columns = {
                column.name
                for column in connection.introspection.get_table_description(cursor, UserSettings._meta.db_table)
            }

        self.assertTrue(codex_columns["base_instructions"].null_ok)
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT base_instructions FROM {CodexInstance._meta.db_table} WHERE id = %s",
                [instance.pk],
            )
            self.assertIsNone(cursor.fetchone()[0])
        self.assertNotIn("coding_agent", settings_columns)

        rolled_back_apps = self._migrate(self.migrate_from).loader.project_state(self.migrate_from).apps
        rolled_back_codex_instance_model = rolled_back_apps.get_model("main", "CodexInstance")
        self.assertEqual(
            rolled_back_codex_instance_model.objects.get(pk=instance.pk).base_instructions,
            "",
        )


class ReasoningEffortDefaultMigrationTests(TransactionTestCase):
    migrate_from = [("main", "0067_workflowsteeringmessage")]
    migrate_to = [("main", "0068_alter_usersettings_reasoning_effort")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_backfills_only_entirely_unset_account_efforts(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(self.migrate_from).apps
        User = old_apps.get_model("auth", "User")
        UserSettings = old_apps.get_model("main", "UserSettings")

        blank_user = User.objects.create(username="blank-effort")
        model_default_user = User.objects.create(username="model-default-effort")
        saved_user = User.objects.create(username="saved-effort")
        UserSettings.objects.create(user=blank_user, reasoning_effort="")
        UserSettings.objects.create(
            user=model_default_user,
            model="gpt-5.6-sol",
            reasoning_effort="",
        )
        UserSettings.objects.create(
            user=saved_user,
            model="gpt-5.6-sol",
            reasoning_effort="xhigh",
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(self.migrate_to).apps
        UserSettings = new_apps.get_model("main", "UserSettings")

        self.assertEqual(
            UserSettings.objects.get(user_id=blank_user.pk).reasoning_effort,
            "high",
        )
        self.assertEqual(
            UserSettings.objects.get(user_id=model_default_user.pk).reasoning_effort,
            "",
        )
        self.assertEqual(
            UserSettings.objects.get(user_id=saved_user.pk).reasoning_effort,
            "xhigh",
        )


class RemovedFeatureMigrationTests(TransactionTestCase):
    migrate_from = [("main", "0068_alter_usersettings_reasoning_effort")]
    migrate_to = [("main", "0069_remove_demo_and_spec_critic")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_retires_running_state_before_removing_feature_schema(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)
        original_prompt = (
            "Implement the accepted request\n\n```python\ndef preserved():\n    return 'exact indentation'\n```"
        )

        old_apps = self._migrate(self.migrate_from).loader.project_state(self.migrate_from).apps
        CodexInstance = old_apps.get_model("main", "CodexInstance")
        ApprovalRequest = old_apps.get_model("main", "ApprovalRequest")
        Project = old_apps.get_model("main", "Project")
        SessionDemo = old_apps.get_model("main", "SessionDemo")
        SessionMetadata = old_apps.get_model("main", "SessionMetadata")
        SystemAgentRun = old_apps.get_model("main", "SystemAgentRun")
        SystemWorkflow = old_apps.get_model("main", "SystemWorkflow")
        UserInputRequest = old_apps.get_model("main", "UserInputRequest")

        recovered_project = Project.objects.create(
            name="Recovered project",
            repo_path="/repo",
        )
        SessionDemo.objects.create(
            thread_id="main-session",
            port=45678,
            container_id="demo-container",
            registration_token="registration-token",
        )
        SessionMetadata.objects.create(
            thread_id="main-session",
            is_hidden_system_session=False,
        )
        spec_main_session = SessionMetadata.objects.create(
            thread_id="spec-main-session",
            is_hidden_system_session=False,
        )
        SessionMetadata.objects.create(
            thread_id="spec-agent",
            is_hidden_system_session=True,
        )
        SessionMetadata.objects.create(
            thread_id="orphan-spec-agent",
            is_hidden_system_session=True,
        )
        SessionMetadata.objects.create(
            thread_id="unrelated-agent",
            is_hidden_system_session=True,
        )
        SessionMetadata.objects.create(
            thread_id="stranded-spec-main-session",
            cwd="/repo",
            project_cleared=True,
        )
        demo_workflow = SystemWorkflow.objects.create(
            kind="demo_deployment",
            main_thread_id="main-session",
            cwd="/repo",
        )
        spec_workflow = SystemWorkflow.objects.create(
            kind="spec_critic",
            main_thread_id="spec-main-session",
            cwd="/repo",
            step="spec_critic_analyzing",
            state={
                "original_prompt": original_prompt,
                "auto_pr_enabled": False,
                "auto_qa_enabled": True,
                "auto_merge_to_local_branch": True,
                "auto_merge_branch": "release",
            },
        )
        blocked_unsettled_workflow = SystemWorkflow.objects.create(
            kind="spec_critic",
            main_thread_id="blocked-unsettled-spec-main-session",
            cwd="/repo",
            status="blocked",
            step="blocked",
            state={
                "original_prompt": "Blocked request with interrupted feedback",
                "failure_surfaced": True,
                "next_user_message_index": 4,
            },
        )
        blocked_settled_workflow = SystemWorkflow.objects.create(
            kind="spec_critic",
            main_thread_id="blocked-settled-spec-main-session",
            cwd="/repo",
            status="blocked",
            step="blocked",
            state={
                "original_prompt": "Blocked request with completed feedback",
                "failure_surfaced": True,
            },
        )

        def create_instance(
            thread_id: str,
            *,
            agent_kind: str,
            workflow_id: int | None,
            purpose: str = "system_agent",
            pid: int = 1,
            prompt: str = "work",
            status: str = "completed",
            user_message_index: int | None = None,
        ) -> Any:
            return CodexInstance.objects.create(
                pid=pid,
                thread_id=thread_id,
                cwd="/repo",
                prompt=prompt,
                events_path="/dev/null",
                purpose=purpose,
                agent_kind=agent_kind,
                workflow_id=workflow_id,
                status=status,
                user_message_index=user_message_index,
            )

        demo_instance = create_instance(
            "main-session",
            agent_kind="",
            workflow_id=demo_workflow.pk,
            pid=12345,
            prompt="Registration token: secret-token",
            status="running",
        )
        spec_instance = create_instance(
            "spec-agent",
            agent_kind="spec_critic_synthesizer",
            workflow_id=spec_workflow.pk,
            pid=23456,
        )
        orphan_spec_instance = create_instance(
            "orphan-spec-agent",
            agent_kind="spec_critic_tests",
            workflow_id=None,
            pid=0,
            status="starting",
        )
        unrelated_instance = create_instance(
            "unrelated-agent",
            agent_kind="qa",
            workflow_id=None,
        )
        blocked_unsettled_feedback = create_instance(
            "blocked-unsettled-spec-main-session",
            agent_kind="spec_critic",
            workflow_id=blocked_unsettled_workflow.pk,
            purpose="system_feedback",
            pid=0,
            status="starting",
            user_message_index=4,
        )
        create_instance(
            "blocked-settled-spec-main-session",
            agent_kind="spec_critic",
            workflow_id=blocked_settled_workflow.pk,
            purpose="system_feedback",
            pid=0,
            status="completed",
        )
        SystemWorkflow.objects.create(
            kind="spec_critic",
            main_thread_id="started-spec-main-session",
            cwd="/repo",
            step="spec_critic_implementation_spawned",
            state={
                "original_prompt": "Already started request",
                "next_user_message_index": 9,
            },
        )
        SystemWorkflow.objects.create(
            kind="spec_critic",
            main_thread_id="stranded-spec-main-session",
            cwd="/repo",
            step="spec_critic_implementation_spawned",
            state={
                "original_prompt": "Stranded before implementation spawn",
                "next_user_message_index": 10,
                "auto_pr_enabled": True,
                "auto_qa_enabled": False,
                "auto_merge_to_local_branch": False,
                "auto_merge_branch": "",
            },
        )
        implementation_instance = create_instance(
            "started-spec-main-session",
            agent_kind="",
            workflow_id=None,
            purpose="user",
            prompt="Already started request",
            status="running",
            user_message_index=9,
        )
        prior_failed_instance = create_instance(
            "reused-index-spec-main-session",
            agent_kind="",
            workflow_id=None,
            purpose="user",
            prompt="An older request that failed before appearing",
            status="failed",
            user_message_index=11,
        )
        reused_index_workflow = SystemWorkflow.objects.create(
            kind="spec_critic",
            main_thread_id="reused-index-spec-main-session",
            cwd="/repo",
            step="spec_critic_implementation_spawned",
            state={
                "original_prompt": "Request whose index was reused",
                "next_user_message_index": 11,
            },
        )
        unrelated_reused_instance = create_instance(
            "reused-index-spec-main-session",
            agent_kind="",
            workflow_id=None,
            purpose="user",
            prompt="An unrelated newer request",
            status="running",
            user_message_index=11,
        )
        self.assertGreaterEqual(
            unrelated_reused_instance.started_at,
            reused_index_workflow.created_at,
        )
        approval = ApprovalRequest.objects.create(
            instance=demo_instance,
            method="item/commandExecution/requestApproval",
            params={},
        )
        input_request = UserInputRequest.objects.create(
            instance=demo_instance,
            method="item/tool/requestUserInput",
            params={},
        )
        for workflow, instance in (
            (demo_workflow, demo_instance),
            (spec_workflow, spec_instance),
        ):
            SystemAgentRun.objects.create(
                workflow=workflow,
                instance=instance,
                thread_id=instance.thread_id,
                agent_kind=instance.agent_kind,
            )

        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        inspect_result = subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "io.hitch.managed": "demo",
                                "io.hitch.session": "main-session",
                                "io.hitch.demo_token": "registration-token",
                            }
                        }
                    }
                ]
            ),
            "",
        )
        missing_result = subprocess.CalledProcessError(
            125,
            ["podman", "inspect", "demo-container"],
            stderr='Error: no such object: "demo-container"',
        )
        podman_results = [
            subprocess.CompletedProcess([], 0, "demo-container\n", ""),
            inspect_result,
            subprocess.CompletedProcess([], 0, "", ""),
            missing_result,
            subprocess.CompletedProcess([], 0, "", ""),
        ]

        def worker_cmdline_parts(instance: Any) -> list[bytes]:
            if instance.pk != spec_instance.pk:
                return []
            return [
                b"/old-release/.venv/bin/python",
                b"/old-release/manage.py",
                b"codex_worker",
                b"--instance-id",
                str(spec_instance.pk).encode(),
            ]

        spawned_feedback_id: int | None = None
        spawned_approval_id: int | None = None

        def stop_worker(instance: Any) -> None:
            nonlocal spawned_approval_id, spawned_feedback_id
            if instance.pk != demo_instance.pk or spawned_feedback_id is not None:
                return
            spawned_feedback = create_instance(
                "spec-agent",
                agent_kind="spec_critic",
                workflow_id=spec_workflow.pk,
                purpose="system_feedback",
                pid=34567,
                status="running",
            )
            spawned_feedback_id = spawned_feedback.pk
            spawned_approval_id = ApprovalRequest.objects.create(
                instance=spawned_feedback,
                method="item/commandExecution/requestApproval",
                params={},
            ).pk

        with (
            patch.object(migration_module, "_podman", side_effect=podman_results) as podman,
            patch.object(
                migration_module,
                "_worker_cmdline_parts",
                side_effect=worker_cmdline_parts,
            ),
            patch.object(
                migration_module,
                "_stop_worker_process",
                side_effect=stop_worker,
            ) as stop_worker_mock,
        ):
            new_apps = self._migrate(self.migrate_to).loader.project_state(self.migrate_to).apps

        self.assertIsNotNone(spawned_feedback_id)
        self.assertIsNotNone(spawned_approval_id)
        self.assertEqual(
            {item.args[0].pk for item in stop_worker_mock.call_args_list},
            {demo_instance.pk, spec_instance.pk, spawned_feedback_id},
        )
        self.assertEqual(
            podman.call_args_list,
            [
                call(
                    [
                        "ps",
                        "-a",
                        "--filter",
                        "label=io.hitch.managed=demo",
                        "--filter",
                        "label=io.hitch.session=main-session",
                        "--filter",
                        "label=io.hitch.demo_token=registration-token",
                        "--format",
                        "{{.ID}}",
                    ]
                ),
                call(["inspect", "demo-container"]),
                call(["rm", "-f", "demo-container"]),
                call(["inspect", "demo-container"]),
                call(
                    [
                        "ps",
                        "-a",
                        "--filter",
                        "label=io.hitch.managed=demo",
                        "--filter",
                        "label=io.hitch.session=main-session",
                        "--filter",
                        "label=io.hitch.demo_token=registration-token",
                        "--format",
                        "{{.ID}}",
                    ]
                ),
            ],
        )
        CodexInstance = new_apps.get_model("main", "CodexInstance")
        ApprovalRequest = new_apps.get_model("main", "ApprovalRequest")
        ProposedSession = new_apps.get_model("main", "ProposedSession")
        SessionMetadata = new_apps.get_model("main", "SessionMetadata")
        SystemAgentRun = new_apps.get_model("main", "SystemAgentRun")
        SystemWorkflow = new_apps.get_model("main", "SystemWorkflow")
        UserInputRequest = new_apps.get_model("main", "UserInputRequest")
        UserSettings = new_apps.get_model("main", "UserSettings")

        with self.assertRaises(LookupError):
            new_apps.get_model("main", "SessionDemo")
        with self.assertRaises(FieldDoesNotExist):
            UserSettings._meta.get_field("spec_critic_enabled")
        self.assertFalse(SystemWorkflow.objects.filter(kind__in=("demo_deployment", "spec_critic")).exists())
        self.assertEqual(
            CodexInstance.objects.filter(pk__in=(demo_instance.pk, spec_instance.pk, orphan_spec_instance.pk)).count(),
            3,
        )
        demo = CodexInstance.objects.get(pk=demo_instance.pk)
        self.assertEqual(demo.status, "failed")
        self.assertEqual(demo.agent_kind, "demo")
        self.assertEqual(demo.prompt, "Registration token: secret-token")
        self.assertEqual(
            CodexInstance.objects.get(pk=orphan_spec_instance.pk).status,
            "failed",
        )
        self.assertEqual(
            CodexInstance.objects.get(pk=blocked_unsettled_feedback.pk).status,
            "failed",
        )
        self.assertEqual(
            CodexInstance.objects.get(pk=spawned_feedback_id).status,
            "failed",
        )
        self.assertFalse(SystemAgentRun.objects.exists())
        self.assertTrue(CodexInstance.objects.filter(pk=unrelated_instance.pk).exists())
        implementation = CodexInstance.objects.get(pk=implementation_instance.pk)
        self.assertIsNone(implementation.workflow_id)
        self.assertEqual(implementation.status, "running")
        self.assertEqual(
            ApprovalRequest.objects.get(pk=approval.pk).decision,
            "cancel",
        )
        self.assertEqual(
            ApprovalRequest.objects.get(pk=spawned_approval_id).decision,
            "cancel",
        )
        self.assertEqual(
            UserInputRequest.objects.get(pk=input_request.pk).response,
            {"answers": {}},
        )
        preserved_request = ProposedSession.objects.get(prompt=original_prompt)
        self.assertEqual(preserved_request.prompt, original_prompt)
        self.assertEqual(preserved_request.inbox_kind, "proposal")
        self.assertNotIn(original_prompt, preserved_request.summary)
        self.assertEqual(preserved_request.source_session_id, spec_main_session.pk)
        self.assertEqual(preserved_request.project_id, recovered_project.pk)
        self.assertIsNone(preserved_request.candidate_session_id)
        self.assertIsNone(preserved_request.source_workflow_id)
        self.assertIs(
            preserved_request.outcome_metadata["resume_source_session"],
            True,
        )
        self.assertEqual(
            {
                key: preserved_request.outcome_metadata[key]
                for key in (
                    "auto_pr_enabled",
                    "auto_qa_enabled",
                    "auto_merge_to_local_branch",
                    "auto_merge_branch",
                )
            },
            {
                "auto_pr_enabled": False,
                "auto_qa_enabled": True,
                "auto_merge_to_local_branch": True,
                "auto_merge_branch": "release",
            },
        )
        stranded_request = ProposedSession.objects.get(prompt="Stranded before implementation spawn")
        self.assertIsNone(stranded_request.candidate_session_id)
        self.assertEqual(stranded_request.source_session.cwd, "/repo")
        self.assertIsNone(stranded_request.project_id)
        self.assertIs(stranded_request.outcome_metadata["auto_pr_enabled"], True)
        self.assertIs(stranded_request.outcome_metadata["auto_qa_enabled"], False)
        blocked_request = ProposedSession.objects.get(prompt="Blocked request with interrupted feedback")
        self.assertEqual(
            blocked_request.source_session.thread_id,
            "blocked-unsettled-spec-main-session",
        )
        self.assertFalse(ProposedSession.objects.filter(prompt="Blocked request with completed feedback").exists())
        self.assertFalse(ProposedSession.objects.filter(prompt="Already started request").exists())
        reused_index_request = ProposedSession.objects.get(prompt="Request whose index was reused")
        self.assertEqual(
            reused_index_request.source_session.thread_id,
            "reused-index-spec-main-session",
        )
        self.assertTrue(CodexInstance.objects.filter(pk=prior_failed_instance.pk).exists())
        self.assertTrue(CodexInstance.objects.filter(pk=unrelated_reused_instance.pk).exists())
        self.assertTrue(SessionMetadata.objects.filter(thread_id="main-session").exists())
        self.assertTrue(SessionMetadata.objects.filter(thread_id="spec-agent").exists())
        self.assertTrue(SessionMetadata.objects.filter(thread_id="orphan-spec-agent").exists())
        self.assertTrue(SessionMetadata.objects.filter(thread_id="unrelated-agent").exists())


class RemovedFeatureRetirementHelperTests(SimpleTestCase):
    def test_worker_event_log_proves_row_ownership_across_release_paths(self) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        with tempfile.TemporaryDirectory() as raw:
            proc_root = Path(raw) / "proc"
            fd_dir = proc_root / "12345" / "fd"
            fd_dir.mkdir(parents=True)
            events_path = Path(raw) / "events" / "7.jsonl"
            events_path.parent.mkdir()
            events_path.write_text("", encoding="utf-8")
            os.symlink(events_path, fd_dir / "4")
            instance = SimpleNamespace(
                pk=7,
                pid=12345,
                events_path=str(events_path),
            )

            self.assertTrue(
                migration_module._worker_has_open_events_file(
                    instance,
                    proc_root=proc_root,
                )
            )

    def test_worker_stop_does_not_signal_an_unrelated_reused_pid(self) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        instance = SimpleNamespace(pk=7, pid=12345, systemd_scope_unit="")
        with (
            patch.object(migration_module, "_worker_cmdline_matches", return_value=False),
            patch.object(
                migration_module,
                "_worker_cmdline_parts",
                return_value=[b"/usr/bin/sleep", b"10"],
            ),
            patch.object(migration_module, "_force_stop_worker") as force_stop,
        ):
            migration_module._stop_worker_process(instance)

        force_stop.assert_not_called()

    def test_force_stop_revalidates_direct_worker_before_group_kill(self) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        instance = SimpleNamespace(pk=7, pid=12345, systemd_scope_unit="")
        root = migration_module._ProcessIdentity(instance.pid, 1, 1000, "S")
        with (
            patch.object(
                migration_module,
                "_worker_cmdline_matches",
                side_effect=[True, False],
            ),
            patch.object(migration_module, "_worker_scope_from_cgroup", return_value=""),
            patch.object(migration_module, "_process_identity", return_value=root),
            patch.object(migration_module.os, "getsid", return_value=instance.pid),
            patch.object(migration_module.os, "killpg") as killpg,
        ):
            migration_module._force_stop_worker(instance)

        killpg.assert_not_called()

    def test_scope_worker_ownership_uses_live_worker_resources_during_pid_handoff(
        self,
    ) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        with tempfile.TemporaryDirectory() as raw:
            proc_root = Path(raw) / "proc"
            worker_root = proc_root / "24680"
            fd_dir = worker_root / "fd"
            fd_dir.mkdir(parents=True)
            scope_unit = "hitch-codex-worker-abc123def456-7.service"
            (worker_root / "cmdline").write_bytes(
                b"\0".join((b"python", b"manage.py", b"codex_worker", b"--instance-id", b"7", b""))
            )
            (worker_root / "cgroup").write_text(
                f"0::/user.slice/{scope_unit}\n",
                encoding="utf-8",
            )
            events_path = Path(raw) / "events" / "7.jsonl"
            events_path.parent.mkdir()
            events_path.write_text("", encoding="utf-8")
            os.symlink(events_path, fd_dir / "4")
            instance = SimpleNamespace(
                pk=7,
                pid=0,
                events_path=str(events_path),
            )

            ownership = migration_module._scope_worker_ownership(
                instance,
                scope_unit,
                proc_root=proc_root,
            )
            (fd_dir / "4").unlink()
            ownership_without_local_resource = (
                migration_module._scope_worker_ownership(
                    instance,
                    scope_unit,
                    proc_root=proc_root,
                )
            )

        self.assertEqual(ownership, "owned")
        self.assertEqual(ownership_without_local_resource, "ambiguous")

    def test_worker_stop_aborts_for_ambiguous_scope_owner(self) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        instance = SimpleNamespace(
            pk=7,
            pid=0,
            events_path="/other-state/events/7.jsonl",
            systemd_scope_unit="hitch-codex-worker-7.service",
        )
        with (
            patch.object(
                migration_module,
                "_scope_worker_ownership",
                return_value="ambiguous",
            ),
            patch.object(migration_module.shutil, "which") as which,
            self.assertRaisesRegex(RuntimeError, "cannot verify ownership"),
        ):
            migration_module._stop_worker_process(instance)

        which.assert_not_called()

    def test_worker_stop_skips_a_reused_legacy_scope(self) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        instance = SimpleNamespace(
            pk=7,
            pid=12345,
            systemd_scope_unit="hitch-codex-worker-7.service",
        )
        with (
            patch.object(migration_module, "_worker_cmdline_matches", return_value=False),
            patch.object(
                migration_module,
                "_scope_worker_ownership",
                return_value="foreign",
            ),
            patch.object(migration_module.shutil, "which") as which,
            patch.object(migration_module.subprocess, "run") as run,
        ):
            migration_module._stop_worker_process(instance)

        which.assert_not_called()
        run.assert_not_called()

    def test_container_cleanup_does_nothing_without_database_registration(
        self,
    ) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        with patch.object(migration_module, "_podman") as podman:
            migration_module._cleanup_removed_feature_containers([])

        podman.assert_not_called()

    def test_container_cleanup_refuses_mismatched_registration_labels(self) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        registration = migration_module._DemoContainerRegistration(
            "local-session", "demo-container", "", "podman", "local-token"
        )
        listed = subprocess.CompletedProcess([], 0, "demo-container\n", "")
        inspected = subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "io.hitch.managed": "demo",
                                "io.hitch.session": "other-deployment-session",
                                "io.hitch.demo_token": "other-token",
                            }
                        }
                    }
                ]
            ),
            "",
        )
        with (
            patch.object(
                migration_module,
                "_podman",
                side_effect=[listed, inspected],
            ) as podman,
            self.assertRaisesRegex(
                RuntimeError,
                "registration does not own it",
            ),
        ):
            migration_module._cleanup_removed_feature_containers([registration])

        self.assertNotIn(call(["rm", "-f", "demo-container"]), podman.call_args_list)

    def test_container_cleanup_removes_tokenless_pre_label_container_by_id(
        self,
    ) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        container_id = "a" * 64
        registration = migration_module._DemoContainerRegistration(
            "legacy-session",
            container_id,
            "hitch-demo-legacy-session-deadbeef",
            "podman",
            "",
        )
        inspected = subprocess.CompletedProcess(
            [],
            0,
            json.dumps([{"Id": container_id, "Config": {"Labels": {}}}]),
            "",
        )
        missing = subprocess.CalledProcessError(
            125,
            ["podman", "inspect", container_id],
            stderr="no such container",
        )
        with patch.object(
            migration_module,
            "_podman",
            side_effect=[
                inspected,
                subprocess.CompletedProcess([], 0, "", ""),
                missing,
            ],
        ) as podman:
            migration_module._cleanup_removed_feature_containers([registration])

        self.assertEqual(
            podman.call_args_list,
            [
                call(["inspect", container_id]),
                call(["rm", "-f", container_id]),
                call(["inspect", container_id]),
            ],
        )

    def test_container_cleanup_refuses_mismatched_tokenless_container_id(
        self,
    ) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        container_id = "a" * 64
        registration = migration_module._DemoContainerRegistration("legacy-session", container_id, "", "podman", "")
        inspected = subprocess.CompletedProcess(
            [],
            0,
            json.dumps([{"Id": "b" * 64, "Config": {"Labels": {}}}]),
            "",
        )
        with (
            patch.object(migration_module, "_podman", return_value=inspected) as podman,
            self.assertRaisesRegex(RuntimeError, "registration does not own it"),
        ):
            migration_module._cleanup_removed_feature_containers([registration])

        self.assertNotIn(call(["rm", "-f", container_id]), podman.call_args_list)

    def test_container_cleanup_fails_if_registered_container_remains(self) -> None:
        migration_module = importlib.import_module("hitch.main.migrations.0069_remove_demo_and_spec_critic")
        registration = migration_module._DemoContainerRegistration("local-session", "demo-container", "", "podman", "")
        inspected = subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                [
                    {
                        "Config": {
                            "Labels": {
                                "io.hitch.managed": "demo",
                                "io.hitch.session": "local-session",
                            }
                        }
                    }
                ]
            ),
            "",
        )
        with (
            patch.object(
                migration_module,
                "_podman",
                side_effect=[
                    inspected,
                    subprocess.CompletedProcess([], 0, "", ""),
                    inspected,
                ],
            ),
            self.assertRaisesRegex(RuntimeError, "remains after removal"),
        ):
            migration_module._cleanup_removed_feature_containers([registration])


class LocalBranchMergeRemovalMigrationTests(TransactionTestCase):
    migrate_from = [("main", "0069_remove_demo_and_spec_critic")]
    migrate_to = [("main", "0070_remove_local_branch_merge")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_removes_configuration_fields_and_stale_metadata(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
        Project = old_apps.get_model("main", "Project")
        AutonomousGoal = old_apps.get_model("main", "AutonomousGoal")
        CodexInstance = old_apps.get_model("main", "CodexInstance")
        ProposedSession = old_apps.get_model("main", "ProposedSession")
        SessionMetadata = old_apps.get_model("main", "SessionMetadata")
        SystemWorkflow = old_apps.get_model("main", "SystemWorkflow")

        project = Project.objects.create(name="Project", repo_path="/repo")
        AutonomousGoal.objects.create(
            project=project,
            title="Goal",
            goal="Improve the project.",
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )
        SessionMetadata.objects.create(
            thread_id="session",
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )
        CodexInstance.objects.create(
            pid=1,
            thread_id="session",
            cwd="/repo",
            events_path="/dev/null",
            auto_merge_to_local_branch=True,
            auto_merge_branch="release",
        )
        proposal = ProposedSession.objects.create(
            project=project,
            title="Proposal",
            outcome_metadata={
                "keep": "value",
                "auto_merge_to_local_branch": True,
                "auto_merge_branch": "release",
                "auto_merge_status": "merged",
                "auto_merge_commit_sha": "abc123",
                "auto_merge_error": "old failure",
            },
        )
        workflow = SystemWorkflow.objects.create(
            kind="pr_qa",
            cwd="/repo",
            step="local_branch_merged",
            state={
                "keep": "value",
                "auto_merge_to_local_branch": True,
                "auto_merge_branch": "release",
                "auto_merge_result": {"changed": True},
                "auto_merge_reviewed_diff": "diff",
                "auto_merge_reviewed_source_tree": "tree",
                "auto_merge_reviewed_target_sha": "target",
                "auto_merge_session_base_sha": "base",
            },
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
        for model_name in ("AutonomousGoal", "CodexInstance", "SessionMetadata"):
            model = new_apps.get_model("main", model_name)
            for field_name in ("auto_merge_to_local_branch", "auto_merge_branch"):
                with self.assertRaises(FieldDoesNotExist):
                    model._meta.get_field(field_name)

        ProposedSession = new_apps.get_model("main", "ProposedSession")
        SystemWorkflow = new_apps.get_model("main", "SystemWorkflow")
        self.assertEqual(
            ProposedSession.objects.get(pk=proposal.pk).outcome_metadata,
            {"keep": "value"},
        )
        migrated_workflow = SystemWorkflow.objects.get(pk=workflow.pk)
        self.assertEqual(migrated_workflow.state, {"keep": "value"})
        self.assertEqual(migrated_workflow.step, "review_completed")


class PrQaWrapperRemovalMigrationTests(TransactionTestCase):
    migrate_from = [("main", "0071_remove_hitch_pr_publication_claim")]
    migrate_to = [("main", "0072_session_pull_request")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_preserves_pr_state_and_retires_in_flight_wrapper_work(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
        ApprovalRequest = old_apps.get_model("main", "ApprovalRequest")
        CodexInstance = old_apps.get_model("main", "CodexInstance")
        SystemAgentRun = old_apps.get_model("main", "SystemAgentRun")
        SystemWorkflow = old_apps.get_model("main", "SystemWorkflow")
        UserInputRequest = old_apps.get_model("main", "UserInputRequest")
        WorkflowSteeringMessage = old_apps.get_model(
            "main", "WorkflowSteeringMessage"
        )

        SystemWorkflow.objects.create(
            kind="pr_qa",
            main_thread_id="pr-thread",
            cwd="/old-repo",
            status="completed",
            state={
                "pr_handoff": {
                    "url": "https://github.com/acme/widgets/pull/11",
                    "pr_number": 11,
                }
            },
        )
        workflow = SystemWorkflow.objects.create(
            kind="pr_qa",
            main_thread_id="pr-thread",
            cwd="/repo",
            status="running",
            step="pr_watch_running",
            state={
                "pr_handoff": {
                    "url": "https://github.com/acme/widgets/pull/12",
                    "repository_full_name": "acme/widgets",
                    "pr_number": 12,
                },
                "pr_gates": {"checks": "pending"},
                "unrelated_wrapper_state": "drop me",
            },
        )
        visible = CodexInstance.objects.create(
            pid=1,
            thread_id="pr-thread",
            cwd="/repo",
            prompt="Publish and watch",
            events_path="/dev/null",
            status="running",
            purpose="user",
            workflow_id=workflow.pk,
            workflow_routing_started_at=timezone.now(),
        )
        hidden = CodexInstance.objects.create(
            pid=2,
            thread_id="hidden-reviewer",
            cwd="/repo",
            prompt="Review",
            events_path="/dev/null",
            status="running",
            purpose="system_agent",
            workflow_id=workflow.pk,
        )
        feedback = CodexInstance.objects.create(
            pid=3,
            thread_id="pr-thread",
            cwd="/repo",
            prompt="Report wrapper failure",
            events_path="/dev/null",
            status="starting",
            purpose="system_feedback",
            workflow_id=workflow.pk,
        )
        run = SystemAgentRun.objects.create(
            workflow=workflow,
            instance=hidden,
            agent_kind="legacy_pr_monitor",
            thread_id="hidden-reviewer",
            status="running",
        )
        approval = ApprovalRequest.objects.create(
            instance=feedback,
            method="item/commandExecution/requestApproval",
        )
        input_request = UserInputRequest.objects.create(
            instance=feedback,
            method="item/tool/requestUserInput",
        )
        WorkflowSteeringMessage.objects.create(
            workflow=workflow,
            prompt="First queued instruction",
        )
        WorkflowSteeringMessage.objects.create(
            workflow=workflow,
            prompt="Second queued instruction",
        )

        review_workflow = SystemWorkflow.objects.create(
            kind="pr_qa",
            main_thread_id="review-thread",
            cwd="/repo",
            status="running",
            state={"open_pr_on_lgtm": False, "review_guidance": True},
        )
        review_turn = CodexInstance.objects.create(
            pid=4,
            thread_id="review-thread",
            cwd="/repo",
            events_path="/dev/null",
            status="starting",
            purpose="user",
            workflow_id=review_workflow.pk,
        )
        publish_workflow = SystemWorkflow.objects.create(
            kind="pr_qa",
            main_thread_id="publish-thread",
            cwd="/repo",
            status="running",
        )
        publish_turn = CodexInstance.objects.create(
            pid=5,
            thread_id="publish-thread",
            cwd="/repo",
            events_path="/dev/null",
            status="starting",
            purpose="user",
            workflow_id=publish_workflow.pk,
        )
        blocked_workflow = SystemWorkflow.objects.create(
            kind="pr_qa",
            main_thread_id="blocked-thread",
            cwd="/repo",
            status="blocked",
            step="blocked",
            state={"error": "Legacy failure"},
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
        ApprovalRequest = new_apps.get_model("main", "ApprovalRequest")
        CodexInstance = new_apps.get_model("main", "CodexInstance")
        SessionPullRequest = new_apps.get_model("main", "SessionPullRequest")
        SystemAgentRun = new_apps.get_model("main", "SystemAgentRun")
        SystemWorkflow = new_apps.get_model("main", "SystemWorkflow")
        UserInputRequest = new_apps.get_model("main", "UserInputRequest")

        record = SessionPullRequest.objects.get(thread_id="pr-thread")
        self.assertEqual(record.cwd, "/repo")
        self.assertEqual(record.state["pr_handoff"]["pr_number"], 12)
        self.assertEqual(record.state["pr_gates"], {"checks": "pending"})
        self.assertNotIn("unrelated_wrapper_state", record.state)

        migrated_visible = CodexInstance.objects.get(pk=visible.pk)
        self.assertEqual(migrated_visible.status, "running")
        self.assertIsNone(migrated_visible.workflow_id)
        self.assertIsNone(migrated_visible.workflow_routing_started_at)
        self.assertEqual(migrated_visible.agent_kind, "pr_watch")
        self.assertEqual(
            CodexInstance.objects.get(pk=review_turn.pk).agent_kind,
            "review_guidance",
        )
        self.assertEqual(
            CodexInstance.objects.get(pk=publish_turn.pk).agent_kind,
            "pr_publish",
        )

        for instance_pk in (hidden.pk, feedback.pk):
            retired = CodexInstance.objects.get(pk=instance_pk)
            self.assertEqual(retired.status, "failed")
            self.assertIsNone(retired.workflow_id)
            self.assertIsNotNone(retired.ended_at)
            self.assertIsNotNone(retired.interrupt_requested_at)
            self.assertEqual(retired.error, "PR/QA wrapper retired during upgrade")
        migrated_run = SystemAgentRun.objects.get(pk=run.pk)
        self.assertEqual(migrated_run.status, "failed")
        self.assertEqual(
            migrated_run.error,
            "PR/QA wrapper retired during upgrade",
        )
        self.assertEqual(
            ApprovalRequest.objects.get(pk=approval.pk).decision,
            "cancel",
        )
        self.assertEqual(
            UserInputRequest.objects.get(pk=input_request.pk).response,
            {"answers": {}},
        )

        migrated_workflow = SystemWorkflow.objects.get(pk=workflow.pk)
        self.assertEqual(migrated_workflow.status, "completed")
        self.assertEqual(migrated_workflow.step, "pr_wrapper_retired")
        self.assertTrue(migrated_workflow.state["pr_qa_wrapper_retired"])
        self.assertEqual(
            migrated_workflow.state["retired_steering_messages"],
            ["First queued instruction", "Second queued instruction"],
        )
        migrated_blocked = SystemWorkflow.objects.get(pk=blocked_workflow.pk)
        self.assertEqual(migrated_blocked.status, "completed")
        self.assertEqual(migrated_blocked.step, "pr_wrapper_retired")
        self.assertTrue(migrated_blocked.state["pr_qa_wrapper_retired"])
        with self.assertRaises(LookupError):
            new_apps.get_model("main", "WorkflowSteeringMessage")


class AutoReviewFieldRemovalMigrationTests(TransactionTestCase):
    migrate_from = [("main", "0073_remove_autonomousgoalmemory")]
    migrate_to = [("main", "0074_remove_codexinstance_auto_review")]

    def _migrate(self, targets: list[tuple[str, str]]) -> MigrationExecutor:
        executor = MigrationExecutor(connection)
        executor.loader.build_graph()
        executor.migrate(targets)
        return executor

    def test_keeps_columns_available_to_workers_started_before_upgrade(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
        legacy_codex_instance = old_apps.get_model("main", "CodexInstance")
        legacy_worker = legacy_codex_instance.objects.create(
            pid=7,
            thread_id="legacy-auto-review-worker",
            cwd="/repo",
            prompt="finish publishing",
            events_path="/dev/null",
            status="running",
            auto_pr_enabled=True,
            auto_qa_enabled=False,
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
        CodexInstance = new_apps.get_model("main", "CodexInstance")
        removed_fields = {
            "auto_pr_enabled",
            "auto_qa_enabled",
            "auto_pr_triggered_at",
            "auto_qa_triggered_at",
        }
        for field_name in removed_fields:
            with self.assertRaises(FieldDoesNotExist):
                CodexInstance._meta.get_field(field_name)

        with connection.cursor() as cursor:
            columns = {
                column.name: column
                for column in connection.introspection.get_table_description(
                    cursor, CodexInstance._meta.db_table
                )
            }
        self.assertLessEqual(removed_fields, columns.keys())
        self.assertTrue(columns["auto_pr_enabled"].null_ok)
        self.assertTrue(columns["auto_qa_enabled"].null_ok)

        triggered_at = timezone.now()
        legacy_codex_instance.objects.filter(pk=legacy_worker.pk).update(
            auto_pr_triggered_at=triggered_at
        )
        legacy_worker.refresh_from_db()
        self.assertTrue(legacy_worker.auto_pr_enabled)
        self.assertEqual(legacy_worker.auto_pr_triggered_at, triggered_at)
