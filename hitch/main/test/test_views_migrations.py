"""Data-migration tests for the views-owned columns."""



from django.core.exceptions import FieldDoesNotExist
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import (
    TransactionTestCase,
)


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

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
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

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
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

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
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

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
        CodexInstance = new_apps.get_model("main", "CodexInstance")

        for thread_id, _approval_mode, _status in rows:
            self.assertFalse(
                CodexInstance.objects.get(thread_id=thread_id).approval_mode_live_editable
            )


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

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
        legacy_codex_instance_model = old_apps.get_model("main", "CodexInstance")
        legacy_instance = legacy_codex_instance_model.objects.create(
            pid=1,
            thread_id="legacy-worker-before-migration",
            cwd="/repo",
            prompt="hi",
            events_path="/dev/null",
            base_instructions="legacy prompt",
        )

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
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
                for column in connection.introspection.get_table_description(
                    cursor, CodexInstance._meta.db_table
                )
            }
            settings_columns = {
                column.name
                for column in connection.introspection.get_table_description(
                    cursor, UserSettings._meta.db_table
                )
            }

        self.assertTrue(codex_columns["base_instructions"].null_ok)
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT base_instructions FROM {CodexInstance._meta.db_table} "
                "WHERE id = %s",
                [instance.pk],
            )
            self.assertIsNone(cursor.fetchone()[0])
        self.assertNotIn("coding_agent", settings_columns)

        rolled_back_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
        rolled_back_codex_instance_model = rolled_back_apps.get_model(
            "main", "CodexInstance"
        )
        self.assertEqual(
            rolled_back_codex_instance_model.objects.get(
                pk=instance.pk
            ).base_instructions,
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

    def test_backfills_only_blank_account_efforts(self) -> None:
        leaf = MigrationExecutor(connection).loader.graph.leaf_nodes("main")
        self.addCleanup(self._migrate, leaf)

        old_apps = self._migrate(self.migrate_from).loader.project_state(
            self.migrate_from
        ).apps
        User = old_apps.get_model("auth", "User")
        UserSettings = old_apps.get_model("main", "UserSettings")

        blank_user = User.objects.create(username="blank-effort")
        saved_user = User.objects.create(username="saved-effort")
        UserSettings.objects.create(user=blank_user, reasoning_effort="")
        UserSettings.objects.create(user=saved_user, reasoning_effort="xhigh")

        new_apps = self._migrate(self.migrate_to).loader.project_state(
            self.migrate_to
        ).apps
        UserSettings = new_apps.get_model("main", "UserSettings")

        self.assertEqual(
            UserSettings.objects.get(user_id=blank_user.pk).reasoning_effort,
            "high",
        )
        self.assertEqual(
            UserSettings.objects.get(user_id=saved_user.pk).reasoning_effort,
            "xhigh",
        )
