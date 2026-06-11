"""Data-migration tests for the views-owned columns."""



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
