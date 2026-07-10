from django.db import migrations


def backfill_session_activity(apps, schema_editor):
    codex_instance = apps.get_model("main", "CodexInstance")
    session_metadata = apps.get_model("main", "SessionMetadata")
    database = schema_editor.connection.alias

    latest_activity_by_thread = {}
    instances = (
        codex_instance.objects.using(database)
        .exclude(thread_id="")
        .values_list("thread_id", "started_at", "ended_at")
        .iterator(chunk_size=1_000)
    )
    for thread_id, started_at, ended_at in instances:
        activity_at = ended_at or started_at
        previous = latest_activity_by_thread.get(thread_id)
        if previous is None or activity_at > previous:
            latest_activity_by_thread[thread_id] = activity_at

    changed = []
    metadata_rows = (
        session_metadata.objects.using(database)
        .filter(thread_id__in=latest_activity_by_thread)
        .iterator(chunk_size=1_000)
    )
    for metadata in metadata_rows:
        activity_at = latest_activity_by_thread[metadata.thread_id]
        if metadata.codex_updated_at is None or activity_at > metadata.codex_updated_at:
            metadata.codex_updated_at = activity_at
            changed.append(metadata)
    session_metadata.objects.using(database).bulk_update(changed, ["codex_updated_at"], batch_size=1_000)


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0062_project_auto_pull_enabled"),
    ]

    operations = [
        migrations.RunPython(backfill_session_activity, migrations.RunPython.noop),
    ]
