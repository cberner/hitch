from django.apps.registry import Apps
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor

_MODES = {"auto_review", "prompt_user", "deny_all", "approve_all"}
_MODE_KEY = "watch_approval_mode"
_OWNER_KEY = "watch_approval_owner_id"


def move_snapshots(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    metadata_model = apps.get_model("main", "SessionMetadata")
    pr_model = apps.get_model("main", "SessionPullRequest")
    database = schema_editor.connection.alias
    for record in pr_model.objects.using(database).all().iterator():
        state = record.state
        if not isinstance(state, dict) or not ({_MODE_KEY, _OWNER_KEY} & state.keys()):
            continue
        mode, owner_id = state.get(_MODE_KEY), state.get(_OWNER_KEY)
        if isinstance(mode, str) and mode in _MODES:
            metadata, _ = metadata_model.objects.using(database).get_or_create(
                thread_id=record.thread_id, defaults={"cwd": record.cwd},
            )
            if type(owner_id) is not int or not 0 < owner_id < 2**63:
                owner_id = None
            metadata_model.objects.using(database).filter(pk=metadata.pk).update(
                approval_snapshot_mode=mode, approval_snapshot_instance_id=owner_id,
            )
        state = {key: value for key, value in state.items() if key not in {_MODE_KEY, _OWNER_KEY}}
        if state:
            pr_model.objects.using(database).filter(pk=record.pk).update(state=state)
        else:
            record.delete(using=database)


def restore_snapshots(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    metadata_model = apps.get_model("main", "SessionMetadata")
    pr_model = apps.get_model("main", "SessionPullRequest")
    database = schema_editor.connection.alias
    for metadata in metadata_model.objects.using(database).exclude(approval_snapshot_mode="").iterator():
        record, _ = pr_model.objects.using(database).get_or_create(
            thread_id=metadata.thread_id, defaults={"cwd": metadata.cwd},
        )
        state = record.state if isinstance(record.state, dict) else {}
        pr_model.objects.using(database).filter(pk=record.pk).update(state={
            **state,
            _MODE_KEY: metadata.approval_snapshot_mode,
            _OWNER_KEY: metadata.approval_snapshot_instance_id,
        })


class Migration(migrations.Migration):
    dependencies = [("main", "0083_cleanup_protection_indexes")]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata", name="approval_snapshot_mode",
            field=models.CharField(max_length=32, blank=True, default=""),
        ),
        migrations.AddField(
            model_name="sessionmetadata", name="approval_snapshot_instance_id",
            field=models.PositiveBigIntegerField(null=True, blank=True),
        ),
        migrations.RunPython(move_snapshots, restore_snapshots),
    ]
