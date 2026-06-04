from django.db import migrations, models


def populate_existing_archived_at(apps, schema_editor):
    session_metadata = apps.get_model("main", "SessionMetadata")
    session_metadata.objects.filter(codex_archived=True, codex_archived_at__isnull=True).update(
        codex_archived_at=models.F("updated_at")
    )


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0054_sessionmetadata_session_list_index"),
    ]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata",
            name="codex_archived_at",
            field=models.DateTimeField(blank=True, db_index=True, null=True),
        ),
        migrations.RunPython(populate_existing_archived_at, migrations.RunPython.noop),
    ]
