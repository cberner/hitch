from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0063_backfill_session_activity"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="codex_error_info",
            field=models.JSONField(blank=True, default=None, null=True),
        ),
    ]
