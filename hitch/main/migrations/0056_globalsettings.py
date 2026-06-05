from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0055_sessionmetadata_codex_archived_at"),
    ]

    operations = [
        migrations.CreateModel(
            name="GlobalSettings",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("disk_usage_max_percent", models.FloatField(default=20.0)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
