from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0018_proposedtask_session_pr"),
    ]

    operations = [
        migrations.CreateModel(
            name="SessionDemo",
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
                ("thread_id", models.CharField(max_length=128, unique=True)),
                ("host", models.CharField(default="127.0.0.1", max_length=255)),
                ("port", models.PositiveIntegerField()),
                ("container_id", models.CharField(blank=True, default="", max_length=128)),
                (
                    "container_name",
                    models.CharField(blank=True, default="", max_length=128),
                ),
                ("runtime", models.CharField(default="podman", max_length=32)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("active", "active"),
                            ("stopped", "stopped"),
                            ("failed", "failed"),
                        ],
                        default="active",
                        max_length=32,
                    ),
                ),
                ("last_error", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["thread_id", "status"],
                        name="sessiondemo_thread_status_idx",
                    ),
                    models.Index(fields=["status"], name="sessiondemo_status_idx"),
                ],
            },
        ),
    ]
