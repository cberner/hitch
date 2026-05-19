from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0011_enable_memories_settings"),
    ]

    operations = [
        migrations.CreateModel(
            name="ArchivedSessionTokenUsage",
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
                ("rollout_path", models.CharField(blank=True, default="", max_length=4096)),
                ("rollout_mtime_ns", models.PositiveBigIntegerField(default=0)),
                ("input_tokens", models.PositiveBigIntegerField(default=0)),
                ("cached_input_tokens", models.PositiveBigIntegerField(default=0)),
                ("output_tokens", models.PositiveBigIntegerField(default=0)),
                ("total_tokens", models.PositiveBigIntegerField(default=0)),
                ("context_tokens", models.PositiveBigIntegerField(default=0)),
                ("model_context_window", models.PositiveBigIntegerField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
