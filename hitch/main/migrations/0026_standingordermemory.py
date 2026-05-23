import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0025_remove_okrs"),
    ]

    operations = [
        migrations.CreateModel(
            name="StandingOrderMemory",
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
                ("title", models.CharField(blank=True, default="", max_length=200)),
                ("summary", models.TextField(blank=True, default="")),
                ("relevant_files", models.JSONField(blank=True, default=list)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "candidate_session",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="standing_order_memories",
                        to="main.sessionmetadata",
                    ),
                ),
                (
                    "source_workflow",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="standing_order_memories",
                        to="main.systemworkflow",
                    ),
                ),
                (
                    "standing_order",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="memories",
                        to="main.standingorder",
                    ),
                ),
            ],
            options={
                "ordering": ["created_at", "id"],
            },
        ),
        migrations.AddIndex(
            model_name="standingordermemory",
            index=models.Index(
                fields=["standing_order", "-created_at"],
                name="main_standi_standin_3b898b_idx",
            ),
        ),
        migrations.AddConstraint(
            model_name="standingordermemory",
            constraint=models.UniqueConstraint(
                condition=models.Q(("source_workflow__isnull", False)),
                fields=("source_workflow",),
                name="uniq_standing_order_memory_workflow",
            ),
        ),
    ]
