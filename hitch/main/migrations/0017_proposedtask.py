import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0016_auto_pr"),
    ]

    operations = [
        migrations.CreateModel(
            name="ProposedTask",
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
                ("title", models.CharField(max_length=200)),
                ("description", models.TextField(blank=True, default="")),
                ("success_criteria", models.TextField(blank=True, default="")),
                ("rationale", models.TextField(blank=True, default="")),
                (
                    "outcome_status",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("", "Not set"),
                            ("accepted", "Accepted"),
                            ("rejected", "Rejected"),
                            ("completed", "Completed"),
                            ("superseded", "Superseded"),
                        ],
                        default="",
                        max_length=32,
                    ),
                ),
                ("outcome_notes", models.TextField(blank=True, default="")),
                ("sort_order", models.PositiveIntegerField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "key_result",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="proposed_tasks",
                        to="main.keyresult",
                    ),
                ),
                (
                    "source_workflow",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="proposed_tasks",
                        to="main.systemworkflow",
                    ),
                ),
            ],
            options={
                "ordering": ["created_at", "sort_order", "id"],
            },
        ),
        migrations.AddIndex(
            model_name="proposedtask",
            index=models.Index(
                fields=["key_result", "created_at"],
                name="main_propos_key_res_66a12d_idx",
            ),
        ),
    ]
