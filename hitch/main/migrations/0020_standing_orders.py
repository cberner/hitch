import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0019_sessiondemo"),
    ]

    operations = [
        migrations.CreateModel(
            name="StandingOrder",
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
                ("goal", models.TextField()),
                (
                    "confidence_threshold",
                    models.CharField(
                        choices=[
                            ("medium", "Medium"),
                            ("high", "High"),
                            ("very_high", "Very high"),
                        ],
                        default="high",
                        max_length=32,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "project",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="standing_orders",
                        to="main.project",
                    ),
                ),
            ],
            options={
                "ordering": ["created_at", "id"],
            },
        ),
        migrations.CreateModel(
            name="ProposedSession",
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
                ("summary", models.TextField(blank=True, default="")),
                (
                    "confidence",
                    models.CharField(
                        choices=[
                            ("medium", "Medium"),
                            ("high", "High"),
                            ("very_high", "Very high"),
                        ],
                        default="medium",
                        max_length=32,
                    ),
                ),
                ("relevant_files", models.JSONField(blank=True, default=list)),
                (
                    "outcome_status",
                    models.CharField(
                        blank=True,
                        choices=[
                            ("", "Not set"),
                            ("accepted", "Accepted"),
                            ("rejected", "Rejected"),
                        ],
                        default="",
                        max_length=32,
                    ),
                ),
                ("outcome_notes", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "accepted_session",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="accepted_standing_order_proposals",
                        to="main.sessionmetadata",
                    ),
                ),
                (
                    "candidate_session",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="standing_order_candidate_proposals",
                        to="main.sessionmetadata",
                    ),
                ),
                (
                    "judge_session",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="standing_order_judge_proposals",
                        to="main.sessionmetadata",
                    ),
                ),
                (
                    "source_workflow",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="proposed_sessions",
                        to="main.systemworkflow",
                    ),
                ),
                (
                    "standing_order",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="proposed_sessions",
                        to="main.standingorder",
                    ),
                ),
            ],
            options={
                "ordering": ["created_at", "id"],
            },
        ),
        migrations.AddIndex(
            model_name="standingorder",
            index=models.Index(
                fields=["project", "created_at"], name="main_standi_project_803130_idx"
            ),
        ),
        migrations.AddIndex(
            model_name="proposedsession",
            index=models.Index(
                fields=["standing_order", "created_at"],
                name="main_propos_standin_66a16b_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="proposedsession",
            index=models.Index(
                fields=["outcome_status", "created_at"],
                name="main_propos_outcome_17d384_idx",
            ),
        ),
    ]
