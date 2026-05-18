import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0009_usersettings_last_selected_repo"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="agent_kind",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="display_author",
            field=models.CharField(blank=True, default="", max_length=128),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="purpose",
            field=models.CharField(
                choices=[
                    ("user", "user"),
                    ("system_agent", "system agent"),
                    ("system_feedback", "system feedback"),
                ],
                default="user",
                max_length=32,
            ),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="output_schema",
            field=models.JSONField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="workflow_id",
            field=models.BigIntegerField(blank=True, db_index=True, null=True),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="user_message_index",
            field=models.PositiveIntegerField(blank=True, db_index=True, null=True),
        ),
        migrations.AddIndex(
            model_name="codexinstance",
            index=models.Index(fields=["purpose"], name="main_codexi_purpose_e781e3_idx"),
        ),
        migrations.CreateModel(
            name="SystemWorkflow",
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
                ("kind", models.CharField(max_length=64)),
                ("main_thread_id", models.CharField(db_index=True, max_length=128)),
                ("cwd", models.CharField(max_length=4096)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("running", "running"),
                            ("blocked", "blocked"),
                            ("completed", "completed"),
                            ("failed", "failed"),
                            ("max_iterations_reached", "max iterations reached"),
                        ],
                        default="running",
                        max_length=64,
                    ),
                ),
                ("step", models.CharField(blank=True, default="", max_length=64)),
                ("iteration", models.PositiveIntegerField(default=0)),
                ("max_iterations", models.PositiveIntegerField(default=3)),
                ("state", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["kind", "main_thread_id", "status"],
                        name="main_system_kind_f8ef00_idx",
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        condition=models.Q(("status", "running")),
                        fields=("kind", "main_thread_id"),
                        name="uniq_running_system_workflow",
                    )
                ],
            },
        ),
        migrations.CreateModel(
            name="SystemAgentRun",
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
                ("agent_kind", models.CharField(max_length=64)),
                ("thread_id", models.CharField(db_index=True, max_length=128)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("starting", "starting"),
                            ("running", "running"),
                            ("completed", "completed"),
                            ("failed", "failed"),
                        ],
                        default="starting",
                        max_length=64,
                    ),
                ),
                ("input", models.JSONField(blank=True, default=dict)),
                ("output", models.JSONField(blank=True, default=dict)),
                ("raw_output", models.TextField(blank=True, default="")),
                ("error", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="system_agent_runs",
                        to="main.codexinstance",
                    ),
                ),
                (
                    "workflow",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="agent_runs",
                        to="main.systemworkflow",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["workflow", "-created_at"],
                        name="main_system_workflo_62f332_idx",
                    ),
                    models.Index(
                        fields=["agent_kind", "status"],
                        name="main_system_agent_k_fa0e6b_idx",
                    ),
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("instance",),
                        name="uniq_system_agent_run_instance",
                    )
                ],
            },
        ),
    ]
