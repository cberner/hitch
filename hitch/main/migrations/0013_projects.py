import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0012_archivedsessiontokenusage"),
    ]

    operations = [
        migrations.CreateModel(
            name="Project",
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
                ("name", models.CharField(max_length=200)),
                ("repo_path", models.CharField(max_length=4096, unique=True)),
                ("git_common_dir", models.CharField(blank=True, default="", max_length=4096)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["name", "repo_path"],
            },
        ),
        migrations.CreateModel(
            name="SessionMetadata",
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
                ("cwd", models.CharField(blank=True, default="", max_length=4096)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "project",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="sessions",
                        to="main.project",
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="usersettings",
            name="selected_project",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="selected_by_settings",
                to="main.project",
            ),
        ),
        migrations.AddIndex(
            model_name="sessionmetadata",
            index=models.Index(fields=["project", "-updated_at"], name="main_sessio_project_0daead_idx"),
        ),
    ]
