from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0015_okrs"),
    ]

    operations = [
        migrations.AddField(
            model_name="project",
            name="auto_pr_mode",
            field=models.CharField(
                choices=[
                    ("follow_global", "Follow global"),
                    ("on", "On"),
                    ("off", "Off"),
                ],
                default="follow_global",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="approval_mode",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="auto_pr_enabled",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="auto_pr_triggered_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="model",
            field=models.CharField(blank=True, default="", max_length=256),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="plan_mode",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="reasoning_effort",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="sandbox_policy",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="sessionmetadata",
            name="auto_pr_enabled",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="auto_pr_enabled",
            field=models.BooleanField(default=False),
        ),
    ]
