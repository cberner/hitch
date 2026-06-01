from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0049_archivedsessiontokenusage_usage_logic_version"),
    ]

    operations = [
        migrations.AddField(
            model_name="usersettings",
            name="visible_session_project_ids",
            field=models.JSONField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="show_no_project_sessions",
            field=models.BooleanField(default=True),
        ),
    ]
