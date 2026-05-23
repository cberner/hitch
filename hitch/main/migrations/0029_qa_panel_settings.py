from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0028_rename_proposed_session_project_index"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="qa_panel_enabled",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="qa_panel_enabled",
            field=models.BooleanField(default=False),
        ),
    ]
