from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0059_sessionmetadata_approval_mode"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="codexinstance",
            name="qa_panel_enabled",
        ),
        migrations.RemoveField(
            model_name="usersettings",
            name="qa_panel_enabled",
        ),
    ]
