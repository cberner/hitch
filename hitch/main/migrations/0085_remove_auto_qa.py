from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [("main", "0084_session_approval_snapshots")]

    operations = [
        migrations.RemoveField(
            model_name="sessionmetadata",
            name="auto_qa_enabled",
        ),
        migrations.RemoveField(
            model_name="usersettings",
            name="auto_qa_enabled",
        ),
    ]
