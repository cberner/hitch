from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("main", "0082_session_model_effort")]

    operations = [
        migrations.AddIndex(
            model_name="sessionpullrequest",
            index=models.Index(fields=["cwd"], name="main_watch_cwd_idx"),
        ),
        migrations.AddIndex(
            model_name="sessionpullrequest",
            index=models.Index(fields=["updated_at"], name="main_watch_updated_idx"),
        ),
        migrations.AddIndex(
            model_name="sessionmetadata",
            index=models.Index(fields=["cwd"], name="main_session_cwd_idx"),
        ),
        migrations.AddIndex(
            model_name="sessionmetadata",
            index=models.Index(fields=["updated_at"], name="main_session_updated_idx"),
        ),
    ]
