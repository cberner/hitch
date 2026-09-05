from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0075_remove_sessionmetadata_derived_stage_pr_refresh_attempted_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata",
            name="archive_local_only",
            field=models.BooleanField(default=False),
        ),
    ]
