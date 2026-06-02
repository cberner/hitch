from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0051_autonomousgoal_deleted_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata",
            name="derived_stage_pr_refresh_attempted_at",
            field=models.DateTimeField(blank=True, db_index=True, null=True),
        ),
    ]
