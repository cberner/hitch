import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0017_proposedtask"),
    ]

    operations = [
        migrations.AddField(
            model_name="proposedtask",
            name="pr_url",
            field=models.URLField(blank=True, default="", max_length=512),
        ),
        migrations.AddField(
            model_name="proposedtask",
            name="session",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="proposed_tasks",
                to="main.sessionmetadata",
            ),
        ),
        migrations.AlterField(
            model_name="proposedtask",
            name="outcome_status",
            field=models.CharField(
                blank=True,
                choices=[
                    ("", "Not set"),
                    ("accepted", "Accepted"),
                    ("rejected", "Rejected"),
                    ("pr_opened", "PR opened"),
                    ("completed", "Completed"),
                    ("superseded", "Superseded"),
                ],
                default="",
                max_length=32,
            ),
        ),
    ]
