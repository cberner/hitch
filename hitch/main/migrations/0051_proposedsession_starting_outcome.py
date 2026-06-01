from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0050_session_project_visibility_settings"),
    ]

    operations = [
        migrations.AlterField(
            model_name="proposedsession",
            name="outcome_status",
            field=models.CharField(
                blank=True,
                choices=[
                    ("", "Not set"),
                    ("starting", "Starting"),
                    ("accepted", "Accepted"),
                    ("rejected", "Rejected"),
                    ("dismissed", "Dismissed"),
                ],
                default="",
                max_length=32,
            ),
        ),
    ]
