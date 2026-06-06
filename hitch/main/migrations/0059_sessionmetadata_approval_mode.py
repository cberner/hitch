from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0058_autonomousgoal_proposal_budget"),
    ]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata",
            name="approval_mode",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
    ]
