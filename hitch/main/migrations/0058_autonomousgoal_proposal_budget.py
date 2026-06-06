from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0057_autonomousgoal_stacked_diff_depth"),
    ]

    operations = [
        migrations.AddField(
            model_name="autonomousgoal",
            name="proposal_budget",
            field=models.PositiveBigIntegerField(blank=True, null=True),
        ),
    ]
