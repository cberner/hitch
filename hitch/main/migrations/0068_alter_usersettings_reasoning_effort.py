from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0067_workflowsteeringmessage"),
    ]

    operations = [
        migrations.AlterField(
            model_name="usersettings",
            name="reasoning_effort",
            field=models.CharField(blank=True, default="high", max_length=32),
        ),
    ]
