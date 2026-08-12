from django.db import migrations, models


def backfill_blank_reasoning_effort(apps, schema_editor):
    UserSettings = apps.get_model("main", "UserSettings")
    UserSettings.objects.filter(model="", reasoning_effort="").update(
        reasoning_effort="high"
    )


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0067_workflowsteeringmessage"),
    ]

    operations = [
        migrations.RunPython(
            backfill_blank_reasoning_effort, migrations.RunPython.noop
        ),
        migrations.AlterField(
            model_name="usersettings",
            name="reasoning_effort",
            field=models.CharField(blank=True, default="high", max_length=32),
        ),
    ]
