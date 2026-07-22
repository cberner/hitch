from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0065_remove_custom_coding_agent_fields"),
    ]

    operations = [
        migrations.AlterField(
            model_name="project",
            name="auto_pull_enabled",
            field=models.BooleanField(default=True),
        ),
    ]
