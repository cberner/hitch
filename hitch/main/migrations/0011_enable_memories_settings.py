from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0010_system_agents"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="enable_memories",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="enable_memories",
            field=models.BooleanField(default=False),
        ),
    ]
