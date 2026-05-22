from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0022_sessiondemo_registration"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="base_instructions",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="coding_agent",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
    ]
