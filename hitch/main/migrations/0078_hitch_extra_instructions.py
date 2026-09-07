from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("main", "0077_recentprompt")]

    operations = [
        migrations.AddField(
            model_name="usersettings",
            name="hitch_extra_instructions",
            field=models.JSONField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name="codexinstance",
            name="hitch_extra_instructions",
            field=models.JSONField(blank=True, default=None, null=True),
        ),
    ]
