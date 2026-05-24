from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0031_usersettings_spec_critic"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="web_search_mode",
            field=models.CharField(blank=True, default="", max_length=16),
        ),
        migrations.AddField(
            model_name="standingorder",
            name="web_search_mode",
            field=models.CharField(
                blank=True,
                choices=[
                    ("", "Codex default"),
                    ("disabled", "Disabled"),
                    ("cached", "Cached"),
                    ("live", "Live"),
                ],
                default="",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="usersettings",
            name="web_search_mode",
            field=models.CharField(blank=True, default="", max_length=16),
        ),
    ]
