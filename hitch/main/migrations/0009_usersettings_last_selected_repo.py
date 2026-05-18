from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0008_userinputrequest"),
    ]

    operations = [
        migrations.AddField(
            model_name="usersettings",
            name="last_selected_repo",
            field=models.CharField(blank=True, default="", max_length=4096),
        ),
    ]
