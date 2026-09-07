from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("main", "0076_sessionmetadata_archive_local_only")]

    operations = [
        migrations.CreateModel(
            name="RecentPrompt",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("prompt", models.TextField()),
            ],
        ),
    ]
