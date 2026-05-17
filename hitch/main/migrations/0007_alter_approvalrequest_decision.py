from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0006_usersettings_use_worktrees"),
    ]

    operations = [
        migrations.AlterField(
            model_name="approvalrequest",
            name="decision",
            field=models.CharField(
                blank=True,
                choices=[
                    ("", "pending"),
                    ("accept", "accept"),
                    ("decline", "decline"),
                    ("cancel", "cancel"),
                ],
                default="",
                max_length=32,
            ),
        ),
    ]
