from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0061_codexinstance_approval_mode_live_editable"),
    ]

    operations = [
        migrations.AddField(
            model_name="project",
            name="auto_pull_enabled",
            field=models.BooleanField(default=False),
        ),
    ]
