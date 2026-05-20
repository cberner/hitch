from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0013_projects"),
    ]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata",
            name="project_cleared",
            field=models.BooleanField(default=False),
        ),
    ]
