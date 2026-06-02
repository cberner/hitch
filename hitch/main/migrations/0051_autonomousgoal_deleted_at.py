from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0050_session_project_visibility_settings"),
    ]

    operations = [
        migrations.AddField(
            model_name="autonomousgoal",
            name="deleted_at",
            field=models.DateTimeField(blank=True, db_index=True, null=True),
        ),
    ]
