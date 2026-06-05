from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0056_globalsettings"),
    ]

    operations = [
        migrations.AddField(
            model_name="autonomousgoal",
            name="stacked_diff_depth",
            field=models.PositiveIntegerField(default=1),
        ),
    ]
