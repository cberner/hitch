from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0020_standing_orders"),
    ]

    operations = [
        migrations.AddField(
            model_name="standingorder",
            name="ambition",
            field=models.CharField(
                choices=[
                    ("incremental", "Incremental"),
                    ("medium", "Medium"),
                    ("high", "High"),
                    ("yolo", "YOLO"),
                ],
                default="incremental",
                max_length=32,
            ),
        ),
    ]
