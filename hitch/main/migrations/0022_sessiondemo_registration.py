from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0021_standingorder_ambition"),
    ]

    operations = [
        migrations.AlterField(
            model_name="sessiondemo",
            name="status",
            field=models.CharField(
                choices=[
                    ("requested", "requested"),
                    ("preparing", "preparing"),
                    ("active", "active"),
                    ("stopped", "stopped"),
                    ("failed", "failed"),
                ],
                default="active",
                max_length=32,
            ),
        ),
        migrations.AddField(
            model_name="sessiondemo",
            name="generation",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="sessiondemo",
            name="registration_token",
            field=models.CharField(blank=True, default="", max_length=128),
        ),
        migrations.AddField(
            model_name="sessiondemo",
            name="logs",
            field=models.TextField(blank=True, default=""),
        ),
    ]
