from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0042_autonomous_goals"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="systemd_scope_unit",
            field=models.CharField(blank=True, default="", max_length=128),
        ),
    ]
