from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0042_autonomous_goals"),
    ]

    operations = [
        migrations.AddField(
            model_name="approvalrequest",
            name="decision_payload",
            field=models.JSONField(blank=True, default=None, null=True),
        ),
    ]
