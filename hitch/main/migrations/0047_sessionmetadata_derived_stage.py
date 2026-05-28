from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0046_codexinstance_workflow_routing_started_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata",
            name="derived_stage",
            field=models.CharField(blank=True, db_index=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="sessionmetadata",
            name="derived_stage_source_mtime_ns",
            field=models.PositiveBigIntegerField(default=0),
        ),
    ]
