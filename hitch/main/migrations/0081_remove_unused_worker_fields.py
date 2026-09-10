from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0080_remove_autonomous_goals"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="codexinstance",
            name="output_schema",
        ),
        migrations.RemoveField(
            model_name="codexinstance",
            name="workflow_routing_started_at",
        ),
    ]
