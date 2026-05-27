from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0045_project_extra_system_prompt"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="workflow_routing_started_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
