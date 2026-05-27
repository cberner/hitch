from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0044_merge_0043_approvalrequest_decision_payload_0043_codexinstance_systemd_scope_unit"),
    ]

    operations = [
        migrations.AddField(
            model_name="project",
            name="extra_system_prompt",
            field=models.TextField(blank=True, default=""),
        ),
    ]
