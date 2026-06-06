from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0060_remove_qa_panel_settings"),
    ]

    operations = [
        migrations.AddField(
            model_name="codexinstance",
            name="approval_mode_live_editable",
            field=models.BooleanField(default=False),
        ),
    ]
