from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("main", "0081_remove_unused_worker_fields")]

    operations = [
        migrations.AddField(
            model_name="sessionmetadata",
            name="model",
            field=models.CharField(blank=True, default="", max_length=256),
        ),
        migrations.AddField(
            model_name="sessionmetadata",
            name="reasoning_effort",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
    ]
