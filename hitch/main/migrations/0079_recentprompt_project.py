import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("main", "0078_hitch_extra_instructions")]

    operations = [
        migrations.AddField(
            model_name="recentprompt",
            name="project",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="main.project",
            ),
        ),
    ]
