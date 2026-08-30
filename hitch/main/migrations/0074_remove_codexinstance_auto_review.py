from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("main", "0073_remove_autonomousgoalmemory")]

    operations = [
        # Retain the columns for one compatibility boundary so workers started
        # before a deploy can finish against the schema they loaded.
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.AlterField(
                    model_name="codexinstance",
                    name="auto_pr_enabled",
                    field=models.BooleanField(default=False, null=True),
                ),
                migrations.AlterField(
                    model_name="codexinstance",
                    name="auto_qa_enabled",
                    field=models.BooleanField(default=False, null=True),
                ),
            ],
            state_operations=[
                migrations.RemoveField(
                    model_name="codexinstance",
                    name="auto_pr_enabled",
                ),
                migrations.RemoveField(
                    model_name="codexinstance",
                    name="auto_qa_enabled",
                ),
                migrations.RemoveField(
                    model_name="codexinstance",
                    name="auto_pr_triggered_at",
                ),
                migrations.RemoveField(
                    model_name="codexinstance",
                    name="auto_qa_triggered_at",
                ),
            ],
        ),
    ]
