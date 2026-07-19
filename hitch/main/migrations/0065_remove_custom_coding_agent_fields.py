from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0064_codexinstance_codex_error_info"),
    ]

    operations = [
        # Detached workers running the previous release still read and write
        # this column. Drop it from ORM state now and from the database later.
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.AlterField(
                    model_name="codexinstance",
                    name="base_instructions",
                    field=models.TextField(blank=True, default="", null=True),
                ),
            ],
            state_operations=[
                migrations.RemoveField(
                    model_name="codexinstance",
                    name="base_instructions",
                ),
            ],
        ),
        migrations.RunSQL(
            sql=migrations.RunSQL.noop,
            reverse_sql=(
                "UPDATE main_codexinstance SET base_instructions = '' "
                "WHERE base_instructions IS NULL"
            ),
        ),
        migrations.RemoveField(
            model_name="usersettings",
            name="coding_agent",
        ),
    ]
