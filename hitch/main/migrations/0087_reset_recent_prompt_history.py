from django.db import migrations


def reset_recent_prompt_history(apps, schema_editor):
    # Legacy entries have no session or turn identity to distinguish initial prompts.
    apps.get_model("main", "RecentPrompt").objects.using(schema_editor.connection.alias).all().delete()


class Migration(migrations.Migration):
    dependencies = [("main", "0086_remove_retired_automation")]

    operations = [
        migrations.RunPython(reset_recent_prompt_history, migrations.RunPython.noop),
    ]
