from django.db import migrations


def remove_obsolete_pr_refresh_state(apps, schema_editor):
    session_pull_request = apps.get_model("main", "SessionPullRequest")
    database = schema_editor.connection.alias
    records = session_pull_request.objects.using(database).all().iterator()
    obsolete_keys = {"hitch_pr_handoff", "pr_stage_refresh"}
    for record in records:
        state = record.state
        if not isinstance(state, dict) or obsolete_keys.isdisjoint(state):
            continue
        cleaned = {
            key: value
            for key, value in state.items()
            if key not in obsolete_keys
        }
        session_pull_request.objects.using(database).filter(pk=record.pk).update(
            state=cleaned
        )


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0074_remove_codexinstance_auto_review"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="sessionmetadata",
            name="derived_stage_pr_refresh_attempted_at",
        ),
        migrations.RunPython(
            remove_obsolete_pr_refresh_state,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
