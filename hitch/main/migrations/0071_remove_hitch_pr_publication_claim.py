from django.db import migrations

_PUBLICATION_CLAIM_KEY = "pr_publication_instance"


def remove_hitch_pr_publication_claims(apps, schema_editor):
    SystemWorkflow = apps.get_model("main", "SystemWorkflow")
    database = schema_editor.connection.alias

    workflows = (
        SystemWorkflow.objects.using(database)
        .filter(kind="pr_qa")
        .only("pk", "state")
    )
    for workflow in workflows.iterator():
        state = workflow.state
        if not isinstance(state, dict) or _PUBLICATION_CLAIM_KEY not in state:
            continue
        workflow.state = {
            key: value
            for key, value in state.items()
            if key != _PUBLICATION_CLAIM_KEY
        }
        workflow.save(update_fields=["state"], using=database)


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0070_remove_local_branch_merge"),
    ]

    operations = [
        migrations.RunPython(
            remove_hitch_pr_publication_claims,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
