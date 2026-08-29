from django.db import migrations, models

_PROPOSAL_METADATA_KEYS = frozenset(
    {
        "auto_merge_branch",
        "auto_merge_commit_sha",
        "auto_merge_error",
        "auto_merge_status",
        "auto_merge_to_local_branch",
    }
)
_WORKFLOW_STATE_KEYS = frozenset(
    {
        "auto_merge_branch",
        "auto_merge_result",
        "auto_merge_reviewed_diff",
        "auto_merge_reviewed_source_tree",
        "auto_merge_reviewed_target_sha",
        "auto_merge_session_base_sha",
        "auto_merge_to_local_branch",
    }
)


def remove_local_branch_merge_data(apps, schema_editor):
    ProposedSession = apps.get_model("main", "ProposedSession")
    SystemWorkflow = apps.get_model("main", "SystemWorkflow")
    database = schema_editor.connection.alias

    for proposal in ProposedSession.objects.using(database).only("pk", "outcome_metadata").iterator():
        metadata = proposal.outcome_metadata
        if not isinstance(metadata, dict) or not _PROPOSAL_METADATA_KEYS.intersection(metadata):
            continue
        proposal.outcome_metadata = {
            key: value for key, value in metadata.items() if key not in _PROPOSAL_METADATA_KEYS
        }
        proposal.save(update_fields=["outcome_metadata"], using=database)

    for workflow in SystemWorkflow.objects.using(database).only("pk", "state").iterator():
        state = workflow.state
        if not isinstance(state, dict) or not _WORKFLOW_STATE_KEYS.intersection(state):
            continue
        workflow.state = {key: value for key, value in state.items() if key not in _WORKFLOW_STATE_KEYS}
        workflow.save(update_fields=["state"], using=database)

    SystemWorkflow.objects.using(database).filter(
        kind="pr_qa",
        step="local_branch_merged",
    ).update(step="review_completed")


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0069_remove_demo_and_spec_critic"),
    ]

    operations = [
        migrations.RunPython(
            remove_local_branch_merge_data,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.RemoveField(
            model_name="autonomousgoal",
            name="auto_merge_branch",
        ),
        migrations.RemoveField(
            model_name="autonomousgoal",
            name="auto_merge_to_local_branch",
        ),
        # Migration 0065 deliberately kept this database column for workers
        # from the previous release. Include it in SQLite table remakes so
        # removing the merge fields does not drop unrelated compatibility data.
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.AddField(
                    model_name="codexinstance",
                    name="base_instructions",
                    field=models.TextField(blank=True, default="", null=True),
                ),
            ],
        ),
        migrations.RemoveField(
            model_name="codexinstance",
            name="auto_merge_branch",
        ),
        migrations.RemoveField(
            model_name="codexinstance",
            name="auto_merge_to_local_branch",
        ),
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.RemoveField(
                    model_name="codexinstance",
                    name="base_instructions",
                ),
            ],
        ),
        migrations.RemoveField(
            model_name="sessionmetadata",
            name="auto_merge_branch",
        ),
        migrations.RemoveField(
            model_name="sessionmetadata",
            name="auto_merge_to_local_branch",
        ),
    ]
