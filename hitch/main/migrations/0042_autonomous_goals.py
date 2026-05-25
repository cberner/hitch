import django.db.models.deletion
from django.db import migrations, models


LEGACY_ACCEPTED_BY = "standing_order_autonomy"
ACCEPTED_BY = "autonomous_goal_autonomy"
LEGACY_AGENT_KIND = "standing_order_run"
AGENT_KIND = "autonomous_goal_run"
LEGACY_JUDGE_AGENT_KIND = "standing_order_judge"
JUDGE_AGENT_KIND = "autonomous_goal_judge"
LEGACY_AGENT_DISPLAY_AUTHOR = "Standing order agent"
AGENT_DISPLAY_AUTHOR = "Autonomous goal agent"
LEGACY_JUDGE_DISPLAY_AUTHOR = "Standing order judge"
JUDGE_DISPLAY_AUTHOR = "Autonomous goal judge"
LEGACY_AGENT_PROMPT_TITLE = "You are Hitch's standing order agent."
AGENT_PROMPT_TITLE = "You are Hitch's autonomous goal agent."
LEGACY_JUDGE_PROMPT_TITLE = "You are Hitch's standing order confidence judge."
JUDGE_PROMPT_TITLE = "You are Hitch's autonomous goal confidence judge."
LEGACY_WORKFLOW_STEPS = {
    "standing_order_candidate_running": "autonomous_goal_candidate_running",
    "standing_order_judge_running": "autonomous_goal_judge_running",
    "standing_order_proposed": "autonomous_goal_proposed",
    "standing_order_draft_started": "autonomous_goal_draft_started",
    "standing_order_skipped": "autonomous_goal_skipped",
}


def _renamed_state(state: object) -> tuple[dict[str, object], bool]:
    if not isinstance(state, dict):
        return {}, False
    renamed = dict(state)
    changed = False
    for old_key, new_key in (
        ("standing_order_id", "autonomous_goal_id"),
        ("standing_order_updated_at", "autonomous_goal_updated_at"),
        ("standing_order_autonomy", "autonomous_goal_autonomy"),
    ):
        if old_key in renamed:
            renamed[new_key] = renamed.pop(old_key)
            changed = True
    if renamed.get("accepted_by") == LEGACY_ACCEPTED_BY:
        renamed["accepted_by"] = ACCEPTED_BY
        changed = True
    return renamed, changed


def _renamed_thread_id(value: object) -> tuple[str, bool]:
    if not isinstance(value, str) or not value.startswith("standing-order:"):
        return "", False
    return value.replace("standing-order:", "autonomous-goal:", 1), True


def rename_existing_autonomous_goal_data(apps, schema_editor) -> None:
    ProposedSession = apps.get_model("main", "ProposedSession")
    SystemWorkflow = apps.get_model("main", "SystemWorkflow")
    SystemAgentRun = apps.get_model("main", "SystemAgentRun")
    CodexInstance = apps.get_model("main", "CodexInstance")
    SessionMetadata = apps.get_model("main", "SessionMetadata")

    for proposal in ProposedSession.objects.exclude(outcome_metadata={}):
        metadata, changed = _renamed_state(proposal.outcome_metadata)
        if changed:
            ProposedSession.objects.filter(pk=proposal.pk).update(
                outcome_metadata=metadata
            )

    for workflow in SystemWorkflow.objects.all():
        updates = {}
        if workflow.kind == LEGACY_AGENT_KIND:
            updates["kind"] = AGENT_KIND
        if workflow.step in LEGACY_WORKFLOW_STEPS:
            updates["step"] = LEGACY_WORKFLOW_STEPS[workflow.step]
        main_thread_id, thread_changed = _renamed_thread_id(workflow.main_thread_id)
        if thread_changed:
            updates["main_thread_id"] = main_thread_id
        state, state_changed = _renamed_state(workflow.state)
        if state_changed:
            updates["state"] = state
        if updates:
            SystemWorkflow.objects.filter(pk=workflow.pk).update(**updates)

    SystemAgentRun.objects.filter(agent_kind=LEGACY_AGENT_KIND).update(
        agent_kind=AGENT_KIND
    )
    SystemAgentRun.objects.filter(agent_kind=LEGACY_JUDGE_AGENT_KIND).update(
        agent_kind=JUDGE_AGENT_KIND
    )
    CodexInstance.objects.filter(agent_kind=LEGACY_AGENT_KIND).update(
        agent_kind=AGENT_KIND
    )
    CodexInstance.objects.filter(agent_kind=LEGACY_JUDGE_AGENT_KIND).update(
        agent_kind=JUDGE_AGENT_KIND
    )
    CodexInstance.objects.filter(display_author=LEGACY_AGENT_DISPLAY_AUTHOR).update(
        display_author=AGENT_DISPLAY_AUTHOR
    )
    CodexInstance.objects.filter(display_author=LEGACY_JUDGE_DISPLAY_AUTHOR).update(
        display_author=JUDGE_DISPLAY_AUTHOR
    )
    SessionMetadata.objects.filter(codex_thread_source="subagent").update(
        is_hidden_system_session=True
    )
    _backfill_hidden_prompt_sessions(SessionMetadata, LEGACY_AGENT_PROMPT_TITLE)
    _backfill_hidden_prompt_sessions(SessionMetadata, AGENT_PROMPT_TITLE)
    _backfill_hidden_judge_sessions(SessionMetadata, LEGACY_JUDGE_PROMPT_TITLE)
    _backfill_hidden_judge_sessions(SessionMetadata, JUDGE_PROMPT_TITLE)


def _backfill_hidden_prompt_sessions(SessionMetadata, title: str) -> None:
    title_marker = "Autonomous goal title:" if "autonomous goal" in title else "Standing order title:"
    goal_marker = "Autonomous goal objective:" if "autonomous goal" in title else "Standing order goal:"
    SessionMetadata.objects.filter(
        codex_name=title,
        codex_preview__startswith=f"{title}\n\n",
        codex_preview__contains=title_marker,
    ).filter(codex_preview__contains=goal_marker).filter(
        codex_preview__contains="Return only JSON matching this shape:"
    ).update(
        is_hidden_system_session=True
    )


def _backfill_hidden_judge_sessions(SessionMetadata, title: str) -> None:
    title_marker = "Autonomous goal title:" if "autonomous goal" in title else "Standing order title:"
    SessionMetadata.objects.filter(
        codex_name=title,
        codex_preview__startswith=f"{title}\n\n",
        codex_preview__contains=title_marker,
    ).filter(codex_preview__contains="Candidate session JSON:").filter(
        codex_preview__contains="Return only JSON matching this shape:"
    ).update(
        is_hidden_system_session=True
    )


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0041_sessionmetadata_hidden_system"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="StandingOrder",
            new_name="AutonomousGoal",
        ),
        migrations.RenameModel(
            old_name="StandingOrderMemory",
            new_name="AutonomousGoalMemory",
        ),
        migrations.RemoveIndex(
            model_name="proposedsession",
            name="main_propos_standin_66a16b_idx",
        ),
        migrations.RemoveIndex(
            model_name="autonomousgoalmemory",
            name="main_standi_standin_3b898b_idx",
        ),
        migrations.RenameField(
            model_name="proposedsession",
            old_name="standing_order",
            new_name="autonomous_goal",
        ),
        migrations.RenameField(
            model_name="autonomousgoalmemory",
            old_name="standing_order",
            new_name="autonomous_goal",
        ),
        migrations.AlterField(
            model_name="autonomousgoal",
            name="project",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="autonomous_goals",
                to="main.project",
            ),
        ),
        migrations.AlterField(
            model_name="proposedsession",
            name="accepted_session",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="accepted_autonomous_goal_proposals",
                to="main.sessionmetadata",
            ),
        ),
        migrations.AlterField(
            model_name="proposedsession",
            name="autonomous_goal",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="proposed_sessions",
                to="main.autonomousgoal",
            ),
        ),
        migrations.AlterField(
            model_name="proposedsession",
            name="candidate_session",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="autonomous_goal_candidate_proposals",
                to="main.sessionmetadata",
            ),
        ),
        migrations.AlterField(
            model_name="proposedsession",
            name="judge_session",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="autonomous_goal_judge_proposals",
                to="main.sessionmetadata",
            ),
        ),
        migrations.AlterField(
            model_name="autonomousgoalmemory",
            name="autonomous_goal",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="memories",
                to="main.autonomousgoal",
            ),
        ),
        migrations.AlterField(
            model_name="autonomousgoalmemory",
            name="candidate_session",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="autonomous_goal_memories",
                to="main.sessionmetadata",
            ),
        ),
        migrations.AlterField(
            model_name="autonomousgoalmemory",
            name="source_workflow",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="autonomous_goal_memories",
                to="main.systemworkflow",
            ),
        ),
        migrations.RenameIndex(
            model_name="autonomousgoal",
            new_name="main_autono_project_9a398c_idx",
            old_name="main_standi_project_803130_idx",
        ),
        migrations.AddIndex(
            model_name="proposedsession",
            index=models.Index(
                fields=["autonomous_goal", "created_at"],
                name="main_propos_autonom_c5613b_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="autonomousgoalmemory",
            index=models.Index(
                fields=["autonomous_goal", "-created_at"],
                name="main_autono_autonom_b8705e_idx",
            ),
        ),
        migrations.RemoveConstraint(
            model_name="autonomousgoalmemory",
            name="uniq_standing_order_memory_workflow",
        ),
        migrations.AddConstraint(
            model_name="autonomousgoalmemory",
            constraint=models.UniqueConstraint(
                condition=models.Q(("source_workflow__isnull", False)),
                fields=("source_workflow",),
                name="uniq_autonomous_goal_memory_workflow",
            ),
        ),
        migrations.RunPython(
            rename_existing_autonomous_goal_data,
            migrations.RunPython.noop,
        ),
    ]
