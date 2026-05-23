from django.db import migrations, models
import django.db.models.deletion


def backfill_proposed_session_projects(apps, _schema_editor):
    ProposedSession = apps.get_model("main", "ProposedSession")
    for proposal in ProposedSession.objects.select_related("standing_order"):
        if proposal.project_id is None and proposal.standing_order_id is not None:
            proposal.project_id = proposal.standing_order.project_id
            proposal.save(update_fields=["project"])


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0025_remove_okrs"),
    ]

    operations = [
        migrations.AddField(
            model_name="proposedsession",
            name="project",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="proposed_sessions",
                to="main.project",
            ),
        ),
        migrations.AddField(
            model_name="proposedsession",
            name="prompt",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="proposedsession",
            name="source_session",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="created_proposals",
                to="main.sessionmetadata",
            ),
        ),
        migrations.AlterField(
            model_name="proposedsession",
            name="standing_order",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="proposed_sessions",
                to="main.standingorder",
            ),
        ),
        migrations.RunPython(backfill_proposed_session_projects, migrations.RunPython.noop),
        migrations.AddIndex(
            model_name="proposedsession",
            index=models.Index(
                fields=["project", "created_at"], name="main_propos_project_6fd310_idx"
            ),
        ),
    ]
