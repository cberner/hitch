from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0023_coding_agent_base_instructions"),
    ]

    operations = [
        migrations.AddField(
            model_name="proposedsession",
            name="inbox_kind",
            field=models.CharField(
                choices=[("proposal", "Proposal"), ("notice", "Notice")],
                default="proposal",
                max_length=16,
            ),
        ),
        migrations.AlterField(
            model_name="proposedsession",
            name="outcome_status",
            field=models.CharField(
                blank=True,
                choices=[
                    ("", "Not set"),
                    ("accepted", "Accepted"),
                    ("rejected", "Rejected"),
                    ("dismissed", "Dismissed"),
                ],
                default="",
                max_length=32,
            ),
        ),
    ]
