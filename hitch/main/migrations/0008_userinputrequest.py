import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0007_alter_approvalrequest_decision"),
    ]

    operations = [
        migrations.CreateModel(
            name="UserInputRequest",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("method", models.CharField(max_length=128)),
                ("params", models.JSONField(blank=True, default=dict)),
                ("response", models.JSONField(blank=True, default=None, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("responded_at", models.DateTimeField(blank=True, null=True)),
                (
                    "instance",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="input_requests",
                        to="main.codexinstance",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["instance", "-created_at"],
                        name="main_userin_instanc_626359_idx",
                    )
                ],
            },
        ),
    ]
