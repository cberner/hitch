from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0024_proposedsession_inbox_kind_and_dismissed"),
    ]

    operations = [
        migrations.DeleteModel(name="ProposedTask"),
        migrations.DeleteModel(name="KeyResult"),
        migrations.DeleteModel(name="Objective"),
    ]
