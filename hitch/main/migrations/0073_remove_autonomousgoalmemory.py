from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [("main", "0072_session_pull_request")]

    operations = [migrations.DeleteModel(name="AutonomousGoalMemory")]
