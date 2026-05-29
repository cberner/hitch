from django.db import migrations


def reset_stage_cache(apps, schema_editor):
    # The pre-fix write path could persist a transient, active-owner-forced
    # stage (e.g. Implementation while a worker ran) into the mtime-keyed
    # cache. Such a row is served verbatim by the read guard once the owner
    # exits, and -- because the read path short-circuits before the write path
    # -- it is never re-derived while the rollout's mtime is unchanged. Reset
    # the cache once so every session re-derives its stage from the rollout on
    # next render; the fixed write path no longer stores transient stages.
    SessionMetadata = apps.get_model("main", "SessionMetadata")
    SessionMetadata.objects.exclude(
        derived_stage="", derived_stage_source_mtime_ns=0
    ).update(derived_stage="", derived_stage_source_mtime_ns=0)


class Migration(migrations.Migration):
    dependencies = [
        ("main", "0047_sessionmetadata_derived_stage"),
    ]

    operations = [
        migrations.RunPython(reset_stage_cache, migrations.RunPython.noop),
    ]
