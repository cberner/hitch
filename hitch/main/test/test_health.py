"""Tests for the Hitch health dashboard (leak + backlog signals)."""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, override
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from hitch.main.models import (
    ApprovalRequest,
    CodexInstance,
    ProposedSession,
    SystemWorkflow,
    UserInputRequest,
)
from hitch.main.runtime import disk_cleanup, health, host_probes, reconciliation, server_lifecycle
from hitch.main.runtime.disk_cleanup import HitchDiskUsage
from hitch.main.runtime.host_probes import (
    LeakedScope,
    ScopeProcess,
    WorkerScopeProbe,
)


def _make_user(username: str = "dev@example.com", password: str = "StrongPass123!") -> Any:
    return get_user_model().objects.create_user(username=username, password=password)


def _make_instance(**kwargs: Any) -> CodexInstance:
    defaults: dict[str, Any] = {
        "pid": 1,
        "thread_id": "t",
        "cwd": "/r",
        "events_path": "/dev/null",
    }
    defaults.update(kwargs)
    return CodexInstance.objects.create(**defaults)


class CollectHealthReportTests(TestCase):
    @override
    def setUp(self) -> None:
        # Pin every collector that reads /proc, the real disk, or host load so
        # report severity reflects only the DB state under test.
        for module, target, value in (
            (reconciliation, "count_running_codex_app_servers", 0),
            (host_probes, "cpu_count", 4),
            (host_probes, "load_average", (0.1, 0.1, 0.1)),
            (host_probes, "runserver_fd_count", 50),
            (host_probes, "runserver_close_wait_count", 0),
            (host_probes, "probe_worker_scopes", WorkerScopeProbe(active_count=0, leaked=[])),
        ):
            patcher = patch.object(module, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        disk_patcher = patch.object(
            disk_cleanup,
            "hitch_home_disk_usage",
            return_value=HitchDiskUsage(used_bytes=1024, limit_bytes=1_000_000, disk_total_bytes=10_000_000),
        )
        disk_patcher.start()
        self.addCleanup(disk_patcher.stop)

    def test_empty_db_is_ok(self) -> None:
        report = health.collect_health_report()

        self.assertEqual(report.overall_severity, health.SEVERITY_OK)
        self.assertEqual(report.overall_label, "OK")
        titles = [section.title for section in report.sections]
        self.assertEqual(
            titles,
            [
                "Worker units (leaks)",
                "Background schedulers",
                "Host / CPU",
                "Leaks",
                "Blocked workflow buckets",
                "Backlogs",
                "Recent Codex failures (24h)",
            ],
        )

    def test_blocked_workflow_raises_overall_to_warn(self) -> None:
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="m",
            cwd="/r",
            status=SystemWorkflow.STATUS_BLOCKED,
        )

        report = health.collect_health_report()

        self.assertEqual(report.overall_severity, health.SEVERITY_WARN)
        metric = _find(report, "blocked_workflows")
        self.assertEqual(metric.value, "1")
        self.assertEqual(metric.severity, health.SEVERITY_WARN)

    def test_stale_blocked_pr_qa_counted(self) -> None:
        workflow = SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id="m",
            cwd="/r",
            status=SystemWorkflow.STATUS_BLOCKED,
        )
        old = timezone.now() - timedelta(days=10)
        # updated_at is auto_now; bypass it with a direct update.
        SystemWorkflow.objects.filter(pk=workflow.pk).update(updated_at=old)

        report = health.collect_health_report()

        self.assertEqual(_find(report, "stale_blocked_workflows").value, "1")

    def test_stuck_turn_flagged(self) -> None:
        instance = _make_instance(status=CodexInstance.STATUS_RUNNING)
        old = timezone.now() - timedelta(hours=8)
        CodexInstance.objects.filter(pk=instance.pk).update(started_at=old)

        report = health.collect_health_report()

        stuck = _find(report, "stuck_turns")
        self.assertEqual(stuck.value, "1")
        self.assertEqual(stuck.severity, health.SEVERITY_WARN)

    def test_pending_handoffs_flagged(self) -> None:
        instance = _make_instance(status=CodexInstance.STATUS_RUNNING)
        ApprovalRequest.objects.create(instance=instance, method="item/commandExecution/requestApproval")
        UserInputRequest.objects.create(instance=instance, method="request_user_input")
        ProposedSession.objects.create(title="proposal")

        report = health.collect_health_report()

        self.assertEqual(_find(report, "pending_approvals").value, "1")
        self.assertEqual(_find(report, "pending_inputs").value, "1")
        self.assertEqual(_find(report, "pending_proposals").value, "1")
        self.assertEqual(report.overall_severity, health.SEVERITY_WARN)

    def test_app_server_surplus_is_danger(self) -> None:
        with patch.object(reconciliation, "count_running_codex_app_servers", return_value=12):
            report = health.collect_health_report()

        metric = _find(report, "app_servers")
        self.assertEqual(metric.value, "12")
        self.assertEqual(metric.severity, health.SEVERITY_DANGER)
        self.assertEqual(report.overall_severity, health.SEVERITY_DANGER)

    def test_disk_over_limit_is_danger(self) -> None:
        with patch.object(
            disk_cleanup,
            "hitch_home_disk_usage",
            return_value=HitchDiskUsage(used_bytes=2_000_000, limit_bytes=1_000_000, disk_total_bytes=10_000_000),
        ):
            report = health.collect_health_report()

        self.assertEqual(_find(report, "hitch_disk").severity, health.SEVERITY_DANGER)

    def test_headline_metric_is_none_when_ok(self) -> None:
        report = health.collect_health_report()

        self.assertEqual(report.overall_severity, health.SEVERITY_OK)
        self.assertIsNone(report.headline_metric)

    def test_headline_metric_surfaces_worst_severity_row(self) -> None:
        with patch.object(
            disk_cleanup,
            "hitch_home_disk_usage",
            return_value=HitchDiskUsage(used_bytes=2_000_000, limit_bytes=1_000_000, disk_total_bytes=10_000_000),
        ):
            report = health.collect_health_report()

        headline = report.headline_metric
        self.assertIsNotNone(headline)
        assert headline is not None
        self.assertEqual(headline.key, "hitch_disk")
        self.assertEqual(headline.severity, health.SEVERITY_DANGER)
        self.assertIn("Headline:", report.copy_text())
        self.assertIn(headline.value, report.copy_text())

    def test_metric_failure_degrades_gracefully(self) -> None:
        with patch.object(reconciliation, "count_running_codex_app_servers",
            side_effect=RuntimeError("boom"),
        ):
            report = health.collect_health_report()

        metric = _find(report, "app_servers")
        self.assertEqual(metric.value, "unavailable")
        self.assertEqual(metric.severity, health.SEVERITY_UNKNOWN)

    def test_copy_text_contains_key_lines(self) -> None:
        text = health.collect_health_report().copy_text()

        self.assertIn("Hitch health report", text)
        self.assertIn("Overall:", text)
        self.assertIn("[Leaks]", text)
        self.assertIn("[Backlogs]", text)
        self.assertIn("[Worker units (leaks)]", text)

    def test_leaked_scope_makes_report_danger(self) -> None:
        probe = WorkerScopeProbe(
            active_count=0,
            leaked=[
                LeakedScope(
                    instance_id=7,
                    scope_unit="hitch-codex-worker-7.service",
                    db_status="failed",
                    processes=[
                        ScopeProcess(
                            pid=4242,
                            comm="encode_benchmark",
                            cmdline="encode_benchmark-add9 --bench",
                            rss_bytes=2 * 1024**3,
                            foreign=True,
                        )
                    ],
                )
            ],
        )
        with patch.object(host_probes, "probe_worker_scopes", return_value=probe):
            report = health.collect_health_report()

        self.assertEqual(report.overall_severity, health.SEVERITY_DANGER)
        summary = _find(report, "leaked_scopes")
        self.assertEqual(summary.value, "1")
        self.assertEqual(summary.severity, health.SEVERITY_DANGER)
        scope_metric = _find(report, "leaked_scope_7")
        self.assertIn("encode_benchmark", scope_metric.detail)
        self.assertIn("status=failed", scope_metric.detail)

    def test_active_scopes_over_core_count_is_danger(self) -> None:
        probe = WorkerScopeProbe(active_count=6, leaked=[])
        with patch.object(host_probes, "probe_worker_scopes", return_value=probe):
            report = health.collect_health_report()

        self.assertEqual(_find(report, "active_scopes").severity, health.SEVERITY_DANGER)

    def test_load_average_over_cores_is_danger(self) -> None:
        with patch.object(host_probes, "load_average", return_value=(9.0, 7.0, 5.0)):
            report = health.collect_health_report()

        self.assertEqual(_find(report, "load_avg").severity, health.SEVERITY_DANGER)

    def test_blocked_buckets_classify_and_respect_benign(self) -> None:
        self._blocked("Command 'gh pr create --fill' failed: no commits")
        self._blocked("worker process exited before reporting completion")
        self._blocked("QA workflow stopped by user")

        report = health.collect_health_report()

        gh = _find(report, "blocked_bucket_gh_pr_create_failures")
        self.assertEqual(gh.value, "1")
        self.assertEqual(gh.severity, health.SEVERITY_WARN)
        self.assertIn("+1 in 24h", gh.detail)
        # Benign buckets never escalate even when fresh.
        stopped = _find(report, "blocked_bucket_stopped_by_user")
        self.assertEqual(stopped.value, "1")
        self.assertEqual(stopped.severity, health.SEVERITY_OK)

    def test_recent_worker_exit_failure_flagged(self) -> None:
        instance = _make_instance(
            status=CodexInstance.STATUS_FAILED,
            error="worker process exited before reporting completion",
        )
        CodexInstance.objects.filter(pk=instance.pk).update(ended_at=timezone.now())

        report = health.collect_health_report()

        worker = _find(report, "worker_exited_24h")
        self.assertEqual(worker.value, "1")
        self.assertEqual(worker.severity, health.SEVERITY_WARN)
        self.assertEqual(_find(report, "failed_24h").value, "1")

    _blocked_seq = 0

    def _blocked(self, error: str) -> None:
        type(self)._blocked_seq += 1
        SystemWorkflow.objects.create(
            kind=SystemWorkflow.KIND_PR_QA,
            main_thread_id=f"thread-{type(self)._blocked_seq}",
            cwd="/r",
            status=SystemWorkflow.STATUS_BLOCKED,
            state={"error": error},
        )


class HealthReportCacheTests(TestCase):
    @override
    def setUp(self) -> None:
        health._report_cache = None
        self.addCleanup(setattr, health, "_report_cache", None)

    @override_settings(TESTING=False)
    def test_report_is_cached_within_ttl(self) -> None:
        with (
            patch.object(
                host_probes,
                "probe_worker_scopes",
                return_value=WorkerScopeProbe(active_count=0, leaked=[]),
            ) as probe,
            patch.object(disk_cleanup, "hitch_home_disk_usage", return_value=None),
            patch.object(reconciliation, "count_running_codex_app_servers", return_value=0),
        ):
            first = health.collect_health_report()
            second = health.collect_health_report()

        # Second call inside the TTL reuses the first build instead of re-walking
        # /proc and ~/.hitch.
        self.assertIs(first, second)
        self.assertEqual(probe.call_count, 1)

    @override_settings(TESTING=True)
    def test_testing_bypasses_cache(self) -> None:
        with patch.object(
            host_probes,
            "probe_worker_scopes",
            return_value=WorkerScopeProbe(active_count=0, leaked=[]),
        ) as probe:
            health.collect_health_report()
            health.collect_health_report()

        self.assertEqual(probe.call_count, 2)


class HealthHelperTests(TestCase):
    def test_human_bytes(self) -> None:
        self.assertEqual(health._human_bytes(0), "0 B")
        self.assertEqual(health._human_bytes(512), "512 B")
        self.assertEqual(health._human_bytes(1536), "1.5 KiB")
        self.assertEqual(health._human_bytes(5 * 1024 * 1024), "5.0 MiB")


class SchedulerHeartbeatTests(TestCase):
    def _status(self, **overrides: Any) -> server_lifecycle.SchedulerStatus:
        now = timezone.now()
        defaults: dict[str, Any] = {
            "name": "hitch-test-scheduler",
            "started": True,
            "started_at": now - timedelta(seconds=300),
            "tick_interval_seconds": 60,
            "tick_count": 5,
            "last_tick_at": now - timedelta(seconds=30),
            "last_tick_errored": False,
            "last_error": "",
            "last_error_at": None,
        }
        defaults.update(overrides)
        return server_lifecycle.SchedulerStatus(**defaults)

    def test_run_tick_records_heartbeat_results_and_errors(self) -> None:
        handle = server_lifecycle.SchedulerHandle(
            thread_name="hitch-test-heartbeat", tick_interval_seconds=5
        )
        self.addCleanup(lambda: server_lifecycle._HANDLES.remove(handle))

        self.assertEqual(handle.run_tick(lambda: "cursor"), "cursor")
        status = handle.status()
        self.assertEqual(status.tick_count, 1)
        self.assertIsNotNone(status.last_tick_at)
        self.assertEqual(status.last_error, "")

        def _boom() -> str:
            raise RuntimeError("boom")

        # One bad tick is swallowed (the scheduler thread must survive) but
        # recorded for the health page; the heartbeat still advances.
        self.assertIsNone(handle.run_tick(_boom))
        status = handle.status()
        self.assertEqual(status.tick_count, 2)
        self.assertTrue(status.last_tick_errored)
        self.assertIn("boom", status.last_error)
        self.assertIsNotNone(status.last_error_at)

        # A successful tick marks recovery while keeping the error on record.
        handle.run_tick(lambda: None)
        status = handle.status()
        self.assertFalse(status.last_tick_errored)
        self.assertIn("boom", status.last_error)

    def test_scheduler_metric_severities(self) -> None:
        healthy = health._scheduler_metric(self._status())
        self.assertEqual(healthy.severity, health.SEVERITY_OK)
        self.assertIn("tick 5", healthy.value)

        not_started = health._scheduler_metric(
            self._status(started=False, started_at=None, tick_count=0, last_tick_at=None)
        )
        self.assertEqual(not_started.severity, health.SEVERITY_OK)
        self.assertEqual(not_started.value, "not started")

        stale = health._scheduler_metric(
            self._status(last_tick_at=timezone.now() - timedelta(seconds=600))
        )
        self.assertEqual(stale.severity, health.SEVERITY_DANGER)
        self.assertIn("dead or wedged", stale.detail)

        never_ticked = health._scheduler_metric(
            self._status(
                started_at=timezone.now() - timedelta(seconds=600),
                tick_count=0,
                last_tick_at=None,
            )
        )
        self.assertEqual(never_ticked.severity, health.SEVERITY_DANGER)

        latest_tick_errored = health._scheduler_metric(
            self._status(
                last_tick_errored=True,
                last_error="RuntimeError('boom')",
                last_error_at=timezone.now() - timedelta(seconds=10),
            )
        )
        self.assertEqual(latest_tick_errored.severity, health.SEVERITY_WARN)
        self.assertIn("boom", latest_tick_errored.detail)

        # A recovered scheduler (latest tick succeeded) is OK, with the old
        # error kept visible in the detail only.
        recovered = health._scheduler_metric(
            self._status(
                last_error="RuntimeError('boom')",
                last_error_at=timezone.now() - timedelta(seconds=10),
            )
        )
        self.assertEqual(recovered.severity, health.SEVERITY_OK)
        self.assertIn("Recovered", recovered.detail)


class HealthDashboardViewTests(TestCase):
    def test_requires_authentication(self) -> None:
        response = self.client.get(reverse("health_dashboard"))

        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("login"), response["Location"])
        self.assertIn(reverse("health_dashboard"), response["Location"])

    def test_renders_for_authenticated_user(self) -> None:
        self.client.force_login(_make_user())

        response = self.client.get(reverse("health_dashboard"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Send this to the assistant")
        self.assertContains(response, "Leaks")
        self.assertContains(response, "Backlogs")
        self.assertContains(response, 'id="health-copy"')
        self.assertContains(response, reverse("profile"))

    @patch("hitch.main.views.common._usage_context", side_effect=RuntimeError("codex down"))
    def test_profile_links_to_health_dashboard(self, _mock_usage: Any) -> None:
        self.client.force_login(_make_user())

        response = self.client.get(reverse("profile"))

        self.assertContains(response, reverse("health_dashboard"))
        self.assertContains(response, "Hitch health dashboard")


def _find(report: health.HealthReport, key: str) -> health.HealthMetric:
    for section in report.sections:
        for metric in section.metrics:
            if metric.key == key:
                return metric
    raise AssertionError(f"metric {key!r} not found in report")


_CGROUP_PREFIX = "0::/user.slice/user-0.slice/user@0.service/hitch.slice/hitch-codex.slice"


def _write_proc_pid(
    proc_root: Path,
    pid: int,
    *,
    scope_unit: str,
    argv: list[str],
    comm: str,
    vmrss_kb: int,
) -> None:
    pid_dir = proc_root / str(pid)
    pid_dir.mkdir(parents=True)
    (pid_dir / "cgroup").write_text(f"{_CGROUP_PREFIX}/{scope_unit}\n")
    (pid_dir / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
    (pid_dir / "comm").write_text(f"{comm}\n")
    (pid_dir / "status").write_text(f"Name:\t{comm}\nVmRSS:\t{vmrss_kb} kB\n")


class WorkerScopeProbeTests(TestCase):
    @override
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.proc = Path(self._tmp.name)
        self.now = timezone.now()

    def _terminal_instance(self, *, status: str, ended_ago: timedelta) -> CodexInstance:
        instance = _make_instance(status=status)
        CodexInstance.objects.filter(pk=instance.pk).update(ended_at=self.now - ended_ago)
        return instance

    def test_orphaned_grandchild_is_leaked(self) -> None:
        instance = self._terminal_instance(status=CodexInstance.STATUS_FAILED, ended_ago=timedelta(minutes=10))
        scope = f"hitch-codex-worker-{instance.pk}.service"
        _write_proc_pid(
            self.proc,
            4242,
            scope_unit=scope,
            argv=["encode_benchmark-add9a2a5", "--bench"],
            comm="encode_benchmark",
            vmrss_kb=2_300_000,
        )

        probe = host_probes.probe_worker_scopes(proc_root=self.proc, now=self.now)

        self.assertEqual(len(probe.leaked), 1)
        leaked = probe.leaked[0]
        self.assertEqual(leaked.instance_id, instance.pk)
        self.assertEqual(leaked.db_status, "failed")
        self.assertEqual(leaked.total_rss_bytes, 2_300_000 * 1024)
        self.assertTrue(leaked.processes[0].foreign)
        self.assertEqual(probe.total_leaked_rss_bytes, 2_300_000 * 1024)
        self.assertEqual(probe.active_count, 0)

    def test_live_worker_in_service_is_active_not_leaked(self) -> None:
        instance = self._terminal_instance(status=CodexInstance.STATUS_FAILED, ended_ago=timedelta(minutes=10))
        scope = f"hitch-codex-worker-{instance.pk}.service"
        _write_proc_pid(
            self.proc,
            55,
            scope_unit=scope,
            argv=["python3", "manage.py", "codex_worker", "--instance-id", str(instance.pk)],
            comm="python3",
            vmrss_kb=120_000,
        )

        probe = host_probes.probe_worker_scopes(proc_root=self.proc, now=self.now)

        self.assertEqual(probe.leaked, [])
        self.assertEqual(probe.active_count, 1)

    def test_legacy_scope_worker_is_still_detected(self) -> None:
        instance = self._terminal_instance(status=CodexInstance.STATUS_FAILED, ended_ago=timedelta(minutes=10))
        scope = f"hitch-codex-worker-{instance.pk}.scope"
        _write_proc_pid(
            self.proc,
            56,
            scope_unit=scope,
            argv=["python3", "manage.py", "codex_worker", "--instance-id", str(instance.pk)],
            comm="python3",
            vmrss_kb=120_000,
        )

        probe = host_probes.probe_worker_scopes(proc_root=self.proc, now=self.now)

        self.assertEqual(probe.leaked, [])
        self.assertEqual(probe.active_count, 1)

    def test_recent_terminal_within_grace_not_leaked(self) -> None:
        instance = self._terminal_instance(status=CodexInstance.STATUS_COMPLETED, ended_ago=timedelta(seconds=5))
        scope = f"hitch-codex-worker-{instance.pk}.service"
        _write_proc_pid(
            self.proc,
            77,
            scope_unit=scope,
            argv=["cargo", "build"],
            comm="cargo",
            vmrss_kb=500_000,
        )

        probe = host_probes.probe_worker_scopes(proc_root=self.proc, now=self.now)

        self.assertEqual(probe.leaked, [])

    def test_running_instance_not_flagged(self) -> None:
        instance = _make_instance(status=CodexInstance.STATUS_RUNNING)
        scope = f"hitch-codex-worker-{instance.pk}.service"
        _write_proc_pid(
            self.proc,
            88,
            scope_unit=scope,
            argv=["cargo", "test"],
            comm="cargo",
            vmrss_kb=500_000,
        )

        probe = host_probes.probe_worker_scopes(proc_root=self.proc, now=self.now)

        self.assertEqual(probe.leaked, [])

    def test_missing_proc_returns_empty(self) -> None:
        probe = host_probes.probe_worker_scopes(proc_root=self.proc / "nope", now=self.now)

        self.assertEqual(probe, WorkerScopeProbe(active_count=0, leaked=[]))


class RunserverSocketProbeTests(TestCase):
    @override
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.proc = Path(self._tmp.name)
        fd_dir = self.proc / "self" / "fd"
        fd_dir.mkdir(parents=True)
        os.symlink("socket:[100]", fd_dir / "3")
        os.symlink("socket:[200]", fd_dir / "4")
        os.symlink("pipe:[999]", fd_dir / "5")

    def _write_net_tcp(self, *rows: str) -> None:
        net = self.proc / "net"
        net.mkdir(exist_ok=True)
        header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"
        (net / "tcp").write_text("\n".join([header, *rows]) + "\n")
        (net / "tcp6").write_text(header + "\n")

    def test_close_wait_counts_only_our_sockets(self) -> None:
        # state 08 == CLOSE_WAIT; inode is field index 9.
        close_wait_ours = "0: 0100007F:1F90 00000000:0000 08 0:0 0:0 0 0 0 100 1"
        close_wait_other = "1: 0100007F:1F91 00000000:0000 08 0:0 0:0 0 0 0 555 1"
        established_ours = "2: 0100007F:1F92 0100007F:ABCD 01 0:0 0:0 0 0 0 200 1"
        self._write_net_tcp(close_wait_ours, close_wait_other, established_ours)

        self.assertEqual(host_probes.runserver_close_wait_count(proc_root=self.proc), 1)

    def test_fd_count(self) -> None:
        self.assertEqual(host_probes.runserver_fd_count(proc_root=self.proc), 3)

    def test_missing_net_tcp_returns_zero(self) -> None:
        self.assertEqual(host_probes.runserver_close_wait_count(proc_root=self.proc), 0)
