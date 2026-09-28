from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from hitch.main.runtime import server_lifecycle


class SchedulerHandleTests(SimpleTestCase):
    @patch("hitch.main.runtime.server_lifecycle.threading.Thread")
    def test_failed_thread_start_is_retryable_and_success_starts_once(
        self, mock_thread: MagicMock
    ) -> None:
        handle = server_lifecycle.SchedulerHandle(thread_name="hitch-test")
        target = MagicMock()
        mock_thread.return_value.start.side_effect = [RuntimeError("unavailable"), None]

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            handle.start(target)
        self.assertTrue(handle.start(target))
        self.assertFalse(handle.start(target))
        self.assertEqual(mock_thread.call_count, 2)
        mock_thread.assert_called_with(target=target, name="hitch-test", daemon=True)

    @patch("hitch.main.runtime.server_lifecycle.close_old_connections")
    def test_ticks_clean_up_connections_and_recover_from_errors(
        self, mock_close: MagicMock
    ) -> None:
        handle = server_lifecycle.SchedulerHandle(thread_name="hitch-test")
        tick = MagicMock(side_effect=[RuntimeError("unavailable"), 42])

        with self.assertLogs(server_lifecycle.logger, level="ERROR"):
            self.assertIsNone(handle.run_tick(tick))
        self.assertEqual(mock_close.call_count, 2)
        self.assertEqual(handle.run_tick(tick), 42)
        self.assertEqual(mock_close.call_count, 4)
