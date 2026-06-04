"""Tests for the shared app-server pool (``borrow_codex`` / ``_SharedCodexPool``).

The pool keeps a small set of long-lived app-servers warm so steady-state
borrows skip the per-request subprocess spawn + state-DB init that used to race
the CODEX_HOME init lock. ``_SharedCodexPool`` is exercised directly (no Django
settings involved); ``borrow_codex``'s bypass-under-TESTING and pooling paths are
covered separately.
"""

import threading
from collections.abc import Callable
from typing import Any, override
from unittest import mock

from django.test import SimpleTestCase, override_settings
from openai_codex import TransportClosedError

from hitch.main import codex_pool

_LOCKED = "app-server closed stdout. stderr_tail=... (code: 5) database is locked"


class _FakeProc:
    def __init__(self) -> None:
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode


class _FakeClient:
    def __init__(self) -> None:
        self._proc: _FakeProc | None = _FakeProc()


class _FakeCodex:
    """Stand-in for ``Codex`` accepting ``Codex(config=...)`` construction and
    matching the context-manager protocol ``open_codex`` relies on. Exposes a
    ``_client._proc`` so the pool's liveness check (``_codex_is_alive``) sees a
    running subprocess until ``mark_dead``/``close``."""

    def __init__(self, **_kwargs: Any) -> None:
        self.closed = False
        self._client = _FakeClient()

    def __enter__(self) -> "_FakeCodex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def mark_dead(self) -> None:
        if self._client._proc is not None:
            self._client._proc._returncode = 1

    def close(self) -> None:
        self.closed = True
        self._client._proc = None


def _closed(codex: Any) -> bool:
    return bool(codex.closed)


def _mark_dead(codex: Any) -> None:
    codex.mark_dead()


def _counting_factory() -> tuple[Callable[[], Any], list[Any]]:
    """A zero-arg factory plus the list of instances it has constructed."""
    built: list[Any] = []

    def factory() -> Any:
        codex = _FakeCodex()
        built.append(codex)
        return codex

    return factory, built


class SharedCodexPoolTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        self.key = codex_pool._pool_key(enable_memories=False, web_search_mode=None)

    def test_checkout_reuses_returned_instance(self) -> None:
        pool = codex_pool._SharedCodexPool()
        factory, built = _counting_factory()

        first = pool.checkout(self.key, factory)
        pool.release(self.key, first, healthy=True)
        second = pool.checkout(self.key, factory)

        self.assertIs(second, first)
        self.assertEqual(len(built), 1)

    def test_distinct_config_keys_do_not_share_instances(self) -> None:
        pool = codex_pool._SharedCodexPool()
        factory, built = _counting_factory()
        key_mem = codex_pool._pool_key(enable_memories=True, web_search_mode=None)

        no_mem = pool.checkout(self.key, factory)
        with_mem = pool.checkout(key_mem, factory)
        pool.release(self.key, no_mem, healthy=True)
        pool.release(key_mem, with_mem, healthy=True)

        self.assertIsNot(no_mem, with_mem)
        # Each key reuses only its own idle instance.
        self.assertIs(pool.checkout(self.key, factory), no_mem)
        self.assertIs(pool.checkout(key_mem, factory), with_mem)
        self.assertEqual(len(built), 2)

    def test_unhealthy_release_drops_and_closes(self) -> None:
        pool = codex_pool._SharedCodexPool()
        factory, built = _counting_factory()

        first = pool.checkout(self.key, factory)
        pool.release(self.key, first, healthy=False)

        self.assertTrue(_closed(first))
        # The dropped instance is not handed back out; a fresh one is built.
        second = pool.checkout(self.key, factory)
        self.assertIsNot(second, first)
        self.assertEqual(len(built), 2)

    def test_idle_never_exceeds_cap(self) -> None:
        pool = codex_pool._SharedCodexPool(max_size=2)
        factory, built = _counting_factory()

        # Three concurrent checkouts force three constructions (idle is empty).
        checked_out = [pool.checkout(self.key, factory) for _ in range(3)]
        for codex in checked_out:
            pool.release(self.key, codex, healthy=True)

        idle = pool._idle.get(self.key)
        idle_count = len(idle) if idle is not None else 0
        self.assertLessEqual(idle_count, 2)
        self.assertEqual(pool._in_use, 0)
        # The over-cap instance was closed rather than pooled.
        self.assertEqual(sum(1 for c in built if _closed(c)), len(built) - idle_count)

    def test_concurrent_checkout_is_exclusive(self) -> None:
        pool = codex_pool._SharedCodexPool()
        factory, _built = _counting_factory()
        barrier = threading.Barrier(2)
        held: list[Any] = []
        lock = threading.Lock()

        def borrow() -> None:
            codex = pool.checkout(self.key, factory)
            with lock:
                held.append(codex)
            barrier.wait()  # hold the checkout until both threads are in

        threads = [threading.Thread(target=borrow) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(len(held), 2)
        self.assertIsNot(held[0], held[1])

    def test_close_all_closes_idle_instances(self) -> None:
        pool = codex_pool._SharedCodexPool()
        factory, built = _counting_factory()

        first = pool.checkout(self.key, factory)
        pool.release(self.key, first, healthy=True)
        pool.close_all()

        self.assertTrue(_closed(first))
        self.assertEqual(pool._idle, {})

    def test_checkout_drops_dead_idle_and_reconnects(self) -> None:
        pool = codex_pool._SharedCodexPool()
        factory, built = _counting_factory()

        first = pool.checkout(self.key, factory)
        pool.release(self.key, first, healthy=True)
        # The idle subprocess exits while pooled (crash/OOM) -- or was returned
        # looking healthy by a borrow whose helper swallowed the transport
        # error. Either way checkout must not hand back the dead instance.
        _mark_dead(first)

        second = pool.checkout(self.key, factory)
        self.assertIsNot(second, first)
        self.assertTrue(_closed(first))
        self.assertEqual(len(built), 2)

    def test_checkout_warm_only_returns_none_when_empty(self) -> None:
        pool = codex_pool._SharedCodexPool()
        # No warm server: must not construct one (that is the contended init).
        self.assertIsNone(pool.checkout_warm_only(self.key))
        self.assertEqual(pool._in_use, 0)

    def test_checkout_warm_only_reuses_and_skips_dead(self) -> None:
        pool = codex_pool._SharedCodexPool()
        factory, built = _counting_factory()
        dead = pool.checkout(self.key, factory)
        live = pool.checkout(self.key, factory)
        # Release dead first so it sits at the pop end and is examined before
        # the live one (checkout pops the right, release appends the left).
        pool.release(self.key, dead, healthy=True)
        pool.release(self.key, live, healthy=True)
        _mark_dead(dead)

        warm = pool.checkout_warm_only(self.key)

        self.assertIs(warm, live)
        self.assertTrue(_closed(dead))
        self.assertEqual(pool._in_use, 1)
        # Never constructs: a second call with nothing warm left returns None.
        self.assertIsNone(pool.checkout_warm_only(self.key))
        self.assertEqual(len(built), 2)

    def test_full_pool_evicts_other_key_rather_than_starving(self) -> None:
        pool = codex_pool._SharedCodexPool(max_size=2)
        factory, built = _counting_factory()
        key_mem = codex_pool._pool_key(enable_memories=True, web_search_mode=None)

        # Fill the pool with two idle servers for the default key.
        a = pool.checkout(self.key, factory)
        b = pool.checkout(self.key, factory)
        pool.release(self.key, a, healthy=True)
        pool.release(self.key, b, healthy=True)

        # A borrow for the other key must end up warm rather than discarded.
        other = pool.checkout(key_mem, factory)
        pool.release(key_mem, other, healthy=True)

        self.assertIn(other, pool._idle.get(key_mem, []))
        self.assertFalse(_closed(other))
        self.assertLessEqual(pool._total_warm(), 2)


class BorrowCodexTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        # borrow_codex uses the module-level singleton; isolate each test.
        saved_pool = codex_pool._SHARED_POOL
        codex_pool._SHARED_POOL = codex_pool._SharedCodexPool()
        self.addCleanup(setattr, codex_pool, "_SHARED_POOL", saved_pool)

    def test_testing_bypass_opens_and_closes_per_borrow(self) -> None:
        # Under TESTING the pool is bypassed: borrow_codex behaves like
        # open_codex so each patched Codex constructs fresh and is closed.
        factory, built = _counting_factory()
        with codex_pool.borrow_codex(lambda **_: factory()) as codex:
            self.assertFalse(_closed(codex))
        self.assertTrue(_closed(codex))
        self.assertEqual(len(built), 1)

    @override_settings(TESTING=False)
    def test_pooling_reuses_warm_server(self) -> None:
        factory, built = _counting_factory()
        codex_class = lambda **_: factory()  # noqa: E731

        with codex_pool.borrow_codex(codex_class) as first:
            self.assertFalse(_closed(first))
        with codex_pool.borrow_codex(codex_class) as second:
            pass

        self.assertIs(second, first)
        self.assertEqual(len(built), 1)
        self.assertFalse(_closed(first))

    @override_settings(TESTING=False)
    def test_pooling_drops_server_after_transport_error(self) -> None:
        factory, built = _counting_factory()
        codex_class = lambda **_: factory()  # noqa: E731

        with (
            self.assertRaises(TransportClosedError),
            codex_pool.borrow_codex(codex_class) as codex,
        ):
            raise TransportClosedError(_LOCKED)

        self.assertTrue(_closed(codex))
        # The dead transport is dropped; the next borrow reconstructs.
        with codex_pool.borrow_codex(codex_class) as replacement:
            self.assertIsNot(replacement, codex)
        self.assertEqual(len(built), 2)


class _ProbeCodex(_FakeCodex):
    """``_FakeCodex`` that records keepalive probe calls."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.thread_list_calls: list[dict[str, Any]] = []

    def thread_list(self, **kwargs: Any) -> object:
        self.thread_list_calls.append(kwargs)
        return object()


class RunBorrowedOpWithRetryTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        saved_pool = codex_pool._SHARED_POOL
        codex_pool._SHARED_POOL = codex_pool._SharedCodexPool()
        self.addCleanup(setattr, codex_pool, "_SHARED_POOL", saved_pool)

    @override_settings(TESTING=False)
    def test_uses_warm_server_without_constructing(self) -> None:
        factory, built = _counting_factory()
        codex_class = lambda **_: factory()  # noqa: E731
        # Pre-warm one server, then return it to the idle pool.
        with codex_pool.borrow_codex(codex_class) as warm:
            pass

        seen: list[Any] = []

        def operation(codex: Any) -> str:
            seen.append(codex)
            return "ok"

        result = codex_pool.run_borrowed_op_with_retry(codex_class, operation)

        self.assertEqual(result, "ok")
        # Reused the warm server; no second construction and no init write.
        self.assertEqual(len(built), 1)
        self.assertIs(seen[0], warm)
        self.assertFalse(_closed(warm))

    @override_settings(TESTING=False)
    def test_falls_back_to_cold_open_when_pool_empty(self) -> None:
        factory, built = _counting_factory()
        codex_class = lambda **_: factory()  # noqa: E731

        seen: list[Any] = []

        def operation(codex: Any) -> str:
            seen.append(codex)
            return "ok"

        result = codex_pool.run_borrowed_op_with_retry(codex_class, operation)

        self.assertEqual(result, "ok")
        # Empty pool: one cold open, run, and close (the cold path owns the server).
        self.assertEqual(len(built), 1)
        self.assertTrue(_closed(seen[0]))

    @override_settings(TESTING=False)
    def test_warm_locked_op_drops_server_and_falls_back(self) -> None:
        factory, built = _counting_factory()
        codex_class = lambda **_: factory()  # noqa: E731
        with codex_pool.borrow_codex(codex_class) as warm:
            pass

        calls = {"n": 0}

        def operation(_codex: Any) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise TransportClosedError(_LOCKED)
            return "ok"

        result = codex_pool.run_borrowed_op_with_retry(codex_class, operation)

        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 2)
        # Warm server poisoned by the locked op is dropped; a fresh one is opened.
        self.assertTrue(_closed(warm))
        self.assertEqual(len(built), 2)

    @override_settings(TESTING=False)
    def test_warm_non_locked_transport_error_propagates(self) -> None:
        factory, built = _counting_factory()
        codex_class = lambda **_: factory()  # noqa: E731
        with codex_pool.borrow_codex(codex_class) as warm:
            pass

        def operation(_codex: Any) -> str:
            raise TransportClosedError("app-server closed stdout. crash")

        with self.assertRaises(TransportClosedError):
            codex_pool.run_borrowed_op_with_retry(codex_class, operation)

        # A non-locked transport error drops the warm server but does not retry.
        self.assertTrue(_closed(warm))
        self.assertEqual(len(built), 1)

    @override_settings(TESTING=False)
    def test_warm_non_locked_error_propagates_without_fallback(self) -> None:
        factory, built = _counting_factory()
        codex_class = lambda **_: factory()  # noqa: E731
        with codex_pool.borrow_codex(codex_class) as warm:
            pass

        def operation(_codex: Any) -> str:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            codex_pool.run_borrowed_op_with_retry(codex_class, operation)

        # The warm server is dropped, but a non-locked error does not retry.
        self.assertTrue(_closed(warm))
        self.assertEqual(len(built), 1)


class CodexPoolKeepaliveTests(SimpleTestCase):
    @override
    def setUp(self) -> None:
        saved_pool = codex_pool._SHARED_POOL
        codex_pool._SHARED_POOL = codex_pool._SharedCodexPool()
        self.addCleanup(setattr, codex_pool, "_SHARED_POOL", saved_pool)

    def test_disabled_under_testing(self) -> None:
        self.assertFalse(codex_pool.start_codex_pool_keepalive())

    def test_starts_one_daemon_thread(self) -> None:
        self.addCleanup(setattr, codex_pool, "_keepalive_started", False)
        codex_pool._keepalive_started = False
        with (
            mock.patch.object(
                codex_pool, "_codex_pool_keepalive_enabled", return_value=True
            ),
            mock.patch("hitch.main.codex_pool.threading.Thread") as thread_cls,
        ):
            started = codex_pool.start_codex_pool_keepalive()
            # Idempotent: a second call does not start another thread.
            again = codex_pool.start_codex_pool_keepalive()

        self.assertTrue(started)
        self.assertFalse(again)
        thread_cls.assert_called_once()
        thread_cls.return_value.start.assert_called_once_with()

    @override_settings(TESTING=False)
    def test_keepalive_enabled_only_for_server_processes(self) -> None:
        with mock.patch("hitch.main.codex_pool.sys.argv", ["manage.py", "test"]):
            self.assertFalse(codex_pool._codex_pool_keepalive_enabled())
        with mock.patch("hitch.main.codex_pool.sys.argv", ["gunicorn", "hitch.wsgi"]):
            self.assertTrue(codex_pool._codex_pool_keepalive_enabled())
        with mock.patch(
            "hitch.main.codex_pool.sys.argv",
            ["manage.py", "runserver", "--noreload"],
        ):
            self.assertTrue(codex_pool._codex_pool_keepalive_enabled())

    @override_settings(TESTING=False)
    def test_tick_warms_and_probes_a_server(self) -> None:
        built: list[_ProbeCodex] = []

        def codex_class(**_kwargs: Any) -> _ProbeCodex:
            codex = _ProbeCodex()
            built.append(codex)
            return codex

        with mock.patch.object(codex_pool, "Codex", codex_class):
            codex_pool._codex_pool_keepalive_tick()

        # One warm server constructed and exercised with a cheap read.
        self.assertEqual(len(built), 1)
        self.assertEqual(
            built[0].thread_list_calls,
            [{"limit": 1, "use_state_db_only": True}],
        )
        # Returned to the pool (warm) rather than closed.
        self.assertFalse(_closed(built[0]))

    @override_settings(TESTING=False)
    def test_tick_swallows_probe_failure(self) -> None:
        def codex_class(**_kwargs: Any) -> _FakeCodex:
            return _FakeCodex()

        with (
            mock.patch.object(codex_pool, "borrow_codex", side_effect=RuntimeError),
            mock.patch.object(codex_pool, "Codex", codex_class),
        ):
            # Must not raise: a failed probe is logged and retried next tick.
            codex_pool._codex_pool_keepalive_tick()

    @override_settings(TESTING=False)
    def test_tick_warms_every_used_key(self) -> None:
        # A memories-enabled session uses a distinct pool key; once seen, the
        # keepalive must keep it warm too, not just the default key.
        mem_key = codex_pool._pool_key(enable_memories=True, web_search_mode=None)
        codex_pool._SHARED_POOL.checkout_warm_only(mem_key)  # records the key
        built: list[_ProbeCodex] = []

        def codex_class(**_kwargs: Any) -> _ProbeCodex:
            codex = _ProbeCodex()
            built.append(codex)
            return codex

        with mock.patch.object(codex_pool, "Codex", codex_class):
            codex_pool._codex_pool_keepalive_tick()

        # One probe for the default key and one for the memories-enabled key.
        self.assertEqual(len(built), 2)
        self.assertTrue(all(c.thread_list_calls for c in built))


class SharedPoolSeenKeyTests(SimpleTestCase):
    def test_warm_target_keys_includes_default_and_seen(self) -> None:
        pool = codex_pool._SharedCodexPool()
        default = codex_pool._pool_key(enable_memories=False, web_search_mode=None)
        # Empty pool still warms the default key so a fresh process keeps one warm.
        self.assertEqual(pool.warm_target_keys(), [default])

        mem_key = codex_pool._pool_key(enable_memories=True, web_search_mode=None)
        pool.checkout_warm_only(mem_key)  # records the key without constructing
        self.assertIn(mem_key, pool.warm_target_keys())
        self.assertIn(default, pool.warm_target_keys())

    def test_warm_target_keys_capped_at_pool_capacity(self) -> None:
        pool = codex_pool._SharedCodexPool(max_size=2)
        default = codex_pool._pool_key(enable_memories=False, web_search_mode=None)
        # More distinct keys than the pool can hold; warming them all would just
        # evict each other and cold-open every tick.
        for mode in ("a", "b", "c", "d"):
            pool.checkout_warm_only((True, mode))

        targets = pool.warm_target_keys()

        self.assertEqual(len(targets), 2)  # == max_size
        self.assertIn(default, targets)
        # Keeps the most-recently-used non-default key, not the oldest.
        self.assertIn((True, "d"), targets)
