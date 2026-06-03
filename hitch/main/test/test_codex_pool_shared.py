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
