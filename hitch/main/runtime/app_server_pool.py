"""Warm Codex app-server pooling and retrying open/borrow entry points.

A bounded pool of initialized app-servers keyed by config (enable_memories,
model overrides) so request-path reads do not cold-open a server - the
cold open both adds latency and races the per-turn worker on the
CODEX_HOME init write (the "database is locked" failure). A keepalive
daemon keeps one warm server healthy across idle stretches.

Worker spawning, reconciliation, and path helpers stay in ``codex_pool``
and are reached through the module object.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from typing import Any

from django.conf import settings
from openai_codex import AppServerConfig, Codex, TransportClosedError
from openai_codex.api import Thread
from openai_codex.generated.v2_all import ThreadResumeParams
from pydantic import ValidationError

from hitch.main.runtime import codex_pool, server_lifecycle
from hitch.main.runtime.db import is_database_locked_error

logger = logging.getLogger(__name__)

_APPSERVER_START_MAX_ATTEMPTS = 10

_APPSERVER_WORKER_START_MAX_ATTEMPTS = 24

_APPSERVER_START_BACKOFF_BASE_SECONDS = 0.2

_APPSERVER_START_BACKOFF_MAX_SECONDS = 5.0

_THREAD_RESUME_PARAM_ALIASES = {
    "approval_policy": "approvalPolicy",
    "approvals_reviewer": "approvalsReviewer",
    "base_instructions": "baseInstructions",
    "developer_instructions": "developerInstructions",
    "model_provider": "modelProvider",
    "service_tier": "serviceTier",
}

def _start_codex_with_retry(factory: Callable[[], Codex]) -> Codex:
    """Call ``factory`` to construct a ``Codex``, retrying a locked init.

    ``factory`` is a zero-arg closure (typically ``lambda: Codex(config=...)``)
    so the call site keeps referencing its own module-local ``Codex`` symbol --
    important both for clarity and so tests that patch ``<module>.Codex`` still
    intercept construction. Only a ``TransportClosedError`` carrying the state
    DB's "database is locked" message is retried (the transient state-DB
    migration race); any other startup failure propagates immediately.
    """
    last_error: TransportClosedError | None = None
    for attempt in range(_APPSERVER_START_MAX_ATTEMPTS):
        try:
            return factory()
        except TransportClosedError as exc:
            if not is_database_locked_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Codex app-server state DB locked on start (attempt %s/%s)",
                attempt + 1,
                _APPSERVER_START_MAX_ATTEMPTS,
            )
        if attempt + 1 < _APPSERVER_START_MAX_ATTEMPTS:
            backoff = min(
                _APPSERVER_START_BACKOFF_BASE_SECONDS * (2**attempt),
                _APPSERVER_START_BACKOFF_MAX_SECONDS,
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error

def run_codex_op_with_retry(
    factory: Callable[[], Codex],
    operation: Callable[[Codex], codex_pool.T],
) -> codex_pool.T:
    """Open a fresh app-server, run ``operation`` against it, and retry the whole
    open+operation when a contended CODEX_HOME state DB surfaces.

    ``_start_codex_with_retry`` only guards *construction*. But the Codex
    runtime's state-DB migration/backfill path (no SQLITE_BUSY retry,
    openai/codex#20213) is also reached lazily by operations like
    ``thread_resume`` -- resuming a thread persisted by another worker migrates
    that thread's rows -- and a lock there *exits the app-server mid-operation*,
    surfacing as a ``TransportClosedError`` the construction-only retry never
    sees. Because the server is gone, recovery means reconstructing it, so this
    is a single retry loop spanning both construction and the operation: each
    attempt builds a fresh app-server and then runs ``operation`` against it. A
    locked ``TransportClosedError`` from *either* phase is retried by this one
    loop, bounded at ``_APPSERVER_START_MAX_ATTEMPTS`` rather than nesting
    ``_start_codex_with_retry``'s loop inside this one. ``operation`` must
    therefore be idempotent (it may run more than once) -- safe for reads like
    ``thread_resume``/``thread_list``, not for turn starts. Non-locked errors
    (including ``Http404`` the operation may raise) propagate immediately.
    """
    last_error: TransportClosedError | None = None
    for attempt in range(_APPSERVER_START_MAX_ATTEMPTS):
        try:
            codex = factory()
            with codex as entered:
                return operation(entered)
        except TransportClosedError as exc:
            if not is_database_locked_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Codex app-server state DB locked during open+operation "
                "(attempt %s/%s)",
                attempt + 1,
                _APPSERVER_START_MAX_ATTEMPTS,
            )
        if attempt + 1 < _APPSERVER_START_MAX_ATTEMPTS:
            backoff = min(
                _APPSERVER_START_BACKOFF_BASE_SECONDS * (2**attempt),
                _APPSERVER_START_BACKOFF_MAX_SECONDS,
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error

def run_borrowed_op_with_retry(
    codex_factory: Callable[..., Codex],
    operation: Callable[[Codex], codex_pool.T],
    *,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
) -> codex_pool.T:
    """Run ``operation`` against a *warm* pooled app-server when one exists,
    falling back to a retrying cold open only when the pool is empty.

    ``run_codex_op_with_retry`` always cold-opens a fresh app-server, so every
    call pays the CODEX_HOME init write that contends on the state-DB writer lock
    -- the failure mode behind "failed to initialize sqlite state runtime ...
    database is locked" on request paths like the session-detail resume. This
    instead borrows an already-initialized server from the shared pool first, so
    the steady-state request does *no* init write at all. ``operation`` must be
    idempotent (a warm server that dies on a locked op is dropped and the call
    falls back to the cold path, so it may run more than once) -- safe for reads
    like ``thread_resume``/``thread_list``, not for turn starts.
    """
    config = codex_pool.app_server_config(
        enable_memories=enable_memories, web_search_mode=web_search_mode
    )
    if _shared_pool_enabled():
        key = _pool_key(enable_memories, web_search_mode)
        warm = _SHARED_POOL.checkout_warm_only(key)
        if warm is not None:
            healthy = True
            try:
                return operation(warm)
            except TransportClosedError as exc:
                healthy = False
                if not is_database_locked_error(exc):
                    raise
                # The warm server exited on a locked op; drop it and fall through
                # to a fresh cold open (which retries the locked init itself).
                logger.warning(
                    "warm app-server state DB locked during borrowed op; "
                    "falling back to a fresh open"
                )
            except BaseException:
                healthy = False
                raise
            finally:
                _SHARED_POOL.release(key, warm, healthy=healthy)
    return run_codex_op_with_retry(lambda: codex_factory(config=config), operation)

def start_codex(config: AppServerConfig) -> Codex:
    """Construct a long-lived Codex app-server with ``_start_codex_with_retry``.

    For callers that own and reuse one app-server across many operations (e.g.
    the background scheduler) rather than opening a fresh one per use; the
    caller is responsible for ``close()``. Reusing a single app-server keeps its
    state DB initialized once instead of racing a new init on every operation.
    """
    return _start_codex_with_retry(lambda: Codex(config=config))

@contextlib.contextmanager
def open_codex(factory: Callable[[], Codex]) -> Generator[Codex]:
    """Open a Codex app-server, tolerating a contended state-DB init.

    Replaces ``with Codex(config=config) as codex`` with
    ``with open_codex(lambda: Codex(config=config)) as codex``: a locked
    state-DB init is retried (see ``_start_codex_with_retry``), then the
    constructed session's own ``__enter__``/``__exit__`` run as usual so it is
    closed on exit.
    """
    codex = _start_codex_with_retry(factory)
    with codex as entered:
        yield entered

@contextlib.contextmanager
def open_codex_resumed(
    factory: Callable[[], Codex],
    *,
    thread_id: str,
    resume_kwargs: dict[str, Any] | None = None,
    configure: Callable[[Codex], None] | None = None,
) -> Generator[tuple[Codex, Any]]:
    """Open an app-server, run ``configure`` then ``thread_resume``, retrying a
    locked CODEX_HOME state DB across the whole sequence, and yield the live
    ``(codex, thread)`` so the caller can start a (non-idempotent) turn.

    ``open_codex`` only retries *construction*. But the state-DB
    migration/backfill path (no SQLITE_BUSY retry, openai/codex#20213) is also
    reached lazily by ``thread_resume`` -- resuming a thread another worker
    persisted migrates its rows -- and a lock there exits the app-server
    mid-resume as a ``TransportClosedError`` the construction-only retry never
    sees. This retries construction+configure+resume together. ``thread_resume``
    is idempotent, so re-running it is safe. ``configure`` runs once per attempt
    against the attempt's fresh server (e.g. to install approval/notification
    handlers); a retried attempt discards its server, so any helper threads
    ``configure`` starts must tolerate that server closing under them (the
    worker's goal forwarder exits cleanly when its source transport closes).
    """
    resume_kwargs = resume_kwargs or {}
    last_error: TransportClosedError | None = None
    for attempt in range(_APPSERVER_WORKER_START_MAX_ATTEMPTS):
        codex = None
        entered = None
        try:
            codex = factory()
            entered = codex.__enter__()
            if configure is not None:
                configure(entered)
            thread = _thread_resume_tolerating_sdk_metadata(
                entered, thread_id=thread_id, resume_kwargs=resume_kwargs
            )
        except TransportClosedError as exc:
            if entered is not None and codex is not None:
                codex.__exit__(type(exc), exc, exc.__traceback__)
            if not is_database_locked_error(exc):
                raise
            last_error = exc
            logger.warning(
                "Codex app-server state DB locked during worker open+resume "
                "(attempt %s/%s)",
                attempt + 1,
                _APPSERVER_WORKER_START_MAX_ATTEMPTS,
            )
            if attempt + 1 < _APPSERVER_WORKER_START_MAX_ATTEMPTS:
                time.sleep(
                    min(
                        _APPSERVER_START_BACKOFF_BASE_SECONDS * (2**attempt),
                        _APPSERVER_START_BACKOFF_MAX_SECONDS,
                    )
            )
            continue
        except BaseException as exc:
            if entered is not None and codex is not None:
                codex.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            yield entered, thread
        except BaseException as exc:
            codex.__exit__(type(exc), exc, exc.__traceback__)
            raise
        codex.__exit__(None, None, None)
        return
    assert last_error is not None
    raise last_error

def _thread_resume_tolerating_sdk_metadata(
    codex: Codex,
    *,
    thread_id: str,
    resume_kwargs: dict[str, Any],
) -> Any:
    try:
        return codex.thread_resume(thread_id, **resume_kwargs)
    except ValidationError as exc:
        if not _can_raw_resume_after_validation_error(exc):
            raise
        logger.warning(
            "Codex SDK could not validate thread/resume metadata for %s; "
            "falling back to raw resume",
            thread_id,
        )
        return _raw_thread_resume(codex, thread_id=thread_id, resume_kwargs=resume_kwargs)

def _can_raw_resume_after_validation_error(exc: ValidationError) -> bool:
    """Resume only needs ``thread.id``; tolerate SDK drift in sibling metadata."""
    for error in exc.errors():
        loc = error.get("loc", ())
        if isinstance(loc, tuple) and loc and loc[0] == "thread":
            return False
    return True

def _raw_thread_resume(
    codex: Codex,
    *,
    thread_id: str,
    resume_kwargs: dict[str, Any],
) -> Thread:
    raw_request = getattr(getattr(codex, "_client", None), "_request_raw", None)
    if not callable(raw_request):
        raise RuntimeError("Codex client does not expose raw requests")
    response = raw_request("thread/resume", _thread_resume_payload(thread_id, resume_kwargs))
    if not isinstance(response, dict):
        raise RuntimeError("thread/resume response must be an object")
    thread = response.get("thread")
    if not isinstance(thread, dict):
        raise RuntimeError("thread/resume response missing thread object")
    resumed_thread_id = thread.get("id")
    if not isinstance(resumed_thread_id, str) or not resumed_thread_id:
        raise RuntimeError("thread/resume response missing thread id")
    return Thread(codex._client, resumed_thread_id)

def _thread_resume_payload(thread_id: str, resume_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        _THREAD_RESUME_PARAM_ALIASES.get(key, key): value
        for key, value in resume_kwargs.items()
    }
    params = ThreadResumeParams(thread_id=thread_id, **kwargs)
    dumped = params.model_dump(by_alias=True, exclude_none=True, mode="json")
    if not isinstance(dumped, dict):
        raise TypeError("thread resume params did not dump to an object")
    return dumped

_SHARED_POOL_MAX = 4

_ConfigKey = tuple[bool, str | None]

def _pool_key(enable_memories: bool, web_search_mode: str | None) -> _ConfigKey:
    return (enable_memories, codex_pool._normalized_web_search_mode(web_search_mode))

def _close_quietly(codex: Codex) -> None:
    with contextlib.suppress(Exception):
        codex.close()

def _codex_is_alive(codex: Codex) -> bool:
    """Best-effort check that the app-server subprocess is still running.

    A pooled server is long-lived, so an idle one may have exited (crash, OOM
    kill) since it was returned -- or been returned looking healthy by a borrow
    whose helper swallowed the ``TransportClosedError``. Handing such a dead
    server back would surface an error to a single request that the old
    open-per-call path never hit, so the pool drops it on checkout and reuses a
    live one instead. Reaches into the SDK client's process handle the way our
    call sites already reach into ``codex._client``; ``poll()`` is a cheap,
    non-blocking liveness probe (no app-server round-trip).
    """
    proc = getattr(getattr(codex, "_client", None), "_proc", None)
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False

class _SharedCodexPool:
    """Bounded pool of long-lived app-servers with exclusive checkout.

    Only one borrower drives a given ``Codex`` at a time, so reuse never relies
    on the SDK being safe for concurrent stdin writes from multiple request
    threads. Checkout skips (and closes) idle servers whose subprocess has died
    so a stale transport never reaches a borrower; a borrow that dies mid-use
    drops its instance -- mirroring ``_SchedulerCodex.reset`` -- so the next
    borrow reconnects. The cap bounds total warm servers; a full pool evicts an
    idle server from *another* config key rather than refusing to keep this
    key's, so no key is starved of a warm server. Checkouts past the cap
    construct (and on return close) a private server rather than blocking.
    """

    def __init__(self, max_size: int = _SHARED_POOL_MAX) -> None:
        self._lock = threading.Lock()
        self._idle: dict[_ConfigKey, deque[Codex]] = {}
        self._in_use = 0
        self._max = max_size
        # Config keys borrowed recently, in LRU order (oldest first), so the
        # keepalive knows which keys to keep warm -- a memories-enabled session
        # uses a different key than the default and would otherwise cold-open on
        # its first render after idle. A dict is used as an ordered set.
        self._seen_keys: dict[_ConfigKey, None] = {}

    def _note_key(self, key: _ConfigKey) -> None:
        """Record ``key`` as most-recently used. Caller holds ``self._lock``."""
        self._seen_keys.pop(key, None)
        self._seen_keys[key] = None

    def _total_warm(self) -> int:
        return self._in_use + sum(len(idle) for idle in self._idle.values())

    def warm_target_keys(self) -> list[_ConfigKey]:
        """Keys the keepalive should keep warm, capped at pool capacity.

        Warming more keys than the pool can hold (``_max``) would have each tick
        cold-open the keys that the previous tick evicted -- reintroducing the
        init churn the keepalive exists to avoid. So return the default key
        (always kept warm) plus the most-recently-used other keys, up to ``_max``
        total.
        """
        default = _pool_key(enable_memories=False, web_search_mode=None)
        with self._lock:
            recent = [k for k in reversed(self._seen_keys) if k != default]
        return [default, *recent[: max(self._max - 1, 0)]]

    def _pop_idle_other_key(self, key: _ConfigKey) -> Codex | None:
        """Pop the oldest idle server belonging to a different config key."""
        for other_key, servers in self._idle.items():
            if other_key != key and servers:
                return servers.pop()
        return None

    def checkout(self, key: _ConfigKey, factory: Callable[[], Codex]) -> Codex:
        dead: list[Codex] = []
        with self._lock:
            self._note_key(key)
            idle = self._idle.get(key)
            reused: Codex | None = None
            while idle:
                candidate = idle.pop()
                if _codex_is_alive(candidate):
                    reused = candidate
                    break
                dead.append(candidate)
            self._in_use += 1
        for stale in dead:
            _close_quietly(stale)
        if reused is not None:
            return reused
        # Construct outside the structure lock: a cold start spawns a
        # subprocess and may retry a locked state-DB init.
        try:
            return _start_codex_with_retry(factory)
        except BaseException:
            with self._lock:
                self._in_use -= 1
            raise

    def checkout_warm_only(self, key: _ConfigKey) -> Codex | None:
        """Check out a live idle server without ever constructing one.

        Returns ``None`` when no warm server is available rather than cold-opening
        (and so re-initializing the CODEX_HOME state DB). Lets callers prefer a
        warm server on the request path and fall back to their own retrying cold
        open only when the pool is empty -- the construction write is exactly what
        contends on the state-DB writer lock, so skipping it when a warm server
        exists avoids the lock entirely.
        """
        dead: list[Codex] = []
        reused: Codex | None = None
        with self._lock:
            self._note_key(key)
            idle = self._idle.get(key)
            while idle:
                candidate = idle.pop()
                if _codex_is_alive(candidate):
                    reused = candidate
                    break
                dead.append(candidate)
            if reused is not None:
                self._in_use += 1
        for stale in dead:
            _close_quietly(stale)
        return reused

    def release(self, key: _ConfigKey, codex: Codex, *, healthy: bool) -> None:
        to_close: Codex | None = None
        with self._lock:
            self._in_use -= 1
            if healthy:
                if self._total_warm() < self._max:
                    self._idle.setdefault(key, deque()).appendleft(codex)
                    return
                # Pool full: keep this key's server by evicting an idle server
                # from another key. If only this key is idle we are already at
                # our share, so close the returning server instead.
                to_close = self._pop_idle_other_key(key)
                if to_close is not None:
                    self._idle.setdefault(key, deque()).appendleft(codex)
            if to_close is None:
                to_close = codex
        _close_quietly(to_close)

    def close_all(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, {}
        for servers in idle.values():
            for codex in servers:
                _close_quietly(codex)

_SHARED_POOL = _SharedCodexPool()
atexit.register(_SHARED_POOL.close_all)

def _shared_pool_enabled() -> bool:
    # Under tests each case patches ``Codex`` fresh and expects construction per
    # call, so caching pooled instances across cases would leak mocks. The pool
    # itself is covered directly by test_codex_pool_shared.
    return not getattr(settings, "TESTING", False)

_KEEPALIVE_INTERVAL_SECONDS = 30

_keepalive = server_lifecycle.SchedulerHandle(
    thread_name="hitch-codex-pool-keepalive",
    tick_interval_seconds=_KEEPALIVE_INTERVAL_SECONDS,
)

def _codex_pool_keepalive_enabled() -> bool:
    """Whether this process serves requests (and so uses the shared pool).

    Independent of the background schedulers: the keepalive must run wherever the
    request-path pool is enabled, even on a server that disabled the maintenance
    scheduler (e.g. ``HITCH_WORKFLOW_MAINTENANCE_SCHEDULER=0`` because maintenance
    runs elsewhere). Mirrors the schedulers' "real server process" gate so it
    never starts under management commands, migrations, or tests.
    """
    return server_lifecycle.background_work_enabled(
        include_wsgi_server_commands=True
    )

def start_codex_pool_keepalive() -> bool:
    """Start a daemon that keeps one warm pooled app-server present and healthy.

    The shared pool only fills when a request borrows, and an idle pooled server
    can die (laptop sleep, OOM, a Codex-side idle exit) with nothing noticing
    until the next checkout cold-opens a replacement. After an idle stretch that
    makes the first request -- and the session-detail resume -- cold-open and
    race the per-turn worker on the CODEX_HOME init write, which is the
    "database is locked" users hit first thing in the morning. This periodically
    borrows each used config key and runs one cheap read, so an initialized
    server is already warm when the user returns and a dead one is rebuilt
    *before* they hit it rather than on their request.
    """
    if not _codex_pool_keepalive_enabled():
        return False
    return _keepalive.start(_codex_pool_keepalive_loop)

def _codex_pool_keepalive_loop() -> None:
    stop = threading.Event()
    while True:
        _keepalive.run_tick(_codex_pool_keepalive_tick)
        stop.wait(_KEEPALIVE_INTERVAL_SECONDS)

def _codex_pool_keepalive_tick() -> None:
    """Borrow and exercise one server per used config key with a cheap read.

    Borrowing reconstructs a dead pooled server (checkout drops one whose
    subprocess has exited) and the read both keeps the app-server from idling out
    and surfaces a wedged-but-alive server -- a failed probe makes ``borrow_codex``
    drop it, so the next tick rebuilds a healthy one. Every key ever borrowed is
    kept warm, not just the default: a memories-enabled session uses a different
    pool key and would otherwise cold-open (and race the state-DB init lock) on
    its first detail render after idle. Best-effort: a failure on one key is
    logged and retried next tick rather than killing the daemon.
    """
    for enable_memories, web_search_mode in _SHARED_POOL.warm_target_keys():
        try:
            with borrow_codex(
                Codex,
                enable_memories=enable_memories,
                web_search_mode=web_search_mode,
            ) as codex:
                codex.thread_list(limit=1, use_state_db_only=True)
        except Exception:
            logger.warning(
                "codex pool keepalive probe failed for key "
                "(enable_memories=%s, web_search_mode=%s)",
                enable_memories,
                web_search_mode,
                exc_info=True,
            )

@contextlib.contextmanager
def borrow_codex(
    codex_factory: Callable[..., Codex],
    *,
    enable_memories: bool = False,
    web_search_mode: str | None = None,
) -> Generator[Codex]:
    """Borrow a warm, already-initialized app-server from the shared pool.

    ``codex_factory`` is the caller's ``Codex`` symbol (constructed as
    ``codex_factory(config=...)``); keeping construction at the call site lets
    callers tune the config and keeps test patches on the caller's module
    effective. Steady-state borrows reuse an idle long-lived server with no
    subprocess spawn; cold construction still goes through
    ``_start_codex_with_retry`` so a genuine state-DB init race is retried. The
    pool owns the server's lifecycle, so -- unlike ``open_codex`` -- the yielded
    server is not entered/closed per borrow.
    """
    config = codex_pool.app_server_config(
        enable_memories=enable_memories, web_search_mode=web_search_mode
    )

    def factory() -> Codex:
        return codex_factory(config=config)

    if not _shared_pool_enabled():
        with open_codex(factory) as codex:
            yield codex
        return

    key = _pool_key(enable_memories, web_search_mode)
    codex = _SHARED_POOL.checkout(key, factory)
    healthy = True
    try:
        yield codex
    except BaseException:
        # A failed borrow may have killed the transport; drop rather than reuse.
        healthy = False
        raise
    finally:
        _SHARED_POOL.release(key, codex, healthy=healthy)
