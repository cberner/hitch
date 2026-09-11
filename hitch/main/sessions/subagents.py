"""Discover native Codex descendants without resuming their threads."""

from dataclasses import dataclass
from typing import Any

from django.core.cache import cache
from openai_codex import Codex

from hitch.main.runtime import app_server_pool, codex_pool


@dataclass(frozen=True)
class Subagent:
    id: str
    parent_id: str
    name: str
    role: str
    path: str


def list_subagents(thread_id: str) -> list[Subagent]:
    cache_key = f"session-subagents:{codex_pool.codex_home_dir()}:{thread_id}"
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return cached
    agents = app_server_pool.run_borrowed_op_with_retry(
        Codex, lambda codex: _read_subagents(codex, thread_id)
    )
    cache.set(cache_key, agents, timeout=5)
    return agents


def _read_subagents(codex: Codex, thread_id: str) -> list[Subagent]:
    candidates: dict[str, Subagent] = {}
    for archived in (False, True):
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            response = codex._client._request_raw(
                "thread/list",
                {
                    "ancestorThreadId": thread_id,
                    "sourceKinds": ["subAgentThreadSpawn"],
                    # Polling must not trigger filesystem repair or contend with workers.
                    "useStateDbOnly": True,
                    "archived": archived,
                    "sortKey": "created_at",
                    "sortDirection": "asc",
                    "limit": 100,
                    "cursor": cursor,
                },
            )
            data = response.get("data") if isinstance(response, dict) else None
            if not isinstance(response, dict) or not isinstance(data, list):
                raise ValueError("Invalid Codex subagent list")
            for raw in data:
                if isinstance(raw, dict) and (agent := _subagent(raw)) is not None:
                    candidates[agent.id] = agent
            next_cursor = response.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    # Validate ancestry even on older runtimes that ignore the list filter.
    result: list[Subagent] = []
    parents = {thread_id}
    while children := [a for a in candidates.values() if a.parent_id in parents and a.id not in parents]:
        result.extend(children)
        parents.update(a.id for a in children)
    return result


def _subagent(raw: dict[str, Any]) -> Subagent | None:
    source = raw.get("source")
    subagent = source.get("subAgent") if isinstance(source, dict) else None
    spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
    spawn = spawn if isinstance(spawn, dict) else {}
    parent_id = raw.get("parentThreadId") or spawn.get("parent_thread_id")
    thread_id = raw.get("id")
    if not isinstance(thread_id, str) or not thread_id or not isinstance(parent_id, str) or not parent_id:
        return None
    name = raw.get("agentNickname") or spawn.get("agent_nickname") or raw.get("name")
    role = raw.get("agentRole") or spawn.get("agent_role")
    path = raw.get("path")
    return Subagent(
        id=thread_id,
        parent_id=parent_id,
        name=name if isinstance(name, str) and name else f"Subagent {thread_id[:8]}",
        role=role if isinstance(role, str) else "",
        path=path if isinstance(path, str) else "",
    )
