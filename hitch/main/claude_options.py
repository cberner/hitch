"""Build ``ClaudeAgentOptions`` for a Hitch Claude Code worker turn.

This is the Claude-backend analog of ``codex_pool.app_server_config`` plus the
per-turn knob translation that ``codex_worker`` does inline. Keeping it here as
pure functions makes the mapping (Hitch settings -> SDK options) unit-testable
without spawning a worker.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    McpSdkServerConfig,
    PermissionMode,
    PermissionResultAllow,
    PermissionResultDeny,
    SandboxSettings,
    ToolPermissionContext,
)

# Claude model ids offered for the Claude Code backend. The Codex backend pulls
# its list from the app-server; Claude has no equivalent listing API, so this
# static set is the source of truth for the settings dialog and validation.
CLAUDE_MODELS: tuple[tuple[str, str], ...] = (
    ("claude-opus-4-8", "Opus 4.8"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("claude-haiku-4-5-20251001", "Haiku 4.5"),
)
DEFAULT_CLAUDE_MODEL = CLAUDE_MODELS[0][0]
VALID_CLAUDE_MODELS = {value for value, _label in CLAUDE_MODELS}

# Tools that only read state. They are auto-approved (added to allowed_tools)
# so the interactive ``can_use_tool`` callback only ever fires for actions that
# mutate the workspace or run commands -- mirroring Codex's auto-reviewer, which
# escalates command execution and file changes but not reads. ``ExitPlanMode``
# is deliberately excluded: it is the plan-mode approval boundary, so it must
# reach ``can_use_tool`` for the user to approve leaving plan mode.
READ_ONLY_TOOLS: tuple[str, ...] = (
    "Read",
    "Glob",
    "Grep",
    "NotebookRead",
    "TodoWrite",
    "WebFetch",
)
# Tools that write to the workspace. Disallowed entirely under the read-only
# sandbox policy.
WRITE_TOOLS: tuple[str, ...] = (
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
)
_WEB_SEARCH_TOOL = "WebSearch"
_WEB_FETCH_TOOL = "WebFetch"
_BASH_TOOL = "Bash"

# Codex sandbox policy strings (cookie/CLI values) -> behaviour.
SANDBOX_READ_ONLY = "readOnly"
SANDBOX_WORKSPACE_WRITE = "workspaceWrite"
SANDBOX_DANGER_FULL_ACCESS = "dangerFullAccess"

# Codex approval mode strings.
APPROVAL_APPROVE_ALL = "approve_all"
APPROVAL_DENY_ALL = "deny_all"

# Codex reasoning-effort strings that line up with a Claude effort level. Other
# values (e.g. Codex's "minimal") are dropped so the CLI uses its default.
_EFFORT_VALUES = frozenset({"low", "medium", "high", "xhigh", "max"})

CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    "Any",  # Awaitable[PermissionResultAllow | PermissionResultDeny]
]


def claude_bin() -> str | None:
    """Resolve the local ``claude`` CLI, or None if it is not on PATH."""
    return shutil.which("claude")


def map_effort(reasoning_effort: str | None) -> str | None:
    if not reasoning_effort:
        return None
    value = reasoning_effort.strip().lower()
    return value if value in _EFFORT_VALUES else None


def resolve_permission_mode(
    *, plan_mode: bool, sandbox_policy: str | None, approval_mode: str | None
) -> PermissionMode:
    """Map Hitch's plan/sandbox/approval knobs onto a Claude permission mode."""
    if plan_mode:
        return "plan"
    # ``dangerFullAccess`` only widens filesystem access; it must not bypass the
    # approval gate. Only an explicit ``approve_all`` removes the gate.
    if approval_mode == APPROVAL_APPROVE_ALL:
        return "bypassPermissions"
    return "default"


def resolve_tool_lists(
    *, sandbox_policy: str | None, web_search_mode: str | None
) -> tuple[list[str], list[str]]:
    """Return ``(allowed_tools, disallowed_tools)`` for a turn.

    Read-only tools are auto-approved. When web access is switched off, both
    ``WebSearch`` and ``WebFetch`` are blocked (``WebFetch`` can also reach
    external pages). The read-only sandbox blocks write tools outright.
    """
    allowed = list(READ_ONLY_TOOLS)
    disallowed: list[str] = []
    web_search_off = bool(web_search_mode) and web_search_mode in {"disabled", "off"}
    if web_search_off:
        disallowed.extend([_WEB_SEARCH_TOOL, _WEB_FETCH_TOOL])
        allowed = [tool for tool in allowed if tool != _WEB_FETCH_TOOL]
    else:
        allowed.append(_WEB_SEARCH_TOOL)
    if sandbox_policy == SANDBOX_READ_ONLY:
        # Block file-edit tools AND Bash: a shell command can mutate the
        # workspace just as a write tool can, so a read-only session must deny
        # both regardless of the approval mode.
        disallowed.extend(WRITE_TOOLS)
        disallowed.append(_BASH_TOOL)
        allowed = [tool for tool in allowed if tool != _BASH_TOOL]
    return allowed, disallowed


def resolve_sandbox_settings(sandbox_policy: str | None) -> SandboxSettings | None:
    """Map a Codex sandbox policy onto Claude ``SandboxSettings``.

    ``workspaceWrite`` confines edits to the repo, so the bash sandbox is
    enabled to keep approved/auto-approved shell commands from reaching the
    host filesystem outside ``cwd``. ``readOnly`` already blocks Bash and the
    write tools in :func:`resolve_tool_lists`, and ``dangerFullAccess`` is the
    deliberate opt-out, so neither needs a sandbox here. ``None`` (Codex
    default) leaves the SDK at its own default.
    """
    if sandbox_policy == SANDBOX_WORKSPACE_WRITE:
        return SandboxSettings(enabled=True)
    return None


def build_options(
    *,
    cwd: str,
    model: str | None,
    reasoning_effort: str | None = None,
    sandbox_policy: str | None = None,
    approval_mode: str | None = None,
    web_search_mode: str | None = None,
    plan_mode: bool = False,
    base_instructions: str | None = None,
    output_schema: dict[str, Any] | None = None,
    resume_session_id: str | None = None,
    session_id: str | None = None,
    mcp_server: McpSdkServerConfig | None = None,
    can_use_tool: CanUseTool | None = None,
) -> ClaudeAgentOptions:
    """Assemble ``ClaudeAgentOptions`` for one worker turn."""
    allowed, disallowed = resolve_tool_lists(
        sandbox_policy=sandbox_policy, web_search_mode=web_search_mode
    )
    mcp_servers: dict[str, Any] = {}
    if mcp_server is not None:
        mcp_servers[mcp_server["name"]] = mcp_server
        from hitch.main.claude_tools import PROPOSE_SESSION_TOOL_NAME

        allowed.append(PROPOSE_SESSION_TOOL_NAME)

    system_prompt: Any = None
    if base_instructions and base_instructions.strip():
        system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": base_instructions,
        }

    options = ClaudeAgentOptions(
        cwd=cwd,
        model=model or None,
        permission_mode=resolve_permission_mode(
            plan_mode=plan_mode,
            sandbox_policy=sandbox_policy,
            approval_mode=approval_mode,
        ),
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        mcp_servers=mcp_servers,
        # Load user/project/local settings so CLAUDE.md memory and project MCP
        # config apply, matching how a developer's own ``claude`` runs behave.
        setting_sources=["user", "project", "local"],
        can_use_tool=can_use_tool,
    )
    if can_use_tool is not None:
        # The Python SDK requires a PreToolUse hook alongside ``can_use_tool``;
        # without one the control stream can close before the permission
        # callback is invoked. The hook just lets execution continue so the
        # callback decides.
        options.hooks = {"PreToolUse": [HookMatcher(hooks=[_continue_pre_tool_use])]}
    if system_prompt is not None:
        options.system_prompt = system_prompt
    effort = map_effort(reasoning_effort)
    if effort is not None:
        options.effort = effort  # type: ignore[assignment]
    sandbox = resolve_sandbox_settings(sandbox_policy)
    if sandbox is not None:
        options.sandbox = sandbox
    if output_schema is not None:
        options.output_format = output_schema
    if resume_session_id:
        options.resume = resume_session_id
    elif session_id:
        # Fix the session id on the first run so the CodexInstance.thread_id and
        # the Claude transcript id agree without a discovery round-trip.
        options.session_id = session_id
    cli = claude_bin()
    if cli:
        options.cli_path = cli
    return options


async def _continue_pre_tool_use(
    _input_data: Any, _tool_use_id: str | None, _context: Any
) -> Any:
    """Let tool execution proceed to the ``can_use_tool`` permission callback."""
    return {"continue_": True}


def allow_result(updated_input: dict[str, Any] | None = None) -> PermissionResultAllow:
    return PermissionResultAllow(updated_input=updated_input)


def deny_result(message: str) -> PermissionResultDeny:
    return PermissionResultDeny(message=message)
