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

# Context-window sizes (tokens) used to render the "X% of context" gauge for
# Claude sessions. Claude has no rollout file or app-server API that reports the
# active model's window, so these static figures stand in. They are the standard
# published API context windows; the larger 1M beta windows are not assumed.
CLAUDE_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}
_DEFAULT_CLAUDE_CONTEXT_WINDOW = 200_000


def context_window_for(model: str | None) -> int:
    """Return the context-window size (tokens) for a Claude model id."""
    if not model:
        return _DEFAULT_CLAUDE_CONTEXT_WINDOW
    return CLAUDE_MODEL_CONTEXT_WINDOWS.get(
        model.strip(), _DEFAULT_CLAUDE_CONTEXT_WINDOW
    )

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
# Runs a background script under Bash permission rules; gated with Bash.
_MONITOR_TOOL = "Monitor"
# Runs host commands natively (Windows, or ``CLAUDE_CODE_USE_POWERSHELL_TOOL=1``);
# gated like Bash so a read-only sandbox blocks command execution either way.
_POWERSHELL_TOOL = "PowerShell"
# Creates/switches an isolated git worktree (writing under ``.claude/worktrees``).
# Claude marks it as not requiring permission, so it never reaches
# ``can_use_tool`` -- meaning it bypasses both the cwd guard and the bash sandbox
# and can write outside ``cwd``. The only way to keep a confining sandbox
# authoritative is to disallow it outright unless the user opted into
# ``dangerFullAccess``.
_ENTER_WORKTREE_TOOL = "EnterWorktree"

# Codex sandbox policy strings (cookie/CLI values) -> behaviour.
SANDBOX_READ_ONLY = "readOnly"
SANDBOX_WORKSPACE_WRITE = "workspaceWrite"
SANDBOX_DANGER_FULL_ACCESS = "dangerFullAccess"

# Codex approval mode strings.
APPROVAL_APPROVE_ALL = "approve_all"
APPROVAL_DENY_ALL = "deny_all"
APPROVAL_AUTO_REVIEW = "auto_review"

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
    # ``bypassPermissions`` skips ``can_use_tool`` entirely. ``SandboxSettings``
    # only sandboxes *bash* -- the SDK's own Write/Edit tools are confined via the
    # permission callback, not the sandbox -- so bypassing under ``workspaceWrite``
    # would let approved file edits escape ``cwd``. Only the deliberate
    # ``dangerFullAccess`` opt-out fully bypasses; otherwise keep ``can_use_tool``
    # authoritative (it auto-approves under ``approve_all`` but confines edits).
    if (
        approval_mode == APPROVAL_APPROVE_ALL
        and sandbox_policy == SANDBOX_DANGER_FULL_ACCESS
    ):
        return "bypassPermissions"
    return "default"


def resolve_tool_lists(
    *, sandbox_policy: str | None, web_search_mode: str | None
) -> tuple[list[str], list[str]]:
    """Return ``(allowed_tools, disallowed_tools)`` for a turn.

    Read-only tools are auto-approved. Web access is blocked -- both
    ``WebSearch`` and ``WebFetch`` (which also reaches live external pages) --
    whenever the user did not opt into live web. Claude has no cached-search
    mode, so ``cached`` (like ``disabled``) must not grant live web access. The
    read-only sandbox blocks write tools outright.
    """
    allowed = list(READ_ONLY_TOOLS)
    disallowed: list[str] = []
    # Only an explicit ``live`` opt-in grants web. Claude has no cached-search
    # mode, and the Codex "default" (empty) setting must not silently grant live
    # external access -- so everything other than ``live`` blocks both WebSearch
    # and the (live-fetching) WebFetch.
    web_search_on = web_search_mode == "live"
    if web_search_on:
        allowed.append(_WEB_SEARCH_TOOL)
    else:
        disallowed.extend([_WEB_SEARCH_TOOL, _WEB_FETCH_TOOL])
        allowed = [tool for tool in allowed if tool != _WEB_FETCH_TOOL]
    if sandbox_policy == SANDBOX_READ_ONLY:
        # Block file-edit tools AND command-capable tools: a shell command can
        # mutate the workspace just as a write tool can, so a read-only session
        # must deny them regardless of the approval mode. ``Monitor`` runs a
        # background script under Bash permission rules, and ``PowerShell`` runs
        # host commands natively, so both are gated too -- otherwise ``approve_all``
        # would let them run commands despite the read-only sandbox.
        disallowed.extend(WRITE_TOOLS)
        disallowed.append(_BASH_TOOL)
        disallowed.append(_MONITOR_TOOL)
        disallowed.append(_POWERSHELL_TOOL)
        allowed = [tool for tool in allowed if tool != _BASH_TOOL]
    # ``EnterWorktree`` is auto-approved by Claude (it never reaches
    # ``can_use_tool``) and writes a new worktree that can land outside ``cwd``,
    # so it escapes both the read-only block above and the workspace-write cwd
    # guard / bash sandbox. Disallow it unless the user explicitly opted out of
    # confinement with ``dangerFullAccess`` -- otherwise a "read-only" or
    # workspace-confined session could still create files on the host.
    if sandbox_policy != SANDBOX_DANGER_FULL_ACCESS:
        disallowed.append(_ENTER_WORKTREE_TOOL)
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
        # enabled: confine bash command execution to the sandbox.
        # allowUnsandboxedCommands=False: a command must not escape the sandbox
        #   via ``dangerouslyDisableSandbox`` -- otherwise under approve_all
        #   (bypassPermissions) it could run unsandboxed with no approval gate,
        #   defeating the workspace-write boundary.
        # autoAllowBashIfSandboxed=False: keep Hitch's ``can_use_tool`` approval
        #   gate authoritative rather than auto-approving sandboxed bash.
        return SandboxSettings(
            enabled=True,
            allowUnsandboxedCommands=False,
            autoAllowBashIfSandboxed=False,
        )
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
    load_filesystem_settings: bool = True,
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
        # Hidden system-agent runs (QA/spec/autonomous) disable this: a repo
        # ``.claude/settings*.json`` can register shell *hooks* that run in the
        # SDK outside ``can_use_tool``, so an untrusted repo could execute
        # commands during a read-only/propose-only hidden run despite the tool
        # gating. Those runs get no filesystem settings at all.
        setting_sources=(
            ["user", "project", "local"] if load_filesystem_settings else []
        ),
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
        # The SDK only forwards a schema to the CLI (``--json-schema``) when
        # ``output_format`` is shaped ``{"type": "json_schema", "schema": ...}``;
        # a bare JSON Schema is silently ignored, leaving no ``structured_output``
        # for the workflow parser. Wrap it unless the caller already did.
        if output_schema.get("type") == "json_schema":
            options.output_format = output_schema
        else:
            options.output_format = {"type": "json_schema", "schema": output_schema}
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
