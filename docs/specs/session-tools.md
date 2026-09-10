# Session Tools Spec

Status: Draft

## 1. Overview

Hitch exposes immutable, thread-scoped dynamic tools. Visible coding sessions
receive user-session tools; selected hidden workflow roles receive only the
tools for that role.

## 2. Requirements

- `SESSIONTOOLS-quota-registration`: Newly created visible coding sessions
  register `hitch.get_codex_quota`, with no arguments. Hidden system-agent
  sessions do not receive it. Existing threads need a new session because
  dynamic tool registration is immutable.
- `SESSIONTOOLS-quota-fresh`: Every invocation synchronously requests
  `account/rateLimits/read` through a separate app-server transport. It must
  bypass Hitch's quota caches, background refresh, and refresh throttles, and
  must never fall back to cached data after an error.
- `SESSIONTOOLS-quota-result`: The tool returns JSON containing `fetched_at`
  (UTC response receipt time) and `rate_limits` with the account's primary and
  secondary windows. Each available window includes `used_percent`,
  `remaining_percent` (clamped to 0–100), `window_duration_mins`, and `resets_at`
  (Unix seconds). Missing windows and reset times remain null, never inferred
  as unlimited quota. Missing both windows or a failed request returns an error.
- `SESSIONTOOLS-quota-stopping`: The tool description tells agents to recheck
  quota during work governed by a user's quota threshold and stop when that
  threshold is reached. A failed check leaves quota unknown. This is a polling
  tool, not an automatic hard cap on usage between checks.
- `SESSIONTOOLS-rename-registration`: Newly created visible coding sessions
  register `hitch.rename_session`; hidden system-agent sessions do not. Dynamic
  tool registration is immutable, so sessions created before the tool was
  available require a new session to use it.
- `SESSIONTOOLS-rename-current`: The tool renames only the invoking session. Its
  input does not accept a session or thread identifier.
- `SESSIONTOOLS-rename-input`: The tool requires one `name` string, trims outer
  whitespace, rejects an empty result, and limits names to 200 characters.
- `SESSIONTOOLS-rename-persistence`: A successful call persists the name through
  Codex and then updates Hitch's cached session-list title.
- `SESSIONTOOLS-rename-failure`: If Codex rejects the rename because the thread
  is archived or unknown, the tool returns an error and does not update Hitch's
  cached title.
- `SESSIONTOOLS-role-registration`: A new thread's dynamic tools are selected
  from its purpose and agent kind when the thread is created. Ordinary hidden
  system sessions receive no tools, AG candidates receive only candidate AG
  tools, and AG reviewers receive only reviewer AG tools.
- `SESSIONTOOLS-role-authorization`: Tool handlers repeat the purpose, agent
  kind, workflow, invoking thread, active-run, and terminal-state checks. A
  registered tool is not authorization by itself.
- `SESSIONTOOLS-ag-candidate`: AG candidate tools are `hitch.get_goal`,
  `hitch.list_goal_sessions`, `hitch.review`, `hitch.propose_session`, and
  `hitch.no_proposal`. The candidate form of `hitch.propose_session` accepts no
  arguments and publishes only the candidate approved by `hitch.review`.
- `SESSIONTOOLS-ag-reviewer`: AG reviewer tools are `hitch.approve` and
  `hitch.deny`.
- `SESSIONTOOLS-ag-scope`: AG tool inputs never select another workflow, goal,
  or session. Scope comes exclusively from the invoking worker context.

## 3. Success Criteria

- `SESSIONTOOLS-quota-success`: Consecutive calls issue separate backend
  requests and reflect changed quota even when Hitch has cached a different
  value. A subsequent failed fetch returns an error without the earlier quota.
- `SESSIONTOOLS-rename-success`: Calling `hitch.rename_session` with a valid
  name changes the invoking session's persisted and cached names.
- `SESSIONTOOLS-rename-validation`: Invalid names fail without attempting a
  rename.
- `SESSIONTOOLS-role-isolation-success`: A candidate cannot invoke reviewer or
  visible-session tools, a reviewer cannot invoke candidate or visible-session
  tools, and a visible session cannot invoke AG workflow tools.
