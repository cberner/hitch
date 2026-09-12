# Session Goals

Status: Implemented

Codex's native thread goal belongs to an ordinary coding session. It does not
create the retired Autonomous Goals scheduler or background review workflows.

- `SG-create`: The session menu offers Set goal when there is no goal. Starting or
  resuming a goal requires an idle, writable, unarchived session.
- `SG-panel`: An existing goal is a compact, expandable panel near the composer,
  including while idle. It shows the objective, native status, recorded work
  time, tokens used, and optional token budget. Subagent views hide the main
  agent's panel. Transcript pagination does not hide the goal or its controls.
- `SG-controls`: Expanded controls offer Pause, Resume, Edit budget, and Clear as
  appropriate. Pause interrupts current work and prevents goal continuation.
  Clear removes the objective without interrupting the current turn. Completed
  goals only offer Clear. To change an objective, clear it and set a new goal.
  Stop still interrupts a turn after its goal has been cleared. Only an explicit
  Pause is recorded as a clean interruption.
- `SG-accounting`: Creating a goal explicitly starts work. Budget changes preserve
  the objective, status, and native accounting. An empty budget removes the token
  limit. Budget-limited goals require a larger budget or no limit before resuming.
  There is no duration limit or automatic timed restart.
- `SG-worker`: Running changes go through the owning worker and require
  acknowledgement. Idle changes use the native goal API without resuming the
  thread. Starting and resuming use a detached worker that follows native
  continuation turns, with the session's model, effort, execution permissions,
  and instructions.
- `SG-persistence`: Goal-controlled sessions keep a stable Codex SQLite home so
  native state and accounting survive worker restarts. Agent-created goals are
  preserved before a temporary worker home is released; their controls become
  available when that turn finishes. Historical event logs and old pooled
  databases are not searched to recover goals. Workers and idle controls lease
  the dedicated home exclusively. Once a cleared goal's Codex process closes,
  its databases are removed; later workers reuse the empty dedicated home.
- `SG-live`: Live updates preserve expanded state and unsaved dialog and composer
  text. Expanded state and composer text survive goal start/pause/resume reloads.
  A turn ending defers its page refresh until the open goal dialog closes.
  Errors stay next to the controls. Read-only sessions have no goal controls.
