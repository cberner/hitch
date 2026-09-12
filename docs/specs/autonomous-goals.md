# Autonomous Goals Retirement

Status: Removed

Autonomous goals are no longer supported. Hitch does not create, schedule,
review, or publish background goal runs. Their pages, actions, tools, settings,
and goal configuration storage are removed.

Upgrades dismiss unresolved goal proposals and notices, retire active goal
workers and workflows without a shutdown grace delay, release owned snapshot
refs, and remove goal-specific proposal relations. Accepted
sessions, their worktrees, historical system-session logs, and token usage remain
available. Historical system threads stay hidden from the ordinary session list;
previously accepted sessions stay visible. Retired workflow records do not
contribute live waiting badges, health backlogs, or worktree ownership; active
workers and visible/proposed sessions retain their normal cleanup protections.

Ordinary coding-session proposals remain supported by the
[Inbox spec](inbox.md). Codex's own in-session goals and controls remain supported by the
[Session Goals spec](session-goals.md).
