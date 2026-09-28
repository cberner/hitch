# Autonomous Goals Retirement

Status: Removed

Hitch no longer creates, schedules, reviews, or publishes background goal runs.
Their pages, actions, tools, settings, and configuration storage are removed.

Upgrades discard retired workflow and agent-run records, automation worker
records and their approval/input records, and Inbox notices. No historical
system-session log or audit UI is retained. Historical migration files remain
so existing installations can upgrade.

Before removing the records, upgrades archive known retired system threads in
Hitch using a durable local archive override. Codex index refreshes must not
unarchive them or remove these overrides. Missing metadata is created from
retired worker records when necessary. Archived threads may appear when the
user enables Show archived; ordinary index views exclude them.

Accepted proposals remain ordinary user sessions and retain their current
archive state and checkout protections. Native Codex subagents remain excluded
from the main session list and available through the Agent picker. Retired
threads follow the ordinary archived-session checkout retention policy.

Ordinary coding-session proposals remain supported by the
[Inbox spec](inbox.md). Codex's own in-session goal events remain part of the
[session transcript](session-transcript.md).
