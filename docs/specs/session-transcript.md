# Session Transcript

Status: Implemented

## Overview

The session transcript keeps the agent's narrative visible while reducing the
space used by repetitive reasoning, command, and web-search activity.

## Requirements

- `ST-thinking-visible`: Thinking messages are always rendered as normal,
  top-level transcript entries and are never hidden by an activity toggle.
- `ST-activity-runs`: Each consecutive run of two or more Reasoning, Command,
  and Web search messages is rendered as one collapsible group. Thinking
  messages and every other transcript entry end the current group.
- `ST-latest-default`: Activity groups are collapsed by default. A collapsed
  group shows its count, its toggle, and only its latest message; expanding the
  group reveals the earlier messages without duplicating the latest message.
- `ST-live-consistency`: Completed transcripts and live-streamed transcript
  updates use the same grouping boundaries and default state.
- `ST-replay-compaction`: Initial stream replay omits historical text deltas
  when the same agent, plan, or reasoning item has a completed snapshot in the
  replay window. Deltas for incomplete items remain available so reconnecting
  clients can recover their current text.
- `ST-bounded-diff-spool`: Live worker event logs omit cumulative
  `turn/diff/updated` snapshots because the active page does not expose a diff
  preview. After the turn, the page reload builds its stable preview directly
  from the worktree. Disk-pressure cleanup removes these obsolete snapshots
  from oversized logs created by older Hitch versions.
- `ST-live-detail-authority`: When a live session requires detail sanitization,
  every tool-detail snapshot and delta is rendered through the same policy.
  Sensitive command, file, and reasoning details use safe placeholders.
- `ST-history-preview`: Large sessions initially render a bounded preview of
  recent persisted user and agent messages, including read-only system and
  autonomous-goal logs. Scrolling upward loads older preview pages.
- `ST-history-active-fallback`: If an active large-session worker's event log
  has not claimed its original user item, persisted messages remain visible in
  both preview and full-history renders. The rollout owns transcript rendering
  for that page lifecycle; SSE still replays the complete worker log for goal,
  plan, approval, and input state, but its transcript items remain hidden to
  avoid cross-source duplication. Older-history fragments and specialized live
  roots, including the demo-agent view, inherit that same owner rather than
  independently re-detecting it; a selected demo-agent panel renders the
  rollout-owned active entries next to its hidden live transcript root.
- `ST-history-full`: A visible up-arrow reloads the canonical full transcript
  and positions the reader at its beginning, including while a worker is
  active. Activity, synthesized rows, and oversized message bodies remain
  authoritative in this full view.

## Success Criteria

- A reader can scan every Thinking message without opening a toggle.
- Long consecutive runs of reasoning, commands, and web searches occupy one
  message row plus a compact toggle until expanded.
