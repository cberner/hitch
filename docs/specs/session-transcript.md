# Session Transcript

Status: Implemented

## Overview

The session transcript keeps the agent's narrative visible while reducing the
space used by repetitive reasoning and command activity.

## Requirements

- `ST-thinking-visible`: Thinking messages are always rendered as normal,
  top-level transcript entries and are never hidden by an activity toggle.
- `ST-activity-runs`: Each consecutive run of two or more Reasoning and Command
  messages is rendered as one collapsible group. Thinking messages and every
  other transcript entry end the current group.
- `ST-latest-default`: Activity groups are collapsed by default. A collapsed
  group shows its count, its toggle, and only its latest message; expanding the
  group reveals the earlier messages without duplicating the latest message.
- `ST-live-consistency`: Completed transcripts and live-streamed transcript
  updates use the same grouping boundaries and default state.
- `ST-replay-compaction`: Initial stream replay omits historical text deltas
  when the same agent, plan, or reasoning item has a completed snapshot in the
  replay window. Deltas for incomplete items remain available so reconnecting
  clients can recover their current text.
- `ST-live-detail-authority`: When a live session requires detail sanitization,
  every tool-detail snapshot and delta is rendered through the same policy.
  Sensitive command, file, and reasoning details use safe placeholders.

## Success Criteria

- A reader can scan every Thinking message without opening a toggle.
- Long consecutive runs of reasoning and commands occupy one message row plus
  a compact toggle until expanded.
