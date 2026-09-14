# Session Transcript

Status: Implemented

## Overview

The session transcript keeps the agent's narrative visible while reducing the
space used by repetitive reasoning, command, file-change, and web-search activity.

## Requirements

- `ST-cleanup-resume-coordination`: Disk-pressure cleanup serializes removal of
  each managed worktree with session lifecycle operations and worker startup,
  including sessions sharing that checkout. Recheck current session, worker,
  proposal, and PR-watch protections after acquiring the removal lease; skip
  busy checkouts. Starting a worker in a managed checkout already removed by
  cleanup fails before creating its worker row.
  Proposal registration and indexing observed sessions participate in the same
  coordination. Indexing a changed cwd holds both old and incoming checkout leases
  in stable path order. Removal locks use existing directory inodes so a full
  filesystem cannot prevent cleanup by blocking creation of new lock files.
  Protection rechecks use indexed checkout and recent-update lookups for session
  metadata and PR watches, retaining path aliases without rebuilding the entire
  session protection snapshot per removal.
- `ST-shared-assets`: Session styles and scripts load as external shared assets.
  Browsers must revalidate them on reuse; validators cover the current complete
  rendered content, including shared snippets. Session-specific configuration
  stays in the HTML and assets do not include request or user data.
- `ST-agent-picker`: Visible session detail pages offer an Agent drop-down with
  Main agent and its native Codex subagents, including nested and archived
  descendants. Use nicknames and roles when available. Unrelated threads and
  historical Hitch system sessions without native ancestry are excluded.
- `ST-agent-transcripts`: Selecting a subagent displays its read-only transcript
  on the same page. Preserve the main agent's draft and keep its stream separate.
  Switching back restores the main conversation and its controls. Selection
  survives a page refresh through the `agent` query parameter. Large subagent
  previews offer a full transcript with commands, reasoning, and web searches.
- `ST-agent-refresh`: Discover newly spawned agents while the main turn runs and
  refresh the selected subagent's latest messages. Older messages load on demand;
  browsing older history pauses automatic transcript refresh until Return to
  latest is selected. A parent finishing defers its page reload while a subagent
  is selected, until switching to Main agent or choosing Return to latest.
  Discovery or read failures show a retryable message without
  blocking the main conversation. Viewing agents never resumes their threads.
- `ST-user-questions`: Blocking user input is available in ordinary Default
  sessions as well as Plan mode. Pending questions remain answerable until
  the user responds or stops the worker; elapsed time must not supply an
  empty answer or resume the agent. Approval mode does not answer questions.
  Hidden background workers do not enable Default-mode user input.
- `ST-nonblocking-questions`: Honor Codex's `isBlocking` distinction. A
  nonblocking question leaves generation, streamed progress, steering, and
  subsequent questions available. Blocking questions still wait for a real
  response while leaving the transport responsive. Missing `isBlocking` keeps
  legacy blocking presentation. Pending nonblocking questions do not mark a
  running session as waiting for input.
- `ST-async-message-questions`: Also recognize structured `agentMessage`
  notifications with `delivery: "async"` and title/options questions. Present
  them through the same nonblocking question controls, replacing their plain
  text live message. Deliver explicit answers as user input to the originating
  turn, with the question titles as context; Skip sends no input. Duplicate
  notifications do not duplicate questions or answers, including when controls
  replay alongside rollout-rendered or lazily loaded history. A rejected
  answer displays a delivery failure without disrupting unrelated work or
  questions. Ordinary messages,
  including older event logs without question controls, stay visible.
- `ST-question-runtime`: The pinned SDK and its bundled Codex executable
  support nonblocking questions without a separately installed CLI. Verify the
  bundled app-server's question schema as part of the test suite.
- `ST-question-answers`: Accept string answers, lists of strings, and Codex's
  canonical `{"answers": ["text"]}` value for each question. Reject other
  shapes with HTTP 400 before resolving the request, so a successful submission
  never silently drops an answer on its way to Codex.
- `ST-question-lifecycle`: Each question request has an independent durable
  response. Multiple requests may remain pending and be answered in any order;
  reconnecting replays their state without duplicating controls. Never answer
  automatically on a timer. A server cancellation, completed originating turn,
  stopped worker, or closed transport closes unanswered questions. A closed
  request rejects later answers and must not appear to have been answered.
- `ST-question-workflows`: Show whether a question needs an answer before
  continuing or can be answered while Codex works. Keep a pending-question
  shortcut near the composer, support multiline free-text answers, and preserve
  the main composer draft when questions arrive or are answered. Suggested
  choices may preselect the first option, but are not submitted automatically;
  Submit and Skip are explicit user actions.
- `ST-reader-isolation`: Displaying a session or refreshing its metadata must
  not acquire a Codex thread writer lease. Browser reads remain available
  before, during, and after a detached worker turn.
- `ST-concurrent-message-start`: Recheck active workers after acquiring the
  session lifecycle lock before starting any follow-up, Plan, review, or PR turn.
  If another worker has started and steering has not delivered the message,
  return a retryable conflict instead of launching a competing writer. Dispose
  of uploads owned by the rejected request.
  Message submissions, review/PR starts, and background PR follow-ups use one
  startup service that owns the lifecycle lease and active-worker check through
  settings resolution, worker creation, and acceptance or rollback bookkeeping.
  A busy background claim defers delivery; startup claims permit only one launch
  attempt and cannot be reused after their lease ends.
- `ST-archive-writer-conflict`: If another Codex process holds the session's
  writer lease, archiving or unarchiving preserves local state and asks the user
  to close the session in that process and retry. AJAX requests return 409;
  form submissions redirect to the session with an error message. Retrying
  after the writer releases the session applies the requested archive state.
  A failed Undo keeps its retry action available beyond the normal grace period.
- `ST-startup-failure`: A tracked session whose worker failed before creating
  readable history still displays its saved prompt and failure instead of
  returning a server error.
- `ST-turn-notices`: The main agent's live work strip shows a compact,
  expandable notice while Codex retries a model connection or checks a response
  before releasing it. Repeated updates replace the current notice. Retries end
  when model output resumes, including reasoning and plan generation. Response
  checks end when a message or new tool call is released, or buffering explicitly
  ends; reasoning and plan generation alone do not end a response check. Both
  notices clear when the turn ends. Heartbeats and background command output do
  not imply recovery. Replaying a worker log restores the current notice without
  leaving recovered warnings.
  These notices remain separate from Hitch's browser connection status and
  are hidden while viewing a subagent.
- `ST-capacity-retries`: A turn that fails with `serverOverloaded` keeps its
  worker active and automatically continues the existing task after 1 minute,
  doubling the delay on each capacity failure up to 30 minutes. Further capacity
  failures retry every 30 minutes until recovery or Stop. Show the next delay
  in the live retry notice. Preserve the current model, effort, and session
  policies; do not repeat the original prompt or attachments. Stop cancels the
  wait promptly. Messages submitted during backoff are queued for the next
  attempt, which reloads saved model and effort changes. Other errors retain
  their existing terminal behavior.
- `ST-turn-failures`: Terminal cybersecurity-policy and model
  connection failures use a clear title and next-step guidance, with the original
  Codex message in expandable details. Other failures retain their original
  message. Previously recorded capacity failures offer the existing model-and-effort
  dialog on writable sessions. Dialog actions never resend a prompt or change
  the model automatically.
- `ST-archive-missing-history`: A saved session without a rollout can be
  archived and restored in Hitch when Codex reports that its rollout is missing.
  Preserve its saved details and usage, and retain the local archive state
  across index refreshes. A successful Codex archive or unarchive returns
  authority to Codex. Unknown sessions and sessions with an existing rollout
  do not use this fallback; active-work and writer-conflict checks still apply.
- `ST-thinking-visible`: Thinking messages are always rendered as normal,
  top-level transcript entries and are never hidden by an activity toggle.
- `ST-instructions-visible`: Visible session pages expose the saved personal/
  project and Hitch developer instructions for the active or latest turn in
  a collapsible section. Legacy turns without separate Hitch instructions
  say none were recorded. User messages render their stored contents without
  stripping historical automatic instructions or other recognized text.
- `ST-activity-runs`: Each consecutive run of two or more Reasoning, Command,
  File change, and Web search messages is rendered as one collapsible group.
  Thinking messages and every other transcript entry end the current group.
- `ST-latest-default`: Activity groups are collapsed by default. A collapsed
  group shows its count, its toggle, and only its latest message; expanding the
  group reveals the earlier messages without duplicating the latest message.
- `ST-command-preview`: Persisted long commands initially include only a short
  preview, including the latest command in an activity group. Expanding a command
  fetches its full text with HTTP caching disabled; collapsing restores the short
  preview. Failed loads are retryable and obsolete responses cannot overwrite a
  later expansion. Command links identify the source call, so appended activity
  cannot change which command expands. This also applies to subagent and
  unindexed transcripts. SDK-only history and legacy commands without source
  identities stay inline.
- `ST-live-consistency`: Completed transcripts and live-streamed transcript
  updates use the same grouping boundaries and default state.
- `ST-replay-compaction`: Initial stream replay omits historical text deltas
  when the same agent, plan, or reasoning item has a completed snapshot in the
  replay window. Deltas for incomplete items remain available so reconnecting
  clients can recover their current text.
- `ST-bounded-diff-spool`: Live worker event logs omit cumulative
  `turn/diff/updated` snapshots because the active page does not expose a diff
  preview. After the turn, opening the diff builds its preview directly
  from the worktree. Disk-pressure cleanup removes these obsolete snapshots
  from oversized logs created by older Hitch versions.
- `ST-diff-on-demand`: Inactive session pages offer View diff without building
  or rendering a diff during the initial page request. Each dialog opening reads
  the current worktree with HTTP caching disabled. Show loading, empty, and
  retryable failure states; an older request must not overwrite a newer opening.
  Unreadable working trees and failed Git reads must never appear as a clean
  diff. A running turn keeps the diff unavailable.
- `ST-agent-math`: Agent and Thinking messages render TeX enclosed by explicit
  `\(...\)`, `\[...\]`, or `$$...$$` delimiters as mathematical notation.
  Rendering applies consistently to persisted history, lazily loaded entries,
  and a live agent message once that message is complete. A separate response
  metadata field is not required.
- `ST-math-safety`: Math is rendered only outside code elements. Single dollar
  signs remain prose because they are ambiguous, malformed TeX remains visible
  as its source, and the renderer does not trust TeX commands that request
  external resources or unsafe HTML.
- `ST-history-preview`: Large sessions initially render a bounded preview of
  recent persisted user and agent messages, including read-only historical
  system-session logs. Scrolling upward loads older preview pages.
- `ST-message-records`: Preview and full-history readers support both legacy
  message events and completed `UserMessage`/`AgentMessage` snapshots. Preserve
  user turn boundaries, image markers, and agent phases, and render assistant
  responses only once when also persisted as response items. Oversized snapshots
  use the same bounded preview placeholders as legacy messages.
- `ST-history-active-fallback`: If an active large-session worker's event log
  has not claimed its original user item, persisted messages remain visible in
  both preview and full-history renders. The rollout owns transcript rendering
  for that page lifecycle; SSE still replays the complete worker log for goal,
  plan, approval, and input state, but its transcript items remain hidden to
  avoid cross-source duplication. Older-history fragments and specialized live
  roots inherit that same owner rather than independently re-detecting it.
- `ST-history-full`: A visible up-arrow reloads the canonical full transcript
  and positions the reader at its beginning, including while a worker is
  active. Activity, synthesized rows, and oversized message bodies remain
  authoritative in this full view.
## Success Criteria

- A reader can scan every Thinking message without opening a toggle.
- Long consecutive runs of reasoning, commands, file changes, and web searches
  occupy one message row plus a compact toggle until expanded.
- Explicitly delimited mathematical notation is readable without exposing TeX
  control sequences, including in Thinking messages from long-running turns.
