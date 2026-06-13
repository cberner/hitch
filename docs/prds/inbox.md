# Inbox and Proposals PRD

Status: Draft

## 1. Overview

### 1.1 Purpose

The Inbox is Hitch's review surface for asynchronous work and automation
messages. It lets users decide what to do with proposed follow-up sessions and
acknowledge notices from Hitch background systems.

### 1.2 Definitions

- Inbox item: Any unresolved item shown in the Inbox.
- Proposal: An Inbox item that can become a normal user-visible session.
- Notice: A dismissible Inbox message that informs the user about automation
  state, such as automation failing or finding no proposal.
- Source session: The Codex session or system session that created the item.
- Candidate session: A background session whose work can be continued when a
  Proposal is accepted.
- Outcome: The final user decision for an Inbox item: accepted, rejected, or
  dismissed.

## 2. Goals and Non-Goals

### 2.1 Goals

- Give users a single place to review asynchronous work from Codex and Hitch
  automation.
- Let users accept useful work into a normal session, reject it with a reason,
  or dismiss it.
- Let Hitch surface non-proposal messages without forcing them into session
  acceptance flows.
- Preserve enough source metadata to explain where an item came from and clean up
  background resources safely.

### 2.2 Non-Goals

- The Inbox is not a general chat or notification center for every app event.
- Notices do not start sessions.
- Resolved Inbox items do not remain in the active Inbox.

## 3. User Stories

- As a user, I can see pending proposed sessions and automation messages in one
  Inbox.
- As a user, I can accept a Proposal and continue it as a normal session.
- As a user, I can reject a Proposal with feedback that future automation can
  use.
- As a user, I can dismiss notices such as automation failures or no-proposal
  messages.
- As Codex, I can create an Inbox Proposal when I identify useful follow-up
  work.

## 4. Requirements

### 4.1 Inbox Items

- `INBOX-item-kinds`: The Inbox supports at least two item kinds: Proposal and
  Notice.
- `INBOX-unresolved-only`: The active Inbox shows only unresolved items.
- `INBOX-project-visibility`: Inbox items respect the same visible-project
  filtering used by the session list.
- `INBOX-nav-count`: The primary navigation shows the count of visible unresolved
  Inbox items.
- `INBOX-source-metadata`: Inbox items preserve source metadata such as project,
  source session, source workflow, source automation, candidate session, judge
  session, outcome metadata, and relevant files when available.

### 4.2 Proposals

- `INBOX-proposal-purpose`: A Proposal represents a suggested follow-up session
  that the user can start, reject, or dismiss.
- `INBOX-proposal-content`: A Proposal includes title, summary, prompt,
  confidence, project, and relevant files when available.
- `INBOX-proposal-display`: The Inbox displays Proposal source, confidence,
  relevant files, stack labels, token usage, stack-stop reason, and candidate or
  judge logs when available.
- `INBOX-proposal-accept`: Accepting a Proposal starts or continues a normal
  user-visible session using the Proposal prompt and source context.
- `INBOX-proposal-candidate`: If a Proposal has a candidate session with a
  distinct worktree, accepting it continues from that candidate work instead of
  starting from the base project worktree.
- `INBOX-proposal-reject`: Rejecting a Proposal requires a user-provided reason.
- `INBOX-proposal-dismiss`: Dismissing a Proposal resolves it without requiring a
  reason.

### 4.3 Notices and Messages

- `INBOX-notice-purpose`: A Notice is an Inbox message that informs the user
  about automation state but cannot be accepted as a session.
- `INBOX-notice-dismiss-only`: The Inbox UI exposes only Dismiss for Notice
  items.
- `INBOX-notice-system-cleanup`: System cleanup may also resolve Notice items
  when their source automation is deleted or otherwise cleaned up.
- `INBOX-notice-logs`: Notices may link to candidate or judge logs when those
  logs help explain the automation result.

### 4.4 Codex-Created Proposals

- `INBOX-codex-tool`: Hitch exposes a `hitch.propose_session` tool that lets a
  Codex session create a Proposal Inbox item.
- `INBOX-tool-authorization`: Codex may use the proposal tool whenever it decides
  a follow-up session would be useful.
- `INBOX-tool-fields`: The tool requires title, summary, and prompt, and accepts
  optional relevant files and confidence.
- `INBOX-tool-project-match`: Tool-created Proposals are assigned to the Hitch
  project that matches the session cwd.
- `INBOX-tool-source-session`: Tool-created Proposals record the source Codex
  thread when available.
- `INBOX-tool-fallback`: If the dynamic tool is unavailable, Codex may use the
  `propose_session` management command through the configured Hitch environment
  fallback.

### 4.5 Outcomes and Concurrency

- `INBOX-outcome-one-way`: Final Inbox item outcomes are one-way transitions
  from unresolved to accepted, rejected, or dismissed.
- `INBOX-accept-start-claim`: Starting a Proposal may use a provisional accepted
  claim while the session is being created.
- `INBOX-accept-rollback`: If a provisional accepted claim expires or session
  creation fails, Hitch may roll the item back to unresolved so it remains
  actionable.
- `INBOX-no-reopen`: The Inbox must not allow a resolved item to be reopened by
  submitting an unresolved outcome, except for the provisional claim rollback in
  `INBOX-accept-rollback`.
- `INBOX-race-safe`: Concurrent accept/reject/dismiss attempts must resolve an
  item exactly once.
- `INBOX-accepted-metadata`: Accepted items record the user resolution and the
  accepted session id/thread when available.
- `INBOX-rejected-metadata`: Rejected items record the user resolution and
  rejection reason.
- `INBOX-dismissed-metadata`: Dismissed items record the user resolution.

### 4.6 Automation Integration

- `INBOX-ag-reference`: User-actionable, published AG-created Proposals and
  Notices appear in the Inbox, but AG-specific proposal, notice, blocking,
  retry, hidden intermediate stack records, and cleanup behavior is owned by
  [Autonomous Goals PRD](autonomous-goals.md).

### 4.7 UX Requirements

- `INBOX-empty-state`: The Inbox has a clear empty state when there are no visible
  unresolved items.
- `INBOX-visible-projects`: Users can adjust visible project filtering from the
  Inbox.
- `INBOX-do-it-dialog`: Starting a Proposal lets the user review or edit the
  starting message before creating the session.
- `INBOX-reject-dialog`: Rejecting a Proposal asks for a reason before resolving
  it.
- `INBOX-item-actions`: Proposal and Notice actions must match their item kind so
  notices cannot be accepted and proposals can be accepted, rejected, or
  dismissed.

## 5. Success Criteria

- `INBOX-accept-visible-session`: Accepting a Proposal creates or continues a
  user-visible session and removes the item from the active Inbox.
- `INBOX-reject-requires-reason`: Rejecting a Proposal without a reason is not
  allowed.
- `INBOX-notice-dismiss`: A Notice can be dismissed but cannot be accepted or
  rejected.
- `INBOX-notice-cleanup-success`: System cleanup can dismiss unresolved Notice
  items without requiring an Inbox UI action.
- `INBOX-accept-rollback-success`: If Proposal session creation fails or a
  provisional start claim expires, the item returns to the active Inbox.
- `INBOX-tool-creates-item`: A valid `hitch.propose_session` call creates a
  visible Proposal tied to the current Hitch project and source thread.
- `INBOX-resolution-race`: If two tabs resolve the same Inbox item, exactly one
  decision succeeds and the loser receives an already-resolved error.
- `INBOX-project-filter-success`: Hidden projects' unresolved Inbox items do not
  appear in the Inbox count or Inbox list.
