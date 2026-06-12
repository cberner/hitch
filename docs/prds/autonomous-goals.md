# Autonomous Goals PRD

Status: Draft

## 1. Overview

### 1.1 Purpose

Autonomous Goals (AGs) let Hitch pursue a user-provided project goal in the
background until there is a milestone worth showing the user.

The main milestone is a Proposal: a proposed session containing the work done so
far. Proposals appear in the Inbox for the user to accept, reject, or dismiss.

### 1.2 Definitions

- AG: A project-scoped background goal with goal text, autonomy settings, stack
  depth, optional token budget, and run configuration.
- System session: A Hitch-owned background Codex session spawned by an AG.
- Proposal: A proposed session in the Inbox containing AG-produced work and the
  prompt/context needed to continue it as a normal session.
- Stack: A bounded chain of AG system sessions that continue from prior
  background work.
- Token budget: A cap on total background tokens used by one AG run.
- Done state: A terminal Hitch session state such as `Done: Merged` or
  `Done: Closed`.

## 2. Goals and Non-Goals

### 2.1 Goals

- Let users delegate recurring or open-ended project work to Hitch.
- Keep background work bounded by quota, stack depth, and token budget.
- Avoid user interruption until there is a Proposal, failure, or blocked state.
- Make every AG's current state and reason visible in the UI.
- Keep AGs independent while serializing AG background execution.

### 2.2 Non-Goals

- AGs do not bypass the Inbox.
- AGs do not replace normal interactive sessions.
- AGs do not silently apply work to the user's primary branch unless an explicit
  auto-merge setting allows it.

## 3. User Stories

- As a user, I can create an AG so Hitch works toward a project goal while I do
  other work.
- As a user, I can see why an AG is running, queued, blocked, stopped, or ready.
- As a user, I can accept a Proposal to continue the background work as a normal
  session.
- As a user, I can bound AG work with stack depth and token budget.

## 4. Requirements

### 4.1 Configuration

- `AG-creation`: Users can create, edit, delete, and manually run project-scoped AGs.
- `AG-config`: Each AG stores goal text, autonomy, ambition, confidence threshold,
  web search, auto-QA, auto-proposal, stack depth, and optional token budget.
- `AG-stack-default`: If stack depth is unset, the effective stack depth is `1`.
- `AG-no-budget-single-run`: If token budget is unset, Hitch runs at most one background system
  session for a no-proposal attempt, then stops.

### 4.2 Background Execution

- `AG-system-sessions`: AGs spawn Hitch-owned system sessions to produce Proposals.
- `AG-noninteractive`: AG system sessions run without user intervention or interactive
  approval prompts.
- `AG-background-queue`: AG-owned background sessions are globally queued so only one executes
  at a time.
- `AG-independent-lifecycles`: Queueing does not make AG lifecycles dependent. Each AG remains
  independent except for the shared one-at-a-time execution queue.
- `AG-stack-continuation`: With stack depth greater than `1` and a token budget set, an AG may
  continue from prior background work until it reaches stack depth, exhausts
  token budget, produces a user-actionable Proposal, or hits a terminal failure.
- `AG-stack-limit`: AGs must not continue past configured stack depth.
- `AG-budget-limit`: AGs must not start another automatic background session when doing so
  would exceed the configured token budget.

### 4.3 Quota

- `AG-low-quota-blocks-auto`: Hitch must not automatically start an AG background session when quota
  is too low.
- `AG-low-quota-formula`: Quota is too low when actual remaining weekly quota is below `50%` of
  the linearly expected remaining quota for the current point in the weekly
  window.
- `AG-quota-guard-scope`: The quota guard applies to automatic starts, automatic stack
  continuations, automatic retries, and other non-manual AG-owned starts.
- `AG-manual-quota-override`: Manual Run is a user override and can start an AG even when quota is
  below the automatic-start threshold.
- `AG-quota-unverified`: If quota cannot be verified, automatic starts should fail safe and the
  UI should explain that quota could not be checked.

### 4.4 Proposals

- `AG-proposal-inbox`: AG-created Proposals appear in the Inbox.
- `AG-proposal-content`: A Proposal includes title, summary, prompt, confidence, relevant files,
  and metadata for AG, stack, and budget lineage.
- `AG-proposal-acceptance`: Accepting a Proposal starts a normal user-visible session with the
  proposed work and prompt.
- `AG-proposal-resolution`: Rejecting or dismissing a Proposal records the outcome and stops any
  background continuation tied to that Proposal.

### 4.5 Accepted Proposal Blocking

- `AG-accepted-session-block`: Once a Proposal is accepted, the producing AG must not run more
  background sessions until the accepted session is Done or archived.
- `AG-inactive-accepted-session-block`: If the accepted session is inactive but not Done or archived, the AG
  remains blocked indefinitely.
- `AG-accepted-block-visible`: The AG page must clearly show when `AG-accepted-session-block` or `AG-inactive-accepted-session-block` is the reason an
  AG is blocked.
- `AG-accepted-block-scope`: Accepted-session blocking applies only to the AG that produced the
  Proposal. Other AGs remain independently eligible, subject to `AG-background-queue`.

### 4.6 UI State

- `AG-state-badge`: Every AG row shows a state badge with a detailed reason.
- `AG-state-taxonomy`: The UI distinguishes at least: ready, queued, running, low quota,
  pending Proposal, accepted session active, accepted session inactive but not
  Done/archived, continuing stack, stack reached, token budget reached, no
  Proposal found, blocked, and failed.
- `AG-state-detail`: State detail explains the next expected action, such as waiting for
  quota recovery, waiting for Inbox action, waiting for accepted session
  completion, or requiring manual retry.

### 4.7 Cleanup and Failure Handling

- `AG-failure-surfacing`: AG failures surface as AG UI state and, when user action is needed,
  Inbox notices.
- `AG-delete-cleanup`: Deleting an AG stops future starts and cleans up or dismisses
  unresolved AG-owned background work.
- `AG-resource-cleanup`: AG-created worktrees and resources are cleaned up when no longer
  needed.
- `AG-preserve-accepted-work`: Accepted Proposal work must not be cleaned up while the accepted user
  session still depends on it.

## 5. Success Criteria

- `AG-accept-stack-default`: An unset stack depth displays and behaves as stack depth `1`.
- `AG-accept-no-budget-single-run`: An AG with no token budget runs exactly one background session for a
  no-proposal attempt, then shows a stopped/no-proposal state.
- `AG-accept-stack-limit`: An AG with stack depth `3` and sufficient token budget runs no more
  than three background system-session iterations.
- `AG-accept-low-quota-auto`: Low quota prevents automatic AG starts and shows a low-quota reason.
- `AG-accept-manual-quota-override`: Manual Run can start an AG below the automatic quota threshold.
- `AG-accept-background-queue`: Multiple eligible AGs execute only one AG-owned background session at a
  time, with queued state shown for waiting AGs.
- `AG-accept-accepted-session-block`: Accepting a Proposal starts a user-visible session and blocks only the
  producing AG until the session is Done or archived.
- `AG-accept-inactive-accepted-session-block`: If the accepted session is inactive but not Done or archived, the AG
  page clearly explains that the AG remains blocked.
- `AG-accept-visible-state`: Every visible AG has a citeable state and detailed explanation.
