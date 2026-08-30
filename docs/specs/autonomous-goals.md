# Autonomous Goals Spec

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
- AGs do not apply work directly to the user's local branches.

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
- `AG-auto-qa-guidance`: When an accepted AG session has Auto-QA enabled, its
  completed implementation turn starts one normal coding turn that recommends
  the optional native `hitch_reviewer` subagent. Hitch registers the role in
  the promoted coding worker's app-server configuration before resuming the
  candidate; it does not require delegation or start a hidden QA
  verdict-and-repair loop.
- `AG-stack-default`: If stack depth is unset, the effective stack depth is `1`.
- `AG-stack-depth-range`: Supported stack depth values are integers from `1`
  through `100`. Hitch must reject stack depth values above `100`.
- `AG-no-budget-single-run`: If token budget is unset, Hitch runs at most one background system
  session for a no-proposal attempt, then stops.

### 4.2 Background Execution

- `AG-system-sessions`: AGs spawn Hitch-owned system sessions to produce Proposals.
- `AG-noninteractive`: AG system sessions run without user intervention or interactive
  approval prompts.
- `AG-tool-driven-protocol`: New AG workflows are driven by role-scoped Hitch
  tools. Candidate and judge final prose is transcript content only and never
  changes workflow state.
- `AG-candidate-tools`: Candidate sessions receive only the AG candidate tools:
  read the current goal, list prior goal sessions, submit a candidate for
  judgment, and finish with no proposal. The tools infer the current workflow;
  they do not accept a goal, workflow, session, or thread identifier.
- `AG-history-sessions`: Listing goal history returns lightweight metadata and
  the Codex rollout path for prior candidate and accepted sessions in the
  current AG lineage. It excludes judge and unrelated system sessions. Hitch
  does not summarize those transcripts; the candidate may inspect any returned
  rollout directly with ordinary read-only filesystem operations.
- `AG-judge-tools`: Every judge session receives only `approve` and `deny`.
  Both record a confidence and may include feedback. Judge final prose is
  ignored.
- `AG-judge-limit`: A candidate may request judgment at most twice. The first
  denial is returned to the candidate so it can address the feedback before
  its final request. After a second denial, only no-proposal completion is
  available.
- `AG-judge-snapshot`: Requesting judgment snapshots the exact candidate
  checkout and retains it with a Hitch-owned Git ref before starting a
  read-only judge in a worktree pinned to that commit. Approval publishes the
  judged candidate data and snapshot, not later candidate mutations. Hitch
  releases the ref only after the snapshot is transferred to an accepted
  session or the proposal is denied, replaced, rejected, or dismissed.
- `AG-protocol-recovery`: If a candidate turn finishes without requesting
  judgment or declaring no proposal, Hitch resumes the same hidden thread with
  a state-aware protocol reminder. Hitch permits at most three such recovery
  turns per candidate. A judge that finishes without approving or denying gets
  one protocol reminder; a second omission becomes a failed judgment and is
  returned to the waiting candidate.
- `AG-role-isolation`: Candidate and judge tools are immutable thread-scoped
  capabilities. Candidate and judge threads always remain hidden and are never
  promoted to user sessions.
- `AG-in-flight-upgrade`: Workflows created before the tool-driven protocol are
  retired when their worker next finishes, or when Hitch detects a stranded
  spawn. Hitch records a failure notice and cleans up the obsolete candidate
  checkout; it does not retain the structured-output candidate, judge, memory,
  or transcript-summarization implementation.
- `AG-background-queue`: AG-owned background sessions are globally queued so only one executes
  at a time.
- `AG-manual-start-admission`: Manual Run and Run all do not create durable queued work.
  They request admission to the global AG queue, start one eligible background session if
  the queue is idle, and show visible retry feedback if another AG is already running.
- `AG-independent-lifecycles`: Queueing does not make AG lifecycles dependent. Each AG remains
  independent except for the shared one-at-a-time execution queue.
- `AG-stack-continuation`: A token budget allows repeated background attempts,
  even when effective stack depth is `1`. Hitch may retry a failed candidate
  turn while budget remains. If a Proposal is produced and stack depth is
  greater than `1`, Hitch may continue from prior background work until it
  reaches stack depth, exhausts token budget, produces a user-actionable
  Proposal, or hits a terminal failure.
- `AG-stack-limit`: AGs must not continue past configured stack depth.
- `AG-budget-limit`: AGs must not start another automatic background session when doing so
  would exceed the configured token budget.
- `AG-no-proposal-terminal`: Calling `hitch.no_proposal` is an explicit terminal
  decision for the current AG workflow. Hitch records the notice or publishes
  the preceding completed stack proposal and does not spend remaining budget on
  another candidate.

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
- `AG-proposal-acceptance`: Accepting a Proposal starts a fresh normal
  user-visible thread with the proposed work and prompt. Implementation
  proposals start in a fresh worktree at the approved snapshot; propose-only
  results start in the selected repository. Acceptance never resumes or
  reveals the hidden candidate thread.
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
- `AG-state-taxonomy`: The UI distinguishes these primary states: Ready, No
  Quota, Queued, Running, Waiting, Blocked, Stopped, and Failed. No Quota means
  automatic work is paused by quota; Queued means the AG is waiting in the
  one-at-a-time background queue; Waiting means a Proposal is in the Inbox.
- `AG-state-detail`: State detail explains the next expected action, such as waiting for
  quota recovery, waiting in the background queue, waiting for Inbox action,
  waiting for accepted session completion, or requiring manual retry.

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
- `AG-accept-stack-depth-limit`: An AG with stack depth `3` creates or continues
  no more than three stack levels. Failed-turn retries consume token budget but
  do not count as new stack levels.
- `AG-accept-budget-retries`: An AG with stack depth `1` and a token budget can
  retry failed candidate turns until it produces a Proposal, exhausts token
  budget, receives `no_proposal`, or hits a terminal failure.
- `AG-accept-low-quota-auto`: Low quota prevents automatic AG starts and shows a low-quota reason.
- `AG-accept-manual-quota-override`: Manual Run can start an AG below the automatic quota threshold.
- `AG-accept-background-queue`: Multiple eligible AGs execute only one AG-owned background session at a
  time. Automatic starts waiting behind active AG work show queued state; manual starts
  show visible retry feedback when the queue is occupied.
- `AG-accept-run-all-admission`: Run all starts at most one eligible AG per request and leaves
  the remaining manual goals eligible for a later request.
- `AG-accept-accepted-session-block`: Accepting a Proposal starts a user-visible session and blocks only the
  producing AG until the session is Done or archived.
- `AG-accept-inactive-accepted-session-block`: If the accepted session is inactive but not Done or archived, the AG
  page clearly explains that the AG remains blocked.
- `AG-accept-visible-state`: Every visible AG has a citeable state and detailed explanation.
