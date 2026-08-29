# Agent Execution Permissions Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Define how Hitch controls local Codex execution, escalation approvals, and non-interactive system-session behavior.

### 1.2 Definitions

- Sandbox policy: The file and execution boundary for a session: Codex default, read-only, workspace-write, or full access.
- Approval mode: How Hitch resolves Codex escalation requests.
- Escalation: A Codex request for Hitch to approve, decline, cancel, or amend a command/file-change decision.
- User session: A user-visible Codex session.
- System session: A Hitch-owned background session for workflows such as Autonomous Goals or PR monitoring.
- Separate worktree: A git worktree distinct from the associated user session's active checkout.

## 2. Goals and Non-Goals

### 2.1 Goals

- Give users clear sandbox and approval controls.
- Make `Approve all (dangerous)` explicit: no prompts, no policy amendments, and fail closed when plain acceptance is unavailable.
- Keep prompts predictable across Auto review, Always prompt, Deny all, and Approve all.
- Keep system sessions least-privileged, isolated when write-capable, and non-interactive.
- Persist escalation decisions safely and race-free.

### 2.2 Non-Goals

- Replace Codex's internal sandbox or reviewer.
- Prompt users for system-session escalations.
- Relax sandboxing because an escalation failed.

## 3. User Stories

- As a user, I can choose read-only, workspace-write, or full-access execution.
- As a user, I can choose Auto review, Always prompt, Deny all, or Approve all behavior.
- As a user, I can change a user session's approval behavior without restarting it.
- As a user, background workflows never interrupt me with approval prompts.
- As a workflow author, I can run read-only system analysis and inherit permissions only when writes and isolation require it.

## 4. Requirements

### 4.1 Sandbox Policies

- `PERM-sandbox-options`: Hitch exposes Codex default, Read only, Workspace write, and Danger - full access.
- `PERM-sandbox-default`: If unset, Hitch lets Codex use its default sandbox; managed worktree sessions may default to Workspace write.
- `PERM-read-only`: Read only sessions cannot write unless Codex offers and Hitch sends an accepting escalation decision.
- `PERM-workspace-write`: Workspace write sessions may read/write inside the workspace; outside actions require an offered accepting decision.
- `PERM-danger-full-access`: Danger - full access intentionally allows unrestricted local execution permitted by Codex and the host.
- `PERM-danger-labeling`: Full-access controls must be visibly dangerous in the UI.

### 4.2 Approval Modes

- `PERM-approval-options`: Hitch exposes Auto review (default), Always prompt for approval, Deny all escalations, and Approve all (dangerous).
- `PERM-auto-review`: Auto review uses Codex's reviewer; user sessions prompt only when Codex routes the escalation to Hitch.
- `PERM-prompt-user`: Always prompt routes every command/file-change escalation to a user-visible prompt.
- `PERM-deny-all`: Deny all sends an offered non-accepting decision, or fails closed if none is offered, with no prompt.
- `PERM-approve-all`: Approve all accepts only offered plain, non-persistent accepting decisions, with no prompt.
- `PERM-approve-all-no-prompts`: Approve all must not create approval prompts.
- `PERM-approve-all-no-policy-amendment`: Approve all must not send structured policy-amending decisions; if plain acceptance is unavailable, it fails closed.

### 4.3 Scope and Inheritance

- `PERM-global-defaults`: Settings-page sandbox and approval choices apply to new user sessions and future turns unless overridden.
- `PERM-session-approval-override`: A user session may override the global approval mode.
- `PERM-proposal-accept-permissions`: Accepted Proposals re-resolve sandbox and approval as user sessions; system-only state must not carry over.
- `PERM-live-approval-update`: Updating a running user session's approval mode affects future Hitch-routed escalations; non-routed server-side decisions use the new mode on the next turn or routed escalation.
- `PERM-live-approval-resolve`: Switching to Deny all or Approve all resolves compatible pending Hitch prompts according to the new mode.
- `PERM-sandbox-start-scope`: Sandbox policy is selected when a session or turn starts; later setting changes do not affect that turn except through offered escalation decisions.

### 4.4 User Escalation Flow

- `PERM-escalation-triggers`: Hitch handles Codex command-execution and file-change approval requests.
- `PERM-pending-request`: Interactive escalations create durable pending requests and emit session events.
- `PERM-pending-replay`: Session views render unresolved pending requests from durable state on load; live events are only notifications.
- `PERM-prompt-detail`: Prompts show enough command/file-change, session, project, and workspace/target context for an informed decision.
- `PERM-prompt-policy-amendment`: Policy-amending accepting decisions must disclose scope and persistence and require explicit confirmation.
- `PERM-available-decisions`: Hitch may offer only decisions Codex included in the escalation request.
- `PERM-resolved-decision-validation`: Hitch validates every resolved decision against the request's offered decisions; unavailable decisions fail closed.
- `PERM-automatic-offered-decisions`: Automatic resolutions send only offered decisions; if the intended decision is unavailable, Hitch uses the safest offered non-accepting decision or fails closed.
- `PERM-structured-decisions`: Structured decisions, including policy amendments, must be validated before being sent to Codex.
- `PERM-decision-options`: User prompts support accept, decline, and cancel when available.
- `PERM-timeout-decline`: Unanswered interactive approvals resolve with an offered non-accepting decision or fail closed.
- `PERM-stop-decline`: Stopping a session blocked on approval resolves the request with an offered non-accepting decision or fails closed.
- `PERM-stop-quiet-turn`: Stopping an active session requests Codex cancellation without waiting for another stream event, including while a command is running silently; a later Stop may force-kill a turn that does not cancel.
- `PERM-decision-race-safe`: Concurrent decisions for one request resolve exactly once; later attempts receive already-resolved results.

### 4.5 System Sessions

- `PERM-system-read-only-preferred`: System sessions use Read only when writes are unnecessary, including AG summarization, evaluation, classification, synthesis, and monitoring.
- `PERM-system-inherit-sandbox`: Write-capable system sessions with an associated user session use a sandbox no broader than that user session unless an explicit workflow policy allows more; writes require a separate worktree or equivalent isolation.
- `PERM-system-sandbox-consent-cap`: Without an explicit workflow policy, system sessions cannot exceed the associated user session's sandbox; they must downgrade, run read-only, or fail closed.
- `PERM-system-sandbox-fresh-consent`: Delayed system sessions must re-check current sandbox settings and workflow policy immediately before using write-capable or broader sandboxing.
- `PERM-system-danger-full-access`: System sessions must not inherit Danger - full access by default; full access requires an explicit product requirement and isolation boundary.
- `PERM-system-inherit-approval-limits`: System sessions may inherit Approve all only through `PERM-system-approve-all-worktree`; Deny all becomes non-interactive denial or fail-closed.
- `PERM-system-approve-all-fresh-consent`: System sessions may inherit Approve all only after re-checking current approval mode and workflow policy immediately before start.
- `PERM-system-approve-all-revocation`: If current approval mode and workflow policy no longer permit Approve all, the system session denies escalations or fails closed.
- `PERM-system-safe-fallback`: Without an associated user session, write-capable system sessions use the narrowest safe sandbox plus isolation, or run read-only/fail closed.
- `PERM-system-sandbox-inputs`: System-session inputs kept outside the workspace must use private ephemeral storage visible to the selected sandbox and must be reclaimed after completion or a bounded crash-recovery window.
- `PERM-system-no-associated-full-access`: System sessions without an associated user session cannot use Danger - full access unless an explicit product requirement and isolation boundary allow it.
- `PERM-system-no-user-escalation`: System sessions never surface execution approval prompts to users.
- `PERM-system-escalation-denied`: System-session escalations are denied unless covered by `PERM-system-approve-all-worktree`; system sessions must not fall back to server-side Auto review approval.
- `PERM-system-deny-all-inheritance`: If the associated user session uses Deny all, the system session denies or fails closed unless the Approve all worktree exception applies.
- `PERM-system-approve-all-worktree`: System sessions may inherit Approve all only when created in a separate worktree before execution and scoped to that worktree or an explicit product exception.
- `PERM-system-approve-all-no-primary`: System sessions must not inherit Approve all while running in the associated user session's active checkout.
- `PERM-system-approve-all-outside-denied`: Inherited Approve all must deny effects outside the separate worktree, including host, process, remote, repository-hosting, deployment, and network-side effects, unless an explicit scoped product exception allows them.
- `PERM-system-visible-failure`: Denied escalations and permission/sandbox fail-closed paths surface as workflow state or Inbox notices; session-visible failure is enough only for user-visible sessions.

### 4.6 UX Requirements

- `PERM-settings-copy`: Settings copy explains sandbox policy versus approval mode.
- `PERM-session-copy`: Session UI shows effective sandbox, approval mode, whether each follows defaults or overrides them, and dangerous labels where relevant.
- `PERM-danger-confirmation`: Dangerous settings require explicit selection and are never defaults or reset selections.
- `PERM-approval-events`: Session streams show approval-requested and approval-resolved events for interactive prompts.
- `PERM-automatic-approval-events`: Automatic approvals, denials, and fail-closed resolutions create audit entries on the session detail page without prompts.
- `PERM-hidden-automatic-approval-audit`: Hidden system-session automatic approval audit appears on that system session's detail page; workflow state or Inbox notices may link to it.
- `PERM-noninteractive-no-noise`: Non-interactive approvals/denials do not prompt, but denial and fail-closed failures remain diagnosable.

## 5. Success Criteria

- `PERM-accept-session-effective-sandbox-visible`: Users can see and change effective sandbox and approval behavior, with clear dangerous labels.
- `PERM-accept-sandbox-write-needs-acceptance`: Sandbox boundaries are enforced; writes or outside-workspace actions require an offered accepting decision.
- `PERM-accept-prompt-flow`: Interactive escalations show sufficient context, survive refresh from durable state, and apply the user's decision.
- `PERM-accept-approve-all-no-amendment`: Approve all uses only plain, non-persistent accept decisions offered by Codex; otherwise it fails closed instead of prompting or amending policy.
- `PERM-accept-user-decision-validated`: User and automatic decisions must be offered by Codex; unavailable, timed-out, stopped, or racing decisions fail safe.
- `PERM-accept-stop-quiet-turn`: A user can stop a turn that is blocked in a silent command without waiting for that command to emit output.
- `PERM-accept-proposal-permissions-reset`: Accepted Proposals re-resolve sandbox and approval as user sessions.
- `PERM-accept-automatic-approval-audit`: Non-interactive approvals, denials, and fail-closed outcomes are visible on the session detail page; hidden system-session outcomes can be reached from workflow state or Inbox.
- `PERM-accept-system-read-only`: System sessions are least-privileged: read-only when possible, isolated when write-capable, and never broader than allowed by user state or workflow policy.
- `PERM-accept-system-denies-escalation`: When a system session cannot auto-approve an escalation, Hitch denies or fails closed; it does not prompt the user or fall back to Auto review approval.
- `PERM-accept-system-approve-all-worktree`: When a system session inherits Approve all, it must run in a separate worktree; approved effects stay within allowed scope, and active-checkout or network-side effects are denied unless explicitly excepted.
