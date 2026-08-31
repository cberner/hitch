# Pull Request Watch Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Define the agent-invoked pull-request watch tool and the minimal durable state
that lets Hitch display and refresh a session's registered PR.

### 1.2 Definitions

- PR watch tool: The visible coding-agent tool named `hitch.watch_pr`.
- Watch invocation: One bounded, read-only observation loop run by Hitch when
  the coding agent invokes the tool.
- Registered PR: The PR identity and latest observation stored for a session by
  an eligible watch invocation.
- Actionable result: A watch result that reports a merge conflict, requested
  review change, failed CI check, or newly observed review/PR feedback.
- Ready result: A watch result whose mergeability, review, and CI gates pass.

## 2. Goals and Non-Goals

### 2.1 Goals

- Let the coding agent publish with Codex's built-in capability and decide how
  to respond to watch results.
- Keep GitHub polling inside a bounded tool call and return structured evidence
  to the invoking agent.
- Retain only the durable PR identity, gate/result snapshot, ownership, stage
  refresh, and post-merge Auto-pull state that the UI needs.
- Preserve ordinary steering, stopping, and session settings without a PR/QA
  state machine.

### 2.2 Non-Goals

- Hitch does not push branches, create or update PRs, decide whether feedback is
  valid, or launch repair turns.
- The watch tool does not edit files, commit, push, comment, resolve threads,
  merge, or otherwise mutate local or GitHub state.
- Hitch does not keep a publication, retry, backoff, monitor-agent,
  feedback-agent, steering, recovery, or failure-narration workflow.

## 3. Requirements

### 3.1 Tool Availability and Contract

- `PRWATCH-tool-registration`: Visible coding sessions register the dynamic
  tool `hitch.watch_pr`; hidden system-agent sessions do not.
- `PRWATCH-upgrade-capability`: Dynamic-tool availability is stored in the
  Codex thread. A pre-tool session that lacks `hitch.watch_pr` rejects a manual
  PR task with a clear instruction to start a new session; an automatic PR
  trigger remains unclaimed and does not start an unusable turn.
- `PRWATCH-url-input`: The tool accepts one required full GitHub pull-request
  URL and rejects non-PR URLs or missing repository working directories.
- `PRWATCH-read-only`: An invocation uses non-interactive `gh` reads for PR
  metadata, review threads, and status checks and performs no mutations.
- `PRWATCH-bounded`: An invocation polls while gates remain pending for at most
  30 minutes and bounds every individual GitHub command. Normal turn
  cancellation interrupts the polling wait.
- `PRWATCH-results`: The tool returns structured JSON with `status`, `summary`,
  `feedback`, `feedback_fingerprint`, `pr`, `gates`, and `blockers`. Status is
  one of `ready`, `terminal`, `action_required`, `attention`, or `timed_out`.
- `PRWATCH-return-conditions`: An invocation returns when the PR is terminal,
  all deterministic gates pass, an actionable gate is blocked, new feedback
  needs assessment, the watch times out, or a GitHub command fails.
- `PRWATCH-untrusted-input`: PR comments, review bodies, thread text, and CI
  details are identified as untrusted data. The coding agent assesses them
  before acting.
- `PRWATCH-repeat-suppression`: A registered session PR remembers its last
  feedback fingerprint so a repeated invocation does not immediately return
  the same pending feedback in a hot loop.

### 3.2 Registration and Ownership

- `PRWATCH-task-registration`: PR publication and PR watch tasks register
  durable UI state. An ordinary visible coding turn may also register when the
  requested PR is already the session's current registration or passes
  publication validation. Calls from review-guidance turns still return
  observations without claiming or altering the session's registered PR.
- `PRWATCH-publication-validation`: A publication task, or an ordinary coding
  turn establishing, replacing, or reviving a historical registered PR,
  verifies that the PR is open and that its head repository, branch, and commit
  match the publishing checkout, including configured SSH host aliases, before
  registration.
- `PRWATCH-validation-snapshot`: An ordinary coding turn binds publication
  validation to the exact registration snapshot observed before validation. If
  another invocation changes that record, or a newer visible turn is recorded
  after an absent-record snapshot, the older call fails safely and asks the
  agent to retry instead of replacing newer state.
- `PRWATCH-follow-up-identity`: A PR watch/fix turn may update only the PR
  already registered to its session. A publication turn or validated ordinary
  coding turn may replace an older identity after passing checkout validation.
- `PRWATCH-register-before-poll`: The eligible invocation atomically records
  the PR URL and number for the Hitch UI before entering the polling loop.
- `PRWATCH-exact-owner`: Each registration records the invoking instance and
  message index. An older instance cannot displace a newer owner or superseding
  turn, and a late result is discarded if a newer invocation has taken
  ownership, so stale polling cannot overwrite a newer PR or result.
- `PRWATCH-publishing-display`: While a new publication turn is active, Hitch
  hides an older registered PR from that session's current UI until that exact
  turn calls `hitch.watch_pr`. Registration then reveals the new or validated
  identity.

### 3.3 Agent-Driven Follow-Up

- `PRWATCH-agent-owner`: One ordinary visible coding turn publishes through
  Codex's built-in PR capability, invokes `hitch.watch_pr`, and owns the
  resulting watch/fix cycle.
- `PRWATCH-agent-decisions`: The coding agent decides whether feedback is
  valid, makes warranted changes, runs tests, commits and publishes follow-up
  changes, and chooses when to invoke the tool again.
- `PRWATCH-no-framework-loop`: `/pr`, `/pr-now`, `/fix-pr`, Auto-PR, and
  equivalent tasks do not launch a hidden monitor, parse a monitor verdict,
  schedule framework backoff, or launch a feedback repair turn.
- `PRWATCH-setting-inheritance`: Publication/watch turns retain the coding
  session's model, reasoning effort, developer instructions, sandbox, approval,
  memory, and web-search settings.
- `PRWATCH-user-control`: Steering and Stop use normal visible-turn behavior;
  no durable framework steering queue is created.

### 3.4 Durable Display and Refresh

- `PRWATCH-session-record`: Hitch stores at most one `SessionPullRequest` row
  per thread. It contains the registered handoff, latest gates/result,
  ownership, stage-refresh bookkeeping, and optional Auto-pull result rather
  than executable workflow state.
- `PRWATCH-lifecycle`: A later unrelated visible turn marks the registered PR
  historical so it no longer controls the current URL or stage. A PR watch
  follow-up preserves it, and a later successful task registration makes its
  validated identity current again.
- `PRWATCH-stage`: A registered open PR displays the PR stage. A refreshed or
  observed merged/closed PR displays Done: Merged or Done: Closed. An active
  tagged turn takes precedence as the QA or PR stage.
- `PRWATCH-stage-refresh`: Hitch may perform bounded, rate-limited, off-request
  `gh` reads to refresh a registered open PR's display state. Refreshes preserve
  PR identity and cannot introduce a different PR.
- `PRWATCH-neutral-turn-end`: Finishing a visible task without a ready or
  terminal result does not claim that deterministic gates passed; the durable
  snapshot and later stage refresh remain authoritative.
- `PRWATCH-auto-pull`: If the tool observes a merged PR, an enabled project may
  run post-merge Auto-pull against its separately verified default repository
  checkout. Failure or skip is recorded without altering the PR result.

### 3.5 Upgrade

- `PRWATCH-wrapper-retirement`: Upgrade copies the latest valid PR handoff for
  each thread from legacy PR/QA wrapper state into `SessionPullRequest` and
  retires every running or blocked wrapper.
- `PRWATCH-in-flight-upgrade`: Active user-visible wrapper turns are detached,
  tagged as review, publication, or watch tasks, and allowed to settle as
  ordinary turns. Obsolete hidden wrapper workers and their pending prompts are
  made terminal so removed routing cannot resume them.
- `PRWATCH-steering-upgrade`: Queued legacy steering text is retained only as
  audit data on the retired wrapper and is not replayed. The legacy steering
  model is removed.

## 4. Success Criteria

- `PRWATCH-accept-tool-driven`: The visible coding turn publishes the PR,
  registers it by invoking `hitch.watch_pr`, and follows it through that tool,
  with no Hitch publication path, hidden monitor, or feedback agent.
- `PRWATCH-accept-bounded`: Pending gates are polled inside the invocation and
  return `timed_out` after the bounded watch window.
- `PRWATCH-accept-agent-fix`: Actionable evidence returns to the invoking agent,
  which can fix, test, publish, and invoke the tool again in the same turn.
- `PRWATCH-accept-safe-state`: Only the exact owning invocation updates its
  session record; review-guidance calls and stale results cannot overwrite it,
  and an ordinary coding turn cannot establish a PR without validating the
  publishing checkout.
- `PRWATCH-accept-no-wrapper`: Review and PR shortcuts create ordinary turns and
  no `pr_qa` workflow or framework steering row.
