# Pull Request Watch Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Define the agent-invoked pull-request watch tool that replaces Hitch's
framework-driven PR monitor and feedback loop.

### 1.2 Definitions

- PR watch tool: The visible coding-agent tool named `hitch.watch_pr`.
- Watch invocation: One bounded, read-only observation loop started when the
  coding agent invokes the PR watch tool.
- Actionable result: A watch result that reports a merge conflict, requested
  review change, failed CI check, or newly observed review/PR feedback.
- Ready result: A watch result whose mergeability, review, and CI gates pass.

## 2. Goals and Non-Goals

### 2.1 Goals

- Let the coding agent decide when to watch a PR and how to respond to results.
- Keep GitHub polling bounded and expose structured evidence to the invoking
  agent.
- Preserve PR follow-up, user steering, stopping, session settings, stage
  tracking, and post-merge Auto-pull without a hidden monitor agent.

### 2.2 Non-Goals

- Hitch does not decide whether review feedback is valid or automatically
  launch a repair turn.
- The watch tool does not edit files, commit, push, comment, resolve threads,
  merge, or otherwise mutate local or GitHub state.
- Hitch does not keep a framework-owned retry, backoff, monitor-agent, or
  feedback-agent loop.

## 3. Requirements

### 3.1 Tool Availability and Contract

- `PRWATCH-tool-registration`: Visible coding sessions register the dynamic
  tool `hitch.watch_pr`; hidden system-agent sessions do not.
- `PRWATCH-upgrade-compatibility`: Dynamic-tool registration is immutable
  thread metadata. A session created before `hitch.watch_pr` was available
  rejects PR workflow activation with a clear instruction to start a new
  session. An in-flight workflow on a removed monitor step is blocked during
  reconciliation instead of restarting hidden monitor work.
- `PRWATCH-url-input`: The tool accepts one required full GitHub pull-request
  URL and rejects non-PR URLs or missing repository working directories.
- `PRWATCH-read-only`: An invocation uses non-interactive `gh` reads for PR
  metadata, review threads, and status checks and performs no mutations.
- `PRWATCH-bounded`: An invocation polls while gates remain pending, for at
  most 30 minutes, and bounds each individual GitHub command.
- `PRWATCH-results`: The tool returns structured JSON with `status`, `summary`,
  `feedback`, `feedback_fingerprint`, `pr`, `gates`, and `blockers`. Status is
  one of `ready`, `terminal`, `action_required`, `attention`, or `timed_out`.
- `PRWATCH-return-conditions`: An invocation returns when the PR is terminal,
  all deterministic gates pass, an actionable gate is blocked, new feedback
  needs assessment, the watch times out, or a GitHub command fails.
- `PRWATCH-untrusted-input`: PR comments, review bodies, thread text, and CI
  details are explicitly identified as untrusted data. The coding agent must
  assess them before acting.
- `PRWATCH-repeat-suppression`: A workflow remembers the last feedback
  fingerprint so re-invoking the tool does not immediately return the same
  pending feedback in a hot loop.
- `PRWATCH-standalone`: A visible coding agent may invoke the tool outside a PR
  workflow. Workflow state is recorded only when the invocation belongs to the
  active watch step of that exact thread, checkout, and workflow.

### 3.2 Agent-Driven Follow-Up

- `PRWATCH-agent-owner`: After PR publication, Hitch starts one visible coding
  turn that owns the watch/fix cycle and instructs it to invoke
  `hitch.watch_pr`.
- `PRWATCH-agent-decisions`: The coding agent decides whether feedback is
  valid, makes any warranted changes, runs tests, commits and pushes, manages
  review threads when appropriate, and chooses when to invoke the tool again.
- `PRWATCH-no-framework-loop`: `/pr`, `/pr-now`, `/fix-pr`, Auto-PR, and
  equivalent publication workflows do not launch a
  hidden PR monitor, parse a monitor verdict, schedule framework backoff, or
  launch a separate feedback repair turn.
- `PRWATCH-setting-inheritance`: The visible follow-up turn retains the coding
  session's model, reasoning effort, developer instructions, sandbox, approval,
  memory, and web-search settings.
- `PRWATCH-user-control`: User steering is durably queued and takes precedence
  when the current follow-up turn settles. After steering is handled, the
  visible agent resumes the watch with the tool. Stop follows the normal
  visible-turn cancellation and force-stop behavior.

### 3.3 Completion and Stage Tracking

- `PRWATCH-ready-completion`: A successful visible turn whose latest tool
  result is `ready` completes at the PR-ready stage.
- `PRWATCH-terminal-completion`: A terminal tool observation completes at the
  PR-closed stage and preserves whether the PR merged or closed.
- `PRWATCH-neutral-completion`: A successful turn without a ready or terminal
  result completes neutrally rather than claiming that the PR passed its
  gates; later stage refresh may still observe terminal GitHub state.
- `PRWATCH-auto-pull`: If the tool observes a merged PR, an enabled project may
  run the existing post-merge Auto-pull against its separately verified default
  repository checkout. Auto-pull failure does not undo workflow completion.

## 4. Success Criteria

- `PRWATCH-accept-tool-driven`: A newly published PR is followed by one visible
  coding turn that invokes `hitch.watch_pr`, with no hidden monitor or feedback
  agent run.
- `PRWATCH-accept-bounded`: Pending gates are polled inside the invocation and
  return `timed_out` after the bounded watch window.
- `PRWATCH-accept-agent-fix`: Actionable evidence returns to the invoking
  coding agent, which can fix, test, push, and invoke the tool again in the same
  turn.
- `PRWATCH-accept-safe-state`: Tool output updates only its owning active PR
  workflow and cannot overwrite another thread's or checkout's workflow state.
