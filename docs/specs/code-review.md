# Code Review Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Define Hitch's native Codex reviewer role and the ordinary visible-agent turns
used by QA and pull-request shortcuts.

### 1.2 Definitions

- Reviewer role: The native Codex subagent role named `hitch_reviewer` that
  Hitch registers for visible coding workers.
- Review run: A native Codex subagent spawned by the coding agent with the
  reviewer role.
- Review-guidance turn: An ordinary visible coding turn that recommends the
  reviewer role without requiring its use.
- Review trigger: `/qa`, `/pr`, Auto-QA, or Auto-PR.

## 2. Goals and Non-Goals

### 2.1 Goals

- Let the coding agent decide when an independent review is useful.
- Keep review delegation inside Codex's native subagent lifecycle.
- Preserve review and PR shortcuts as visible, steerable, stoppable agent turns.
- Preserve the source session's effective model and execution settings.

### 2.2 Non-Goals

- Hitch does not decide whether a review finding is valid.
- Hitch does not require review before completion or PR publication.
- Hitch does not run a hidden verdict, repair, retry, or review workflow.
- Hitch does not build a second app-server or tool protocol around native
  subagents.

## 3. Requirements

### 3.1 Native Reviewer Availability

- `REVIEW-role-registration`: Every visible coding worker registers
  `hitch_reviewer` in that worker's app-server configuration and enables
  Codex's native multi-agent feature. Hidden system-agent workers do not receive
  the role.
- `REVIEW-resume-availability`: Registration is process-scoped rather than
  persisted in a rollout. The role is available on new, resumed, pre-upgrade,
  promoted, and visible system-feedback coding threads without modifying the
  stored thread.
- `REVIEW-agent-controlled`: The coding agent decides whether to spawn the role
  when the user or task prompt authorizes native subagent use. A reviewer may be
  used more than once.
- `REVIEW-native-lifecycle`: Codex owns reviewer creation, execution, event
  routing, model inheritance, interruption, and cleanup. Hitch does not start a
  nested app-server or expose a fallback review tool.
- `REVIEW-advisory-findings`: Findings return to the invoking coding agent as
  advisory output. The coding agent assesses them; Hitch does not parse a
  verdict.

### 3.2 Reviewer Behavior

- `REVIEW-complete-change-set`: The reviewer inspects the complete current
  change set relative to an appropriate merge base, including committed,
  staged, unstaged, and untracked changes plus relevant surrounding code and
  tests.
- `REVIEW-read-only`: The role overlays `sandbox_mode = "read-only"` and
  `approval_policy = "never"`, and instructs the reviewer not to mutate files
  or external state.
- `REVIEW-setting-inheritance`: The role does not override model or reasoning
  effort, so the native subagent inherits the effective parent settings. Other
  capabilities follow Codex's native subagent inheritance rules.
- `REVIEW-comprehensive-result`: The reviewer returns all concrete actionable
  findings in one pass, ordered by severity with precise file locations, or
  clearly states that it found none.

### 3.3 Ordinary Review and PR Turns

- `REVIEW-trigger-guidance`: `/qa` starts one ordinary visible review-guidance
  turn on the selected session checkout. The turn recommends, but does not
  require, `hitch_reviewer` and asks the coding agent to assess findings, fix
  valid issues, and run relevant tests.
- `REVIEW-pr-guidance`: `/pr` starts one ordinary visible turn that combines
  optional review guidance with PR preparation, Codex-owned publication, and
  the agent-invoked `hitch.watch_pr` cycle.
- `REVIEW-pr-now`: `/pr-now` starts the same publication/watch task without the
  optional review guidance.
- `REVIEW-fix-pr`: `/fix-pr` starts an ordinary visible follow-up turn for the
  PR currently registered to the session. It does not infer or create a PR when
  no registered identity exists.
- `REVIEW-auto-guidance`: Auto-QA or Auto-PR adds the corresponding review or
  publication/watch instruction to the original non-Plan coding turn. The
  coding agent completes it in that turn; Hitch does not launch a follow-up
  turn or track a framework trigger.
- `REVIEW-trigger-settings`: Manual review/PR turns retain the session's model,
  reasoning effort, developer instructions, sandbox, approval, memory,
  web-search, and message-index settings. Automatic guidance runs under the
  original coding turn's settings.
- `REVIEW-selected-checkout`: Review and PR tasks inspect the selected session
  checkout as it exists. Starting them does not Auto-pull or create a separate
  clean worktree first.
- `REVIEW-ordinary-control`: Steering is sent directly to the active worker and
  falls back to a normal follow-up if the worker settles during delivery. Stop
  uses the normal visible-turn graceful and force-stop behavior. There is no
  framework steering queue or recovery narration.
- `REVIEW-visible-stage`: Active tagged review and publication turns display QA
  and PR stages respectively and appear in the ordinary session transcript.
- `REVIEW-no-framework-loop`: These triggers create no PR/QA `SystemWorkflow`,
  hidden reviewer, framework-managed repair turn, or repeated review loop.
- `REVIEW-no-local-merge`: Review and PR tasks never apply session changes
  directly to another local branch.

## 4. Success Criteria

- `REVIEW-accept-available`: New, resumed, promoted, and visible
  system-feedback coding turns expose `hitch_reviewer`; hidden system agents do
  not.
- `REVIEW-accept-native`: Delegating review uses Codex's native `spawn_agent`
  path and returns findings to the coding agent without Hitch review-runtime
  processes or callbacks.
- `REVIEW-accept-ordinary-turns`: `/qa`, `/pr`, `/pr-now`, `/fix-pr`, Auto-QA,
  and Auto-PR run as ordinary visible turns with normal steering, stopping, and
  failure display.
- `REVIEW-accept-no-loop`: Reviewer findings alone never cause Hitch to start a
  framework-managed repair or another review run.
