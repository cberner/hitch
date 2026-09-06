# Code Review Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Define the ordinary visible-agent turns used by QA and pull-request shortcuts
and their optional delegation through Codex's native subagents.

### 1.2 Definitions

- Review run: An inspection of the current changes by the coding agent or a
  native Codex subagent it chooses to involve.
- Review-guidance turn: An ordinary visible coding turn that asks the coding
  agent to review changes and delegate review as it sees fit.
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
- Hitch does not maintain a custom reviewer role or review rubric.

## 3. Requirements

### 3.1 Native Review Delegation

- `REVIEW-native-configuration`: New and resumed workers use Codex's native
  agent configuration. Hitch does not register a reviewer role or override
  multi-agent feature settings for review.
- `REVIEW-agent-controlled`: Review guidance authorizes the coding agent to
  delegate review to Codex subagents as it sees fit. The coding agent decides
  whether to delegate, which agents to use, and whether another pass is useful.
- `REVIEW-native-lifecycle`: Codex owns reviewer creation, execution, event
  routing, model inheritance, interruption, and cleanup. Hitch does not start a
  nested app-server or expose a fallback review tool.
- `REVIEW-advisory-findings`: Findings return to the invoking coding agent as
  advisory output. The coding agent assesses them; Hitch does not parse a
  verdict.

### 3.2 Review Behavior

- `REVIEW-complete-change-set`: The coding agent reviews the complete current
  change set relative to an appropriate merge base, including committed,
  staged, unstaged, and untracked changes plus relevant surrounding code and
  tests.
- `REVIEW-native-settings`: Delegated agents follow Codex's native configuration
  and inheritance rules. Hitch supplies no review-specific model, reasoning,
  sandbox, approval, or developer-instruction overrides.
- `REVIEW-assess-and-fix`: The coding agent assesses delegated findings, fixes
  valid issues, and runs relevant tests before finishing.

### 3.3 Ordinary Review and PR Turns

- `REVIEW-trigger-guidance`: `/qa` starts one ordinary visible review-guidance
  turn on the selected session checkout. The turn leaves delegation to the
  coding agent and asks it to assess findings, fix valid issues, and run
  relevant tests.
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

- `REVIEW-accept-native-configuration`: New, resumed, promoted, and visible
  system-feedback coding workers start without Hitch reviewer-role or
  multi-agent feature overrides.
- `REVIEW-accept-native`: Delegating review uses Codex's native `spawn_agent`
  path and returns findings to the coding agent without Hitch review-runtime
  processes or callbacks.
- `REVIEW-accept-ordinary-turns`: `/qa`, `/pr`, `/pr-now`, `/fix-pr`, Auto-QA,
  and Auto-PR run as ordinary visible turns with normal steering, stopping, and
  failure display.
- `REVIEW-accept-no-loop`: Reviewer findings alone never cause Hitch to start a
  framework-managed repair or another review run.
