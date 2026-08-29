# Code Review Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Define Hitch's native Codex reviewer subagent and the optional review guidance
that replaces the framework-driven QA verdict-and-repair loop.

### 1.2 Definitions

- Reviewer role: The native Codex subagent role named `hitch_reviewer` that
  Hitch registers for visible coding workers.
- Review run: A native Codex subagent spawned by the coding agent with the
  reviewer role.
- QA trigger: A manual or automatic workflow event that previously started
  Hitch's hidden QA loop, including `/qa`, `/pr`, Auto-QA, and Auto-PR.
- Review-guidance turn: A normal coding turn that recommends the reviewer
  subagent without requiring its use.

## 2. Goals and Non-Goals

### 2.1 Goals

- Let the coding agent decide when an independent review is useful.
- Keep review delegation inside Codex's native subagent lifecycle.
- Give the reviewer the coding session's checkout and effective model settings
  while preventing ordinary filesystem writes.
- Preserve QA, PR, and local-merge triggers without orchestrating a hidden
  review-and-repair loop.

### 2.2 Non-Goals

- Hitch does not decide whether a review finding is valid.
- Hitch does not require review before work can complete, merge locally, or
  proceed to PR publication.
- Hitch does not build a second app-server, tool protocol, cancellation router,
  or capability sandbox around native subagents.

## 3. Requirements

### 3.1 Native Reviewer Availability

- `REVIEW-role-registration`: Every visible coding worker registers
  `hitch_reviewer` in that worker's app-server configuration and enables
  Codex's native multi-agent feature. Hidden system-agent workers do not receive
  the role.
- `REVIEW-resume-availability`: Registration is process-scoped rather than
  persisted in a rollout. The role is therefore available on new, resumed,
  pre-upgrade, promoted, and visible system-feedback coding threads without
  modifying the stored thread.
- `REVIEW-agent-controlled`: The coding agent, rather than Hitch's workflow,
  decides whether to spawn the role when the user or review-guidance prompt
  authorizes native subagent use. A reviewer may be used more than once.
- `REVIEW-native-lifecycle`: Codex owns reviewer creation, execution, event
  routing, model inheritance, interruption, and cleanup. Hitch does not start a
  nested app-server or expose a fallback MCP server for review.
- `REVIEW-advisory-findings`: Findings return to the invoking coding agent as
  advisory subagent output. The coding agent assesses them and Hitch does not
  parse them into a verdict.

### 3.2 Reviewer Behavior

- `REVIEW-complete-change-set`: The role asks the reviewer to inspect the
  complete current change set relative to an appropriate merge base, including
  committed, staged, unstaged, and untracked changes plus relevant surrounding
  code and tests.
- `REVIEW-read-only`: The role overlays `sandbox_mode = "read-only"` and
  `approval_policy = "never"`, and instructs the reviewer not to mutate files
  or external state.
- `REVIEW-setting-inheritance`: The role does not override model or reasoning
  effort, so the native subagent inherits the effective parent settings. Other
  capabilities follow Codex's native subagent inheritance rules.
- `REVIEW-comprehensive-result`: The reviewer returns all concrete actionable
  findings in one pass, ordered by severity with precise file locations, or
  states clearly that it found none.

### 3.3 QA and PR Triggers

- `REVIEW-trigger-guidance`: When a QA trigger would previously have started
  Hitch's framework-driven QA loop, Hitch instead starts one review-guidance
  turn on the coding session.
- `REVIEW-trigger-optional`: The guidance recommends, but does not require,
  spawning `hitch_reviewer` and leaves the decision to the coding agent.
- `REVIEW-no-framework-loop`: New workflows do not launch a hidden QA reviewer,
  parse a QA verdict, automatically send findings to a repair agent, or repeat
  those steps. Pre-upgrade framework-QA workflow states are not resumed.
- `REVIEW-guidance-settings`: The guidance turn retains the coding session's
  model, reasoning effort, developer instructions, sandbox, approval, memory,
  and web-search settings.
- `REVIEW-guidance-approval-modes`: Auto-QA and Auto-PR start guidance under the
  source session's approval mode, including Always prompt and Deny all. These
  modes do not suppress the automatic trigger.
- `REVIEW-qa-completion`: A QA-only workflow completes after its guidance turn
  succeeds, whether or not review was delegated.
- `REVIEW-pr-handoff`: A PR workflow proceeds to publication and one visible,
  agent-driven `hitch.watch_pr` follow-up turn after its guidance turn succeeds.
- `REVIEW-local-merge-handoff`: A configured local auto-merge captures the final
  change set after guidance, applies that exact change set to the target branch,
  and reports the branch and commit. Messages use neutral review-guidance
  terminology and do not claim QA approval.
- `REVIEW-guidance-steering`: User steering during QA-only or local-merge
  guidance resumes guidance without PR preparation, commit, push, or PR
  publication instructions. Guidance failures are attributed to the review
  workflow.
- `REVIEW-qa-display`: `/qa` is displayed as a coding-agent inspection with an
  optional reviewer subagent.
- `REVIEW-pr-now`: `/pr-now` continues to skip review guidance and proceeds
  directly to PR preparation.

## 4. Success Criteria

- `REVIEW-accept-available`: New, resumed, promoted, and visible
  system-feedback coding turns expose `hitch_reviewer`; hidden system agents do
  not.
- `REVIEW-accept-native`: Delegating review uses Codex's native `spawn_agent`
  path and returns findings to the coding agent without Hitch review-runtime
  processes or tool callbacks.
- `REVIEW-accept-optional`: `/qa`, `/pr`, Auto-QA, and Auto-PR each use one
  coding turn that recommends but does not require the reviewer.
- `REVIEW-accept-no-loop`: Reviewer findings alone never cause Hitch to start a
  framework-managed repair or another review run.
