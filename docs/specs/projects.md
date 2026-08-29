# Projects Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Define Hitch Projects and project-level settings.

### 1.2 Definitions

- Project: A named local repository checkout that Hitch can associate with sessions.
- Project default repository checkout: The project's configured `repo_path`, distinct from a session worktree when a session runs in a separate checkout.

## 2. Requirements

- `PROJECT-settings`: Projects may expose settings that control future project-scoped behavior.
- `PROJECT-auto-pull`: Projects expose an Auto-pull setting that is enabled by default for new projects. When enabled, Hitch runs a non-interactive fast-forward pull of the project default branch advertised by `origin/HEAD` from `origin` before creating a new coding session, so any managed session worktree contains the latest remote default branch. A clean checkout that is ahead of the remote default branch is accepted and keeps its local commits. Review-guidance and PR-preparation turns started with `/qa` or `/pr` are exempt because they must inspect the selected checkout's current changes. A pull failure blocks coding-session creation with a diagnosable error. After an agent-invoked `hitch.watch_pr` observes a GitHub PR merge for a session, Hitch may also pull in the project default repository checkout, not in the session worktree, without recursing into submodules or fetching tags. Post-merge Auto-pull must skip when the project checkout cannot be verified as a separate checkout for the workflow's repository, and its failures or skips must not undo the completed PR workflow.

## 3. Success Criteria

- `PROJECT-accept-auto-pull`: Post-merge Auto-pull runs only after `hitch.watch_pr` observes a GitHub PR merge for the session, targets a verified separate project default repository checkout, uses a fast-forward pull of `origin/HEAD` from `origin` without submodule recursion or tag fetching, and records success, failure, or skip without changing the completed PR workflow state.
- `PROJECT-accept-auto-pull-before-session`: Starting a new coding session for a project with Auto-pull enabled first attempts to fast-forward the project checkout's default branch from `origin`; an ahead-only checkout proceeds without changing or discarding local commits. Hitch does not create a thread, workflow, or managed worktree if that pull fails. Starting a `/qa` review-guidance turn or `/pr` workflow does not pull or modify the checkout first.
