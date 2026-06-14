# Projects PRD

Status: Draft

## 1. Overview

### 1.1 Purpose

Define Hitch Projects and project-level settings.

### 1.2 Definitions

- Project: A named local repository checkout that Hitch can associate with sessions.
- Project default repository checkout: The project's configured `repo_path`, distinct from a session worktree when a session runs in a separate checkout.

## 2. Requirements

- `PROJECT-settings`: Projects may expose settings that control future project-scoped behavior.
- `PROJECT-auto-pull`: Projects expose an Auto-pull setting. When enabled, after Hitch's PR monitor observes a GitHub PR merge for the session, Hitch may run a non-interactive fast-forward pull of the project default branch advertised by `origin/HEAD` from `origin` in the project default repository checkout, not in the session worktree, without recursing into submodules or fetching tags. Hitch must skip Auto-pull when the project checkout cannot be verified as a separate checkout for the workflow's repository. Pull failures or skips must not undo the completed PR workflow and must remain diagnosable from session state.

## 3. Success Criteria

- `PROJECT-accept-auto-pull`: Auto-pull runs only after Hitch's PR monitor observes a GitHub PR merge for the session, targets a verified separate project default repository checkout, uses a fast-forward pull of `origin/HEAD` from `origin` without submodule recursion or tag fetching, and records success, failure, or skip without changing the completed PR workflow state.
