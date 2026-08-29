# Session Tools Spec

Status: Draft

## 1. Overview

Hitch exposes dynamic tools that let Codex act on the visible coding session
that owns the current turn.

## 2. Requirements

- `SESSIONTOOLS-rename-registration`: Newly created visible coding sessions
  register `hitch.rename_session`; hidden system-agent sessions do not. Dynamic
  tool registration is immutable, so sessions created before the tool was
  available require a new session to use it.
- `SESSIONTOOLS-rename-current`: The tool renames only the invoking session. Its
  input does not accept a session or thread identifier.
- `SESSIONTOOLS-rename-input`: The tool requires one `name` string, trims outer
  whitespace, rejects an empty result, and limits names to 200 characters.
- `SESSIONTOOLS-rename-persistence`: A successful call persists the name through
  Codex and then updates Hitch's cached session-list title.
- `SESSIONTOOLS-rename-failure`: If Codex rejects the rename because the thread
  is archived or unknown, the tool returns an error and does not update Hitch's
  cached title.

## 3. Success Criteria

- `SESSIONTOOLS-rename-success`: Calling `hitch.rename_session` with a valid
  name changes the invoking session's persisted and cached names.
- `SESSIONTOOLS-rename-validation`: Invalid names fail without attempting a
  rename.
