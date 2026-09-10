# Hitch Extra Instructions

Status: Implemented

## Overview

Hitch supplies its workflow guidance through an editable developer-instruction
setting, separately from personal/project instructions and user messages.

## Requirements

- `HI-questions`: Default Hitch guidance directs agents to use blocking
  `request_user_input` when asking the user a question and wait for the answer,
  or end the turn with the question if that tool is unavailable. Do not use
  asynchronous questions to continue past an unanswered question.
- `HI-editable-default`: Settings expose a prefilled Hitch extra instructions
  field and a Reset to defaults button. An unset value uses the current built-in
  defaults; a custom value replaces them; an explicitly empty value disables
  added Hitch instructions. Reset fills the field without submitting the form.
- `HI-persistence`: Account settings and signed guest cookies preserve the
  distinction between defaults and explicitly empty overrides. Save validates
  the character limit and the guest cookie's encoded size. Old forms that omit
  the field preserve the stored value.
- `HI-workflow`: The saved Hitch instructions include the selected turn's
  workflow: None, Plan, Auto-QA, or Auto-PR. Auto-PR takes precedence over Auto-QA;
  Plan suppresses both automatic workflows in the defaults. Proposal PR titles
  are supplied only for Auto-PR. Overrides may customize the guidance; no
  default instructions are silently appended to an override.
- `HI-turn-snapshot`: Each visible turn started from Hitch saves its effective
  Hitch instructions separately from the personal/project developer prompt.
  A settings change applies to the next started turn, including an existing
  session's follow-up. Steering a running turn does not change its snapshot.
- `HI-delivery`: Thread creation and worker resume combine the two saved
  instruction fields through Codex's developer-instruction channel. Clearing
  Hitch instructions explicitly replaces previously persisted guidance even
  when the resulting developer prompt is empty. Legacy and hidden workers
  without a Hitch snapshot preserve their existing instruction handling.
- `HI-existing-baseline`: New sessions preserve Codex's configured developer
  instructions when no personal/project override is supplied. Imported or
  legacy sessions with no saved baseline recover only explicitly tagged
  developer instructions from their history. If bounded history cannot
  establish that baseline, preserve Codex's existing instructions without
  applying Hitch guidance; the session instruction section exposes this
  exception. Do not infer instructions from prose or replace an unknown
  baseline with current configuration.
- `HI-user-text`: Automatic workflows do not append anything to the user's
  prompt. Stored historical messages remain intact and visible. Manual
  review/PR shortcuts remain explicit requests and do not depend on the setting.
- `HI-visible`: A session exposes the saved instruction fields for its active
  or latest turn, including the selected workflow, without recomputing them
  from current settings. Explicitly empty snapshots show Disabled; legacy
  snapshots say no separate Hitch instructions were recorded.

## Success Criteria

- Starting and resuming sessions preserves the submitted prompt while supplying
  default, custom, or disabled Hitch instructions as configured.
- Users can inspect effective instructions, replace the defaults, clear them,
  and restore them without any transcript text filtering.
