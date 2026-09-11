# Session Model and Effort

Status: Implemented

## Overview

The session actions menu provides a Model and effort dialog for the main agent.
It saves a model and reasoning effort for that session without changing account
defaults. Service tiers are outside this feature's scope.

## Requirements

- `MODEL-selection`: Offer the Codex model catalog and filter reasoning efforts
  to those supported by the selected model. Validate the pair when saving.
  Resolve Model default to the model's advertised default effort.
- `MODEL-persistence`: Save the pair together in session metadata. All subsequent
  visible coding turns, including Plan, review, and PR follow-ups, use it. Plan
  turns use medium only when no effort was selected. Existing sessions without
  an override retain their previous model selection behavior.
- `MODEL-live`: Forward changes to the owning worker's current turn through
  `turn/settings/update`. Never resume a second writer to change live settings.
  Require Codex's applied acknowledgement before reporting a live update.
  Already captured model calls and existing subagents retain their settings.
- `MODEL-pending`: If live updates are unsupported, rejected, unacknowledged, or
  race turn completion, retain the selection for the next turn and explain
  that the running turn has not confirmed it. Display the active settings
  separately from pending settings. Do not interrupt or start a turn to save.
- `MODEL-interface`: Saving preserves the transcript, selected agent, and draft.
  Display failures in the dialog and successful saves on the session page.
  Read-only session pages do not expose mutation controls.

## Success Criteria

- A saved model/effort pair reaches later turns, including automated follow-ups,
  and the active turn when supported.
- Unsupported pairs are rejected without changing saved settings.
- Pending settings are never presented as confirmed active settings.
