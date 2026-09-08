# New Session

Status: Implemented

## Requirements

- `NS-prompt-history`: A small history icon directly below the `/` menu button opens a
  scrollable list of the 20 most recent nonblank user prompts saved by Hitch,
  newest first, from the project currently selected in the new-session form.
  Changing the project immediately updates the list. Include follow-up prompts and
  archived sessions; exclude automated turns and hidden system sessions.
- `NS-history-storage`: Save accepted browser submissions, including steering
  during an active turn, before adding worker instructions. Keep only the last
  20 nonblank submissions per project. Bare-repo sessions and legacy history
  without a recorded project share the bare-repo history; deleting a project
  deletes its history. History starts empty on upgrade and collects new
  submissions going forward, without importing old worker prompts. History bookkeeping
  failures must not reject input already accepted by a worker.
- `NS-prompt-reuse`: Selecting a history entry replaces the prompt box with its
  complete saved text, closes the list, and focuses the box for editing without
  submitting the form or changing session options. Long entries have compact
  previews. An empty history displays an explanatory message.
- `NS-history-keyboard`: The history icon and entries support keyboard use.
  Escape closes the list and returns focus to the icon; clicking outside closes
  it without changing the prompt.
