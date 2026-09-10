# Session List

Status: Implemented

## Requirements

- `LIST-refresh`: Session and system-session lists fetch updated rows in place
  every five seconds while visible, with an initial follow-up after one second
  to pick up background index refreshes. Names, activity ordering, archive
  state, and stages update using the current page's filters and pagination.
- `LIST-refresh-interaction`: Refreshes preserve open rename forms, menus,
  pointer gestures, focused row controls, and archive/Undo operations. A response
  captured before an intervening interaction or completed rename/archive
  mutation must be discarded. Newly inserted rows retain their action handlers.
  Interacting rows remain attached in their
  existing positions while other rows refresh; an open interaction must not
  suspend updates for the entire list.
- `LIST-refresh-navigation`: Refreshes pause in hidden tabs and resume when
  visible or restored through browser history. Responses prevent HTTP caching.
  Failed requests retain the current list and retry on the next interval.
- `LIST-pr-authority`: List refreshes use the registered PR snapshot and do not
  independently poll GitHub; the PR watch specification remains authoritative.
