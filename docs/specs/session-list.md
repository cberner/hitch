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
- `LIST-stage-reads`: When current worker, input, or registered PR state determines
  a stage, deriving its badge must not read the session transcript.

- `LIST-shared-stage-policy`: Session list and detail use the same PR visibility,
  stage precedence, and cache policy. Pending input and active workers take
  precedence over registered PR state, then valid cached rollout stages and
  transcript-derived stages. A replacement publication hides the prior PR until
  its own registration; historical and empty PR records supply no displayed PR.
  Only New, Plan, Implementation, and QA caches with the current rollout mtime
  may be reused. PR and input-derived stages require current authoritative state.
  Partial detail history uses a valid cache or the bounded preview and its
  leading user-message context when current worker/input/PR state cannot resolve
  the stage. Stage resolution must not turn a paginated read into a full-rollout
  load. Partial or unreadable history and transient worker/input stages must not
  overwrite the cache.
  Full-history and registered-PR stages remain available for durable classification.
  Each view keeps its existing badge formatting.
