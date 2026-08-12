# Usage Spec

Status: Draft

## 1. Overview

### 1.1 Purpose

Usage shows Hitch token usage without making page loads wait on expensive token
counting. The page should render quickly from cached token counts, show clearly
when those counts are stale, and refresh the displayed values in place with
asynchronous browser polling until a refresh sweep has completed.

The same Usage sections appear on `/usage/` and in the profile page, so this
spec covers both surfaces.

### 1.2 Definitions

- Token usage: Codex token counts derived from rollout `token_count` events.
  Hitch displays non-cached input tokens, output tokens, and cached input tokens.
- Cached token usage: Persisted per-session token counts and per-day usage
  buckets. These are the source for request-time Usage rendering.
- Usage sweep: A background refresh pass that checks indexed sessions, repairs
  missing rollout paths when possible, parses rollouts whose cached token usage
  is missing or stale, writes updated cache rows, and records that the checked
  rows were swept.
- Stale token usage: Cached token usage that may no longer match the latest
  indexed session source, rollout file, or token-counting logic version.
- Usage status: The backend-reported state that tells the browser whether the
  displayed token usage is fresh, still refreshing, terminally partial, failed,
  unavailable, and whether polling should continue.
- User session: A normal user-visible Codex session.
- HITCH system session: A hidden or system-owned session used by Hitch
  automation, except for system-created work that Hitch intentionally exposes
  as user-visible session work.

## 2. Goals and Non-Goals

### 2.1 Goals

- Keep `/usage/` and profile usage sections fast enough that cached Usage
  renders target about 100 ms under normal local-database conditions.
- Render token counts and charts from cached database state only.
- Show cached token numbers even when they are stale, with an explicit
  refreshing indicator.
- Refresh stale or missing token counts with asynchronous JavaScript calls after
  initial render.
- Stop browser polling after the backend completes a token-usage refresh sweep
  for the requested view.
- Let every token-usage section expand to a stacked per-day bar chart.

### 2.2 Non-Goals

- Usage does not require an aggregate token-usage snapshot table. Request-time
  aggregation over cached per-session database rows is acceptable when it meets
  the performance target.
- Usage does not persist quota or rate-limit data in the token cache. Those
  sections remain backed by live account data through a short-lived process
  snapshot and stay outside the token-usage refresh contract.
- Usage does not provide billing-grade accounting beyond the token counts Codex
  records in rollouts.
- Usage does not add team, user, or organization-level breakdowns.

## 3. User Stories

- As a user, I can open Usage or Profile without waiting for Hitch to parse
  historical rollout files.
- As a user, I can see the last known token counts immediately, even when Hitch
  is refreshing them.
- As a user, I can leave the page open and watch stale token counts update
  without manually refreshing the browser.
- As a user, I can expand each token-usage section to understand usage by day.

## 4. Requirements

### 4.1 Scope and Data Sources

- `USAGE-surfaces`: The shared Usage sections on `/usage/` and `/profile/` must
  follow this spec.
- `USAGE-token-source`: Token usage is derived from Codex rollout
  `token_count` events and cached per session. Cached input tokens must remain
  a displayed breakdown, not be double-counted into non-cached input.
- `USAGE-daily-source`: Headline counts and daily buckets for one session must
  be derived from the same rollout snapshot when a cache row is refreshed, so a
  daily chart cannot disagree with the headline count for that cache row.
- `USAGE-system-buckets`: Usage totals must split user sessions and HITCH
  system sessions using the same hidden/system-session classification used by
  the session and automation UI.
- `USAGE-project-scope`: Profile's active-project usage summary must use the
  currently selected project and must distinguish all selected-project usage
  from selected-project system-session usage.
- `USAGE-project-profile-only`: Active-project token sections are profile-only.
  They must not appear on `/usage/` unless a future spec explicitly changes the
  Usage page information architecture.
- `USAGE-quota-live`: Quota and rate-limit sections are not governed by the
  token cache. Initial page rendering must use the current short-lived,
  process-local snapshot and must not wait for the live account request. A
  missing or stale snapshot must schedule a coalesced background fetch. A cold
  page must show a refreshing state while that fetch is active, then show the
  refreshed snapshot on a later request or the unavailable state after a
  terminal failure.

### 4.2 Cached Rendering and Performance

- `USAGE-cache-only-token-render`: Initial HTML rendering of token usage must
  read cached database state only. It must not synchronously parse rollout
  files, repair missing Codex paths, call Codex for token counts, or wait for a
  token-usage refresh worker.
- `USAGE-db-aggregation-ok`: Request-time aggregation over cached per-session
  database rows is allowed, provided the Usage render path still targets about
  100 ms under normal local-database conditions.
- `USAGE-no-render-file-sweep`: The render path should avoid per-session
  rollout-file reads or filesystem sweeps. Freshness checks that require
  inspecting rollout files belong in the usage sweep, not in the blocking page
  render.
- `USAGE-stale-show-cache`: Structurally usable cached token rows must
  contribute their last known counts and daily buckets even when they are stale.
  The UI must mark the displayed token usage as refreshing instead of replacing
  stale cached counts with zero.
- `USAGE-missing-cache`: Sessions with no usable token cache row may contribute
  zero until the first sweep computes their counts. Their absence must keep the
  token usage in a refreshing or partial state until a sweep completes.
- `USAGE-index-coverage`: All-sessions token totals require complete indexed
  coverage of both active and archived sessions. If Hitch has never completed
  that coverage, the UI may show the token-usage unavailable state while the
  index refresh is pending, but that pending state is part of the async Usage
  refresh flow in `USAGE-index-polling`.
- `USAGE-stale-logic-version`: A cache row produced by an older token-counting
  logic version is stale. If the row is structurally usable, Hitch may display
  its last known values while a sweep recomputes it, but the UI must not present
  those values as fresh.

### 4.3 Asynchronous Refresh

- `USAGE-refresh-needed-state`: The rendered Usage context must expose a usage
  status and an explicit `should_poll`-style signal. The status must distinguish
  at least fresh, refreshing, partial-refreshing, partial-checked, unavailable,
  and failed states so the browser does not infer polling behavior from a vague
  partial label.
- `USAGE-refresh-on-load`: If `should_poll` is true, the page must start
  asynchronous JavaScript polling after the initial render. Terminal
  partial-checked, failed, or unavailable states with `should_poll` false must
  not restart polling.
- `USAGE-refresh-endpoint`: Hitch must expose an idempotent asynchronous
  endpoint for token-usage refresh status. A request to this endpoint should
  schedule or join a backend usage sweep when one is needed.
- `USAGE-index-polling`: The async Usage refresh flow must also cover pending
  active or archived session-index coverage. A first-time page that initially
  renders token usage unavailable because indexing is incomplete should poll
  until the index refresh finishes, then either render cached token usage or
  continue into the token-usage sweep state.
- `USAGE-refresh-coalescing`: Concurrent page renders or polling requests must
  not start duplicate token-usage sweeps for the same pending work. Existing
  in-flight refresh work should be reused.
- `USAGE-poll-update`: Poll responses must let the browser update token counts,
  refresh indicators, and token-usage charts without a full-page reload.
- `USAGE-stop-after-sweep`: Browser polling must stop once the backend reports
  that a usage sweep has completed for the requested view, even if some rows
  remain unavailable because their rollout could not be found or parsed.
- `USAGE-refresh-failure`: If a sweep completes with rows that could not be
  refreshed, the UI should return a terminal partial-checked or failed status
  with `should_poll` false and present the token usage as last-known or partial
  rather than spinning indefinitely.
- `USAGE-running-sessions`: A completed sweep means the displayed counts are
  current as of that sweep. The page does not need to keep polling forever just
  because a live session might spend more tokens later.

### 4.4 UI Behavior

- `USAGE-refresh-indicator`: When displayed token usage is stale, partial, or
  being refreshed, the UI must show a visible refreshing indicator near the
  affected token-usage section.
- `USAGE-refresh-accessibility`: Refresh indicators and async count updates
  should be accessible to assistive technology without being excessively noisy.
- `USAGE-token-sections`: `/usage/` and `/profile/` must both represent these
  shared token-usage sections when their data is available: All sessions,
  Sessions, and HITCH system.
- `USAGE-profile-token-sections`: `/profile/` must additionally represent these
  profile-only token-usage sections when a current project and project usage
  data are available: Active project all sessions and Active project system
  sessions.
- `USAGE-chart-all-token-sections`: Every token-usage section in
  `USAGE-token-sections` and `USAGE-profile-token-sections` must be expandable
  to a usage-by-day bar chart on the surfaces where that section is visible
  when it has daily usage data.
- `USAGE-chart-stacked`: Daily charts must use stacked bars with separate
  segments for non-cached input, output, and cached input tokens.
- `USAGE-chart-scale`: Each chart should scale its bars against the maximum
  daily total in that chart, so small sections remain readable independently of
  larger sections.
- `USAGE-chart-accessibility`: Expandable chart controls must expose keyboard
  interaction and `aria-expanded` state. Sections without chart data must not
  pretend to be expandable controls.
- `USAGE-chart-update`: Async refreshes must update chart data along with the
  headline counts. If practical, the browser should preserve the user's current
  expanded/collapsed state across an in-place update.
- `USAGE-no-quota-charts`: Quota and rate-limit sections should not expand into
  usage-by-day charts under this spec.
- `USAGE-responsive`: Usage cards, charts, and refresh indicators must remain
  usable on narrow mobile-width layouts.

### 4.5 Edge Cases

- `USAGE-zero-usage`: A complete sweep over sessions with no token-count events
  should render zero usage without endless polling.
- `USAGE-missing-rollout-with-cache`: If a rollout path is missing but Hitch has
  a usable cached row, Hitch should keep showing the last known cached values
  and mark them stale or partial until a sweep checks the row.
- `USAGE-missing-rollout-no-cache`: If a rollout path is missing and no usable
  cache exists, that session contributes zero after the sweep and the UI should
  not poll indefinitely for it.
- `USAGE-path-repair`: Sweeps may repair missing token-usage paths through
  Codex resume/index metadata, but path repair must happen off the blocking
  render path.
- `USAGE-cache-corruption`: Malformed cached daily buckets should not break the
  page. Hitch should ignore malformed bucket data for charts while preserving
  usable headline counts when possible.

## 5. Implementation Notes

- The current `ArchivedSessionTokenUsage` model is a suitable per-session cache
  even though its name references archived sessions. The spec does not require
  renaming it.
- The current stale-file-backed behavior intentionally needs to change:
  lifetime aggregation should not drop a usable stale cache row to zero while a
  refresh is pending.
- The async refresh endpoint can return either JSON data or a rendered HTML
  partial. Reusing the existing Usage partial is acceptable if it keeps the
  browser update simple and avoids duplicating formatting logic.
- The backend will likely need a sweep-level status signal, not just per-row
  `usage_last_checked_at`, so polling can stop after a completed sweep and show
  partial/failure states deterministically. That status should include an
  explicit polling flag instead of requiring the client to infer polling from
  labels such as partial.
- The async status endpoint may coordinate both session-index and token-cache
  refresh work. Keeping both under one Usage polling contract prevents a page
  from getting stuck in the initial unavailable state after index coverage
  completes.
- Tests should verify that `/usage/` and `/profile/` do not synchronously parse
  rollouts for token usage during cached renders.

## 6. Success Criteria

- `USAGE-accept-fast-cached-render`: With complete indexed coverage and cached
  token rows, `/usage/` and `/profile/` render token usage from database cache
  without synchronous rollout parsing or token refresh work.
- `USAGE-accept-stale-visible`: A stale cached token row still contributes its
  last known counts and daily buckets to the rendered totals, and the page shows
  a refreshing indicator.
- `USAGE-accept-async-refresh`: After a stale render, browser polling triggers
  or joins a background sweep, updates counts and charts in place, and stops
  after the sweep completes.
- `USAGE-accept-index-polling`: If initial active or archived index coverage is
  incomplete, browser polling continues through the index refresh and updates
  the Usage section once cached token usage can be rendered.
- `USAGE-accept-refresh-failure`: If a sweep cannot refresh some sessions, the
  page stops polling and shows last-known or partial token usage rather than an
  indefinite spinner.
- `USAGE-accept-all-token-charts`: All sessions, Sessions, HITCH system, Active
  project all sessions, and Active project system sessions each expose a
  stacked usage-by-day chart when daily data exists on the surfaces where those
  sections are visible.
- `USAGE-accept-no-quota-charts`: Quota and rate-limit sections keep their
  background-refreshed live rate-limit presentation and do not gain daily usage
  charts.
- `USAGE-accept-accessible-expansion`: Chart expansion works by mouse and
  keyboard, uses accurate `aria-expanded` state, and leaves chartless sections
  non-interactive.
