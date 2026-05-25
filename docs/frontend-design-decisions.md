# Frontend Design Decisions

This document records stable UI decisions for HITCH so new pages and features stay consistent.

## Navigation and Page Actions

- The primary navigation menu belongs in the upper left of every authenticated application page.
- Page-level menus, when a page needs one, belong in the upper right of the top bar.
- Page-level actions should live in that upper-right page menu instead of the page body header. For example, autonomous goal actions such as "New goal" and "Run all" are menu items.
- Page body headers should identify the current view and relevant context. They should not become a second top bar for global or page-level commands.

