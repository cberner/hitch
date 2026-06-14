# Spec Guidelines

Specs in this directory are the authoritative description of how Hitch features
behave. Agents should read relevant specs before changing behavior and flag any
user request that contradicts an existing spec.

## Expected Format

Use Markdown with these sections when applicable:

- Overview
- Goals and Non-Goals
- Requirements
- Success Criteria

Include `Status: Draft` or another clear status near the top.

## Requirement Slugs

Use stable, human-readable slugs for citeable requirements, such as
`AG-low-quota-blocks-auto`.

When editing a spec:

- Do not renumber or rename existing slugs unless the requirement itself is
  intentionally being replaced.
- Prefer adding a new slug for new behavior.
- If a requirement is removed, delete the slug rather than reusing it for a
  different meaning.
- Keep slugs short, specific, and feature-prefixed.
