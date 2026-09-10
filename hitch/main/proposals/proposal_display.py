"""Display helpers for proposed follow-up sessions."""

from hitch.main.models import ProposedSession
from hitch.main.sessions.session_settings import (
    _BARE_REPO_PROJECT_VALUE,
    _project_for_proposed_session,
    _target_cwd_for_proposed_session,
)


def _attach_proposed_session_display_state(
    proposed_sessions: list[ProposedSession],
) -> None:
    for proposed_session in proposed_sessions:
        files = proposed_session.relevant_files
        proposed_session.display_files = (  # type: ignore[attr-defined]
            [item for item in files if isinstance(item, str) and item.strip()] if isinstance(files, list) else []
        )
        proposed_session.session_prompt = _proposed_session_prompt(  # type: ignore[attr-defined]
            proposed_session
        )
        project = _project_for_proposed_session(proposed_session)
        target_cwd = _target_cwd_for_proposed_session(proposed_session)
        proposed_session.accept_project_id = (  # type: ignore[attr-defined]
            project.pk if project is not None else _BARE_REPO_PROJECT_VALUE if target_cwd else ""
        )
        proposed_session.accept_cwd = (  # type: ignore[attr-defined]
            "" if project is not None else target_cwd
        )
        auto_pr_enabled, auto_qa_enabled = _auto_review_settings_for_proposed_session(proposed_session)
        proposed_session.accept_auto_pr = auto_pr_enabled  # type: ignore[attr-defined]
        proposed_session.accept_auto_qa = auto_qa_enabled  # type: ignore[attr-defined]
        metadata = _proposal_metadata(proposed_session)
        proposed_session.accept_auto_review_explicit = (  # type: ignore[attr-defined]
            "auto_pr_enabled" in metadata or "auto_qa_enabled" in metadata
        )


def _proposed_session_prompt(proposed_session: ProposedSession) -> str:
    if proposed_session.prompt.strip():
        return proposed_session.prompt
    parts = [
        "Go ahead and implement this proposed session.",
        "",
        "Source: Coding agent proposal",
    ]
    parts.extend(["", f"Proposed session: {proposed_session.title}"])
    if proposed_session.summary:
        parts.extend(["", f"Summary:\n{proposed_session.summary}"])
    files = proposed_session.display_files  # type: ignore[attr-defined]
    if files:
        parts.extend(["", "Relevant files:", *[f"- {file}" for file in files]])
    return "\n".join(parts)


def _auto_review_settings_for_proposed_session(
    proposed_session: ProposedSession,
) -> tuple[bool, bool]:
    metadata = _proposal_metadata(proposed_session)
    if "auto_pr_enabled" in metadata or "auto_qa_enabled" in metadata:
        auto_pr_enabled = metadata.get("auto_pr_enabled") is True
        auto_qa_enabled = metadata.get("auto_qa_enabled") is True and not auto_pr_enabled
        return auto_pr_enabled, auto_qa_enabled
    return False, False


def _proposal_metadata(proposed_session: ProposedSession) -> dict[str, object]:
    return proposed_session.outcome_metadata if isinstance(proposed_session.outcome_metadata, dict) else {}
