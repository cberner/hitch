"""The web layer, split by domain. Endpoints are re-exported here for urls.py;
helpers are re-exported for compatibility with existing attribute access."""

# --- Backward-compatibility re-exports for the former monolith ``views`` module.
# Tests and older callers reach these helpers (and a few module/class objects)
# through ``hitch.main.views.<name>`` attribute access or ``patch.object(views,
# ...)``; bind them here from their relocated split modules.
from hitch.main.sessions import (
    session_index as session_index,
)
from hitch.main.sessions.session_pr_plan import (
    _auto_merge_to_local_branch_for_session as _auto_merge_to_local_branch_for_session,
)
from hitch.main.sessions.session_pr_plan import (
    _auto_pr_enabled_for_session as _auto_pr_enabled_for_session,
)
from hitch.main.sessions.session_pr_plan import (
    _auto_qa_enabled_for_session as _auto_qa_enabled_for_session,
)
from hitch.main.sessions.session_pr_plan import (
    _claude_fix_pr_url as _claude_fix_pr_url,
)
from hitch.main.sessions.session_pr_plan import (
    _claude_pr_observation_for_session as _claude_pr_observation_for_session,
)
from hitch.main.sessions.session_pr_plan import (
    _claude_user_message_index as _claude_user_message_index,
)
from hitch.main.sessions.session_pr_plan import (
    _thread_plan_mode_state as _thread_plan_mode_state,
)
from hitch.main.sessions.session_pr_plan import (
    _workflow_after_main_lifecycle as _workflow_after_main_lifecycle,
)
from hitch.main.sessions.session_resume import (
    _session_is_claude as _session_is_claude,
)
from hitch.main.sessions.session_settings import (
    _allowed_session_cwds as _allowed_session_cwds,
)
from hitch.main.sessions.session_settings import (
    _candidate_thread_backend as _candidate_thread_backend,
)
from hitch.main.sessions.session_settings import (
    _claude_workflow_common as _claude_workflow_common,
)
from hitch.main.sessions.session_settings import (
    _model_for_thread_backend as _model_for_thread_backend,
)
from hitch.main.sessions.session_settings import (
    _stored_settings as _stored_settings,
)
from hitch.main.sessions.session_stage_refresh import (
    _refresh_session_pr_stage as _refresh_session_pr_stage,
)
from hitch.main.sessions.settings_cookies import (
    SettingsValues as SettingsValues,
)
from hitch.main.sessions.token_usage import (
    _claude_thread_ids as _claude_thread_ids,
)
from hitch.main.sessions.token_usage import (
    _claude_token_usage_for as _claude_token_usage_for,
)
from hitch.main.sessions.token_usage import (
    _usage_token_cache_state as _usage_token_cache_state,
)
from hitch.main.sessions.token_usage import (
    _usage_token_refresh_needed as _usage_token_refresh_needed,
)
from hitch.main.sessions.token_usage import (
    _usage_token_refresh_work_batches as _usage_token_refresh_work_batches,
)
from hitch.main.sessions.token_usage import (
    _UsageTokenRefreshCandidate as _UsageTokenRefreshCandidate,
)
from hitch.main.views.account import (
    _import_cookie_settings_to_user as _import_cookie_settings_to_user,
)
from hitch.main.views.account import (
    _parse_nuked_count as _parse_nuked_count,
)
from hitch.main.views.account import (
    _profile_usage_context as _profile_usage_context,
)
from hitch.main.views.account import (
    health_dashboard as health_dashboard,
)
from hitch.main.views.account import (
    login as login,
)
from hitch.main.views.account import (
    logout as logout,
)
from hitch.main.views.account import (
    nuke_codex as nuke_codex,
)
from hitch.main.views.account import (
    profile as profile,
)
from hitch.main.views.account import (
    register as register,
)
from hitch.main.views.common import (
    _DEBUG_CHAT_PROJECT_NAME as _DEBUG_CHAT_PROJECT_NAME,
)
from hitch.main.views.common import (
    _DEBUG_CHAT_PROMPT_TEMPLATE as _DEBUG_CHAT_PROMPT_TEMPLATE,
)
from hitch.main.views.common import (
    _INTERMEDIATE_DETAIL_CACHE as _INTERMEDIATE_DETAIL_CACHE,
)
from hitch.main.views.common import (
    _INTERMEDIATE_DETAIL_CACHE_LOCK as _INTERMEDIATE_DETAIL_CACHE_LOCK,
)
from hitch.main.views.common import (
    _INTERMEDIATE_DETAIL_CACHE_MAX_SIZE as _INTERMEDIATE_DETAIL_CACHE_MAX_SIZE,
)
from hitch.main.views.common import (
    _MAX_BIGAUTOFIELD as _MAX_BIGAUTOFIELD,
)
from hitch.main.views.common import (
    _NAME_MAX_LEN as _NAME_MAX_LEN,
)
from hitch.main.views.common import (
    _PLAN_APPROVAL_PROMPT as _PLAN_APPROVAL_PROMPT,
)
from hitch.main.views.common import (
    _PLAN_MODE_REASONING_EFFORT as _PLAN_MODE_REASONING_EFFORT,
)
from hitch.main.views.common import (
    _PLAN_REVISION_PROMPT as _PLAN_REVISION_PROMPT,
)
from hitch.main.views.common import (
    _PROJECT_NAME_MAX_LEN as _PROJECT_NAME_MAX_LEN,
)
from hitch.main.views.common import (
    _SESSION_INTERMEDIATE_DEMO_CONTEXT_SALT as _SESSION_INTERMEDIATE_DEMO_CONTEXT_SALT,
)
from hitch.main.views.common import (
    _THREAD_LIST_FETCH_LIMIT as _THREAD_LIST_FETCH_LIMIT,
)
from hitch.main.views.common import (
    _THREAD_LIST_USE_STATE_DB_ONLY as _THREAD_LIST_USE_STATE_DB_ONLY,
)
from hitch.main.views.common import (
    _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT as _USAGE_SESSION_INDEX_REFRESH_IN_FLIGHT,
)
from hitch.main.views.common import (
    _USAGE_SESSION_INDEX_REFRESH_LOCK as _USAGE_SESSION_INDEX_REFRESH_LOCK,
)
from hitch.main.views.common import (
    Codex as Codex,
)
from hitch.main.views.common import (
    UsageContext as UsageContext,
)
from hitch.main.views.common import (
    UsageSessionIndexState as UsageSessionIndexState,
)
from hitch.main.views.common import (
    _all_threads as _all_threads,
)
from hitch.main.views.common import (
    _append_approval_resolved_events as _append_approval_resolved_events,
)
from hitch.main.views.common import (
    _apply_live_approval_mode_to_instances as _apply_live_approval_mode_to_instances,
)
from hitch.main.views.common import (
    _apply_proposed_session_title_to_session_metadata as _apply_proposed_session_title_to_session_metadata,
)
from hitch.main.views.common import (
    _ApprovalResolvedEvent as _ApprovalResolvedEvent,
)
from hitch.main.views.common import (
    _attach_lazy_intermediate_context as _attach_lazy_intermediate_context,
)
from hitch.main.views.common import (
    _base_instructions_for_settings as _base_instructions_for_settings,
)
from hitch.main.views.common import (
    _cache_intermediate_detail as _cache_intermediate_detail,
)
from hitch.main.views.common import (
    _cleanup_saved_input_images as _cleanup_saved_input_images,
)
from hitch.main.views.common import (
    _debug_chat_database_path as _debug_chat_database_path,
)
from hitch.main.views.common import (
    _debug_chat_new_session_url as _debug_chat_new_session_url,
)
from hitch.main.views.common import (
    _debug_chat_project as _debug_chat_project,
)
from hitch.main.views.common import (
    _developer_instructions_for_project as _developer_instructions_for_project,
)
from hitch.main.views.common import (
    _ensure_private_dir as _ensure_private_dir,
)
from hitch.main.views.common import (
    _has_input_image_uploads as _has_input_image_uploads,
)
from hitch.main.views.common import (
    _input_image_extension_from_header as _input_image_extension_from_header,
)
from hitch.main.views.common import (
    _intermediate_detail_cache_key as _intermediate_detail_cache_key,
)
from hitch.main.views.common import (
    _is_allowed_session_cwd as _is_allowed_session_cwd,
)
from hitch.main.views.common import (
    _metadata_rows_for_usage as _metadata_rows_for_usage,
)
from hitch.main.views.common import (
    _models_for_plan_mode_fallback as _models_for_plan_mode_fallback,
)
from hitch.main.views.common import (
    _new_session_proposal_start_claim_filter as _new_session_proposal_start_claim_filter,
)
from hitch.main.views.common import (
    _next_message_config as _next_message_config,
)
from hitch.main.views.common import (
    _plan_mode_model_from_models as _plan_mode_model_from_models,
)
from hitch.main.views.common import (
    _posted_input_image_uploads as _posted_input_image_uploads,
)
from hitch.main.views.common import (
    _posted_project as _posted_project,
)
from hitch.main.views.common import (
    _prevent_stale_cache as _prevent_stale_cache,
)
from hitch.main.views.common import (
    _project_for_cwd as _project_for_cwd,
)
from hitch.main.views.common import (
    _project_for_thread as _project_for_thread,
)
from hitch.main.views.common import (
    _proposed_session_inbox_count as _proposed_session_inbox_count,
)
from hitch.main.views.common import (
    _proposed_session_inbox_queryset as _proposed_session_inbox_queryset,
)
from hitch.main.views.common import (
    _proposed_session_thread_title as _proposed_session_thread_title,
)
from hitch.main.views.common import (
    _recover_stale_new_session_proposal_start_claims as _recover_stale_new_session_proposal_start_claims,
)
from hitch.main.views.common import (
    _refresh_usage_session_index_best_effort as _refresh_usage_session_index_best_effort,
)
from hitch.main.views.common import (
    _rename_codex_thread_from_proposal as _rename_codex_thread_from_proposal,
)
from hitch.main.views.common import (
    _render_session_detail as _render_session_detail,
)
from hitch.main.views.common import (
    _safe_next_url as _safe_next_url,
)
from hitch.main.views.common import (
    _save_posted_input_images as _save_posted_input_images,
)
from hitch.main.views.common import (
    _schedule_session_index_refresh as _schedule_session_index_refresh,
)
from hitch.main.views.common import (
    _schedule_usage_session_index_refresh_if_needed as _schedule_usage_session_index_refresh_if_needed,
)
from hitch.main.views.common import (
    _session_approval_mode_context as _session_approval_mode_context,
)
from hitch.main.views.common import (
    _session_intermediate_demo_context as _session_intermediate_demo_context,
)
from hitch.main.views.common import (
    _session_template_thread as _session_template_thread,
)
from hitch.main.views.common import (
    _SessionTemplateThread as _SessionTemplateThread,
)
from hitch.main.views.common import (
    _settings_context as _settings_context,
)
from hitch.main.views.common import (
    _settle_live_pending_approval_requests as _settle_live_pending_approval_requests,
)
from hitch.main.views.common import (
    _start_usage_session_index_refresh_thread as _start_usage_session_index_refresh_thread,
)
from hitch.main.views.common import (
    _stop_autonomous_goal_stack_after_proposal_resolution as _stop_autonomous_goal_stack_after_proposal_resolution,
)
from hitch.main.views.common import (
    _stream_url_for as _stream_url_for,
)
from hitch.main.views.common import (
    _thread_cwd as _thread_cwd,
)
from hitch.main.views.common import (
    _thread_resume_missing_or_invalid as _thread_resume_missing_or_invalid,
)
from hitch.main.views.common import (
    _uploaded_input_image_extension as _uploaded_input_image_extension,
)
from hitch.main.views.common import (
    _usage_context as _usage_context,
)
from hitch.main.views.common import (
    _usage_session_index_refresh_needed as _usage_session_index_refresh_needed,
)
from hitch.main.views.common import (
    _usage_session_index_state as _usage_session_index_state,
)
from hitch.main.views.common import (
    logger as logger,
)
from hitch.main.views.goals import (
    _AUTONOMOUS_GOAL_TITLE_MAX_LEN as _AUTONOMOUS_GOAL_TITLE_MAX_LEN,
)
from hitch.main.views.goals import (
    _autonomous_goal_cleanup_proposal_filter as _autonomous_goal_cleanup_proposal_filter,
)
from hitch.main.views.goals import (
    _cleanup_proposed_session_candidate_worktree as _cleanup_proposed_session_candidate_worktree,
)
from hitch.main.views.goals import (
    _dismiss_unresolved_autonomous_goal_proposals as _dismiss_unresolved_autonomous_goal_proposals,
)
from hitch.main.views.goals import (
    autonomous_goal_run_log as autonomous_goal_run_log,
)
from hitch.main.views.goals import (
    autonomous_goals as autonomous_goals,
)
from hitch.main.views.goals import (
    create_autonomous_goal as create_autonomous_goal,
)
from hitch.main.views.goals import (
    delete_autonomous_goal as delete_autonomous_goal,
)
from hitch.main.views.goals import (
    edit_autonomous_goal as edit_autonomous_goal,
)
from hitch.main.views.goals import (
    run_autonomous_goal as run_autonomous_goal,
)
from hitch.main.views.goals import (
    run_autonomous_goals as run_autonomous_goals,
)
from hitch.main.views.goals import (
    update_proposed_session_outcome as update_proposed_session_outcome,
)
from hitch.main.views.messages import (
    _DEFAULT_COLLABORATION_MODE as _DEFAULT_COLLABORATION_MODE,
)
from hitch.main.views.messages import (
    _PLAN_ACTION_APPROVE as _PLAN_ACTION_APPROVE,
)
from hitch.main.views.messages import (
    _PLAN_ACTION_REVISE as _PLAN_ACTION_REVISE,
)
from hitch.main.views.messages import (
    _VALID_PLAN_ACTIONS as _VALID_PLAN_ACTIONS,
)
from hitch.main.views.messages import (
    _codex_followup_model as _codex_followup_model,
)
from hitch.main.views.messages import (
    _duplicate_saved_input_images as _duplicate_saved_input_images,
)
from hitch.main.views.messages import (
    _metadata_cwd_is_disallowed as _metadata_cwd_is_disallowed,
)
from hitch.main.views.messages import (
    _send_claude_follow_up as _send_claude_follow_up,
)
from hitch.main.views.messages import (
    _start_claude_fix_pr_workflow as _start_claude_fix_pr_workflow,
)
from hitch.main.views.messages import (
    _start_claude_qa_workflow as _start_claude_qa_workflow,
)
from hitch.main.views.messages import (
    _start_claude_spec_critic_follow_up as _start_claude_spec_critic_follow_up,
)
from hitch.main.views.messages import (
    _stored_model_and_effort as _stored_model_and_effort,
)
from hitch.main.views.messages import (
    _TurnRejectedError as _TurnRejectedError,
)
from hitch.main.views.messages import (
    send_message as send_message,
)
from hitch.main.views.new_session import (
    _accept_proposed_session_for_session as _accept_proposed_session_for_session,
)
from hitch.main.views.new_session import (
    _auto_merge_to_local_branch_for_proposal as _auto_merge_to_local_branch_for_proposal,
)
from hitch.main.views.new_session import (
    _candidate_proposal_continuation_prompt as _candidate_proposal_continuation_prompt,
)
from hitch.main.views.new_session import (
    _candidate_session_to_continue_from_proposal as _candidate_session_to_continue_from_proposal,
)
from hitch.main.views.new_session import (
    _candidate_thread_user_message_index as _candidate_thread_user_message_index,
)
from hitch.main.views.new_session import (
    _claim_candidate_proposal_start as _claim_candidate_proposal_start,
)
from hitch.main.views.new_session import (
    _claim_new_session_proposal_start as _claim_new_session_proposal_start,
)
from hitch.main.views.new_session import (
    _cleanup_worktree_quietly as _cleanup_worktree_quietly,
)
from hitch.main.views.new_session import (
    _finish_candidate_proposal_start as _finish_candidate_proposal_start,
)
from hitch.main.views.new_session import (
    _finish_new_session_proposal_start_claim as _finish_new_session_proposal_start_claim,
)
from hitch.main.views.new_session import (
    _new_session_post_settings as _new_session_post_settings,
)
from hitch.main.views.new_session import (
    _NewSessionTarget as _NewSessionTarget,
)
from hitch.main.views.new_session import (
    _next_user_message_index_for_candidate_thread as _next_user_message_index_for_candidate_thread,
)
from hitch.main.views.new_session import (
    _post_new_session as _post_new_session,
)
from hitch.main.views.new_session import (
    _posted_bool_override as _posted_bool_override,
)
from hitch.main.views.new_session import (
    _posted_new_session_coding_agent as _posted_new_session_coding_agent,
)
from hitch.main.views.new_session import (
    _posted_new_session_target as _posted_new_session_target,
)
from hitch.main.views.new_session import (
    _posted_proposed_session_for_new_session as _posted_proposed_session_for_new_session,
)
from hitch.main.views.new_session import (
    _posted_web_search_override as _posted_web_search_override,
)
from hitch.main.views.new_session import (
    _prefill_bare_repo_cwd_for_new_session_page as _prefill_bare_repo_cwd_for_new_session_page,
)
from hitch.main.views.new_session import (
    _prefill_project_for_new_session_page as _prefill_project_for_new_session_page,
)
from hitch.main.views.new_session import (
    _proposed_session_for_new_session_page as _proposed_session_for_new_session_page,
)
from hitch.main.views.new_session import (
    _remember_repo_and_redirect as _remember_repo_and_redirect,
)
from hitch.main.views.new_session import (
    _render_new_session_page as _render_new_session_page,
)
from hitch.main.views.new_session import (
    _reset_candidate_proposal_start_claim as _reset_candidate_proposal_start_claim,
)
from hitch.main.views.new_session import (
    _reset_new_session_proposal_start_claim as _reset_new_session_proposal_start_claim,
)
from hitch.main.views.new_session import (
    _start_candidate_proposal_session as _start_candidate_proposal_session,
)
from hitch.main.views.new_session import (
    new_session as new_session,
)
from hitch.main.views.session_actions import (
    _apply_live_session_approval_mode as _apply_live_session_approval_mode,
)
from hitch.main.views.session_actions import (
    _mark_workflow_failed as _mark_workflow_failed,
)
from hitch.main.views.session_actions import (
    register_session_demo as register_session_demo,
)
from hitch.main.views.session_actions import (
    session_demo_proxy as session_demo_proxy,
)
from hitch.main.views.session_actions import (
    session_demo_proxy_root as session_demo_proxy_root,
)
from hitch.main.views.session_actions import (
    set_session_approval_mode as set_session_approval_mode,
)
from hitch.main.views.session_actions import (
    set_session_archived as set_session_archived,
)
from hitch.main.views.session_actions import (
    set_session_name as set_session_name,
)
from hitch.main.views.session_actions import (
    set_session_project as set_session_project,
)
from hitch.main.views.session_actions import (
    start_session_demo as start_session_demo,
)
from hitch.main.views.session_detail import (
    _cached_intermediate_detail as _cached_intermediate_detail,
)
from hitch.main.views.session_detail import (
    _rollout_intermediate_entry_for_detail as _rollout_intermediate_entry_for_detail,
)
from hitch.main.views.session_detail import (
    _session_intermediate_allows_demo_entries as _session_intermediate_allows_demo_entries,
)
from hitch.main.views.session_detail import (
    session as session,
)
from hitch.main.views.session_detail import (
    session_intermediate as session_intermediate,
)
from hitch.main.views.session_detail import (
    session_stream as session_stream,
)
from hitch.main.views.session_list import (
    _SESSION_PAGE_SIZE as _SESSION_PAGE_SIZE,
)
from hitch.main.views.session_list import (
    QAActivityPageState as QAActivityPageState,
)
from hitch.main.views.session_list import (
    SessionListPage as SessionListPage,
)
from hitch.main.views.session_list import (
    SessionPageSource as SessionPageSource,
)
from hitch.main.views.session_list import (
    ThreadListPage as ThreadListPage,
)
from hitch.main.views.session_list import (
    VisibleSessionPage as VisibleSessionPage,
)
from hitch.main.views.session_list import (
    _add_thread_derived_hidden_ids as _add_thread_derived_hidden_ids,
)
from hitch.main.views.session_list import (
    _clear_cursor_params as _clear_cursor_params,
)
from hitch.main.views.session_list import (
    _cursor_done_param as _cursor_done_param,
)
from hitch.main.views.session_list import (
    _cursor_offset_param as _cursor_offset_param,
)
from hitch.main.views.session_list import (
    _materialized_session_list_page_from_codex as _materialized_session_list_page_from_codex,
)
from hitch.main.views.session_list import (
    _materialized_visible_session_page as _materialized_visible_session_page,
)
from hitch.main.views.session_list import (
    _merged_session_list_page_from_codex as _merged_session_list_page_from_codex,
)
from hitch.main.views.session_list import (
    _next_sessions_url as _next_sessions_url,
)
from hitch.main.views.session_list import (
    _page_has_cross_page_qa_activity as _page_has_cross_page_qa_activity,
)
from hitch.main.views.session_list import (
    _peek_source_session as _peek_source_session,
)
from hitch.main.views.session_list import (
    _pop_source_session as _pop_source_session,
)
from hitch.main.views.session_list import (
    _positive_int as _positive_int,
)
from hitch.main.views.session_list import (
    _project_for_thread_cached as _project_for_thread_cached,
)
from hitch.main.views.session_list import (
    _qa_activity_page_state as _qa_activity_page_state,
)
from hitch.main.views.session_list import (
    _request_uses_codex_cursor as _request_uses_codex_cursor,
)
from hitch.main.views.session_list import (
    _request_uses_index_cursor as _request_uses_index_cursor,
)
from hitch.main.views.session_list import (
    _session_index_sources_complete as _session_index_sources_complete,
)
from hitch.main.views.session_list import (
    _session_list_page as _session_list_page,
)
from hitch.main.views.session_list import (
    _session_list_page_from_codex as _session_list_page_from_codex,
)
from hitch.main.views.session_list import (
    _session_list_page_from_codex_or_warm_index as _session_list_page_from_codex_or_warm_index,
)
from hitch.main.views.session_list import (
    _session_list_page_from_index as _session_list_page_from_index,
)
from hitch.main.views.session_list import (
    _session_list_page_from_warm_index as _session_list_page_from_warm_index,
)
from hitch.main.views.session_list import (
    _session_page_source as _session_page_source,
)
from hitch.main.views.session_list import (
    _session_row_for_thread as _session_row_for_thread,
)
from hitch.main.views.session_list import (
    _SessionListQuery as _SessionListQuery,
)
from hitch.main.views.session_list import (
    _set_cursor_params as _set_cursor_params,
)
from hitch.main.views.session_list import (
    _source_next_cursor as _source_next_cursor,
)
from hitch.main.views.session_list import (
    _system_session_list_page_from_index as _system_session_list_page_from_index,
)
from hitch.main.views.session_list import (
    _thread_list_page as _thread_list_page,
)
from hitch.main.views.session_list import (
    _valid_codex_session_id as _valid_codex_session_id,
)
from hitch.main.views.session_list import (
    _visible_session_page_from_codex as _visible_session_page_from_codex,
)
from hitch.main.views.session_list import (
    inbox as inbox,
)
from hitch.main.views.session_list import (
    index as index,
)
from hitch.main.views.session_list import (
    system_session as system_session,
)
from hitch.main.views.session_list import (
    system_sessions as system_sessions,
)
from hitch.main.views.session_list import (
    usage as usage,
)
from hitch.main.views.settings import (
    _VALID_PROJECT_AUTO_PR_MODES as _VALID_PROJECT_AUTO_PR_MODES,
)
from hitch.main.views.settings import (
    _apply_live_global_approval_mode as _apply_live_global_approval_mode,
)
from hitch.main.views.settings import (
    _associate_existing_sessions_with_project as _associate_existing_sessions_with_project,
)
from hitch.main.views.settings import (
    _creatable_project_repos as _creatable_project_repos,
)
from hitch.main.views.settings import (
    _matching_project_exists as _matching_project_exists,
)
from hitch.main.views.settings import (
    _parse_disk_usage_max_percent as _parse_disk_usage_max_percent,
)
from hitch.main.views.settings import (
    _save_disk_usage_max_percent as _save_disk_usage_max_percent,
)
from hitch.main.views.settings import (
    _validate_settings_against_models as _validate_settings_against_models,
)
from hitch.main.views.settings import (
    edit_project as edit_project,
)
from hitch.main.views.settings import (
    new_project as new_project,
)
from hitch.main.views.settings import (
    update_archived_session_visibility as update_archived_session_visibility,
)
from hitch.main.views.settings import (
    update_settings as update_settings,
)
from hitch.main.views.settings import (
    update_visible_session_projects as update_visible_session_projects,
)
from hitch.main.workflows import (
    system_agents as system_agents,
)
from hitch.main.workflows.pr_stage import (
    _latest_pr_workflow_for_thread as _latest_pr_workflow_for_thread,
)
