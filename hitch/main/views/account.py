"""Auth, profile, health dashboard, and admin endpoints."""
from typing import Any

from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseForbidden,
)
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from hitch.main.models import (
    UserSettings,
)
from hitch.main.runtime import health, reconciliation
from hitch.main.sessions.project_visibility import (
    _metadata_by_thread_id as _metadata_by_thread_id,
)
from hitch.main.sessions.session_settings import (
    _authenticated_user,
    _settings_for_user,
    _stored_settings,
)
from hitch.main.sessions.settings_cookies import (
    _apply_cookie_updates,
    _settings_cookie_updates,
    _valid_cookie_setting_updates,
)
from hitch.main.views import common


@require_http_methods(["GET", "POST"])
def register(request: HttpRequest) -> HttpResponse:
    if _authenticated_user(request) is not None:
        return redirect("index")
    form: Any
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            _import_cookie_settings_to_user(request, user)
            auth_login(request, user)
            response = redirect("index")
            _apply_cookie_updates(
                response, _settings_cookie_updates(_stored_settings(request))
            )
            return response
    else:
        form = UserCreationForm()
    return render(
        request,
        "register.html",
        {"form": form, "login_url": reverse("login")},
    )

@require_http_methods(["GET", "POST"])
def login(request: HttpRequest) -> HttpResponse:
    if _authenticated_user(request) is not None:
        return redirect("index")
    next_url = common._safe_next_url(request)
    form: Any
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            _import_cookie_settings_to_user(request, user)
            auth_login(request, user)
            response = redirect(next_url or "index")
            _apply_cookie_updates(
                response, _settings_cookie_updates(_stored_settings(request))
            )
            return response
    else:
        form = AuthenticationForm(request)
    return render(
        request,
        "login.html",
        {
            "form": form,
            "next": next_url,
            "register_url": reverse("register"),
        },
    )

@require_http_methods(["GET"])
def profile(request: HttpRequest) -> HttpResponse:
    user = _authenticated_user(request)
    usage_context = _profile_usage_context(request)
    profile_name = user.get_username() if user is not None else "anonymous"
    response = render(
        request,
        "profile.html",
        {
            "profile_name": profile_name,
            "profile_status": "Signed in" if user is not None else "Signed out",
            "logout_url": reverse("logout") if user is not None else "",
            "nuke_codex_url": reverse("nuke_codex") if user is not None else "",
            "health_url": reverse("health_dashboard") if user is not None else "",
            "nuked_count": _parse_nuked_count(request.GET.get("nuked")),
            **usage_context.template_context,
        },
    )
    _apply_cookie_updates(response, usage_context.cookie_updates)
    return response

def _parse_nuked_count(raw: str | None) -> int | None:
    """Parse the ``?nuked=N`` confirmation count the nuke action redirects with.

    Returns ``None`` (render no confirmation) for a missing or malformed value
    so a hand-edited URL cannot inject arbitrary text into the page.
    """
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None

def _profile_usage_context(request: HttpRequest) -> common.UsageContext:
    try:
        return common._usage_context(request)
    except Exception:
        common.logger.exception("failed to load profile usage context; showing empty usage state")
    settings_context = common._settings_context(_stored_settings(request), [])
    return common.UsageContext(
        template_context={
            "login_url": reverse("login"),
            "register_url": reverse("register"),
            "rate_limits": None,
            "lifetime_usage": None,
            **settings_context,
        },
        cookie_updates={},
    )

@require_http_methods(["GET"])
def health_dashboard(request: HttpRequest) -> HttpResponse:
    """Hitch health dashboard: leak and backlog signals on one page.

    Linked from the bottom of the profile page. Requires authentication since
    it exposes operational internals. The copy block is built to be long-pressed
    and pasted into a chat with the assistant when diagnosing issues.
    """
    if _authenticated_user(request) is None:
        return redirect(f"{reverse('login')}?next={reverse('health_dashboard')}")
    report = health.collect_health_report()
    return render(
        request,
        "health.html",
        {
            "report": report,
            "copy_text": report.copy_text(),
            "profile_url": reverse("profile"),
        },
    )

@require_http_methods(["POST"])
def logout(request: HttpRequest) -> HttpResponse:
    values = _stored_settings(request) if _authenticated_user(request) is not None else None
    auth_logout(request)
    response = redirect("index")
    if values is not None:
        _apply_cookie_updates(response, _settings_cookie_updates(values))
    return response

@require_http_methods(["POST"])
def nuke_codex(request: HttpRequest) -> HttpResponse:
    """SIGKILL every Codex app-server Hitch started, then return to the profile.

    Manual cleanup for leaked app-servers contending on the shared CODEX_HOME
    state-DB lock. The killed count is round-tripped through a query param so
    the profile page can confirm the outcome.
    """
    if _authenticated_user(request) is None:
        return HttpResponseForbidden("authentication required")
    killed = reconciliation.nuke_codex_app_servers()
    return redirect(f"{reverse('profile')}?nuked={killed}")

def _import_cookie_settings_to_user(request: HttpRequest, user: Any) -> UserSettings:
    settings = _settings_for_user(user)
    updates: list[str] = []
    for field, value in _valid_cookie_setting_updates(request).items():
        if getattr(settings, field) != value:
            setattr(settings, field, value)
            updates.append(field)
    if updates:
        settings.save(update_fields=[*updates, "updated_at"])
    return settings
