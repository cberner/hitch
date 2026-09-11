(() => {
    const picker = document.querySelector("[data-agent-picker]");
    if (!picker) return;
    const select = picker.querySelector("[data-agent-select]");
    const panel = document.querySelector("[data-agent-transcript]");
    const heading = panel.querySelector("[data-agent-heading]");
    const entries = panel.querySelector("[data-agent-entries]");
    const older = panel.querySelector("[data-agent-older]");
    const earlierButton = panel.querySelector("[data-agent-earlier]");
    const fullButton = panel.querySelector("[data-agent-full]");
    const latestButton = panel.querySelector("[data-agent-latest]");
    const feedback = document.querySelector("[data-agent-feedback]");
    let selected = new URL(location.href).searchParams.get("agent") || "";
    if (selected === picker.dataset.mainAgentId) selected = "";
    let nextUrl = "";
    let browsingHistory = false;
    let parentEnded = false;
    let lastHtml = null;
    let controller = null;
    let timer = null;

    function showFeedback(message) {
        feedback.textContent = message;
        feedback.hidden = !message;
    }

    function showSelection() {
        document.body.classList.toggle("viewing-subagent", Boolean(selected));
        panel.hidden = !selected;
        const url = new URL(location.href);
        if (selected) url.searchParams.set("agent", selected);
        else url.searchParams.delete("agent");
        history.replaceState(null, "", url);
    }

    function formatTranscript(root) {
        document.dispatchEvent(new CustomEvent("hitch:transcript-updated", { detail: { root } }));
    }

    function renderAgents(agents) {
        const options = [new Option("Main agent", "")];
        for (const agent of agents) {
            const nested = agent.parent_id !== picker.dataset.mainAgentId ? "↳ " : "";
            options.push(new Option(nested + agent.name + (agent.role ? ` (${agent.role})` : ""), agent.id));
        }
        if (selected && !agents.some(agent => agent.id === selected)) {
            options.push(new Option("Selected subagent", selected));
        }
        if (options.length !== select.options.length || options.some((option, index) =>
            option.value !== select.options[index].value || option.text !== select.options[index].text
        )) select.replaceChildren(...options);
        select.value = selected;
    }

    async function refresh({ previous = false, full = false } = {}) {
        clearTimeout(timer);
        if (controller) controller.abort();
        const current = new AbortController();
        controller = current;
        const url = new URL(previous ? nextUrl : picker.dataset.agentsUrl, location.href);
        if (selected) url.searchParams.set("agent", selected);
        if (full) url.searchParams.set("history", "all");
        earlierButton.disabled = true;
        panel.setAttribute("aria-busy", "true");
        try {
            const response = await fetch(url, {
                signal: current.signal,
                headers: { "X-Requested-With": "XMLHttpRequest" },
            });
            if (!response.ok) throw new Error("Unable to load agents. Use Refresh to try again.");
            const data = await response.json();
            if (current.signal.aborted) return;
            renderAgents(data.agents);
            showFeedback("");
            if (selected && data.selected === selected) {
                if (!previous && !full && browsingHistory) return;
                heading.textContent = data.name + (data.role ? ` · ${data.role}` : "");
                if (previous) {
                    const page = document.createElement("div");
                    page.innerHTML = data.html;
                    const previousHeight = document.documentElement.scrollHeight;
                    older.prepend(page);
                    formatTranscript(page);
                    window.scrollBy(0, document.documentElement.scrollHeight - previousHeight);
                    browsingHistory = true;
                    latestButton.hidden = false;
                } else if (full || data.html !== lastHtml) {
                    const selection = window.getSelection();
                    // Keep the history cursor tied to the messages actually on screen.
                    if (!full && selection && !selection.isCollapsed && panel.contains(selection.anchorNode)) return;
                    const openDetails = [...entries.querySelectorAll("details")].map(detail => detail.open);
                    entries.innerHTML = data.html.trim() ? data.html : "<p>No messages from this subagent yet.</p>";
                    entries.querySelectorAll("details").forEach((detail, index) => {
                        detail.open = Boolean(openDetails[index]);
                    });
                    formatTranscript(entries);
                    lastHtml = data.html;
                }
                if (full) {
                    older.replaceChildren();
                    browsingHistory = true;
                    latestButton.hidden = false;
                }
                nextUrl = data.next_url;
                earlierButton.hidden = !nextUrl;
                fullButton.hidden = !data.partial;
            }
        } catch (error) {
            if (controller === current && error.name !== "AbortError") showFeedback(error.message);
        } finally {
            if (controller === current) {
                controller = null;
                earlierButton.disabled = false;
                panel.removeAttribute("aria-busy");
                if (picker.dataset.active === "true" || selected) {
                    timer = setTimeout(() => {
                        if (!document.hidden && !browsingHistory) refresh();
                    }, 5000);
                }
            }
        }
    }

    function resetTranscript() {
        browsingHistory = false;
        nextUrl = "";
        lastHtml = null;
        older.replaceChildren();
        entries.replaceChildren();
        earlierButton.hidden = true;
        fullButton.hidden = true;
        latestButton.hidden = !parentEnded;
        heading.textContent = "Loading subagent…";
    }

    select.addEventListener("change", () => {
        selected = select.value;
        resetTranscript();
        showSelection();
        if (!selected && parentEnded) {
            location.reload();
            return;
        }
        refresh();
    });
    picker.querySelector("[data-agents-refresh]").addEventListener("click", () => refresh());
    earlierButton.addEventListener("click", () => refresh({ previous: true }));
    fullButton.addEventListener("click", () => refresh({ full: true }));
    latestButton.addEventListener("click", () => {
        if (parentEnded) {
            location.reload();
            return;
        }
        resetTranscript();
        refresh();
    });
    document.addEventListener("hitch:session-ended", event => {
        if (!selected) return;
        event.preventDefault();
        parentEnded = true;
        picker.dataset.active = "false";
        latestButton.hidden = false;
    });
    document.addEventListener("visibilitychange", () => {
        if (!document.hidden && !browsingHistory) refresh();
    });
    window.addEventListener("pagehide", () => {
        clearTimeout(timer);
        if (controller) controller.abort();
    });
    resetTranscript();
    showSelection();
    renderAgents([]);
    refresh();
})();
