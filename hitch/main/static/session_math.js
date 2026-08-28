(function () {
    "use strict";

    const MATH_BODY_SELECTOR = [
        ".message.agent:not(.streaming) > .body",
        ".message.thinking:not(.streaming) > .body",
    ].join(", ");
    const OPTIONS = {
        delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true },
        ],
        throwOnError: true,
        strict: "ignore",
        trust: false,
        maxExpand: 1000,
        maxSize: 20,
    };

    function mathBodies(root) {
        const bodies = [];
        if (root instanceof Element && root.matches(MATH_BODY_SELECTOR)) bodies.push(root);
        if (typeof root.querySelectorAll === "function") {
            bodies.push(...root.querySelectorAll(MATH_BODY_SELECTOR));
        }
        return bodies;
    }

    function render(root) {
        if (typeof window.renderMathInElement !== "function") return;
        for (const body of mathBodies(root)) {
            if (body.dataset.mathRendered === "true") continue;
            const source = body.innerText || body.textContent || "";
            window.renderMathInElement(body, OPTIONS);
            if (body.querySelector(".katex")) body.dataset.copyText = source;
            body.dataset.mathRendered = "true";
        }
    }

    window.hitchMath = { render };
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => render(document));
    } else {
        render(document);
    }
})();
