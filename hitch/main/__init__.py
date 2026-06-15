"""The ``hitch.main`` Django app package.

Some modules were relocated out of this package into sub-packages
(``runtime`` and ``workflows``). Tests and older callers still reference the
legacy dotted paths ``hitch.main.codex_pool`` and ``hitch.main.system_agents``
(both for ``from hitch.main import ...`` and ``mock.patch("hitch.main.<name>.X")``).

To keep those working without re-importing the relocated modules at Django
app-discovery time (which would trigger circular imports / premature app
loading), the legacy names are resolved lazily via :pep:`562` ``__getattr__``
and, on first access, the resolved module object is registered in
``sys.modules`` under the legacy dotted name so ``importlib.import_module`` (used
by ``mock.patch``) finds the *same* module object.
"""

import sys
from types import ModuleType

_LEGACY_MODULE_ALIASES = {
    "codex_pool": "hitch.main.runtime.codex_pool",
    "system_agents": "hitch.main.workflows.system_agents",
}


def __getattr__(name: str) -> ModuleType:  # PEP 562 module hook
    target = _LEGACY_MODULE_ALIASES.get(name)
    if target is not None:
        import importlib

        module = importlib.import_module(target)
        # Register under the legacy dotted name so ``mock.patch`` resolving
        # e.g. ``hitch.main.codex_pool`` lands on this same module object.
        sys.modules.setdefault(f"hitch.main.{name}", module)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'hitch.main' has no attribute {name!r}")
