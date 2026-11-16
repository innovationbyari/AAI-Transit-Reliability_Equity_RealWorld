"""Agentic AI for Public Transit Reliability and Equity — runtime loader.

The original Colab notebook keeps every section in one shared namespace (later cells use
names defined by earlier ones, and some functions rebind module globals). To keep the
behaviour identical, `load()` executes each section module, in notebook order, inside a
single module object: ``transit_ai.runtime``.

    from transit_ai import load
    rt = load()                 # runs Setup → E1 → CV → A → E2 → C → D → E3 → B → E4 → E5
    demo = rt.build_dashboard() # Gradio Blocks
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES: list[str] = json.loads((HERE / "_modules.json").read_text())


def param(name, default):
    """Notebook form fields (#@param) become environment-variable overrides, cast to the default's type."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


_RUNTIME: types.ModuleType | None = None


def load(verbose: bool = True) -> types.ModuleType:
    """Run all section modules once and return the shared runtime module."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    import matplotlib
    matplotlib.use("Agg")  # no display on a server; plt.show() becomes a no-op

    rt = types.ModuleType("transit_ai.runtime")
    rt.__file__ = str(HERE / "__init__.py")
    sys.modules[rt.__name__] = rt  # pydantic / dataclasses resolve annotations through sys.modules
    ns = rt.__dict__
    ns["param"] = param
    # SETUP 2 (installing Ollama inside Colab) is not part of the web app; allow it only if a server is reachable.
    ns["USE_OLLAMA"] = param("USE_OLLAMA", False)
    ns["OLLAMA_MODEL"] = param("OLLAMA_MODEL", "llama3.2:3b")

    for name in MODULES:
        path = HERE / f"{name}.py"
        if verbose:
            print(f"── loading {name}", flush=True)
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    _RUNTIME = rt
    return rt
