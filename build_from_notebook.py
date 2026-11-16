"""Split the Colab notebook into deployable modules (run once; output in ../repo)."""
import json, re, sys, pathlib, shutil

NB = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
cells = json.load(open(NB))["cells"]
src = lambda i: "".join(cells[i]["source"])

MODULES = [
    ("setup_llm", "SETUP 3 · Live LLM client (OpenAI → OpenRouter → Hugging Face → templates)", [6]),
    ("foundations", "Section E1 · Shared foundations: helpers, versioned policy rules, synthetic network, scenarios", [8, 9, 10, 12]),
    ("vision", "Section CV · Computer vision passenger counting (counts only, no images kept)", [14]),
    ("data_inputs", "Section A · Live connectors (NJ Transit, Census ACS, NWS), data inputs, SLM, hand-off adapter", [16, 17, 18]),
    ("engine", "Section E2 · Deterministic validation, exception paths and scoring engine", [20]),
    ("ethics", "Section C · Ethics training and compliance monitor (schemas, tools, agent, rosters)", [22, 23, 24, 25]),
    ("conflict_of_interest", "Section D · Conflict-of-interest detection (schemas, agents, disclosures)", [28, 29, 30]),
    ("governance", "Integration · governance_check(): ethics gate (C) + conflict-of-interest gate (D)", [33]),
    ("case_store", "Section E3 · Shared case state, explanation, human review, outcomes, policy governance", [35, 36]),
    ("agents_workflow", "Section B · Typed agents, rule steps, HITL forms, agent-level runner, audit trail", [39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 67]),
    ("outputs", "Section E4 · Outputs: interactive map, tables and charts, export", [100]),
    ("handlers", "Section E5 · Dashboard actions (UI-free; the Streamlit layout is in streamlit_app.py)", [101]),
]

# Server-friendly defaults that differ from the notebook
OVERRIDES = {"SHOW_CV_PREVIEW": "False"}

PARAM = re.compile(r'^(?P<name>[A-Z_][A-Z0-9_]*)\s*=\s*(?P<val>.+?)\s*#@param.*$')

def transform(code, cell_no):
    out = []
    for line in code.split("\n"):
        if line.startswith("!"):            # Colab shell magics (pip install) → requirements.txt
            out.append("# " + line + "   (handled by requirements.txt)")
            continue
        if line.startswith("#@title"):
            out.append("# " + "-" * 100 + "\n# " + line[len("#@title"):].strip() + f"   [notebook cell {cell_no}]\n# " + "-" * 100)
            continue
        m = PARAM.match(line)
        if m:
            name, val = m["name"], OVERRIDES.get(m["name"], m["val"])
            out.append(f'{name} = param("{name}", {val})  # env-var override; notebook form field')
            continue
        out.append(line)
    code = "\n".join(out)
    if cell_no == 67:   # drop the notebook-only display block at the end of B·9
        code = code.split("\nif wf is not None and wf.case:")[0] + "\n"
    if cell_no == 100:  # CARTO basemaps now need an API key: send CARTO_API_KEY, or fall back to OpenStreetMap tiles
        code = code.replace(
            'm = folium.Map(location=list(CITY_CENTER), zoom_start=13, tiles="CartoDB positron", control_scale=True)',
            'm = folium.Map(location=list(CITY_CENTER), zoom_start=13, tiles=None, control_scale=True)\n    add_basemap(m)')
        code = code.replace("\ndef build_map(case):", BASEMAP + "\n\ndef build_map(case):", 1)
    if cell_no == 101:  # keep the dashboard's actions and helpers; drop the Gradio layout (now streamlit_app.py)
        code = strip_gradio(code)
    code = code.replace("in Colab Secrets", "in the app secrets").replace("Colab Secrets", "app secrets (environment variables)")
    return code

BASEMAP = '''
def add_basemap(m):
    """CARTO Positron with the CARTO_API_KEY secret (free key: carto.com/basemaps/apikey); OpenStreetMap otherwise."""
    key = read_secret("CARTO_API_KEY")
    if key:
        folium.TileLayer(
            tiles="https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}{r}.png?key=" + key,
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            name="CARTO Positron", max_zoom=20, subdomains="abcd").add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
'''


def strip_gradio(code):
    import ast
    tree = ast.parse(code)
    drop = {"refresh", "action", "build_dashboard", "_notify", "CSS"}
    lines = code.split("\n")
    kill = set()
    for n in tree.body:
        name = getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
        is_gr_import = isinstance(n, ast.Import) and any(a.name == "gradio" for a in n.names)
        is_build = isinstance(n, (ast.Assign, ast.Expr)) and "build_dashboard()" in ast.get_source_segment(code, n) or \
            isinstance(n, ast.Expr) and "Dashboard built" in ast.get_source_segment(code, n)
        if name in drop or is_gr_import or is_build:
            start = n.lineno - 1 - len(getattr(n, "decorator_list", []))
            kill.update(range(start, n.end_lineno))
    out = "\n".join(l for i, l in enumerate(lines) if i not in kill)
    if "gr." in out:
        raise SystemExit("gradio reference left in handlers: " + ", ".join(sorted(set(l.strip()[:60] for l in out.split("\n") if "gr." in l))))
    return re.sub(r"\n{4,}", "\n\n\n", out)


HEADER = '''"""{title}

Generated from the notebook (cells {cells}). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa
'''

pkg = OUT / "transit_ai"; pkg.mkdir(parents=True, exist_ok=True)
for mod, title, idx in MODULES:
    body = "\n\n\n".join(transform(src(i), i) for i in idx)
    if mod == "agents_workflow":
        body = "wf = None   # notebook B·8 runner is not used in the web app\n\n" + body
    (pkg / f"{mod}.py").write_text(HEADER.format(title=title, cells=", ".join(map(str, idx))) + "\n" + body.rstrip() + "\n")
# Hand-written modules (not generated from the notebook) load last, in this order.
EXTRA_MODULES = ["live_ops"]
(OUT / "transit_ai" / "_modules.json").write_text(json.dumps([m for m, _, _ in MODULES] + EXTRA_MODULES))
print("wrote", len(MODULES), "modules")
