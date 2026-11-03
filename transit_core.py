"""transit_core — Agentic AI for Public Transit Reliability and Equity (shared library).

Generated from the notebook's core cells (one source of truth). Used by:
  * the Colab notebook (Run all),
  * app_gradio.py    (Gradio / Hugging Face Spaces),
  * app_streamlit.py (Streamlit Community Cloud).

Live data only: NJ Transit GTFS-realtime + static GTFS, US Census ACS, National Weather Service. Inputs that no live
source provides are left missing and are never imputed. Secrets are read from Colab Secrets, Streamlit secrets or
environment variables: NJ_TRANSIT_USERNAME, NJ_TRANSIT_PASSWORD, CENSUS_API_KEY (optional), NWS_CONTACT_EMAIL
(optional), and optional LLM keys OPENAI_API_KEY / OPENROUTER_API_KEY / HF_API_KEY.
"""


# ====================================================================================================
# cell: s3_llm_core
# ====================================================================================================
#@title SETUP 3 · Live LLM client (OpenAI by default → OpenRouter → Hugging Face → templates)
import os, json, requests
from datetime import datetime, timezone
try:
    USE_OLLAMA
except NameError:
    USE_OLLAMA = False
try:
    OLLAMA_MODEL
except NameError:
    OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
iso = globals().get("iso") or (lambda dt: dt.isoformat(timespec="seconds"))
utcnow = globals().get("utcnow") or (lambda: datetime.now(timezone.utc))

LLM_PROVIDER = "openai"  #@param ["openai", "openrouter", "huggingface", "ollama"]
LLM_FALLBACKS = "openrouter, huggingface"  #@param {type:"string"}
LLM_CUSTOM_MODEL = ""  #@param {type:"string"}
LLM_STRUCT_MODE = "auto"  #@param ["auto", "native", "prompt"]
LLM_MAX_CALLS = 300  #@param {type:"integer"}
# Hosted apps (Streamlit / Gradio) can override the form values with environment variables of the same name.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", LLM_PROVIDER)
LLM_FALLBACKS = os.environ.get("LLM_FALLBACKS", LLM_FALLBACKS)
LLM_CUSTOM_MODEL = os.environ.get("LLM_CUSTOM_MODEL", LLM_CUSTOM_MODEL)
LLM_MAX_CALLS = int(os.environ.get("LLM_MAX_CALLS", LLM_MAX_CALLS))
import re, time, warnings
from datetime import date
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Keys are read only from Colab Secrets or environment variables; they are never printed or saved.
CATALOG = {
    "openai": {
        "label": "OpenAI", "url": "https://api.openai.com/v1/chat/completions", "key_name": "OPENAI_API_KEY",
        "key_help": "https://platform.openai.com/api-keys", "token_param": "max_completion_tokens",
        "models": [("gpt-4.1-mini", "Fast, low cost, reliable JSON-schema output")],
    },
    "openrouter": {
        "label": "OpenRouter", "url": "https://openrouter.ai/api/v1/chat/completions", "key_name": "OPENROUTER_API_KEY",
        "key_help": "https://openrouter.ai/keys", "token_param": "max_tokens",
        "models": [("openrouter/free", "Free-models router: picks an available free model")],
    },
    "huggingface": {
        "label": "Hugging Face", "url": "https://router.huggingface.co/v1/chat/completions", "key_name": "HF_API_KEY", "alt_key_names": ["HF_TOKEN"],
        "key_help": "https://huggingface.co/settings/tokens", "token_param": "max_tokens",
        "models": [("Qwen/Qwen2.5-72B-Instruct", "Open model with reliable structured output")],
    },
    "ollama": {
        "label": "Ollama (local, inside this Colab runtime)", "url": f"{OLLAMA_URL}/v1/chat/completions",
        "key_name": None, "key_help": "", "token_param": "max_tokens",
        "models": [(OLLAMA_MODEL, "Local model pulled in SETUP 2 (no key, no cost)")],
    },
}


def read_secret(name):
    """Colab Secrets, then Streamlit secrets (st.secrets), then environment variables
    (Hugging Face Spaces / Gradio and most hosts expose secrets as environment variables)."""
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v.strip()
    except Exception:
        pass
    try:
        import streamlit as _st
        if name in _st.secrets:
            return str(_st.secrets[name]).strip()
    except Exception:
        pass
    return (os.getenv(name) or "").strip()


_hf_key = read_secret("HF_API_KEY") or read_secret("HF_TOKEN")
if _hf_key:
    os.environ.setdefault("HF_TOKEN", _hf_key)          # lets the FLAN-T5 download in A2 authenticate too


def _provider_entry(name, model_override=""):
    cfg = CATALOG[name]
    key = "ollama" if cfg["key_name"] is None else next(
        (v for v in (read_secret(k) for k in [cfg["key_name"]] + cfg.get("alt_key_names", [])) if v), "")
    usable = bool(key) and (name != "ollama" or USE_OLLAMA)
    return {"provider": name, "label": cfg["label"], "url": cfg["url"], "key": key,
            "model": model_override or cfg["models"][0][0], "token_param": cfg["token_param"],
            "native_ok": True, "usable": usable, "timeout": 180 if name == "ollama" else 90}


_chain_names = [LLM_PROVIDER] + [p.strip() for p in LLM_FALLBACKS.split(",") if p.strip() in CATALOG and p.strip() != LLM_PROVIDER]
if USE_OLLAMA and "ollama" not in _chain_names:      # SETUP 2 started a local model: use it as the first fallback
    _chain_names.insert(1, "ollama")
LLM_CHAIN = [_provider_entry(n, LLM_CUSTOM_MODEL.strip() if n == LLM_PROVIDER else "") for n in _chain_names]
LLM_CHAIN = [p for p in LLM_CHAIN if p["usable"]]
LLM_READY = bool(LLM_CHAIN)
LLM = {"provider": LLM_CHAIN[0]["provider"] if LLM_READY else None, "model": LLM_CHAIN[0]["model"] if LLM_READY else None,
       "struct_mode": LLM_STRUCT_MODE, "native_used": False, "last_provider": None, "last_model": None}
LLM_STATS = {"calls": 0, "by_provider": {}, "failures": [], "fallbacks_used": 0, "provider_errors": {}}
print("LLM providers available (in order):", [f"{p['label']} · {p['model']}" for p in LLM_CHAIN] or "none → template text only")
for n in _chain_names:
    cfg = CATALOG[n]
    if cfg["key_name"] and not any(read_secret(k) for k in [cfg["key_name"]] + cfg.get("alt_key_names", [])):
        print(f"  {cfg['label']}: no {cfg['key_name']} in secrets (skipped)")


def _scrub(text):
    """Error bodies can echo part of a key; never print anything that looks like one."""
    return re.sub(r"(sk-|or-|hf_)[A-Za-z0-9_\-\*]{4,}", "[key hidden]", str(text))[:200]


class LLMUnavailable(RuntimeError):
    pass


def response_format_for(schema):
    """Native JSON-schema mode: hand the Pydantic schema to the provider."""
    return {"type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": False, "schema": schema.model_json_schema()}}


def _call_one(p, messages, temperature, max_tokens, schema):
    headers = {"Authorization": f"Bearer {p['key']}", "Content-Type": "application/json"}
    payload = {"model": p["model"], "messages": messages, "temperature": temperature, p["token_param"]: max_tokens}
    if p["provider"] == "openrouter":
        payload["reasoning"] = {"exclude": True}
    if schema is not None and LLM["struct_mode"] in ("native", "auto") and p["native_ok"]:
        payload["response_format"] = response_format_for(schema)
    last_error = None
    for attempt in range(3):
        try:
            r = requests.post(p["url"], headers=headers, json=payload, timeout=p["timeout"])
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise LLMUnavailable(f"auth failed (HTTP {r.status_code}); check {CATALOG[p['provider']]['key_name']}")
            if r.status_code == 429 and "insufficient_quota" in r.text:
                raise LLMUnavailable("quota exhausted (HTTP 429 insufficient_quota); add credit or billing for this key")
            if r.status_code in (400, 404, 415, 422) and "response_format" in payload and LLM["struct_mode"] == "auto":
                p["native_ok"] = False
                payload.pop("response_format")
                continue
            if r.status_code in (400, 404, 422):
                raise LLMUnavailable(f"request rejected (HTTP {r.status_code}): {_scrub(r.text)}")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if content:
                if "response_format" in payload:
                    LLM["native_used"] = True
                return content
            last_error = "empty response"
        except LLMUnavailable:
            raise
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
    raise LLMUnavailable(last_error or "no response")


def _looks_like_reasoning(text):
    """Some free models return their working notes instead of an answer; those are not used."""
    head = text.lstrip()[:400].lower()
    return any(k in head for k in ("analyze the request", "thinking process", "let me think", "**goal:**", "step 1: understand"))


def call_llm(messages, temperature=0.2, max_tokens=1800, schema=None):
    """One chat request. Tries each provider in LLM_CHAIN in order and records which one answered."""
    if not LLM_READY:
        raise LLMUnavailable("No LLM provider configured (add a key in Colab Secrets). Deterministic rules still work.")
    if LLM_STATS["calls"] >= LLM_MAX_CALLS:
        raise LLMUnavailable(f"LLM call budget reached ({LLM_MAX_CALLS}); raise LLM_MAX_CALLS to continue")
    errors = []
    for i, p in enumerate(LLM_CHAIN):
        if p.get("disabled"):                 # circuit breaker: a provider that failed auth/quota/model is skipped
            errors.append(f"{p['label']}: skipped ({p['disabled']})")
            continue
        try:
            LLM_STATS["calls"] += 1
            out = _call_one(p, messages, temperature, max_tokens, schema)
            if schema is None and _looks_like_reasoning(out):
                raise LLMUnavailable("returned its reasoning instead of an answer (not used)")
            LLM_STATS["by_provider"][p["label"]] = LLM_STATS["by_provider"].get(p["label"], 0) + 1
            LLM_STATS["fallbacks_used"] += int(i > 0)
            LLM.update(last_provider=p["label"], last_model=p["model"])
            return out
        except LLMUnavailable as e:
            errors.append(f"{p['label']}: {e}")
            LLM_STATS["provider_errors"][p["label"]] = _scrub(e)
            if re.search(r"auth failed|quota exhausted|request rejected", str(e)):
                p["disabled"] = _scrub(e)
    LLM_STATS["failures"].append({"time": iso(utcnow()), "errors": errors})
    raise LLMUnavailable("All LLM providers failed — " + " | ".join(errors))


def llm_generate(prompt, system, max_tokens=320):
    """Plain-text generation for agent narratives and notes (never used in any score or routing decision)."""
    txt = call_llm([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=max_tokens)
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    if not txt:
        raise LLMUnavailable("empty response")
    return txt


def llm_source():
    return f"{LLM['last_provider']} · {LLM['last_model']}" if LLM.get("last_provider") else "template"


def extract_json(text):
    """Find the JSON object in a model reply."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in the reply")
    return json.loads(text[start:end + 1])


def run_structured(schema, system, user, retries=2):
    """Get JSON that fits a Pydantic schema (native mode if enabled), validate it, retry with the error on failure."""
    schema_hint = json.dumps(schema.model_json_schema(), indent=1)
    messages = [
        {"role": "system", "content": system + "\n\nReturn ONLY one JSON object (no prose, no markdown) "
                                              "that matches this JSON schema:\n" + schema_hint},
        {"role": "user", "content": user},
    ]
    for attempt in range(retries + 1):
        raw = call_llm(messages, schema=schema)
        try:
            return schema.model_validate(extract_json(raw)), attempt
        except (ValueError, ValidationError) as e:          # bad JSON or schema mismatch
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That output was invalid: {str(e)[:500]}\nReturn corrected JSON only."}]
    raise RuntimeError(f"{schema.__name__}: model never produced valid JSON after {retries + 1} tries. "
                       f"Try a different model or output mode.")


def mode_used():
    """Which structured-output mode actually did the work in this session."""
    if LLM["struct_mode"] == "prompt":
        return "prompt + validate"
    return "native JSON-schema" if LLM["native_used"] else "prompt + validate (native unavailable)"


def draw_workflow(steps, notes=None, title=""):
    """Draw the workflow as boxes and arrows (identical helper in C and D, kept once).
    steps = [(key, label, kind)], kind in input/llm/tool/output."""
    from matplotlib.patches import FancyBboxPatch
    notes = notes or {}
    colors = {"input": "#E5E7EB", "llm": "#DBEAFE", "tool": "#FEF3C7", "output": "#DCFCE7"}
    tags = {"input": "INPUT", "llm": "LLM AGENT", "tool": "PYTHON TOOL", "output": "OUTPUT"}
    n, w, gap = len(steps), 1.9, 0.5
    fig, ax = plt.subplots(figsize=(min(17, 2.3 * n), 2.6))
    ax.set_xlim(0, n * (w + gap)); ax.set_ylim(0, 2.6); ax.axis("off")
    for i, (key, label, kind) in enumerate(steps):
        x = i * (w + gap) + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 1.15, boxstyle="round,pad=0.04", fc=colors[kind], ec="#334155", lw=1.2))
        ax.text(x + w / 2, 1.50, label, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2, 1.05, tags[kind], ha="center", va="center", fontsize=7, color="#475569")
        if key in notes:
            ax.text(x + w / 2, 0.45, notes[key], ha="center", va="center", fontsize=8, color="#0F766E")
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 1.32), xytext=(x + w + 0.02, 1.32),
                        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.4))
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    plt.tight_layout(); plt.show()


# ====================================================================================================
# cell: e1_imports
# ====================================================================================================
#@title E1 · Imports, runtime settings and helpers
import json, math, random, zlib, copy, threading, io, zipfile, os, html as _html, statistics
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests

try:
    USE_OLLAMA
except NameError:
    USE_OLLAMA = False
try:
    OLLAMA_MODEL
except NameError:
    OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# Service area: Newark Penn Station. Live vehicles within NJT_NEARBY_KM of this point are offered as cases, and the
# weather connectors use it. Override with SERVICE_AREA_LAT / SERVICE_AREA_LON environment variables.
SERVICE_AREA_LATLON = (float(os.environ.get("SERVICE_AREA_LAT", 40.7347)), float(os.environ.get("SERVICE_AREA_LON", -74.1642)))
CITY_CENTER = SERVICE_AREA_LATLON
# 1.0 = real review deadlines (5/10/20 min). Set e.g. 0.1 to make deadlines 10x shorter for demos.
DEMO_DEADLINE_SCALE = 1.0  #@param {type:"number"}
DEMO_DEADLINE_SCALE = float(os.environ.get("DEMO_DEADLINE_SCALE", DEMO_DEADLINE_SCALE))
# Every timestamp the notebook shows or saves is in this time zone (EST/EDT for New Jersey).
LOCAL_TZ_NAME = "America/New_York"  #@param {type:"string"}
LOCAL_TZ_NAME = os.environ.get("LOCAL_TZ_NAME", LOCAL_TZ_NAME)
from zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
OUTPUT_DIR = os.environ.get("TRANSIT_OUTPUT_DIR", "transit_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    """ISO time in the local zone, e.g. 2026-11-02T01:41:00-05:00 (the offset shows EST or EDT)."""
    return dt.astimezone(LOCAL_TZ).isoformat(timespec="seconds")


def local_now():
    return utcnow().astimezone(LOCAL_TZ)


def local_label(dt=None):
    """Readable local time with its zone name, e.g. '2026-11-02 01:41 EST'."""
    return (dt or utcnow()).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z")


def parse_iso(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def r3(x):
    return None if x is None else round(float(x), 3)


def n_benefit(x, L, U):
    """n(x) = clip((x - L) / (U - L), 0, 1). Missing values return None (never imputed silently)."""
    if x is None:
        return None
    if U == L:
        return 0.0
    return float(min(1.0, max(0.0, (float(x) - L) / (U - L))))


def n_cost(x, L, U):
    """ninv(x) = 1 - n(x) for cost or delay burden."""
    v = n_benefit(x, L, U)
    return None if v is None else 1.0 - v

KM_LAT = 110.574


def km_between(a, b):
    """Approximate distance in km between two (lat, lon) points (fine at city scale)."""
    lat0 = math.radians((a[0] + b[0]) / 2)
    dx = (b[1] - a[1]) * 111.320 * math.cos(lat0)
    dy = (b[0] - a[0]) * KM_LAT
    return math.hypot(dx, dy)


class LiveDataUnavailable(RuntimeError):
    """Raised when a required live source (NJ Transit, Census, NWS) gave no data. There is no sample fallback."""


# ====================================================================================================
# cell: e1_policy
# ====================================================================================================
#@title E1 · Policy rules (flowchart section 4) — versioned assumptions
DEFAULT_POLICY = {
    "version": "1.0.0",
    # Confidence C = 0.40Q + 0.25F + 0.20G + 0.15M
    "confidence_weights": {"Q": 0.40, "F": 0.25, "G": 0.20, "M": 0.15},
    "confidence_threshold": 0.70,
    "source_quality": {"vehicle_apc": 1.00, "official_alert": 0.90, "trusted_external": 0.80,
                       "public_report_corroborated": 0.40},
    # Source precedence: vehicle data > official alert > trusted external feed > corroborated public report
    "source_precedence": ["vehicle_apc", "official_alert", "trusted_external", "public_report"],
    "official_sources": ["vehicle_apc", "official_alert"],   # sources allowed to verify a hard trigger (C-gate bypass)
    "single_source_agreement": 0.50,                         # G when only one source reports a value (assumption)
    "conflict_threshold_min": 8,                             # delay disagreement that counts as a conflict
    "conflict_penalty": 0.15,                                # C reduction while a conflict is unresolved
    "freshness_limit_s": 300,
    # Impact and priority
    "R_weights": {"D": 0.30, "H": 0.25, "K": 0.25, "E": 0.20},
    "V_weights": {"N": 0.35, "I": 0.25, "B": 0.20, "T": 0.20},
    "A_weights": {"S": 0.35, "L": 0.30, "X": 0.20, "U": 0.15},
    "P_weights": {"R": 0.40, "V": 0.35, "A": 0.25},
    "significance_R": 0.50,
    "hard_trigger": {"delay_min": 15, "gap_headway_ratio": 2.0, "unboardable_load_factor": 1.0,
                     "essential_severity_min": 0.8},
    # Priority-adjusted scenario rank: Si = Ci x [0.30DR + (0.15+0.10P)EB + (0.15+0.10P)AP + (0.25-0.10P)OF + (0.15-0.10P)CE]
    "scenario_base": {"DR": 0.30, "EB": 0.15, "AP": 0.15, "OF": 0.25, "CE": 0.15},
    "scenario_shift": {"DR": 0.0, "EB": 0.10, "AP": 0.10, "OF": -0.10, "CE": -0.10},
    # Documented normalization bounds (L, U)
    "bounds": {
        "delay_min": [0, 30], "headway_ratio": [1, 3], "transit_dependence": [0.05, 0.60],
        "income_disadvantage": [0.05, 0.40], "disability": [0.05, 0.25], "alternatives": [0, 3],
        "missed_transfers": [0, 40], "passups": [0, 40], "age_s": [0, 300], "agreement_spread_min": [0, 15],
        "dr_min": [0, 20], "eb_riders": [0, 300], "cost_usd": [0, 1000],
    },
    # Published categorical lookup tables
    "lookups": {
        "event_severity": {"none": 0.0, "minor": 0.3, "major": 0.7, "emergency": 1.0},
        "time_sensitivity": {"off_peak": 0.3, "midday": 0.4, "peak": 0.8, "shift_change": 0.9, "last_service": 1.0},
        "essential_severity": {"none": 0.0, "retail": 0.3, "school": 0.6, "employment_center": 0.6,
                               "clinic": 0.8, "hospital": 1.0, "dialysis": 1.0},
        "option_certainty": {"do_nothing": 1.00, "rider_alerts": 0.95, "protect_connection": 0.85,
                             "reroute": 0.75, "dispatch_second_bus": 0.80},
    },
    # Human review
    "deadlines_min": {"severe": 5, "significant": 10, "moderate": 20},
    "severity_tiers": {"severe_P": 0.70, "significant_P": 0.50},
    "high_impact": {"actions": ["reroute", "dispatch_second_bus"], "P_min": 0.70, "cost_min": 500},
    "max_revisions": 2,
    "max_operator_checks": 2,
    "review": {"min_reason_chars": 15, "fast_response_flag_s": 30, "approval_rate_flag": 0.90, "min_n_for_pattern": 5},
    # Planner defaults (constraints the planner can revise)
    "planner_defaults": {"budget_cap_usd": 1000, "max_hold_min": 5, "max_deadhead_min": 30,
                         "allow_alternate_depot": False, "excluded_options": []},
    "bus_hour_cost_usd": 160,
    # Review deadline on a revision: False = the original deadline still applies (dashboard default);
    # True = every rebuilt brief starts a new deadline.
    "deadline_resets_on_revision": False,
    # Reviewer governance gates built from Sections C (ethics training) and D (conflict of interest)
    "governance_gates": {"enabled": True,
                         "unknown_person": "warn",                    # "warn" or "block" names not on the training roster
                         "training_block_statuses": ["NON_COMPLIANT"],
                         "training_gate_decisions": ["Approve", "Revise"],  # Reject (→ safe default) is always allowed
                         "coi_gate_decisions": ["Approve"],
                         "coi_threshold": 0.70},
}

REVIEW_CHECKLIST = [
    "I checked the verified evidence, sources and timestamps",
    "I viewed every alternative and its component scores",
    "I considered equity and access impacts on affected areas",
    "I confirmed resource, road, policy and safety constraints",
]


def validate_policy(p):
    errs = []
    for k in ["confidence_weights", "R_weights", "V_weights", "A_weights", "P_weights", "scenario_base"]:
        s = sum(p[k].values())
        if abs(s - 1) > 1e-9:
            errs.append(f"{k} sums to {s:.4f}, not 1")
    if abs(sum(p["scenario_shift"].values())) > 1e-9:
        errs.append("scenario_shift must sum to 0 so Si weights always sum to 1")
    for c, b in p["scenario_base"].items():
        sh = p["scenario_shift"].get(c, 0)
        if not (0 <= b + min(0, sh) and b + max(0, sh) <= 1):
            errs.append(f"weight for {c} leaves [0,1] when P moves 0→1")
    for k, (L, U) in p["bounds"].items():
        if U <= L:
            errs.append(f"bound {k}: U must exceed L")
    return errs


def get_path(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def set_path(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


class PolicyManager:
    """Versioned policy store. Changes only via policy-owner approval; every version kept for rollback."""

    def __init__(self, policy):
        assert not validate_policy(policy), validate_policy(policy)
        self.versions = [{"version": policy["version"], "policy": copy.deepcopy(policy), "released_at": iso(utcnow()),
                          "approved_by": "initial release", "reason": "Initial documented policy assumptions",
                          "changes": [], "backtest_summary": ""}]
        self.current_index = 0

    @property
    def current(self):
        return self.versions[self.current_index]["policy"]

    def get(self, version):
        for v in self.versions:
            if v["version"] == version:
                return v["policy"]
        raise KeyError(version)

    def _bump(self):
        major, minor, patch = map(int, self.versions[-1]["version"].split("."))
        return f"{major}.{minor + 1}.0"

    def release(self, new_policy, approved_by, reason, changes, backtest_summary=""):
        new_policy = copy.deepcopy(new_policy)
        new_policy["version"] = self._bump()
        errs = validate_policy(new_policy)
        if errs:
            raise ValueError("; ".join(errs))
        self.versions.append({"version": new_policy["version"], "policy": new_policy, "released_at": iso(utcnow()),
                              "approved_by": approved_by, "reason": reason, "changes": changes,
                              "backtest_summary": backtest_summary})
        self.current_index = len(self.versions) - 1
        return new_policy["version"]

    def rollback(self, to_version, by, reason):
        old = self.get(to_version)
        return self.release(old, by, f"ROLLBACK to {to_version}: {reason}",
                            [{"path": "*", "old": self.current["version"], "new": f"rules of {to_version}"}])


# ====================================================================================================
# cell: cv_core
# ====================================================================================================
#@title CV · Computer vision passenger counting (pretrained person detector → anonymized counts)
USE_CV = True  #@param {type:"boolean"}
USE_CV = os.environ.get("TRANSIT_USE_CV", str(USE_CV)).strip().lower() in ("1", "true", "yes")
CV_SCORE_THRESHOLD = 0.75  #@param {type:"number"}
CV_VIDEO_FRAMES = 8  #@param {type:"integer"}
SHOW_CV_PREVIEW = True  #@param {type:"boolean"}
import numpy as np

# Sample transit imagery, downloaded at runtime. Only person counts leave this cell; images are not saved.
#  - bus.jpg: riders at the curb beside a city bus (Ultralytics sample asset, AGPL-3.0)
#  - vtest.avi: overhead camera of pedestrians on a walkway (OpenCV sample data), used as a station-approach camera
CV_SOURCES = [
    {"camera_id": "VISION-STOP-NEWARK-PENN", "kind": "image", "role": "bus stop curb camera",
     "url": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
     "stop_id": "NEWARK_PENN_BUS_LANE", "capacity": 50,
     "note": "generic street photo used as a stand-in for a Newark Penn curb camera"},
    {"camera_id": "VISION-APPROACH-NEWARK-PENN", "kind": "video", "role": "station approach camera",
     "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
     "stop_id": "NEWARK_PENN_APPROACH", "capacity": 50,
     "note": "public pedestrian video used as a stand-in for a station approach camera"},
]
CV_DIR = os.path.join(OUTPUT_DIR, "cv_cache")      # downloaded samples only; nothing derived from them is written here
os.makedirs(CV_DIR, exist_ok=True)
_DETECTOR = {"name": None, "model": None, "error": []}


def load_detector():
    """Faster R-CNN (MobileNetV3, COCO) from torchvision; OpenCV's HOG people detector if torchvision weights fail."""
    if _DETECTOR["name"]:
        return _DETECTOR["name"]
    try:
        import torch, torchvision
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
        w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        m = fasterrcnn_mobilenet_v3_large_fpn(weights=w).eval()
        _DETECTOR.update(name=f"torchvision Faster R-CNN MobileNetV3 ({w.name}, COCO)", model=("torch", m, w))
        return _DETECTOR["name"]
    except Exception as e:
        _DETECTOR["error"].append(f"torchvision: {type(e).__name__}: {str(e)[:120]}")
    try:
        import cv2
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _DETECTOR.update(name="OpenCV HOG + linear SVM people detector (fallback)", model=("hog", hog, None))
        return _DETECTOR["name"]
    except Exception as e:
        _DETECTOR["error"].append(f"opencv: {type(e).__name__}: {str(e)[:120]}")
    return None


def detect_people(rgb):
    """rgb: HxWx3 uint8 array → list of (x1, y1, x2, y2, score) for detected people."""
    kind, model, weights = _DETECTOR["model"]
    if kind == "torch":
        import torch
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        with torch.no_grad():
            out = model([weights.transforms()(t)])[0]
        person = weights.meta["categories"].index("person")
        return [(*map(float, b), float(s)) for b, s, l in zip(out["boxes"], out["scores"], out["labels"])
                if int(l) == person and float(s) >= CV_SCORE_THRESHOLD]
    import cv2
    # HOG uses a fixed 64x128 window: small people need the full-size frame, large people a downscaled one
    h, w = rgb.shape[:2]
    hog_threshold = 0.5   # HOG scores are SVM margins, not probabilities
    keep = []
    for f in sorted({1.0, min(1.0, 480.0 / max(h, w))}):
        img = cv2.resize(rgb, (int(w * f), int(h * f))) if f < 1 else rgb
        rects, wts = model.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), winStride=(4, 4), padding=(8, 8), scale=1.05)
        keep += [(x / f, y / f, (x + bw) / f, (y + bh) / f, float(sc)) for (x, y, bw, bh), sc in zip(rects, np.ravel(wts))
                 if float(sc) >= hog_threshold]
    return _nms(keep)


def _nms(boxes, iou=0.4):
    """Drop overlapping duplicate detections of the same person."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in boxes:
        def ov(a, c):
            ix = max(0, min(a[2], c[2]) - max(a[0], c[0])); iy = max(0, min(a[3], c[3]) - max(a[1], c[1]))
            inter = ix * iy
            return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter + 1e-9)
        if all(ov(b, k) < iou for k in kept):
            kept.append(b)
    return kept


def _download(url):
    path = os.path.join(CV_DIR, os.path.basename(url))
    if not os.path.exists(path):
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def _frames(src, path):
    import cv2
    if src["kind"] == "image":
        img = cv2.imread(path)
        if img is None:
            raise ValueError("unreadable image")
        yield 0, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    for idx in np.linspace(0, n - 1, max(1, CV_VIDEO_FRAMES)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield int(idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def _privacy_preview(rgb, boxes, block=None):
    """Privacy: the WHOLE frame is pixelated (so a person the detector missed is not identifiable either),
    then detection boxes are drawn on top. Display only; never saved or passed downstream."""
    import cv2
    h, w = rgb.shape[:2]
    block = block or max(10, w // 60)
    small = cv2.resize(rgb, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    for x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (250, 204, 21), max(2, w // 200))
    return out


def run_computer_vision(sources=CV_SOURCES):
    """Each camera → anonymized aggregate: people counted, load factor, detector confidence. No images or identities kept."""
    results = {"detector": None, "cameras": [], "errors": [], "ran_at": iso(utcnow()), "previews": []}
    if not USE_CV:
        results["errors"].append("computer vision disabled")
        return results
    results["detector"] = load_detector()
    if not results["detector"]:
        results["errors"] += _DETECTOR["error"]
        return results
    for src in sources:
        try:
            path = _download(src["url"])
            counts, confs, preview = [], [], None
            for idx, rgb in _frames(src, path):
                boxes = detect_people(rgb)
                counts.append(len(boxes))
                confs += [b[4] for b in boxes]
                if preview is None or len(boxes) > preview[1]:
                    preview = (_privacy_preview(rgb, boxes), len(boxes), idx)
            if not counts:
                raise ValueError("no frames read")
            people = int(round(statistics.median(counts)))
            results["cameras"].append({
                "camera_id": src["camera_id"], "role": src["role"], "stop_id": src["stop_id"],
                "frames_analyzed": len(counts), "people_per_frame": counts, "person_count": people,
                "peak_count": max(counts), "capacity": src["capacity"], "load_factor": round(people / src["capacity"], 2),
                "mean_detector_score": round(float(np.mean(confs)), 2) if confs else None,   # probability for Faster R-CNN; SVM margin for HOG
                "observed_at": iso(utcnow()), "identities_stored": False, "images_stored": False, "note": src.get("note", ""),
            })
            if SHOW_CV_PREVIEW and preview is not None:
                results["previews"].append((src["camera_id"], preview))
        except Exception as e:
            results["errors"].append(f"{src['camera_id']}: {type(e).__name__}: {str(e)[:120]}")
    return results


def show_cv_previews(results):
    if not results["previews"]:
        return
    fig, axes = plt.subplots(1, len(results["previews"]), figsize=(6 * len(results["previews"]), 5))
    axes = np.atleast_1d(axes)
    for ax, (cam, (img, n, idx)) in zip(axes, results["previews"]):
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"{cam}\n{n} {'person' if n == 1 else 'people'} detected (frame {idx}) · frame pixelated for privacy", fontsize=9)
    plt.tight_layout(); plt.show()


CV_RESULTS = {"detector": None, "cameras": [], "errors": ["computer vision not run yet"], "ran_at": None}


def refresh_cv():
    """Run the detector on the configured camera sources and keep only anonymized counts."""
    global CV_RESULTS
    res = run_computer_vision()
    res.pop("previews", None)          # pixelated previews are display-only
    CV_RESULTS = res
    return CV_RESULTS


# ====================================================================================================
# cell: a1_core
# ====================================================================================================
#@title A1 · Live data connectors: NJ Transit GTFS-realtime, US Census ACS, National Weather Service
NJT_GTFS_SYSTEM = "GTFSG2"  #@param ["GTFSG2", "GTFS"]
NJT_NEARBY_KM = 2.0  #@param {type:"number"}
ACS_YEAR = 2023  #@param {type:"integer"}
NJT_GTFS_SYSTEM = os.environ.get("NJT_GTFS_SYSTEM", NJT_GTFS_SYSTEM)
NJT_NEARBY_KM = float(os.environ.get("NJT_NEARBY_KM", NJT_NEARBY_KM))
ACS_YEAR = int(os.environ.get("ACS_YEAR", ACS_YEAR))
import re
from datetime import datetime, timezone, timedelta

# Every input comes from one of three live services. There is no sample mode: when a service gives no data, the
# field stays missing (the engine never imputes it) and the case says so.
LAT, LON = SERVICE_AREA_LATLON           # Newark Penn Station by default
LIVE = True


def _ago(seconds):
    return iso(datetime.now(timezone.utc) - timedelta(seconds=seconds))


# ---------------------------------------------------------------- NJ Transit GTFS / GTFS-realtime
# Login: form post → {"Authenticated": "True", "UserToken": ...}. Data calls send the token as a multipart form field
# and return GTFS-realtime protobuf (trip updates, vehicle positions, alerts). The token is kept in memory only.
NJT_BASE = "https://pcsdata.njtransit.com/api/{system}/"
_NJT = {"token": None, "system": None}


def njt_authenticate():
    user, pw = read_secret("NJ_TRANSIT_USERNAME"), read_secret("NJ_TRANSIT_PASSWORD")
    if not (user and pw):
        return "skipped (no NJ_TRANSIT_USERNAME / NJ_TRANSIT_PASSWORD secret)"
    last = "not attempted"
    for system in dict.fromkeys([NJT_GTFS_SYSTEM, "GTFSG2", "GTFS"]):
        url = NJT_BASE.format(system=system) + "authenticateUser"
        for mode in ("form", "multipart"):
            try:
                if mode == "form":
                    r = requests.post(url, data={"username": user, "password": pw}, timeout=20)
                else:
                    r = requests.post(url, files={"username": (None, user), "password": (None, pw)}, timeout=20)
                r.raise_for_status()
                body = r.json()
            except (requests.RequestException, ValueError) as e:
                last = f"{system}: request failed ({type(e).__name__})"
                continue
            token = body.get("UserToken") if isinstance(body, dict) else None
            if str(body.get("Authenticated", "")).lower() == "true" and token:
                _NJT.update(token=token, system=system)
                return f"authenticated ({system})"
            last = f"{system}: rejected (no token returned)"
    return last


def njt_post(endpoint, system=None, timeout=30):
    """POST an authenticated request and return the raw bytes (protobuf for feeds, a zip for getGTFS)."""
    url = NJT_BASE.format(system=system or _NJT["system"]) + endpoint
    r = requests.post(url, files={"token": (None, _NJT["token"])}, timeout=timeout)
    r.raise_for_status()
    return r.content


def njt_feed(endpoint):
    from google.transit import gtfs_realtime_pb2
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(njt_post(endpoint))
    return feed


def _text(ts):
    return ts.translation[0].text.strip() if ts and len(ts.translation) else ""


def parse_trip_updates(tu):
    updates = {}
    for ent in (tu.entity if tu is not None else []):
        if not ent.HasField("trip_update"):
            continue
        t = ent.trip_update
        stus = [s for s in t.stop_time_update if s.HasField("arrival") or s.HasField("departure")]
        first = stus[0] if stus else None
        delay = None
        if first is not None:
            delay = first.arrival.delay if first.HasField("arrival") else first.departure.delay
        elif t.HasField("delay"):
            delay = t.delay
        updates[t.trip.trip_id] = {"delay_s": delay, "cancelled": t.trip.schedule_relationship == t.trip.CANCELED,
                                   "route_id": t.trip.route_id,
                                   "next_stop_id": first.stop_id if first is not None else None,
                                   "next_stop_seq": first.stop_sequence if first is not None and first.stop_sequence else None}
    return updates


def parse_alerts(al):
    out = []
    for ent in (al.entity if al is not None else []):
        if ent.HasField("alert"):
            a = ent.alert
            out.append({"alert_id": ent.id, "routes": sorted({ie.route_id for ie in a.informed_entity if ie.route_id}),
                        "stops": sorted({ie.stop_id for ie in a.informed_entity if ie.stop_id}),
                        "header": _text(a.header_text), "description": _text(a.description_text)[:600],
                        "effect": a.Effect.Name(a.effect) if a.effect else None})
    return out


def fetch_njt_service():
    """Every live vehicle within NJT_NEARBY_KM of the service area, joined to its trip update, plus all alerts.
    vehicles are sorted with cancelled and most-delayed first; 'service' is the first one (or None)."""
    out = {"status": njt_authenticate(), "feeds": {}, "vehicles": [], "alerts": [], "service": None, "alert": None,
           "nearby_vehicles": 0, "fetched_at": iso(utcnow())}
    if not _NJT["token"]:
        return out
    feeds = {}
    for ep in ("getVehiclePositions", "getTripUpdates", "getAlerts"):
        try:
            feeds[ep] = njt_feed(ep)
            out["feeds"][ep] = f"ok ({len(feeds[ep].entity)} entities)"
        except Exception as e:
            out["feeds"][ep] = f"failed ({type(e).__name__})"
    vp = feeds.get("getVehiclePositions")
    out["alerts"] = parse_alerts(feeds.get("getAlerts"))
    if vp is None:
        return out
    updates = parse_trip_updates(feeds.get("getTripUpdates"))
    nearby = []
    for ent in vp.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        d = km_between((v.position.latitude, v.position.longitude), (LAT, LON))
        if d > NJT_NEARBY_KM:
            continue
        u = updates.get(v.trip.trip_id, {})
        nearby.append({"route_id": v.trip.route_id or u.get("route_id") or "", "trip_id": v.trip.trip_id,
                       "vehicle_id": v.vehicle.id, "lat": v.position.latitude, "lon": v.position.longitude,
                       "distance_km": round(d, 2), "delay_s": u.get("delay_s"), "cancelled": bool(u.get("cancelled", False)),
                       "has_trip_update": v.trip.trip_id in updates,
                       "stop_id": v.stop_id or u.get("next_stop_id"),
                       "stop_seq": (v.current_stop_sequence or None) if v.HasField("current_stop_sequence") else u.get("next_stop_seq"),
                       "timestamp": int(v.timestamp or vp.header.timestamp or 0)})
    nearby.sort(key=lambda n: (not n["cancelled"], -(n["delay_s"] or 0)))
    out.update(vehicles=nearby, nearby_vehicles=len(nearby), service=nearby[0] if nearby else None)
    if nearby:
        out["alert"] = alert_for_route(nearby[0]["route_id"], out["alerts"])
    return out


def alert_for_route(route_id, alerts):
    for a in alerts or []:
        if route_id and route_id in a["routes"]:
            return {k: a[k] for k in ("alert_id", "header", "description", "effect")}
    return None


# ---------------------------------------------------------------- US Census ACS (tract-level equity inputs)
ACS_DETAILED = {"B08201_001E": "households", "B08201_002E": "households_no_vehicle",
                "B08301_001E": "workers", "B08301_010E": "workers_public_transit", "B19013_001E": "median_household_income"}
ACS_SUBJECT = {"S1701_C03_001E": "poverty_rate_pct", "S1810_C03_001E": "disability_rate_pct"}
_CENSUS_GEO_CACHE, _CENSUS_TRACT_CACHE = {}, {}


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_key(v, key)
            if hit is not None:
                return hit
    return None


def _acs(path, variables, geo, key):
    params = {"get": ",".join(["NAME"] + list(variables)), "for": f"tract:{geo['TRACT']}",
              "in": f"state:{geo['STATE']} county:{geo['COUNTY']}"}
    if key:
        params["key"] = key
    r = requests.get(f"https://api.census.gov/data/{ACS_YEAR}/{path}", params=params, timeout=30)
    r.raise_for_status()
    header, row = r.json()[:2]
    vals = dict(zip(header, row))
    out = {"NAME": vals.get("NAME")}
    for code, name in variables.items():
        try:
            v = float(vals.get(code))
            out[name] = None if v < 0 else v          # negative = Census "not available" sentinel
        except (TypeError, ValueError):
            out[name] = None
    return out


def census_tract_at(lat, lon):
    """Census geocoder: the tract containing a point (cached by ~10 m grid)."""
    k = (round(lat, 4), round(lon, 4))
    if k not in _CENSUS_GEO_CACHE:
        r = requests.get("https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
                         params={"x": lon, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current",
                                 "layers": "Census Tracts", "format": "json"}, timeout=30)
        r.raise_for_status()
        tracts = _find_key(r.json(), "Census Tracts")
        _CENSUS_GEO_CACHE[k] = tracts[0] if tracts else None
    return _CENSUS_GEO_CACHE[k]


def acs_for_tract(geo):
    """ACS 5-year equity values for one tract (cached)."""
    gid = geo.get("GEOID") or f"{geo['STATE']}{geo['COUNTY']}{geo['TRACT']}"
    if gid not in _CENSUS_TRACT_CACHE:
        key = read_secret("CENSUS_API_KEY")
        d = _acs("acs/acs5", ACS_DETAILED, geo, key)
        s = _acs("acs/acs5/subject", ACS_SUBJECT, geo, key)
        share = lambda a, b: round(a / b, 3) if (a is not None and b) else None
        _CENSUS_TRACT_CACHE[gid] = {
            "tract": gid, "tract_name": d["NAME"],
            "zero_vehicle_share": share(d["households_no_vehicle"], d["households"]),
            "transit_commute_share": share(d["workers_public_transit"], d["workers"]),
            "poverty_rate": None if s["poverty_rate_pct"] is None else round(s["poverty_rate_pct"] / 100, 3),
            "disability_rate": None if s["disability_rate_pct"] is None else round(s["disability_rate_pct"] / 100, 3),
            "median_household_income": d["median_household_income"],
            "households": d["households"]}
    return _CENSUS_TRACT_CACHE[gid]


def fetch_census(lat, lon):
    res = {"status": "not attempted", "tract": None}
    try:
        geo = census_tract_at(lat, lon)
        if not geo:
            res["status"] = "geocoder returned no tract"
            return res
        res.update(acs_for_tract(geo))
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        res["status"] = f"failed ({type(e).__name__})"
        return res
    res["status"] = f"ok (ACS {ACS_YEAR} 5-year{'' if read_secret('CENSUS_API_KEY') else ', no key'})"
    return res


def census_for_points(points):
    """Equity values for every tract touched by a list of (lat, lon) points; tracts are averaged, weighted by
    households. Missing values stay missing. Returns (summary, per-tract rows, errors)."""
    rows, errors = {}, []
    for lat, lon in points:
        try:
            geo = census_tract_at(lat, lon)
            if geo:
                t = acs_for_tract(geo)
                rows[t["tract"]] = t
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
            errors.append(f"{type(e).__name__}")
    summary = {}
    for field, out in (("zero_vehicle_share", "transit_dependence"), ("poverty_rate", "income_disadvantage"),
                       ("disability_rate", "disability")):
        vals = [(t[field], t.get("households") or 1) for t in rows.values() if t.get(field) is not None]
        summary[out] = round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 3) if vals else None
    summary["areas"] = [t["tract_name"] for t in rows.values()]
    return summary, list(rows.values()), errors


# ---------------------------------------------------------------- National Weather Service (7 connectors)
NWS_BASE = "https://api.weather.gov"


def _nws_headers():
    email = read_secret("NWS_CONTACT_EMAIL")
    return {"User-Agent": f"TransitAIStudentProject ({email})" if email else "TransitAIStudentProject",
            "Accept": "application/geo+json"}


def get_weather(path, log):
    try:
        response = requests.get(NWS_BASE + path, headers=_nws_headers(), timeout=15)
        response.raise_for_status()
        log.append({"connector": path.split("?")[0], "status": "ok"})
        return response.json()
    except (requests.RequestException, ValueError) as e:
        log.append({"connector": path.split("?")[0], "status": f"failed ({type(e).__name__})"})
        return {}


def fetch_weather(lat, lon):
    log, fetched_at = [], _ago(0)
    points = get_weather(f"/points/{lat},{lon}", log)                         # 1 location and grid
    grid = points.get("properties", {})
    office, x, y = grid.get("gridId"), grid.get("gridX"), grid.get("gridY")
    alerts = get_weather(f"/alerts/active?point={lat},{lon}", log)            # 2 active alerts
    observation, hourly, grid_data = {}, {}, {}
    if office is not None and x is not None and y is not None:
        path = f"/gridpoints/{office}/{x},{y}"
        stations = get_weather(path + "/stations", log)                       # 3 nearby stations
        hourly = get_weather(path + "/forecast/hourly", log)                  # 4 hourly forecast
        grid_data = get_weather(path, log)                                    # 5 gridpoint data
        feats = stations.get("features", [])
        sid = feats[0].get("properties", {}).get("stationIdentifier") if feats else None
        if sid:
            observation = get_weather(f"/stations/{sid}/observations/latest", log)  # 6 latest observation
    alert_types = get_weather("/alerts/types", log)                           # 7 alert types
    op = observation.get("properties", {})
    return {
        "source": "NWS live" if any(r["status"] == "ok" for r in log) else "unavailable",
        "fetched_at": fetched_at, "current_conditions": op.get("textDescription"), "observed_at": op.get("timestamp"),
        "active_alerts": [{"event": a.get("properties", {}).get("event"), "severity": a.get("properties", {}).get("severity"),
                           "sent": a.get("properties", {}).get("sent")} for a in alerts.get("features", [])],
        "hourly_forecast": [{"time": p.get("startTime"), "forecast": p.get("shortForecast")}
                            for p in hourly.get("properties", {}).get("periods", [])[:3]],
        "grid_data_available": bool(grid_data), "alert_types_available": bool(alert_types), "connectors": log,
    }


LIVE_CONTEXT = {"nj_transit": {"status": "not fetched yet", "vehicles": [], "alerts": [], "service": None, "alert": None,
                               "feeds": {}, "nearby_vehicles": 0, "fetched_at": None},
                "census": {"status": "not fetched yet", "tract": None},
                "weather": {"source": "not fetched yet", "active_alerts": [], "connectors": []}}


def refresh_live_context(njt=True, census=True, weather=True):
    """Fetch the three live services. Call this before opening cases; re-call to refresh (the apps do this on demand)."""
    if njt:
        LIVE_CONTEXT["nj_transit"] = fetch_njt_service()
    if census:
        LIVE_CONTEXT["census"] = fetch_census(LAT, LON)
    if weather:
        LIVE_CONTEXT["weather"] = fetch_weather(LAT, LON)
    return LIVE_CONTEXT


def live_status_lines():
    n, c, w = LIVE_CONTEXT["nj_transit"], LIVE_CONTEXT["census"], LIVE_CONTEXT["weather"]
    lines = [f"NJ Transit GTFS-RT: {n['status']} · feeds: {n.get('feeds') or '-'} · vehicles within {NJT_NEARBY_KM} km: "
             f"{n.get('nearby_vehicles', 0)} · alerts: {len(n.get('alerts', []))}",
             f"US Census ACS: {c['status']}" + (f" · {c['tract_name']}" if c.get("tract") else ""),
             f"NWS weather: {w.get('source')} · {w.get('current_conditions') or 'conditions not reported'} · active alerts: "
             f"{len(w.get('active_alerts', []))} · connectors ok: {sum(r['status'] == 'ok' for r in w.get('connectors', []))}/"
             f"{len(w.get('connectors', []))}"]
    return lines


# ====================================================================================================
# cell: a1b_gtfs
# ====================================================================================================
#@title A1b · Real transit network: NJ Transit static GTFS (routes, stops, schedules) in a local SQLite cache
NJT_GTFS_ZIP = ""  #@param {type:"string"}
GTFS_MAX_AGE_DAYS = 7  #@param {type:"integer"}
NJT_GTFS_ZIP = os.environ.get("NJT_GTFS_ZIP", NJT_GTFS_ZIP)   # optional: a local path or URL to an NJ Transit GTFS zip
GTFS_MAX_AGE_DAYS = int(os.environ.get("GTFS_MAX_AGE_DAYS", GTFS_MAX_AGE_DAYS))
GTFS_DIR = os.environ.get("TRANSIT_DATA_DIR", os.path.join(OUTPUT_DIR, "gtfs"))
import sqlite3

# The static GTFS gives the real routes, stops, the trip each live bus is running, the scheduled headway and which
# other routes stop nearby (riders' alternatives). It is downloaded once through the NJ Transit API (getGTFS) and
# stored in SQLite, so even the full bus schedule stays on disk instead of in memory (important on Streamlit or
# Hugging Face hosts). Set NJT_GTFS_ZIP to use a zip you downloaded from the NJ Transit developer portal instead.
GTFS_TABLES = {
    "agency": ["agency_id", "agency_name"],
    "routes": ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type", "route_color"],
    "stops": ["stop_id", "stop_code", "stop_name", "stop_lat", "stop_lon"],
    "trips": ["trip_id", "route_id", "service_id", "trip_headsign", "direction_id", "shape_id"],
    "calendar": ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                 "start_date", "end_date"],
    "calendar_dates": ["service_id", "date", "exception_type"],
    "stop_times": ["trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time"],
}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def gtfs_secs(t):
    """'25:10:00' → seconds after the service day's midnight (GTFS times can pass 24:00)."""
    try:
        h, m, s = (int(x) for x in str(t).strip().split(":"))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return None


def _gtfs_zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            return {"stops.txt", "trips.txt", "stop_times.txt", "routes.txt"} <= set(n.split("/")[-1] for n in z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def download_gtfs_zip():
    """Get NJ Transit's static GTFS: NJT_GTFS_ZIP (path or URL) if set, otherwise the authenticated getGTFS endpoint."""
    os.makedirs(GTFS_DIR, exist_ok=True)
    dest = os.path.join(GTFS_DIR, "njt_gtfs.zip")
    src = (NJT_GTFS_ZIP or "").strip()
    if src and os.path.exists(src):
        if not _gtfs_zip_ok(src):
            raise LiveDataUnavailable(f"NJT_GTFS_ZIP={src} is not a GTFS zip")
        return src, f"local file {src}"
    if src.startswith("http"):
        r = requests.get(src, timeout=300)
        r.raise_for_status()
        open(dest, "wb").write(r.content)
        if not _gtfs_zip_ok(dest):
            raise LiveDataUnavailable("the NJT_GTFS_ZIP URL did not return a GTFS zip")
        return dest, f"downloaded from {src}"
    if not _NJT["token"]:
        njt_authenticate()
    if not _NJT["token"]:
        raise LiveDataUnavailable("NJ Transit login failed, and NJT_GTFS_ZIP is not set: cannot get the static GTFS")
    errors = []
    for system in dict.fromkeys([_NJT["system"], "GTFSG2", "GTFS"]):
        try:
            data = njt_post("getGTFS", system=system, timeout=300)
            open(dest, "wb").write(data)
            if _gtfs_zip_ok(dest):
                return dest, f"NJ Transit API {system}/getGTFS"
            errors.append(f"{system}: response was not a GTFS zip")
        except requests.RequestException as e:
            errors.append(f"{system}: {type(e).__name__}")
    raise LiveDataUnavailable("NJ Transit getGTFS failed (" + "; ".join(errors) + "). Download the bus GTFS zip from "
                              "developer.njtransit.com and set NJT_GTFS_ZIP to its path.")


def build_gtfs_db(zip_path, db_path):
    """Load the GTFS text files into SQLite in chunks (bounded memory), then index them."""
    tmp = db_path + ".building"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    with zipfile.ZipFile(zip_path) as z:
        names = {n.split("/")[-1]: n for n in z.namelist()}
        for table, cols in GTFS_TABLES.items():
            if f"{table}.txt" not in names:
                continue
            with z.open(names[f"{table}.txt"]) as fh:
                reader = pd.read_csv(fh, dtype=str, encoding="utf-8-sig", chunksize=250_000,
                                     usecols=lambda c: c.strip() in cols, keep_default_na=False)
                for chunk in reader:
                    chunk.columns = [c.strip() for c in chunk.columns]
                    for c in cols:
                        if c not in chunk:
                            chunk[c] = ""
                    chunk = chunk[cols]
                    if table == "stops":
                        chunk["stop_lat"] = pd.to_numeric(chunk["stop_lat"], errors="coerce")
                        chunk["stop_lon"] = pd.to_numeric(chunk["stop_lon"], errors="coerce")
                    if table == "stop_times":
                        chunk["stop_sequence"] = pd.to_numeric(chunk["stop_sequence"], errors="coerce")
                        chunk["dep_s"] = chunk["departure_time"].map(gtfs_secs)
                    chunk.to_sql(table, con, if_exists="append", index=False)
    cur = con.cursor()
    for sql in ["CREATE INDEX IF NOT EXISTS st_trip ON stop_times(trip_id)",
                "CREATE INDEX IF NOT EXISTS st_stop ON stop_times(stop_id)",
                "CREATE INDEX IF NOT EXISTS tr_route ON trips(route_id)",
                "CREATE INDEX IF NOT EXISTS tr_trip ON trips(trip_id)",
                "CREATE INDEX IF NOT EXISTS sp_id ON stops(stop_id)",
                "CREATE INDEX IF NOT EXISTS sp_ll ON stops(stop_lat, stop_lon)",
                "CREATE TABLE stop_routes AS SELECT DISTINCT st.stop_id AS stop_id, t.route_id AS route_id "
                "FROM stop_times st JOIN trips t ON st.trip_id = t.trip_id",
                "CREATE INDEX IF NOT EXISTS sr_stop ON stop_routes(stop_id)"]:
        cur.execute(sql)
    for t in ("calendar", "calendar_dates", "agency"):       # optional files: make sure the tables exist
        cur.execute(f"CREATE TABLE IF NOT EXISTS {t} ({', '.join(c + ' TEXT' for c in GTFS_TABLES[t])})")
    con.commit()
    con.close()
    os.replace(tmp, db_path)


class GTFSNetwork:
    """Read-only queries on the NJ Transit static GTFS (one SQLite connection per call; safe across app threads)."""

    def __init__(self, db_path, source):
        self.db_path, self.source = db_path, source
        self.built_at = datetime.fromtimestamp(os.path.getmtime(db_path), timezone.utc)
        with self._con() as con:
            self.counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("routes", "stops", "trips")}

    def _con(self):
        con = sqlite3.connect(self.db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def q(self, sql, args=()):
        with self._con() as con:
            return [dict(r) for r in con.execute(sql, args).fetchall()]

    # ---- routes and stops
    def route(self, route_id):
        r = self.q("SELECT r.*, a.agency_name FROM routes r LEFT JOIN agency a ON r.agency_id = a.agency_id "
                   "WHERE r.route_id = ?", (route_id,))
        return r[0] if r else None

    def route_label(self, route_id):
        r = self.route(route_id)
        if not r:
            return str(route_id)
        return " ".join(x for x in (r.get("route_short_name"), r.get("route_long_name")) if x) or str(route_id)

    def stop(self, stop_id):
        r = self.q("SELECT * FROM stops WHERE stop_id = ?", (stop_id,))
        return r[0] if r else None

    def trip(self, trip_id):
        r = self.q("SELECT * FROM trips WHERE trip_id = ?", (trip_id,))
        return r[0] if r else None

    def trip_stops(self, trip_id):
        return self.q("SELECT st.stop_sequence AS seq, st.stop_id, st.departure_time, st.dep_s, s.stop_name AS name, "
                      "s.stop_lat AS lat, s.stop_lon AS lon FROM stop_times st JOIN stops s ON st.stop_id = s.stop_id "
                      "WHERE st.trip_id = ? ORDER BY st.stop_sequence", (trip_id,))

    def affected_stops(self, trip_id, stop_id=None, stop_seq=None, lat=None, lon=None, n=6):
        """The next n stops of the trip from where the bus is now (by stop id, sequence, or nearest stop)."""
        stops = self.trip_stops(trip_id)
        if not stops:
            return []
        i = None
        if stop_id:
            i = next((k for k, s in enumerate(stops) if s["stop_id"] == str(stop_id)), None)
        if i is None and stop_seq is not None:
            i = next((k for k, s in enumerate(stops) if s["seq"] is not None and s["seq"] >= float(stop_seq)), None)
        if i is None and lat is not None:
            i = min(range(len(stops)), key=lambda k: km_between((stops[k]["lat"], stops[k]["lon"]), (lat, lon)))
        return stops[(i or 0):(i or 0) + n]

    def routes_at(self, stop_id):
        return [r["route_id"] for r in self.q("SELECT route_id FROM stop_routes WHERE stop_id = ?", (stop_id,))]

    def stops_near(self, lat, lon, radius_km):
        dlat = radius_km / KM_LAT
        dlon = radius_km / (111.320 * math.cos(math.radians(lat)))
        rows = self.q("SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops WHERE stop_lat BETWEEN ? AND ? "
                      "AND stop_lon BETWEEN ? AND ?", (lat - dlat, lat + dlat, lon - dlon, lon + dlon))
        return [r for r in rows if km_between((r["stop_lat"], r["stop_lon"]), (lat, lon)) <= radius_km]

    def routes_near(self, lat, lon, radius_km, exclude=None):
        out = set()
        for s in self.stops_near(lat, lon, radius_km):
            out.update(self.routes_at(s["stop_id"]))
        out.discard(exclude)
        return sorted(out)

    # ---- schedule
    def services_on(self, d):
        """service_ids running on date d (calendar + calendar_dates exceptions)."""
        ds, wd = d.strftime("%Y%m%d"), WEEKDAYS[d.weekday()]
        on = {r["service_id"] for r in self.q(f"SELECT service_id FROM calendar WHERE {wd} = '1' AND start_date <= ? "
                                              "AND end_date >= ?", (ds, ds))}
        for r in self.q("SELECT service_id, exception_type FROM calendar_dates WHERE date = ?", (ds,)):
            (on.add if r["exception_type"] == "1" else on.discard)(r["service_id"])
        return on

    def _departures(self, route_id, stop_id, now_local):
        """Scheduled departures of a route at a stop, as seconds after today's midnight (yesterday's late trips too)."""
        today = now_local.date()
        out = []
        for day, offset in ((today, 0), (today - timedelta(days=1), -86400)):
            sids = self.services_on(day)
            if not sids:
                continue
            marks = ",".join("?" * len(sids))
            rows = self.q(f"SELECT st.dep_s FROM stop_times st JOIN trips t ON st.trip_id = t.trip_id WHERE st.stop_id = ? "
                          f"AND t.route_id = ? AND t.service_id IN ({marks})", (stop_id, route_id, *sids))
            out += [r["dep_s"] + offset for r in rows if r["dep_s"] is not None]
        return sorted(out)

    def scheduled_headway_min(self, route_id, stop_id, now_local=None, window_min=90):
        """Median scheduled gap between buses of this route at this stop within ±window of now. None if unknown."""
        now_local = now_local or local_now()
        now_s = now_local.hour * 3600 + now_local.minute * 60
        deps = [d for d in self._departures(route_id, stop_id, now_local) if abs(d - now_s) <= window_min * 60]
        gaps = [b - a for a, b in zip(deps, deps[1:]) if b > a]
        return round(statistics.median(gaps) / 60, 1) if gaps else None

    def is_last_trip(self, route_id, stop_id, trip_id, now_local=None):
        """True if no later trip of this route is scheduled at this stop today. None if the schedule cannot tell."""
        now_local = now_local or local_now()
        mine = self.q("SELECT dep_s FROM stop_times WHERE trip_id = ? AND stop_id = ?", (trip_id, stop_id))
        if not mine or mine[0]["dep_s"] is None:
            return None
        deps = self._departures(route_id, stop_id, now_local)
        return not any(d > mine[0]["dep_s"] for d in deps)


_NETWORK = {"net": None, "error": None}


def get_network(force=False):
    """The GTFS network (built on first use and refreshed after GTFS_MAX_AGE_DAYS). Raises LiveDataUnavailable."""
    if _NETWORK["net"] is not None and not force:
        return _NETWORK["net"]
    os.makedirs(GTFS_DIR, exist_ok=True)
    db = os.path.join(GTFS_DIR, "njt_gtfs.sqlite")
    fresh = os.path.exists(db) and (time.time() - os.path.getmtime(db)) < GTFS_MAX_AGE_DAYS * 86400
    src_note = os.path.join(GTFS_DIR, "source.txt")
    if force or not fresh:
        zip_path, source = download_gtfs_zip()
        build_gtfs_db(zip_path, db)
        open(src_note, "w").write(source)
    source = open(src_note).read() if os.path.exists(src_note) else "cached GTFS"
    _NETWORK.update(net=GTFSNetwork(db, source), error=None)
    return _NETWORK["net"]


def network_or_none():
    """The network if it can be loaded; otherwise None, with the reason in _NETWORK['error']."""
    try:
        return get_network()
    except (LiveDataUnavailable, OSError, sqlite3.Error, zipfile.BadZipFile, requests.RequestException) as e:
        _NETWORK["error"] = str(e)
        return None


# ====================================================================================================
# cell: a2_core
# ====================================================================================================
#@title A2 · Data inputs from the live services + small language model + hand-off to the Signal Scout
USE_SLM = True  #@param {type:"boolean"}
USE_LLM_ORGANIZER = True  #@param {type:"boolean"}
AFFECTED_STOPS = 6  #@param {type:"integer"}
USE_SLM = os.environ.get("TRANSIT_USE_SLM", str(USE_SLM)).strip().lower() in ("1", "true", "yes")
USE_LLM_ORGANIZER = os.environ.get("TRANSIT_USE_LLM_ORGANIZER", str(USE_LLM_ORGANIZER)).strip().lower() in ("1", "true", "yes")
AFFECTED_STOPS = int(os.environ.get("AFFECTED_STOPS", AFFECTED_STOPS))
SLM_MODEL_ID = "google/flan-t5-small"
MISSING = "not connected (missing; never imputed)"


# 1. DATA INPUTS — the seven sources of the design. Each comes from a live service or is explicitly missing:
#    service feeds, events  → NJ Transit GTFS-realtime (+ static GTFS for stops, headway, nearby routes)
#    community access       → US Census ACS for every tract the affected stops are in
#    event signals          → NJ Transit alerts + National Weather Service alerts
#    vision ridership       → computer vision counts (only if USE_CV ran); otherwise missing
#    connecting services    → routes from the static GTFS; transfer window and missed transfers are missing
#    incident history       → cases recorded by this deployment (empty at first)
#    operating resources    → missing (no depot, spare-bus or operator feed is connected)
def live_vehicles():
    return LIVE_CONTEXT["nj_transit"].get("vehicles", [])


def pick_vehicle(vehicle_id=None):
    vs = live_vehicles()
    if not vs:
        raise LiveDataUnavailable("No live NJ Transit vehicle within "
                                  f"{NJT_NEARBY_KM} km ({LIVE_CONTEXT['nj_transit'].get('status')}). Refresh the live feed.")
    if vehicle_id is None:
        return vs[0]
    v = next((x for x in vs if str(x["vehicle_id"]) == str(vehicle_id)), None)
    if v is None:
        raise LiveDataUnavailable(f"Vehicle {vehicle_id} is no longer in the live feed. Refresh and pick again.")
    return v


def vehicle_label(v, net=None):
    d = "delay unknown" if v["delay_s"] is None else f"{v['delay_s'] / 60:+.1f} min"
    route = net.route_label(v["route_id"]) if net else v["route_id"]
    return f"Route {route} · vehicle {v['vehicle_id']} · {'CANCELLED · ' if v['cancelled'] else ''}{d} · {v['distance_km']} km away"


def _cv_stop_camera():
    cams = [c for c in CV_RESULTS.get("cameras", []) if c["role"] == "bus stop curb camera"] or CV_RESULTS.get("cameras", [])
    return cams[0] if cams else None


def section_a_live_inputs(vehicle=None, store=None):
    """The seven input sources for one live vehicle (default: the most delayed one near the service area)."""
    v = vehicle if isinstance(vehicle, dict) else pick_vehicle(vehicle)
    net = network_or_none()
    njt = LIVE_CONTEXT["nj_transit"]
    w = LIVE_CONTEXT["weather"]
    now = utcnow()
    # ---- service feed (real-time) + static schedule
    stops = net.affected_stops(v["trip_id"], v.get("stop_id"), v.get("stop_seq"), v["lat"], v["lon"], AFFECTED_STOPS) if net else []
    if not stops:            # trip not in the static GTFS (or GTFS unavailable): the bus position is the only place we know
        stops = [{"seq": None, "stop_id": f"VEHICLE-{v['vehicle_id']}", "name": "current vehicle position",
                  "lat": v["lat"], "lon": v["lon"], "dep_s": None}]
    gtfs_ok = net is not None and not stops[0]["stop_id"].startswith("VEHICLE-")
    headway = net.scheduled_headway_min(v["route_id"], stops[0]["stop_id"]) if gtfs_ok else None
    delay = None if v["delay_s"] is None else max(0.0, round(v["delay_s"] / 60, 1))   # one decimal: 14.96 must not become 15
    route_label = net.route_label(v["route_id"]) if net else v["route_id"]
    route_info = (net.route(v["route_id"]) if net else None) or {}
    service = {"route": v["route_id"], "route_label": route_label, "agency": route_info.get("agency_name") or "NJ Transit",
               "trip_id": v["trip_id"], "vehicle_id": v["vehicle_id"], "delay_min": delay, "cancelled": bool(v["cancelled"]),
               "scheduled_headway_min": headway,
               "headway_source": "NJ Transit static GTFS (median scheduled gap ±90 min)" if headway else MISSING,
               "vehicle_position": f"{v['lat']:.5f},{v['lon']:.5f} ({v['distance_km']} km from the service area)",
               "affected_stops": stops, "gtfs": net.source if net else f"unavailable: {_NETWORK['error']}",
               "source": f"NJ Transit {_NJT['system']} GTFS-realtime (live)",
               "observed_at": iso(datetime.fromtimestamp(v["timestamp"], timezone.utc)) if v.get("timestamp") else iso(now)}
    # ---- event signals: NJ Transit alert for this route + NWS
    alert = alert_for_route(v["route_id"], njt.get("alerts"))
    if alert:
        text = f"{alert['header']}. {alert['description']}".strip(". ")
        events = {"source_type": "official_alert", "report": text, "report_source": "NJ Transit GTFS-realtime alert (live)",
                  "alert_effect": alert.get("effect"), "source": "NJ Transit GTFS-realtime alert (live)",
                  "observed_at": njt.get("fetched_at") or iso(now)}
    else:
        events = {"source_type": "none", "report_source": "generated from the live vehicle feed (no active alert)",
                  "alert_effect": None, "source": "NJ Transit GTFS-realtime vehicle feed (live; no alert)",
                  "report": f"Route {route_label} is running "
                            + ("with an unknown delay" if delay is None else f"{delay:g} minutes late")
                            + " near the service area. No active NJ Transit service alert for this route.",
                  "observed_at": service["observed_at"]}
    w_live = w.get("source") == "NWS live"
    events.update(weather=w.get("current_conditions") if w_live else None, weather_alerts=w.get("active_alerts", []) if w_live else [],
                  weather_source="NWS live" if w_live else f"NWS {w.get('source')}", weather_fetched_at=w.get("fetched_at"))
    if w_live:
        events["source"] += " + NWS weather (live)"
    # ---- vision ridership: computer vision counts, or missing
    cam = _cv_stop_camera()
    if cam:
        vision = {"boardings": None, "boardings_source": MISSING + " (one camera cannot see route boardings)",
                  "waiting_at_stop": cam["person_count"], "load_factor": cam["load_factor"], "capacity": cam["capacity"],
                  "pass_ups": None, "pass_ups_source": "not observable in a still image (missing)", "obstruction": None,
                  "identities_stored": False, "camera_id": cam["camera_id"], "detector": CV_RESULTS["detector"],
                  "source": "computer vision (anonymized counts)", "observed_at": cam["observed_at"]}
    else:
        vision = {"boardings": None, "load_factor": None, "pass_ups": None, "obstruction": None, "identities_stored": False,
                  "source": MISSING + " (computer vision not run)", "observed_at": None}
    # ---- community access: ACS for every tract the affected stops are in
    comm, tracts, errs = census_for_points([(s["lat"], s["lon"]) for s in stops])
    community = {"transit_dependence": comm["transit_dependence"], "income_disadvantage": comm["income_disadvantage"],
                 "disability_barrier": comm["disability"], "tracts": tracts, "essential_destination": None,
                 "essential_destination_source": MISSING,
                 "data_level": (f"{len(tracts)} census tract(s), ACS {ACS_YEAR} 5-year, household-weighted") if tracts else MISSING,
                 "source": "US Census ACS (live)" if tracts else f"US Census ACS unavailable ({', '.join(errs) or 'no tract'})",
                 "observed_at": iso(now) if tracts else None}
    # ---- connecting services: routes nearby the downstream stops (static GTFS); transfer details missing
    conn_routes, transfer_stop = [], None
    if gtfs_ok:
        for s in stops:
            near = net.routes_near(s["lat"], s["lon"], 0.6, exclude=v["route_id"])
            if near:
                conn_routes, transfer_stop = near, s["stop_id"]
                break
    last = net.is_last_trip(v["route_id"], stops[0]["stop_id"], v["trip_id"]) if gtfs_ok else None
    connecting = {"connecting_routes": conn_routes, "transfer_stop_id": transfer_stop,
                  "connecting_headway_min": (net.scheduled_headway_min(conn_routes[0], transfer_stop) if conn_routes else None),
                  "transfer_window_min": None, "missed_transfers": None, "last_service": last,
                  "source": "NJ Transit static GTFS (routes and last-trip flag); transfer window and missed transfers "
                            + MISSING if gtfs_ok else MISSING, "observed_at": iso(now) if gtfs_ok else None}
    # ---- incident history: earlier cases on this route recorded by this deployment
    past = [c for c in (store.all_cases() if store else []) if c["raw"]["route_id"] == v["route_id"]]
    history = {"past_cases_this_deployment": len(past),
               "past_false_alarms": sum(bool((c.get("outcome") or {}).get("false_alarm")) for c in past),
               "source": "case store (this deployment)" if store else MISSING, "observed_at": iso(now)}
    resources = {"available_buses": None, "available_operators": None, "estimated_dispatch_cost": None,
                 "estimated_travel_min": None, "source": MISSING, "observed_at": None}
    return {"service_feeds": service, "vision_ridership": vision, "connecting_services": connecting,
            "incident_history": history, "community_access": community, "operating_resources": resources,
            "event_signals": events}


# 2. STANDARDIZE AND COLLATE INPUT DATA
def collate_inputs(data):
    combined = []
    collated_at = datetime.now(timezone.utc)
    for source_name, source_data in data.items():
        observed = source_data.get("observed_at")
        age_s = int((collated_at - parse_iso(observed)).total_seconds()) if observed else None
        combined.append({"source": source_name, "timestamp": observed or iso(collated_at), "collated_at": iso(collated_at),
                         "age_s": age_s, "data": source_data, "status": "unverified"})
    return combined


# 3. SMALL LANGUAGE MODEL - FLAN-T5 (one short question per field; supporting information only)
_SLM = {"tok": None, "model": None, "error": None}
SLM_QUESTIONS = {
    "route": "Which transit route is affected?",
    "delay": "How many minutes is the delay?",
    "location": "Where is the problem?",
    "cause": "What caused the delay?",
}


def load_slm():
    if _SLM["model"] is not None:
        return True
    if _SLM["error"]:
        return False
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _SLM["tok"] = AutoTokenizer.from_pretrained(SLM_MODEL_ID)
        _SLM["model"] = AutoModelForSeq2SeqLM.from_pretrained(SLM_MODEL_ID)
        return True
    except Exception as e:  # no internet / no transformers: deterministic fields still work
        _SLM["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return False


def slm_answers(report):
    """Returns {field: short answer} or None. Supporting information only: never verified, never scored."""
    if not (USE_SLM and load_slm()):
        return None
    tok, model = _SLM["tok"], _SLM["model"]
    answers = {}
    for field_name, q in SLM_QUESTIONS.items():
        prompt = f"Answer from the report only. If it is not stated, answer 'unknown'.\nReport: {report}\nQuestion: {q}"
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        out = model.generate(**ids, max_new_tokens=24, do_sample=False)
        answers[field_name] = tok.decode(out[0], skip_special_tokens=True).strip()
    return answers


def interpret_event(report):
    a = slm_answers(report)
    if a is None:
        return f"[SLM unavailable — {_SLM['error'] or 'disabled'}; deterministic fields only]"
    return "; ".join(f"{k}: {v}" for k, v in a.items())


# 4. CREATE STRUCTURED EVENT FIELDS (deterministic; numbers never come from a language model)
DELAY_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)",        # a range reports the upper end
    r"(\d+(?:\.\d+)?)\s*-?\s*(?:minutes?|mins?)\s*(?:delay|late|behind)",
    r"delay(?:ed|s)?\s*(?:of|by)?\s*(?:up to|about|around|approximately)?\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)",
]
CAUSE_KEYWORDS = {
    "road_closure": ["road closure", "closed road", "road closed", "detour"],
    "weather": ["snow", "ice", "icy", "storm", "flood", "weather", "heavy rain"],
    "obstruction": ["crash", "collision", "accident", "blocked", "obstruction", "disabled vehicle"],
    "mechanical": ["mechanical", "breakdown", "operator shortage", "vehicle failure"],
    "medical": ["medical", "sick passenger"],
}


def extract_delay(text):
    for p in DELAY_PATTERNS:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            v = max(float(g) for g in m.groups() if g)
            return int(v) if v.is_integer() else v
    return None


def classify_cause(text):
    t = text.lower()
    for cause, kws in CAUSE_KEYWORDS.items():
        # a keyword must START a word ("flood" matches "flooding"), so "ice" inside "service" is not weather
        if any(re.search(rf"\b{re.escape(k)}\w*", t) for k in kws):
            return cause
    return "none"


def slm_cross_check(answers, delay_min, route):
    if not answers:
        return []
    flags = []
    if delay_min is not None and str(delay_min) not in answers.get("delay", ""):
        flags.append(f"SLM delay '{answers.get('delay')}' differs from extracted {delay_min} min")
    if route and str(route).lower() not in answers.get("route", "").lower():
        flags.append(f"SLM route '{answers.get('route')}' differs from feed route {route}")
    return flags


# 5. LLM ORGANIZER (organizes the record for the Signal Scout — does not score, verify or recommend)
def llm_organize(report, slm_text, weather, sources):
    if not USE_LLM_ORGANIZER:
        return None, "disabled"
    if not LLM_READY:
        return None, "skipped (no LLM key)"
    try:
        txt = llm_generate(json.dumps({"report": report, "slm_interpretation": slm_text, "weather": weather,
                                       "data_sources": sources}),
                           "Organize transit incident data for the Signal Scout agent in at most 6 short bullet points. "
                           "Separate confirmed live information from missing information, using the data_sources labels. "
                           "Do not calculate scores, verify incidents, or recommend operational actions.",
                           max_tokens=300)
        bullets = [l.strip() for l in txt.splitlines() if l.strip().startswith(("-", "•", "*")) and len(l.strip()) > 2][:6]
        if not bullets:
            return None, f"discarded: {llm_source()} did not return bullet points"
        return "\n".join(bullets), f"completed ({llm_source()})"
    except LLMUnavailable as e:
        return None, f"failed ({str(e)[:120]})"


def build_signal_scout_input(inputs):
    """Collate → SLM reading → deterministic fields → optional LLM organization → hand-off for the Signal Scout."""
    collated_data = collate_inputs(inputs)
    ev = inputs["event_signals"]
    event_report = ev["report"]
    answers = slm_answers(event_report)
    slm_text = ("; ".join(f"{k}: {v}" for k, v in answers.items()) if answers
                else f"[SLM unavailable — {_SLM['error'] or 'disabled'}; deterministic fields only]")
    text_delay = extract_delay(event_report) if ev.get("source_type") == "official_alert" else None
    route = inputs["service_feeds"]["route"]
    weather_ctx = {"conditions": ev.get("weather"), "alerts": ev.get("weather_alerts", []), "source": ev.get("weather_source")}
    llm_text, llm_status = llm_organize(event_report, slm_text, weather_ctx,
                                        {k: v.get("source", MISSING) for k, v in inputs.items()})
    structured_event = {
        "route": route, "delay_min": text_delay, "cause": classify_cause(event_report), "report_text": event_report,
        "slm_interpretation": slm_text, "slm_answers": answers,
        "slm_cross_check": slm_cross_check(answers, text_delay, inputs["service_feeds"].get("route_label") or route),
        "llm_interpretation": llm_text, "source_type": ev.get("source_type", "none"), "verification_status": "pending",
    }
    return {
        "case_status": "new", "status": "awaiting_verification", "created_at": iso(utcnow()), "route": route,
        "event": structured_event, "all_inputs": collated_data,
        "external_feeds": {
            "nj_transit": {k: v for k, v in LIVE_CONTEXT["nj_transit"].items() if k in ("status", "feeds", "nearby_vehicles", "fetched_at")},
            "census": {k: v for k, v in LIVE_CONTEXT["census"].items()},
            "weather": {k: v for k, v in LIVE_CONTEXT["weather"].items() if k != "connectors"},
            "weather_connectors": LIVE_CONTEXT["weather"].get("connectors", []),
            "computer_vision": {"detector": CV_RESULTS.get("detector"), "cameras": CV_RESULTS.get("cameras", [])},
            "gtfs_static": inputs["service_feeds"].get("gtfs"),
            "source_of_each_input": {k: v.get("source", MISSING) for k, v in inputs.items()},
            "llm": llm_status,
        },
        "verification_required": True, "confidence_score": None, "priority_score": None,
        "audit_log": [{"step": "input_preparation", "status": "completed", "message": "Live data collated for Signal Scout"}],
    }


# ====================================================================================================
# cell: aint_core
# ====================================================================================================
#@title A · INTEGRATION — adapter from Section A's live hand-off to the raw bundle the E2 engine scores
NWS_SEVERITY_RANK = {"Unknown": 0, "Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
NWS_TO_EVENT_SEVERITY = {"Unknown": "minor", "Minor": "minor", "Moderate": "minor", "Severe": "major", "Extreme": "emergency"}
# Published lookup: the severity an official alert's cause is given (a rule, not data).
SEVERITY_BY_CAUSE = {"road_closure": "major", "obstruction": "major", "weather": "minor", "mechanical": "minor",
                     "medical": "minor", "none": "none"}


def transit_time_period(ts=None):
    """Service period from the local (New Jersey) time of the observation."""
    t = (parse_iso(ts) if ts else utcnow()).astimezone(ZoneInfo("America/New_York"))
    h = t.hour + t.minute / 60
    if t.weekday() >= 5:
        return "midday" if 9 <= h < 20 else "off_peak"
    if 6 <= h < 9.5 or 15.5 <= h < 19:
        return "peak"
    if 9.5 <= h < 15.5:
        return "midday"
    return "off_peak"


def _crowding(load):
    if load is None:
        return None
    return "unboardable" if load >= 1.0 else "high" if load >= 0.85 else "medium" if load >= 0.6 else "low"


def signal_scout_input_to_raw(ssi, name=None):
    """Section A record → raw bundle for the Signal Scout / E2 engine. Every value is live or missing (None).
    Numbers come only from the feeds or the deterministic regex; the SLM and LLM text are attached as notes."""
    src = {r["source"]: r for r in ssi["all_inputs"]}
    svc, vis = src["service_feeds"]["data"], src["vision_ridership"]["data"]
    conn, hist = src["connecting_services"]["data"], src["incident_history"]["data"]
    comm, res = src["community_access"]["data"], src["operating_resources"]["data"]
    ev = src["event_signals"]["data"]
    now = utcnow()
    net = network_or_none()
    rid = svc["route"]
    stops = svc["affected_stops"]
    headway = svc.get("scheduled_headway_min")
    delay = svc.get("delay_min")
    cancelled = bool(svc.get("cancelled"))
    # Gap between consecutive buses when one runs `delay` minutes late (or one trip is missing) ÷ scheduled headway.
    # It is derived from the delay, so it is never a second hard trigger (headway_from_delay below).
    gap_ratio = 2.0 if cancelled else (round((headway + delay) / headway, 2) if headway and delay is not None else None)
    cause = ssi["event"]["cause"]
    stype = ev.get("source_type", "none")
    feed_age = src["service_feeds"]["age_s"] or 0

    def obs(source_id, stype_, age_s, fields, corroborated=True, **extra):
        return {"source_id": source_id, "source_type": stype_, "timestamp": iso(now - timedelta(seconds=age_s)),
                "age_s": age_s, "corroborated": corroborated, "fields": {k: v for k, v in fields.items() if v is not None},
                **extra}

    observations = [obs(f"AVL-{rid}", "vehicle_apc", feed_age,
                        {"delay_min": delay, "headway_ratio": gap_ratio, "cancelled": cancelled,
                         "cancellation_risk": 1.0 if cancelled else None})]
    if stype == "official_alert":
        sev = SEVERITY_BY_CAUSE.get(cause, "minor")
        if cause == "none":
            sev = "minor"         # an official alert with no recognised cause still counts as a verified minor event
        observations.append(obs("AGENCY-ALERTS", "official_alert", src["event_signals"]["age_s"] or 0,
                                {"event_type": cause, "event_severity": sev, "event_verified": True,
                                 "delay_min": ssi["event"]["delay_min"]}))
    w_alerts = ev.get("weather_alerts") or []
    if w_alerts:
        top = max(w_alerts, key=lambda a: NWS_SEVERITY_RANK.get(a.get("severity"), 0))
        fetched = ev.get("weather_fetched_at")
        age = int((now - parse_iso(fetched)).total_seconds()) if fetched else 0
        observations.append(obs("NWS-ALERTS", "trusted_external", age,
                                {"event_type": "weather", "event_severity": NWS_TO_EVENT_SEVERITY.get(top.get("severity"), "minor")},
                                nws_event=top.get("event")))
    load = vis.get("load_factor")
    vision = {"camera_feeds": [vis["camera_id"]] if vis.get("camera_id") else [], "boardings": vis.get("boardings"),
              "alightings": None, "capacity": vis.get("capacity"), "load_factor": load, "passups": vis.get("pass_ups"),
              "obstruction": vis.get("obstruction"), "crowding_level": _crowding(load), "anonymized": True,
              "identities_stored": False, "detector": vis.get("detector"), "source": vis.get("source"),
              "waiting_at_stop": vis.get("waiting_at_stop")}
    # Alternatives: fewest other routes within 450 m of any affected stop (static GTFS); missing without GTFS
    real_stops = [s for s in stops if not str(s["stop_id"]).startswith("VEHICLE-")]
    alternatives = min(len(net.routes_near(s["lat"], s["lon"], 0.45, exclude=rid)) for s in real_stops) if (net and real_stops) else None
    agency = svc.get("agency") or "NJ Transit"
    ext_partner = agency
    raw = {
        "scenario_name": name or f"Live · route {svc.get('route_label') or rid} · vehicle {svc.get('vehicle_id')}",
        "route_id": rid, "route_name": svc.get("route_label") or rid, "external_route_id": rid,
        "trip_id": svc.get("trip_id"), "vehicle_id": svc.get("vehicle_id"), "agency": agency,
        "headway_min": headway, "headway_from_delay": True,
        "stops": stops, "affected_stop_ids": [s["stop_id"] for s in stops],
        "vehicle_position": svc.get("vehicle_position"),
        "time_period": "last_service" if conn.get("last_service") else transit_time_period(src["service_feeds"].get("observed_at")),
        "event_type": cause, "observations": observations, "vision": vision,
        "connections": {"connecting_route": (conn.get("connecting_routes") or [None])[0],
                        "connecting_routes": conn.get("connecting_routes") or [], "partner": ext_partner,
                        "transfer_stop_id": conn.get("transfer_stop_id"), "transfer_window_min": conn.get("transfer_window_min"),
                        "connecting_headway_min": conn.get("connecting_headway_min"), "last_service": conn.get("last_service"),
                        "missed_transfers": conn.get("missed_transfers")},
        "community": {"transit_dependence": comm.get("transit_dependence"), "income_disadvantage": comm.get("income_disadvantage"),
                      "disability": comm.get("disability_barrier"),
                      "areas": [t["tract_name"] for t in comm.get("tracts", [])], "tracts": comm.get("tracts", [])},
        "access": {"essential_type": None, "essential_names": [], "alternatives": alternatives},
        "resources": {"depots": [], "bus_hour_cost_usd": APP.PM.current["bus_hour_cost_usd"] if "APP" in globals() else DEFAULT_POLICY["bus_hour_cost_usd"],
                      "source": res.get("source")},
        "roads": {"detour_open": None, "detour_extra_min": None, "segment_open": None, "reroute_allowed": None,
                  "detour_announced": ev.get("alert_effect") == "DETOUR"},
        "faults": {"vision_agent_failures": 0, "backup_available": False},
        "generated_at": iso(now),
        "incident_history": {k: v for k, v in hist.items() if k != "observed_at"},
        "slm_interpretation": ssi["event"]["slm_interpretation"], "slm_cross_check": ssi["event"].get("slm_cross_check", []),
        "llm_interpretation": ssi["event"].get("llm_interpretation"),
        "external_feeds": ssi.get("external_feeds", {}),
        "weather_context": {"conditions": ev.get("weather"), "alerts": w_alerts, "source": ev.get("weather_source")},
        "event_report": ev.get("report"), "handoff_from": "Section A — live inputs",
        "missing_inputs": [k for k, r in src.items() if str(r["data"].get("source", "")).startswith("not connected")],
    }
    return raw


def raw_for_vehicle(vehicle=None, store=None, name=None):
    """One live vehicle → Section A inputs → Signal Scout hand-off → raw bundle. vehicle: a feed row, an id, or None."""
    ssi = build_signal_scout_input(section_a_live_inputs(vehicle, store))
    return signal_scout_input_to_raw(ssi, name)


# ====================================================================================================
# cell: e2_core
# ====================================================================================================
#@title E2 · Deterministic validation, exception paths and scoring engine (Signal Scout → verification → confidence → impact → planner → evaluator)
APPROVED_TOOLS = {"vision_agent", "data_verification", "feed_outage_path", "conflicting_data_path", "confidence_calculation",
                  "impact_analyzer", "significance_rules", "planner_copilot", "scenario_evaluator", "explanation_agent",
                  "outcome_evaluator", "insight_agent"}
SOURCE_REGISTRY = {"AVL-": "vehicle_apc", "APC-": "vehicle_apc", "VISION-": "vehicle_apc", "AGENCY-ALERTS": "official_alert",
                   "NWS-": "trusted_external", "511-": "trusted_external", "GTFSRT-MIRROR-": "trusted_external",
                   "SOCIAL-": "public_report"}
REQUIRED_FIELDS = ["delay_min", "headway_ratio", "cancellation_risk", "load_factor", "passups", "missed_transfers",
                   "event_severity", "transit_dependence", "income_disadvantage", "disability", "essential_type", "alternatives"]
OPTION_LABELS = {"do_nothing": "Do nothing (baseline)", "rider_alerts": "Send rider alerts",
                 "protect_connection": "Protect the connection (hold)", "reroute": "Reroute around blockage",
                 "dispatch_second_bus": "Dispatch a second bus"}


class AgentFailure(Exception):
    pass


class SignalScout:
    """Signal Scout Agent: routes tasks, enforces approved tools, records the trace, exceptions and alerts."""

    def __init__(self):
        self.trace, self.exceptions, self.alerts = [], [], []

    def step(self, stage, result, detail=""):
        self.trace.append({"stage": stage, "result": result, "detail": detail})

    def exception(self, path, tool, detail, outcome):
        self.exceptions.append({"path": path, "tool": tool, "detail": detail, "outcome": outcome})

    def alert(self, to, msg):
        self.alerts.append({"to": to, "message": msg})

    def invoke(self, tool, fn, *args, retry=False, **kw):
        if tool not in APPROVED_TOOLS:
            raise PermissionError(f"Signal Scout blocked unapproved tool '{tool}'")
        attempts = 2 if retry else 1
        last = None
        for a in range(1, attempts + 1):
            try:
                out = fn(*args, attempt=a, **kw) if retry else fn(*args, **kw)
                if a > 1:
                    self.exception("AGENT ERROR", tool, "Recovered on retry with the same inputs", "recovered")
                return out
            except AgentFailure as e:
                last = e
                self.exception("AGENT ERROR", tool, str(e), "retrying once with same inputs" if a < attempts else "failed twice")
        raise last


def vision_agent(raw, attempt=1):
    """Vision Agent (ridership and load). Uses anonymized camera/sensor aggregates; no identities stored."""
    if attempt <= raw.get("faults", {}).get("vision_agent_failures", 0):
        raise AgentFailure(f"Vision Agent inference error on attempt {attempt}")
    v = raw.get("vision") or {}
    if v.get("identities_stored"):
        raise AgentFailure("Vision payload contains identities; rejected")
    return {k: v.get(k) for k in ["boardings", "alightings", "capacity", "load_factor", "passups", "obstruction", "crowding_level"]}


def _source_type_ok(o):
    expected = next((t for p, t in SOURCE_REGISTRY.items() if o["source_id"].startswith(p)), None)
    return expected is not None and expected == o["source_type"]


def verify_observations(raw, policy):
    """Data verification: schema, timestamps, refresh, source identity, completeness."""
    rows = []
    for o in raw.get("observations", []):
        o = copy.deepcopy(o)
        problems = []
        for k in ["source_id", "source_type", "timestamp", "age_s", "fields"]:
            if k not in o:
                problems.append(f"missing {k}")
        try:
            parse_iso(o["timestamp"])
        except Exception:
            problems.append("bad timestamp")
        if not problems and not _source_type_ok(o):
            problems.append("unknown source identity")
        if problems:
            o["status"], o["note"] = "excluded", "; ".join(problems)
        elif o["source_type"] == "public_report" and not o.get("corroborated"):
            o["status"], o["note"] = "excluded", "Uncorroborated public report cannot trigger action"
        elif o["age_s"] > policy["freshness_limit_s"]:
            o["status"], o["note"] = "stale", f"Older than freshness limit ({o['age_s']} s > {policy['freshness_limit_s']} s)"
        else:
            o["status"], o["note"] = "accepted", "Schema, timestamp, freshness and identity verified"
        q_key = "public_report_corroborated" if o["source_type"] == "public_report" else o["source_type"]
        o["quality"] = policy["source_quality"].get(q_key, 0.0) if o["status"] != "excluded" else 0.0
        rows.append(o)
    return rows


def precedence_rank(policy, stype):
    order = policy["source_precedence"]
    return order.index(stype) if stype in order else len(order)


def resolve_fields(evidence, policy):
    """Take each field from the highest-precedence accepted source."""
    acc = sorted([e for e in evidence if e["status"] == "accepted"], key=lambda e: precedence_rank(policy, e["source_type"]))
    fields, origin = {}, {}
    for e in acc:
        for k, v in e["fields"].items():
            if k not in fields and v is not None:
                fields[k], origin[k] = v, e["source_type"]
    return fields, origin, acc


def run_analysis(raw, policy, operator_resolution=None, constraints=None):
    """Pure, deterministic analysis of one raw bundle under one policy version (used live and in backtests)."""
    sc = SignalScout()
    A = {"policy_version": policy["version"], "gate": None, "gate_reason": "", "significant": False, "hard_triggers": [],
         "official_hard_trigger": False, "confidence": None, "scores": None, "normalization": [], "options": [],
         "ranking": [], "missing": [], "tier": None, "conflict": None, "outage": None}
    sc.step("Signal Scout", "case opened", f"Route {raw['route_id']} · {len(raw.get('observations', []))} observations · tools restricted to approved list")

    # ---- Data verification
    evidence = sc.invoke("data_verification", verify_observations, raw, policy)
    sc.step("Data verification", f"{sum(e['status'] == 'accepted' for e in evidence)} accepted / {len(evidence)}",
            "; ".join(f"{e['source_id']}: {e['status']}" for e in evidence))
    for e in evidence:
        if e["note"].startswith("Uncorroborated"):
            sc.exception("SOCIAL REPORT", "data_verification", f"{e['source_id']} excluded: requires corroboration", "excluded")

    # ---- Vision agent (retry once with same inputs, then deterministic safe mode)
    try:
        vis = sc.invoke("vision_agent", vision_agent, raw, retry=True)
        evidence.append({"source_id": f"VISION-{raw['route_id']}", "source_type": "vehicle_apc", "timestamp": raw["generated_at"],
                         "age_s": 15, "corroborated": True, "fields": {k: v for k, v in vis.items() if v is not None},
                         "status": "accepted", "note": "Anonymized station/onboard camera + sensor aggregates (no identities)",
                         "quality": policy["source_quality"]["vehicle_apc"]})
        sc.step("Vision Agent", "ok", f"load factor {vis.get('load_factor')}, pass-ups {vis.get('passups')}")
    except AgentFailure:
        A.update(gate="SAFE_MODE", gate_reason="Vision Agent failed twice → deterministic safe mode; evidence preserved")
        sc.step("Agent error path", "SAFE MODE", A["gate_reason"])
        sc.alert("technical owner", "Vision Agent failed twice with the same inputs; safe mode engaged")
        sc.alert("planner", "Case routed to safe default because an agent failed; no resource action will be taken")
        A.update(evidence=evidence, trace=sc.trace, exceptions=sc.exceptions, alerts=sc.alerts)
        return A

    # ---- Feed outage path
    vehicle = [e for e in evidence if e["source_type"] == "vehicle_apc" and not e["source_id"].startswith("VISION-")]
    if vehicle and all(e["status"] == "stale" for e in vehicle):
        oldest = max(e["age_s"] for e in vehicle)
        for e in vehicle:
            e["note"] = f"FROZEN last verified snapshot, age {e['age_s']} s (kept as evidence, not scored)"
        sc.alert("operator", f"Vehicle feed outage on {raw['route_id']} (snapshot age {oldest} s); backup source requested")
        if raw.get("faults", {}).get("backup_available") and raw.get("backup_observation"):
            b = copy.deepcopy(raw["backup_observation"])
            b.update(status="accepted", note="Backup source after feed outage; re-verified", quality=policy["source_quality"]["trusted_external"])
            evidence.append(b)
            A["outage"] = "backup"
            sc.exception("FEED OUTAGE", "feed_outage_path", f"Snapshot frozen at age {oldest} s; backup {b['source_id']} found", "re-verified with backup")
            sc.step("Feed outage path", "backup check: YES → re-verify", b["source_id"])
        else:
            A["outage"] = "no_backup"
            sc.exception("FEED OUTAGE", "feed_outage_path", f"Snapshot frozen at age {oldest} s; no backup source", "monitoring only")
            sc.step("Feed outage path", "backup check: NO", "Freshness limit exceeded → monitoring only")

    # ---- Conflicting data path
    acc_delay = sorted([e for e in evidence if e["status"] == "accepted" and "delay_min" in e["fields"]],
                       key=lambda e: precedence_rank(policy, e["source_type"]))
    conflict_rows = []
    if len(acc_delay) >= 2:
        win = acc_delay[0]
        for e in acc_delay[1:]:
            dd = abs(e["fields"]["delay_min"] - win["fields"]["delay_min"])
            cc = ("cancelled" in e["fields"] and "cancelled" in win["fields"] and e["fields"]["cancelled"] != win["fields"]["cancelled"])
            if dd > policy["conflict_threshold_min"] or cc:
                conflict_rows.append((e, dd))
    unresolved_conflict = False
    if conflict_rows:
        for e, dd in conflict_rows:
            e["status"] = "quarantined"
            e["note"] = f"Conflicts with {acc_delay[0]['source_id']} by {dd:.0f} min; lower source precedence"
        desc = ", ".join(e["source_id"] for e, _ in conflict_rows)
        if operator_resolution == "accept_precedence":
            A["conflict"] = "resolved"
            sc.exception("CONFLICTING DATA", "conflicting_data_path", f"Quarantined {desc}; operator accepted precedence", "resolved")
            sc.step("Conflicting data path", "RESOLVED by operator", "Precedence winner kept; quarantined sources excluded")
        elif operator_resolution == "unresolved":
            A["conflict"] = "unresolved_final"
            sc.exception("CONFLICTING DATA", "conflicting_data_path", f"Quarantined {desc}; operator could not resolve", "monitoring")
        else:
            A["conflict"] = "pending"
            unresolved_conflict = True
            sc.exception("CONFLICTING DATA", "conflicting_data_path",
                         f"Quarantined {desc}; precedence vehicle > official > trusted external > corroborated public; C reduced",
                         "flagged for operator review")
            sc.alert("operator", "Conflicting data quarantined; operator review required")

    # ---- Resolve fields + context
    fields, origin, acc = resolve_fields(evidence, policy)
    for k, src in [("transit_dependence", "community"), ("income_disadvantage", "community"), ("disability", "community")]:
        if raw.get("community", {}).get(k) is not None:
            fields[k], origin[k] = raw["community"][k], "community (area-level)"
    acc_info = raw.get("access", {})
    if acc_info.get("essential_type") is not None:
        fields["essential_type"], origin["essential_type"] = acc_info["essential_type"], "access"
    if acc_info.get("alternatives") is not None:
        fields["alternatives"], origin["alternatives"] = acc_info["alternatives"], "access"
    conn = raw.get("connections", {})
    if conn.get("missed_transfers") is not None:
        fields["missed_transfers"], origin["missed_transfers"] = conn["missed_transfers"], "connecting services"
    if fields.get("cancelled") and "cancellation_risk" not in fields:
        fields["cancellation_risk"], origin["cancellation_risk"] = 1.0, origin.get("cancelled")
    if "event_severity" not in fields:
        fields["event_severity"], origin["event_severity"] = "none", "default: no event reported"
    elif origin.get("event_severity") not in ("official_alert", "trusted_external"):
        fields["event_severity"], origin["event_severity"] = "none", "unverified event ignored (E = 0)"
    A["fields"], A["field_origin"] = fields, origin
    A["missing"] = [f for f in REQUIRED_FIELDS if fields.get(f) is None]

    # ---- Data ready?
    feed_available = len([e for e in acc if not e["source_id"].startswith("VISION-")]) > 0
    ready = feed_available and not unresolved_conflict
    sc.step("Data ready?", "YES" if ready else "NO",
            f"feed available={feed_available}, agent healthy=True, unresolved conflict={unresolved_conflict}")

    # ---- Confidence C = 0.40Q + 0.25F + 0.20G + 0.15M
    def confidence(include_quarantined):
        pool = [e for e in evidence if e["status"] == "accepted" or (include_quarantined and e["status"] == "quarantined")]
        pool_nv = [e for e in pool if not e["source_id"].startswith("VISION-")]
        if not pool_nv:
            return {"Q": 0.0, "F": 0.0, "G": 0.0, "M": 0.0, "penalty": 0.0, "C": 0.0}
        Q = statistics.mean(e["quality"] for e in pool_nv)
        top = min(pool_nv, key=lambda e: precedence_rank(policy, e["source_type"]))
        F = n_cost(top["age_s"], *policy["bounds"]["age_s"])
        dl = [e["fields"]["delay_min"] for e in pool_nv if "delay_min" in e["fields"]]
        G = n_cost(max(dl) - min(dl), *policy["bounds"]["agreement_spread_min"]) if len(dl) >= 2 else policy["single_source_agreement"]
        M = 1 - len(A["missing"]) / len(REQUIRED_FIELDS)
        w = policy["confidence_weights"]
        pen = policy["conflict_penalty"] if unresolved_conflict else 0.0
        C = max(0.0, w["Q"] * Q + w["F"] * F + w["G"] * G + w["M"] * M - pen)
        return {"Q": r3(Q), "F": r3(F), "G": r3(G), "M": r3(M), "penalty": pen, "C": r3(C)}

    conf = sc.invoke("confidence_calculation", confidence, unresolved_conflict)
    A["confidence"] = conf

    # ---- Hard triggers (from accepted evidence only)
    ht = policy["hard_trigger"]
    lk = policy["lookups"]
    trig = []
    if fields.get("cancelled"):
        trig.append(("cancellation", origin.get("cancelled")))
    if fields.get("delay_min") is not None and fields["delay_min"] >= ht["delay_min"]:
        trig.append((f"delay ≥ {ht['delay_min']} min", origin.get("delay_min")))
    gap_measured = not raw.get("headway_from_delay")   # a gap derived from the delay is not a second trigger
    if gap_measured and fields.get("headway_ratio") is not None and fields["headway_ratio"] >= ht["gap_headway_ratio"]:
        trig.append((f"gap ≥ {ht['gap_headway_ratio']} × headway", origin.get("headway_ratio")))
    if fields.get("event_severity") == "emergency" and origin.get("event_severity") in ("official_alert",):
        trig.append(("verified emergency", "official_alert"))
    if (fields.get("alternatives") == 0 and lk["essential_severity"].get(fields.get("essential_type", "none"), 0) >= ht["essential_severity_min"]
            and (fields.get("cancelled") or (fields.get("delay_min") or 0) >= ht["delay_min"])):
        trig.append(("essential route with no alternative disrupted", origin.get("cancelled") or origin.get("delay_min")))
    if (fields.get("load_factor") or 0) >= ht["unboardable_load_factor"] and (fields.get("passups") or 0) >= 1:
        trig.append(("unboardable crowding", "vehicle_apc"))
    A["hard_triggers"] = [{"trigger": t, "source": s, "official": s in policy["official_sources"]} for t, s in trig]
    A["official_hard_trigger"] = any(t["official"] for t in A["hard_triggers"])

    # ---- Impact analyzer (always computed so monitoring cases record priority)
    if feed_available:
        A["scores"], A["normalization"] = sc.invoke("impact_analyzer", impact_scores, fields, raw, policy)
    sig = bool(A["hard_triggers"]) or (A["scores"] is not None and A["scores"]["R"] >= policy["significance_R"])
    A["significant"] = sig

    # ---- Gates (flowchart order)
    if A["outage"] == "no_backup":
        A.update(gate="MONITORING", gate_reason="Vehicle feed older than the freshness limit and no backup feed → monitoring only")
    elif not feed_available:
        A.update(gate="MONITORING", gate_reason="No verified evidence (e.g. uncorroborated public report only) → continue monitoring")
    elif unresolved_conflict:
        A.update(gate="CONFLICT_REVIEW", gate_reason="Conflict quarantined; operator review required before analysis can proceed")
    elif A["conflict"] == "unresolved_final":
        A.update(gate="MONITORING", gate_reason="Conflict could not be resolved by operator → monitoring")
    elif conf["C"] < policy["confidence_threshold"] and not A["official_hard_trigger"]:
        A.update(gate="MONITORING", gate_reason=f"C = {conf['C']:.2f} < {policy['confidence_threshold']} and no official hard trigger → monitoring")
    else:
        bypass = conf["C"] < policy["confidence_threshold"]
        sc.step("Confidence gate", "PROCEED" + (" (official hard trigger bypass)" if bypass else ""),
                f"C = {conf['C']:.3f} (threshold {policy['confidence_threshold']})")
        if not sig:
            A.update(gate="BELOW_THRESHOLD", gate_reason=f"No hard trigger and R = {A['scores']['R']:.2f} < {policy['significance_R']} → safe default (rider info + monitor)")
        else:
            A.update(gate="SIGNIFICANT", gate_reason=("Hard trigger: " + ", ".join(t["trigger"] for t in A["hard_triggers"])) if A["hard_triggers"]
                     else f"R = {A['scores']['R']:.2f} ≥ {policy['significance_R']}")
    if A["gate"] in ("MONITORING", "CONFLICT_REVIEW"):
        sc.step("Confidence gate", A["gate"], A["gate_reason"])
    if A["scores"]:
        sc.step("Deterministic impact analyzer", "scored", f"R={A['scores']['R']:.3f} V={A['scores']['V']:.3f} A={A['scores']['A']:.3f} P={A['scores']['P']:.3f}")
    sc.step("Significance rules", "SIGNIFICANT" if sig else "not significant", A["gate_reason"] if A["gate"] in ("SIGNIFICANT", "BELOW_THRESHOLD") else "")

    # ---- Severity tier for deadline control
    if A["scores"]:
        P = A["scores"]["P"]
        ts = policy["severity_tiers"]
        if any(t["trigger"] == "verified emergency" for t in A["hard_triggers"]) or P >= ts["severe_P"]:
            A["tier"] = "severe"
        elif P >= ts["significant_P"] or A["hard_triggers"]:
            A["tier"] = "significant"
        else:
            A["tier"] = "moderate"

    # ---- Planner Copilot + Scenario Evaluator (significant cases only)
    if A["gate"] == "SIGNIFICANT":
        cons = {**policy["planner_defaults"], **(constraints or {})}
        opts = sc.invoke("planner_copilot", planner_copilot, fields, raw, A["scores"], policy, cons)
        feas = [o for o in opts if o["feasible"]]
        sc.step("Planner Copilot", f"{len(feas)} feasible / {len(opts)} considered",
                "; ".join(f"{o['label']}: {'feasible' if o['feasible'] else o['infeasible_reason']}" for o in opts))
        ranked = sc.invoke("scenario_evaluator", scenario_evaluator, feas, conf["C"], A["scores"]["P"], policy)
        A["options"], A["ranking"], A["constraints_used"] = opts, ranked, cons
        if not feas:
            sc.step("Planner Copilot", "NO FEASIBLE OPTION", "Escalate to human review")
            sc.alert("planner", "No feasible option; escalated to human review")
        else:
            sc.step("Scenario evaluator", "ranked", " > ".join(f"{r['label']} ({r['Si']:.3f})" for r in ranked))
    A.update(evidence=evidence, trace=sc.trace, exceptions=sc.exceptions, alerts=sc.alerts)
    return A


def impact_scores(fields, raw, policy):
    b, lk = policy["bounds"], policy["lookups"]
    comp, norm = {}, []

    def add(code, group, label, raw_val, val, L=None, U=None, method="n(x)"):
        comp[code] = val
        norm.append({"group": group, "component": code, "feature": label, "raw": raw_val, "L": L, "U": U, "method": method,
                     "normalized": r3(val) if val is not None else None,
                     "note": "MISSING — not imputed; contributes 0 and lowers completeness M" if val is None else ""})

    d = fields.get("delay_min")
    add("D", "R", "delay (min)", d, n_benefit(d, *b["delay_min"]), *b["delay_min"])
    h = fields.get("headway_ratio")
    add("H", "R", "gap / scheduled headway", h, n_benefit(h, *b["headway_ratio"]), *b["headway_ratio"])
    k = fields.get("cancellation_risk")
    add("K", "R", "cancellation risk", k, None if k is None else min(1.0, max(0.0, float(k))), 0, 1, "clip 0–1")
    ev = fields.get("event_severity", "none")
    add("E", "R", f"verified event ({fields.get('event_type', 'none')})", ev, lk["event_severity"].get(ev, 0.0), method="lookup table")
    add("N", "V", "transit dependence (area share)", fields.get("transit_dependence"),
        n_benefit(fields.get("transit_dependence"), *b["transit_dependence"]), *b["transit_dependence"])
    add("I", "V", "income disadvantage (area share)", fields.get("income_disadvantage"),
        n_benefit(fields.get("income_disadvantage"), *b["income_disadvantage"]), *b["income_disadvantage"])
    add("B", "V", "disability barrier (area share)", fields.get("disability"),
        n_benefit(fields.get("disability"), *b["disability"]), *b["disability"])
    tp = raw.get("time_period", "off_peak")
    add("T", "V", "time sensitivity", tp, lk["time_sensitivity"].get(tp, 0.3), method="lookup table")
    es = fields.get("essential_type")
    add("S", "A", "essential service severity", es, None if es is None else lk["essential_severity"].get(es, 0.0), method="lookup table")
    al = fields.get("alternatives")
    add("L", "A", "loss of alternatives (routes nearby)", al, n_cost(al, *b["alternatives"]), *b["alternatives"], "ninv(x)")
    mt = fields.get("missed_transfers")
    add("X", "A", "connection loss (missed transfers)", mt, n_benefit(mt, *b["missed_transfers"]), *b["missed_transfers"])
    pu = fields.get("passups")
    add("U", "A", "unable to board (pass-ups)", pu, n_benefit(pu, *b["passups"]), *b["passups"])

    def wsum(group):
        w = policy[f"{group}_weights"]
        return sum(w[c] * (comp[c] or 0.0) for c in w)

    R, V, A_ = wsum("R"), wsum("V"), wsum("A")
    pw = policy["P_weights"]
    P = pw["R"] * R + pw["V"] * V + pw["A"] * A_
    for row in norm:
        w = policy[f"{row['group']}_weights"][row["component"]]
        row["weight"] = w
        row["contribution"] = r3(w * (comp[row["component"]] or 0.0))
    return {"R": r3(R), "V": r3(V), "A": r3(A_), "P": r3(P), "components": {k: r3(v) for k, v in comp.items()}}, norm


def planner_copilot(fields, raw, scores, policy, cons):
    """Planner Copilot: generates feasible options only, applying resource, road, policy and safety constraints.
    Assigns no scores. An option whose inputs are not available from a live source is listed as infeasible with
    the missing input named (nothing is assumed)."""
    headway = raw.get("headway_min")
    delay = fields.get("delay_min")
    delay_assumed = delay is None and headway is not None
    if delay is None:
        delay = headway or 0.0          # documented assumption: a missing delay is estimated as one scheduled headway
    if fields.get("cancelled") and headway:
        delay = max(delay, headway)    # riders of a cancelled trip wait at least one headway
    riders = (fields.get("boardings") or 0) + (fields.get("passups") or 0)
    V = scores["V"]
    vul = riders * V
    conn = raw.get("connections", {})
    roads = raw.get("roads", {})
    alts = fields.get("alternatives")
    cancelled = bool(fields.get("cancelled"))
    excluded = set(cons.get("excluded_options", []))
    agency = raw.get("agency") or "NJ Transit"
    opts = []

    def opt(oid, feasible, reason, dr, vul_helped, ap, of, cost, notes, extra=None, parties=None):
        o = {"option_id": oid, "type": oid.split("@")[0], "label": OPTION_LABELS[oid.split("@")[0]] + (f" — {extra}" if extra else ""),
             "feasible": feasible, "infeasible_reason": reason, "pred_delay_reduction_min": round(dr, 2),
             "pred_vulnerable_riders_helped": round(vul_helped, 1), "pred_access_protection": round(ap, 3),
             "feasibility": round(of, 3), "cost_usd": round(cost, 0), "notes": notes,
             "assumptions": "Delay missing: scheduled headway used as delay estimate" if delay_assumed else "",
             "parties": parties or [agency]}
        if o["type"] in excluded:
            o["feasible"], o["infeasible_reason"] = False, "Excluded by planner revision"
        opts.append(o)

    opt("do_nothing", True, "", 0, 0, 0.0, 1.0, 0, "Baseline: service recovers on its own")
    alt_note = "" if alts is not None else " (nearby alternatives unknown: no static GTFS)"
    opt("rider_alerts", True, "", min(delay, 2) * 0.5, 0.25 * vul, 0.10 + 0.10 * min(alts or 0, 3) / 3, 0.98, 50,
        "Push verified info to riders, stop displays and paratransit desk" + alt_note)
    # Protect a connection: needs the transfer window and the missed-transfer count, which no live source provides.
    if not conn.get("connecting_route"):
        opt("protect_connection", False, "No connecting route near the affected stops (static GTFS)", 0, 0, 0, 0, 0, "")
    elif conn.get("transfer_window_min") is None or conn.get("missed_transfers") is None:
        opt("protect_connection", False, f"Transfer window and missed transfers to {conn['connecting_route']} are not "
            "available from a live source", 0, 0, 0, 0, 0, "", extra=f"hold {conn['connecting_route']}")
    else:
        hold = max(0.0, delay - conn["transfer_window_min"])
        ok = hold <= cons["max_hold_min"]
        xn = n_benefit(conn["missed_transfers"], *policy["bounds"]["missed_transfers"]) or 0
        opt("protect_connection", ok, "" if ok else f"Hold needed {hold:.0f} min > max hold {cons['max_hold_min']} min",
            0.35 * delay, conn["missed_transfers"] * V, min(1.0, 0.30 + 0.50 * xn + (0.2 if conn.get("last_service") else 0)),
            0.85 - 0.15 * (hold / max(cons["max_hold_min"], 1)), 40 * hold,
            f"Hold {conn['connecting_route']} at {conn['transfer_stop_id']} up to {hold:.0f} min",
            extra=f"hold {conn['connecting_route']}", parties=[conn.get("partner") or agency])
    # Reroute: needs a blockage plus detour road status, extra time and permission (no live source for these).
    blockage = raw.get("event_type") in ("road_closure", "obstruction") or bool(fields.get("obstruction"))
    if not blockage:
        opt("reroute", False, "No verified blockage or closure on the segment", 0, 0, 0, 0, 0, "")
    elif roads.get("detour_announced"):
        opt("reroute", False, "NJ Transit has already announced a detour for this route (see the alert); "
            "the prototype does not plan a second one", 0, 0, 0, 0, 0, "")
    elif cancelled:
        opt("reroute", False, "Trip cancelled; nothing to reroute", 0, 0, 0, 0, 0, "")
    elif None in (roads.get("detour_open"), roads.get("detour_extra_min"), roads.get("reroute_allowed")):
        opt("reroute", False, "Detour road status, extra running time and reroute permission are not available "
            "from a live source", 0, 0, 0, 0, 0, "")
    else:
        reasons = []
        if not roads["detour_open"]:
            reasons.append("detour road closed")
        if not roads["reroute_allowed"]:
            reasons.append("reroute not permitted by policy")
        ex = roads["detour_extra_min"]
        opt("reroute", not reasons, "; ".join(reasons), 0.6 * delay, 0.6 * vul, 0.55, max(0.2, 0.75 - 0.02 * ex),
            100 + 25 * ex, f"Detour adds ~{ex} min; skipped stops get rider alerts")
    # Dispatch a second bus: needs depots with spare buses and operators (no live source is connected).
    deps = raw.get("resources", {}).get("depots") or []
    if not deps:
        opt("dispatch_second_bus", False, "No depot, spare-bus or operator data is connected", 0, 0, 0, 0, 0, "")
        return opts
    stops = raw.get("stops") or []
    mid = (statistics.mean(s["lat"] for s in stops), statistics.mean(s["lon"] for s in stops))
    deps = sorted(deps, key=lambda d: km_between((d["lat"], d["lon"]), mid))
    use = deps if cons.get("allow_alternate_depot") else deps[:1]
    for i, d in enumerate(use):
        dh = d.get("deadhead_min_override") or km_between((d["lat"], d["lon"]), mid) / 25 * 60
        cost = d.get("dispatch_cost_override") or raw["resources"]["bus_hour_cost_usd"] * (dh + 60) / 60
        reasons = []
        if (d.get("spare_buses") or 0) < 1:
            reasons.append("no spare bus")
        if (d.get("operators") or 0) < 1:
            reasons.append("no available operator")
        if dh > cons["max_deadhead_min"]:
            reasons.append(f"deadhead {dh:.0f} min > {cons['max_deadhead_min']} min")
        if cost > cons["budget_cap_usd"]:
            reasons.append(f"cost ${cost:.0f} > budget cap ${cons['budget_cap_usd']}")
        if roads.get("segment_open") is False:
            reasons.append("segment closed to buses")
        passup_relief = 0.5 * (headway or 0) * min(1.0, (fields.get("passups") or 0) / 20)
        dr = max(0.0, delay - 0.5 * dh) * 0.85 + passup_relief
        opt(f"dispatch_second_bus@{d['id']}", not reasons, "; ".join(reasons), dr, 0.85 * vul, 0.80,
            0.10 + 0.85 * (n_cost(dh, 0, 40) or 0), cost, f"Spare bus from {d['name']}, deadhead {dh:.0f} min",
            extra=d["name"] + (" (alternate)" if i > 0 else ""), parties=[d.get("operated_by") or agency])
    return opts


def scenario_weights(P, policy):
    base, shift = policy["scenario_base"], policy["scenario_shift"]
    return {c: base[c] + shift.get(c, 0) * P for c in base}


def scenario_evaluator(feasible_opts, C, P, policy):
    """Si = Ci × Σ w_c(P)·component; tie-break: higher AP, then higher OF, then lower cost. No LLM judgment."""
    b, lk = policy["bounds"], policy["lookups"]
    w = scenario_weights(P, policy)
    out = []
    for o in feasible_opts:
        comp = {"DR": n_benefit(o["pred_delay_reduction_min"], *b["dr_min"]),
                "EB": n_benefit(o["pred_vulnerable_riders_helped"], *b["eb_riders"]),
                "AP": min(1.0, max(0.0, o["pred_access_protection"])), "OF": min(1.0, max(0.0, o["feasibility"])),
                "CE": n_cost(o["cost_usd"], *b["cost_usd"])}
        Ci = C * lk["option_certainty"][o["type"]]
        inner = sum(w[c] * comp[c] for c in w)
        out.append({"option_id": o["option_id"], "label": o["label"], "type": o["type"], **{c: r3(v) for c, v in comp.items()},
                    "Ci": r3(Ci), "Si": round(Ci * inner, 4), "cost_usd": o["cost_usd"],
                    "weights": {c: r3(v) for c, v in w.items()},
                    "weighted": {c: r3(Ci * w[c] * comp[c]) for c in w}})
    out.sort(key=lambda r: (-r["Si"], -r["AP"], -r["OF"], r["cost_usd"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


# ====================================================================================================
# cell: c_schemas
# ====================================================================================================
#@title C · Structured outputs (schemas)
class Official(BaseModel):
    name: str
    department: str = ""
    completed: List[str] = []
    last_training: Optional[str] = Field(default=None, description="ISO date YYYY-MM-DD, or null if never trained")

    @field_validator("last_training")
    @classmethod
    def _iso_date(cls, v):                               # reject malformed dates with a clear message
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"last_training must be YYYY-MM-DD, got {v!r}")
        return v


class ActionItem(BaseModel):
    official: str
    status: Literal["COMPLIANT", "AT_RISK", "NON_COMPLIANT"]
    missing_courses: List[str] = []
    action: str
    priority: Literal["high", "medium", "low"]


class ComplianceReport(BaseModel):                  # the agent's final answer
    answer: str = Field(description="Direct answer to the user's question, max 80 words")
    compliance_rate_pct: float = Field(ge=0, le=100)
    action_items: List[ActionItem]
    ethics_notes: str = Field(description="Fairness / privacy / due-process remarks, max 60 words")


class AgentStep(BaseModel):                         # what the agent says on every turn
    thought: str = Field(description="One short sentence of reasoning")
    action: Literal["tool", "final"]
    tool: Optional[str] = None
    args: Dict[str, Any] = {}
    report: Optional[ComplianceReport] = None       # only when action == "final"


print("✓ Section C schemas defined")


# ====================================================================================================
# cell: c_tools
# ====================================================================================================
#@title C · The tools (plain Python rule engine)
ETH_REQUIRED = ["conflicts", "transparency", "procurement", "data_protection"]
ETH_REFRESH_DAYS = 365
ETH_WARNING_DAYS = 30
ETH_TODAY = None          # None = today's local date (long-running apps stay current); set a date to test a reference day


def eth_today():
    return ETH_TODAY or local_now().date()

ETH_ROSTER: List[Official] = []   # filled with load_eth_roster(); never rebound (tools read this list)


def _status_for(o: Official):
    missing = [c for c in ETH_REQUIRED if c not in o.completed]
    if o.last_training:
        days_since = (eth_today() - date.fromisoformat(o.last_training)).days
        days_until_due = ETH_REFRESH_DAYS - days_since
    else:
        days_since, days_until_due = None, None
    overdue = days_until_due is not None and days_until_due < 0
    if missing or overdue or o.last_training is None:
        status = "NON_COMPLIANT"
    elif days_until_due <= ETH_WARNING_DAYS:
        status = "AT_RISK"
    else:
        status = "COMPLIANT"
    return {"official": o.name, "department": o.department, "status": status, "missing_courses": missing,
            "days_since_training": days_since, "days_until_refresher_due": days_until_due}


def eth_list_officials():
    return [o.name for o in ETH_ROSTER]


def eth_check_official(name):
    for o in ETH_ROSTER:
        if o.name.lower() == str(name).strip().lower():
            return _status_for(o)
    return {"error": f"unknown official '{name}'", "known": eth_list_officials()}


def eth_check_all():
    return [_status_for(o) for o in ETH_ROSTER]


def eth_team_summary():
    rows = eth_check_all()
    counts = {s: sum(r["status"] == s for r in rows) for s in ["COMPLIANT", "AT_RISK", "NON_COMPLIANT"]}
    total = len(rows)
    # AT_RISK people are compliant (refresher due soon), so they count toward the compliance rate.
    rate = round(100 * (counts["COMPLIANT"] + counts["AT_RISK"]) / total, 1) if total else 0.0
    current = round(100 * counts["COMPLIANT"] / total, 1) if total else 0.0
    return {"total": total, **counts, "compliance_rate_pct": rate, "fully_current_pct": current}


ETH_TOOLS = {
    "list_officials": (eth_list_officials, "no args. Returns the names of all officials."),
    "check_official": (eth_check_official, 'args: {"name": "<official name>"}. Returns status, missing courses, days until refresher.'),
    "check_all":      (eth_check_all,      "no args. Returns status for every official (use this first for team questions)."),
    "team_summary":   (eth_team_summary,   "no args. Returns counts per status and the compliance rate."),
}


def load_eth_roster(rows):
    """Replace the roster contents in place."""
    new = [Official(**d) for d in rows]
    if not new:
        raise ValueError("The roster is empty: add at least one person.")
    ETH_ROSTER.clear()
    ETH_ROSTER.extend(new)
    return len(ETH_ROSTER)


print("✓ Section C tools ready:", ", ".join(ETH_TOOLS))


# ====================================================================================================
# cell: c_agent
# ====================================================================================================
#@title C · The single agent (think → act → observe loop) and guardrail
ETH_TRACE = []
ETH_MAX_STEPS = 8


class ComplianceAgent:
    def __init__(self):
        tool_docs = "\n".join(f"- {name}: {doc}" for name, (_, doc) in ETH_TOOLS.items())
        self.system = (
            "You are the Ethics Training Compliance Agent for a public-transit agency. The people on the roster review and "
            "approve service-disruption actions, so their ethics training must be current.\n"
            "You do NOT calculate dates or statuses yourself: you call tools and report what they return.\n\n"
            f"TOOLS:\n{tool_docs}\n\n"
            "Each turn reply with ONE JSON object:\n"
            '  {"thought": "...", "action": "tool", "tool": "<tool name>", "args": {...}}   to call a tool, or\n'
            '  {"thought": "...", "action": "final", "report": {...ComplianceReport...}}    when you have what you need.\n'
            "Rules: call check_all (or team_summary) before answering team-level questions; include EVERY official in action_items; "
            "priority high = NON_COMPLIANT, medium = AT_RISK, low = COMPLIANT; keep tone neutral and fair (same standards for everyone); "
            "never speculate about reasons for non-compliance. Copy status and missing_courses EXACTLY from the tool output.\n"
            "Example of a final turn:\n"
            '{"thought": "done", "action": "final", "report": {"answer": "...", "compliance_rate_pct": 40.0, '
            '"action_items": [{"official": "A", "status": "NON_COMPLIANT", "missing_courses": ["procurement"], '
            '"action": "Complete procurement course", "priority": "high"}], "ethics_notes": "..."}}'
        )

    def run(self, question, verbose=True):
        if not ETH_ROSTER:                                   # never answer about an empty roster
            raise ValueError("The roster is empty — load it with load_eth_roster() first.")
        ETH_TRACE.clear()
        seen_calls = set()
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": f"Question: {question}\nToday's date: {eth_today().isoformat()}"}]
        for step in range(1, ETH_MAX_STEPS + 1):
            t0 = time.time()
            raw = call_llm(messages, schema=AgentStep)
            try:
                s = AgentStep.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                if verbose:
                    print(f"Step {step}: ⚠ invalid output, asking the agent to fix it")
                ETH_TRACE.append({"step": step, "type": "invalid", "detail": str(e)[:80], "seconds": round(time.time() - t0, 1)})
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"Invalid output: {str(e)[:400]}. Reply with corrected JSON only."}]
                continue
            if s.action == "tool":
                fn = ETH_TOOLS.get(s.tool or "")
                if fn is None:
                    result = {"error": f"unknown tool '{s.tool}'", "available": list(ETH_TOOLS)}
                elif (s.tool, json.dumps(s.args, sort_keys=True)) in seen_calls:
                    # a repeated call adds nothing: point the model back to the result it already has
                    result = {"note": f"You already called {s.tool} with these arguments; its result is above. Do not call "
                                      "tools again. Reply now with action \"final\" and the full report."}
                else:
                    seen_calls.add((s.tool, json.dumps(s.args, sort_keys=True)))
                    try:
                        result = fn[0](**s.args)
                    except TypeError as e:
                        result = {"error": f"bad arguments: {e}"}
                if verbose:
                    print(f"Step {step}: 🧠 {s.thought}\n         🔧 {s.tool}({json.dumps(s.args)}) ✓")
                ETH_TRACE.append({"step": step, "type": "tool", "detail": f"{s.tool}({json.dumps(s.args)})", "seconds": round(time.time() - t0, 1)})
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": "TOOL RESULT:\n" + json.dumps(result)}]
            elif s.report is not None:
                ETH_TRACE.append({"step": step, "type": "final", "detail": "ComplianceReport", "seconds": round(time.time() - t0, 1)})
                return s.report
            else:
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": 'action "final" needs a "report" object. Try again.'}]
        raise RuntimeError(f"Agent did not finish within {ETH_MAX_STEPS} steps. Try a stronger model.")


def verify_report(report):
    """GUARDRAIL: compare the LLM's report with the deterministic rule engine."""
    truth = {r["official"].lower(): r for r in eth_check_all()}
    issues = []
    if not truth:
        issues.append("the roster is empty — nothing can be verified")
    seen = set()
    for item in report.action_items:
        t = truth.get(item.official.lower())
        if t is None:
            issues.append(f"'{item.official}' is not on the roster (possible hallucination)")
            continue
        seen.add(item.official.lower())
        if item.status != t["status"]:
            issues.append(f"{item.official}: report says {item.status}, rules say {t['status']}")
        if set(item.missing_courses) != set(t["missing_courses"]):
            issues.append(f"{item.official}: missing courses differ (report {item.missing_courses} vs rules {t['missing_courses']})")
    for name in truth.keys() - seen:
        issues.append(f"{truth[name]['official']} is missing from the report")
    true_rate = eth_team_summary()["compliance_rate_pct"]
    if abs(report.compliance_rate_pct - true_rate) > 1:
        issues.append(f"compliance rate: report {report.compliance_rate_pct}% vs rules {true_rate}%")
    return issues


def reconcile_report(report):
    """GUARDRAIL: the rule engine owns the facts (status, missing courses, priority, rate). Overwrite them; keep the LLM's wording."""
    truth = {r["official"].lower(): r for r in eth_check_all()}
    prio = {"NON_COMPLIANT": "high", "AT_RISK": "medium", "COMPLIANT": "low"}
    fixed, seen = [], set()
    for item in report.action_items:
        t = truth.get(item.official.lower())
        if t is None:
            continue                                   # drop officials the LLM made up
        seen.add(item.official.lower())
        fixed.append(item.model_copy(update={"status": t["status"], "missing_courses": t["missing_courses"],
                                             "priority": prio[t["status"]]}))
    for name, t in truth.items():                      # add officials the LLM forgot
        if name not in seen:
            fixed.append(ActionItem(official=t["official"], status=t["status"], missing_courses=t["missing_courses"],
                                    action="(added by guardrail) see rule-engine status", priority=prio[t["status"]]))
    s = eth_team_summary()
    answer, notes = report.answer, []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", report.answer):
        v = float(m.group(1))
        if abs(v - s["compliance_rate_pct"]) <= 1:
            continue
        if abs(v - s["fully_current_pct"]) <= 1:
            notes.append(f"{m.group(0)} is the fully-current share (COMPLIANT only), not the compliance rate")
        else:
            notes.append(f"{m.group(0)} is not supported by the rule engine")
    verified = (f"[Rule engine] Compliance rate {s['compliance_rate_pct']}% (COMPLIANT + AT_RISK, {s['COMPLIANT'] + s['AT_RISK']} of "
                f"{s['total']}); fully current {s['fully_current_pct']}%; NON_COMPLIANT {s['NON_COMPLIANT']}.")
    if notes:
        answer = verified + " [Guardrail correction: " + "; ".join(notes) + ".] LLM explanation: " + report.answer
    return report.model_copy(update={"action_items": fixed, "compliance_rate_pct": s["compliance_rate_pct"], "answer": answer})


eth_agent = ComplianceAgent()
ETH_WORKFLOW = [("in", "Roster +\nQuestion", "input"), ("agent", "Compliance\nAgent (think)", "llm"),
                ("tools", "Tools\n(act / observe)", "tool"), ("final", "Structured\nReport", "llm"),
                ("guard", "Guardrail\nverify", "tool"), ("out", "Verified\nOutput", "output")]
print("✓ Section C agent + guardrail ready")


# ====================================================================================================
# cell: c_roster
# ====================================================================================================
#@title C · Reviewer roster: your real reviewers (loaded from a file or secret; empty until you add people)
REVIEWER_ROSTER_PATH = "reviewer_roster.json"  #@param {type:"string"}
REVIEWER_ROSTER_PATH = os.environ.get("REVIEWER_ROSTER_PATH", REVIEWER_ROSTER_PATH)
ETH_DEFAULT_QUESTION = "Give me the compliance status of the team and tell me who needs action first and why."
ETH_ROSTER_SOURCE = {"source": "none"}
# Format: a JSON list of {"name", "department", "completed": [course ids from ETH_REQUIRED], "last_training": "YYYY-MM-DD"}.
# The roster is the agency's training record. It is read from (first found): the REVIEWER_ROSTER_JSON secret
# (the JSON text itself), or the file at REVIEWER_ROSTER_PATH. It can also be pasted in the dashboards.
ROSTER_TEMPLATE = [{"name": "", "department": "", "completed": list(ETH_REQUIRED), "last_training": "YYYY-MM-DD"}]


def load_roster_from_config():
    """Load the reviewer roster from the secret or file, if either exists. Returns the number of people loaded."""
    text, where = read_secret("REVIEWER_ROSTER_JSON"), "REVIEWER_ROSTER_JSON secret"
    if not text and os.path.exists(REVIEWER_ROSTER_PATH):
        text, where = open(REVIEWER_ROSTER_PATH).read(), REVIEWER_ROSTER_PATH
    if not text:
        ETH_ROSTER_SOURCE["source"] = "none (add reviewers: governance gate notes every reviewer as 'not on the roster')"
        return 0
    n = load_eth_roster(json.loads(text))
    ETH_ROSTER_SOURCE["source"] = where
    return n


def save_roster(path=None):
    path = path or REVIEWER_ROSTER_PATH
    with open(path, "w") as f:
        json.dump([o.model_dump() for o in ETH_ROSTER], f, indent=1)
    return path


try:
    load_roster_from_config()
except (ValueError, ValidationError) as e:
    ETH_ROSTER_SOURCE["source"] = f"roster could not be read: {e}"


# ====================================================================================================
# cell: d_schemas
# ====================================================================================================
#@title D · Structured outputs (the "contract" between agents)
HOLDING_KINDS = ["direct_stock", "business_ownership", "family_business", "donation_received",
                 "blind_trust", "index_fund", "other"]


class Holding(BaseModel):
    entity: str = Field(description="Company / organisation name")
    kind: Literal["direct_stock", "business_ownership", "family_business", "donation_received",
                  "blind_trust", "index_fund", "other"]
    owner: Literal["official", "spouse", "family", "donor"] = "official"
    value_usd: Optional[float] = None
    note: str = ""


class Decision(BaseModel):
    bill: str = Field(description="Name of the bill / vote / decision")
    subject: str = Field(description="What it is about, in a few words")
    action: str = Field(default="", description="e.g. voted yes, signed contract, chaired hearing")


class ExtractedFacts(BaseModel):                       # Agent 1 output
    official_name: str
    role: str = ""
    holdings: List[Holding] = []
    decisions: List[Decision] = []
    relationships: List[str] = []


class Flag(BaseModel):
    type: Literal["financial_interest", "family_interest", "donor_influence", "other"]
    severity: Literal["low", "medium", "high"]
    entity: str
    decision: str
    reasoning: str
    evidence: List[str] = Field(default=[], description="Short quotes / facts from the case file")


class AnalystOutput(BaseModel):                        # Agent 3 output
    flags: List[Flag] = []
    overall_risk_score: float = Field(ge=0, le=1)
    dismissed_rule_hits: List[str] = Field(default=[], description="Rule-screener hits judged not to be conflicts, with reason")


class ReviewerOutput(BaseModel):                       # Agent 4 output
    verdict_on_analysis: Literal["agree", "partially_agree", "disagree"]
    innocent_explanations: List[str] = Field(default=[], description="Hypothetical explanations, each starting with 'Hypothetical:'")
    mitigating_factors: List[str] = Field(default=[], description="ONLY facts stated in the case file (e.g. recusal, blind trust, competitive bidding). Empty list if none. Never speculate.")
    questions_for_official: List[str] = []


class FinalReport(BaseModel):                          # Agent 5 output
    official: str
    recommendation: Literal["CLEAR", "MONITOR", "REVIEW", "ESCALATE"]
    risk_score: float = Field(ge=0, le=1)
    summary: str = Field(description="2-3 sentences (40-80 words): who, which holding, which decision, why it matters")
    key_flags: List[str] = []
    required_actions: List[str] = []
    due_process_note: str


print("✓ Section D schemas defined:", ", ".join(c.__name__ for c in [ExtractedFacts, AnalystOutput, ReviewerOutput, FinalReport]))


# ====================================================================================================
# cell: d_agents
# ====================================================================================================
#@title D · The agents, the rule-screener tool, and the orchestrator
COI_TRACE = []
COI_TOTAL_STEPS = 5


def coi_traced(name, kind, fn, verbose=True):
    """Run one workflow step, print progress, record timing."""
    i = len(COI_TRACE) + 1
    if verbose:
        print(f"▶ Step {i}/{COI_TOTAL_STEPS}  {name:<22} [{kind}] ...", end=" ", flush=True)
    t0 = time.time()
    try:
        out = fn()
    except Exception:
        COI_TRACE.append({"step": i, "name": name, "kind": kind, "seconds": round(time.time() - t0, 1), "status": "failed"})
        if verbose:
            print("✗ failed")
        raise
    COI_TRACE.append({"step": i, "name": name, "kind": kind, "seconds": round(time.time() - t0, 1), "status": "ok"})
    if verbose:
        print(f"✓ {COI_TRACE[-1]['seconds']}s")
    return out


class COIAgent:                                         # renamed from `Agent` (clashed with Section B)
    def __init__(self, name, system, schema):
        self.name, self.system, self.schema = name, system, schema

    def run(self, user_message, verbose=True):
        def _call():
            obj, retries = run_structured(self.schema, self.system, user_message)
            return obj
        return coi_traced(self.name, "LLM agent", _call, verbose)


coi_extractor = COIAgent("Extractor Agent",
    "You are the Extractor Agent in a government-ethics review pipeline. Read the case file and extract ONLY "
    "facts that are explicitly stated: the official, their holdings/relationships (own, spouse, family, donors), "
    "and the votes/decisions they made or will make. Do not infer or invent. Use kind 'blind_trust' for blind trusts, "
    "'index_fund' for broad diversified funds, 'family_business' for a relative's company, 'donation_received' for donors.",
    ExtractedFacts)

coi_analyst = COIAgent("Conflict Analyst Agent",
    "You are the Conflict Analyst Agent. You receive extracted facts and the output of a deterministic rule screener. "
    "Decide which potential conflicts are REAL. For each flag give type, severity, the entity, the decision, your reasoning, "
    "and evidence taken from the facts. Severity guide: high = direct or family financial stake in the decision; "
    "medium = donor influence or sector-level exposure; low = remote exposure (blind trust, diversified fund). "
    "Put rule hits you judge to be false alarms in dismissed_rule_hits with a reason. If there is no conflict, return an empty flags list. "
    "overall_risk_score is 0 (none) to 1 (severe).",
    AnalystOutput)

coi_reviewer = COIAgent("Reviewer Agent",
    "You are the Reviewer Agent, a devil's advocate protecting due process. Challenge the analyst's flags. Keep FACTS and HYPOTHESES separate: "
    "mitigating_factors may contain ONLY things stated in the extracted facts (e.g. recusal, blind trust, competitive bidding, diversified fund); "
    "if the case states none, return an empty list. Never invent facts. Speculative alternatives go in innocent_explanations and must start with "
    "'Hypothetical:'. Write concrete questions the official should be allowed to answer. Say whether you agree, partially agree, or disagree "
    "with the analysis. Be fair, not lenient.",
    ReviewerOutput)

coi_decider = COIAgent("Decision Agent",
    "You are the Decision Agent. Combine the facts, rule screen, analysis and review into a final report. Recommendation guide: "
    "CLEAR = no material flags; MONITOR = only low-severity flags; REVIEW = medium/high flags with a stated mitigation; "
    "ESCALATE = high-severity flag with no stated mitigation. Only the reviewer's mitigating_factors count as mitigation; hypothetical explanations "
    "and unanswered questions do NOT lower the recommendation. The summary must be 2-3 sentences (40-80 words) naming the official, the holding, "
    "the decision and why it matters. required_actions must be concrete (e.g. 'request recusal statement'). due_process_note must state that the "
    "official may respond before any action and that a human decides.",
    FinalReport)


# ---------- the deterministic tool ----------
COI_SECTORS = {
    "technology": ["tech", "software", "semiconductor", "cloud", "cyber", "antitrust", "artificial intelligence", "data privacy"],
    "healthcare_pharma": ["pharma", "drug", "health", "medical", "biotech", "prescription", "medicare"],
    "energy": ["oil", "gas", "energy", "solar", "pipeline", "coal", "utility", "drilling"],
    "defense": ["defense", "weapon", "aerospace", "military"],           # "procurement" removed (sector-neutral)
    "finance": ["bank", "financ", "insur", "lending", "crypto"],
    # transportation / construction was missing, so transit vendors and road contracts were never screened
    "transportation_infrastructure": ["transit", "bus", "coach", "rail", "paratransit", "shuttle", "fleet", "depot",
                                      "highway", "road", "paving", "construction", "detour", "transportation"],
}
COI_WEIGHTS = {"direct_stock": 1.0, "business_ownership": 1.0, "family_business": 0.8, "donation_received": 0.7,
               "blind_trust": 0.4, "index_fund": 0.2, "other": 0.5}
COI_THRESHOLD = 0.7

_LEGAL_SUFFIX = r"\b(incorporated|inc|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|group|holdings)\b\.?"
_GENERIC_WORDS = {"construction", "services", "service", "sons", "and", "the", "of", "international", "systems",
                  "solutions", "partners", "company", "group", "transport", "transit", "coach", "&"}


def norm_entity(text):
    """normalize names so 'TechCorp Inc.' matches "TechCorp's cloud division"."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"'s\b", "", t)
    t = re.sub(_LEGAL_SUFFIX, " ", t)
    t = re.sub(r"[^a-z0-9& ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def name_matches(entity, text):
    e, t = norm_entity(entity), " " + norm_entity(text) + " "
    if e and f" {e} " in t:
        return True
    core = " ".join(w for w in e.split() if w not in _GENERIC_WORDS)    # e.g. "nair" from "Nair & Sons Construction LLC"
    return len(core) >= 3 and f" {core} " in t


def sectors_of(text):
    t = text.lower()
    return {s for s, kws in COI_SECTORS.items() if any(re.search(r"\b" + re.escape(k), t) for k in kws)}


def rule_screen(facts, decisions=None, threshold=None):
    """Deterministic screen: match each holding against each decision by name or by sector.
    INTEGRATION: `decisions` can be supplied (the options a reviewer is approving) instead of facts.decisions."""
    threshold = COI_THRESHOLD if threshold is None else threshold
    hits = []
    for h in facts.holdings:
        for d in (decisions if decisions is not None else facts.decisions):
            decision_text = f"{d.bill} {d.subject} {d.action}"
            name_match = name_matches(h.entity, decision_text)
            shared = sectors_of(f"{h.entity} {h.note}") & sectors_of(decision_text)
            if name_match or shared:
                score = round(COI_WEIGHTS[h.kind] * (1.0 if name_match else 0.75), 2)
                hits.append({"holding": h.entity, "kind": h.kind, "owner": h.owner, "decision": d.bill,
                             "match": "name" if name_match else f"sector:{'/'.join(sorted(shared))}", "score": score})
    top = max([x["score"] for x in hits], default=0.0)
    return {"hits": hits, "rule_score": top, "threshold": threshold, "over_threshold": top >= threshold}


# ---------- the orchestrator ----------
def coi_run_pipeline(case_text, verbose=True):
    COI_TRACE.clear()
    facts = coi_extractor.run(f"CASE FILE:\n{case_text}", verbose)
    screen = coi_traced("Rule Screener", "python tool", lambda: rule_screen(facts), verbose)
    analysis = coi_analyst.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                               "\n\nRULE SCREENER OUTPUT:\n" + json.dumps(screen, indent=1), verbose)
    review = coi_reviewer.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                              "\n\nANALYST OUTPUT:\n" + analysis.model_dump_json(indent=1), verbose)
    final = coi_decider.run("EXTRACTED FACTS:\n" + facts.model_dump_json(indent=1) +
                            "\n\nRULE SCREEN:\n" + json.dumps(screen) +
                            "\n\nANALYST:\n" + analysis.model_dump_json() +
                            "\n\nREVIEWER:\n" + review.model_dump_json(), verbose)
    return {"facts": facts, "screen": screen, "analysis": analysis, "review": review, "final": final}


def coi_guardrail(result):
    """Consistency check on the final recommendation (two consistency checks + recommendation check)."""
    final, analysis, review, screen = result["final"], result["analysis"], result["review"], result["screen"]
    issues = []
    if any(f.severity == "high" for f in analysis.flags) and not review.mitigating_factors and final.recommendation != "ESCALATE":
        issues.append("high-severity flag with no stated mitigation, but the recommendation is not ESCALATE")
    if final.recommendation == "CLEAR" and analysis.flags:
        issues.append("recommendation is CLEAR although the analyst raised flags")
    if final.recommendation == "CLEAR" and screen["over_threshold"]:
        issues.append(f"recommendation is CLEAR although the deterministic rule score {screen['rule_score']} ≥ threshold {screen['threshold']}")
    return issues


def coi_enforce(result):
    """GUARDRAIL: when the consistency check fails, the stricter recommendation wins (a person still decides)."""
    issues, final = coi_guardrail(result), result["final"]
    rule = float(result["screen"].get("rule_score") or 0)
    if final.risk_score < rule:                     # the stated risk is never below the deterministic rule score
        final = final.model_copy(update={"risk_score": rule})
        issues = issues + [f"risk raised to the rule screener's {rule}"]
    if not any("ESCALATE" in i or "CLEAR" in i for i in issues):
        return final, issues
    new = "ESCALATE" if any("ESCALATE" in i for i in issues) else "REVIEW"
    return final.model_copy(update={"recommendation": new,
                                    "summary": final.summary + f" [Guardrail: changed {final.recommendation} → {new}.]"}), issues


COI_WORKFLOW = [("case", "Case file", "input"), ("Extractor Agent", "1 Extractor", "llm"),
                ("Rule Screener", "2 Rule\nScreener", "tool"), ("Conflict Analyst Agent", "3 Conflict\nAnalyst", "llm"),
                ("Reviewer Agent", "4 Reviewer", "llm"), ("Decision Agent", "5 Decision", "llm"),
                ("report", "Report", "output")]
print("✓ Section D: 4 agents + 1 tool + orchestrator ready")


# ====================================================================================================
# cell: d_disclosures
# ====================================================================================================
#@title D · Conflict-of-interest disclosures of your real reviewers (loaded from a file or secret; empty until added)
STAFF_DISCLOSURES_PATH = "staff_disclosures.json"  #@param {type:"string"}
STAFF_DISCLOSURES_PATH = os.environ.get("STAFF_DISCLOSURES_PATH", STAFF_DISCLOSURES_PATH)
STAFF_DISCLOSURES = {}
STAFF_DISCLOSURES_SOURCE = {"source": "none"}
# Format: {"<person name>": {"text": "<disclosure as filed>", "facts": <ExtractedFacts JSON>}}. "facts" is what the
# Section D Extractor Agent produces from the text; if it is missing it is extracted with the LLM when loaded.
DISCLOSURE_TEMPLATE = {"<name as on the roster>": {"text": "<the disclosure as filed>", "facts": {
    "official_name": "<name>", "role": "<role>", "holdings": [{"entity": "<company>", "kind": "family_business",
                                                               "owner": "spouse", "note": ""}], "relationships": []}}}


def load_disclosures(data):
    """Replace the disclosures in place. data: dict as in DISCLOSURE_TEMPLATE. Returns the number loaded."""
    new = {}
    for name, d in data.items():
        facts = d.get("facts")
        if facts is None:
            if not LLM_READY:
                raise ValueError(f"{name}: no 'facts' given and no LLM to extract them from the text")
            facts = coi_extractor.run(f"CASE FILE:\n{d['text']}", verbose=False)
        else:
            facts = ExtractedFacts(**facts) if isinstance(facts, dict) else facts
        new[name.strip()] = {"text": d.get("text", ""), "facts": facts}
    STAFF_DISCLOSURES.clear()
    STAFF_DISCLOSURES.update(new)
    return len(STAFF_DISCLOSURES)


def load_disclosures_from_config():
    text, where = read_secret("STAFF_DISCLOSURES_JSON"), "STAFF_DISCLOSURES_JSON secret"
    if not text and os.path.exists(STAFF_DISCLOSURES_PATH):
        text, where = open(STAFF_DISCLOSURES_PATH).read(), STAFF_DISCLOSURES_PATH
    if not text:
        STAFF_DISCLOSURES_SOURCE["source"] = "none (the conflict gate notes 'no disclosure on file')"
        return 0
    n = load_disclosures(json.loads(text))
    STAFF_DISCLOSURES_SOURCE["source"] = where
    return n


def disclosures_as_json():
    return {n: {"text": d["text"], "facts": d["facts"].model_dump()} for n, d in STAFF_DISCLOSURES.items()}


try:
    load_disclosures_from_config()
except (ValueError, ValidationError, RuntimeError) as e:
    STAFF_DISCLOSURES_SOURCE["source"] = f"disclosures could not be read: {e}"


# ====================================================================================================
# cell: gov
# ====================================================================================================
#@title INTEGRATION · governance_check(): ethics-training gate (C) + conflict-of-interest gate (D)
def decisions_for_option(option):
    """Turn an option under review into Section D `Decision`s (its outside parties are what gets screened)."""
    parties = ", ".join(option.get("parties", []))
    return [Decision(bill=option["label"], subject=f"{option['type'].replace('_', ' ')} for transit bus service; parties: {parties}",
                     action="approve operational action")]


def decisions_for_proposal(proposal):
    changes = "; ".join(f"{c['path']} {c['old']}→{c['new']}" for c in proposal["changes"])
    return [Decision(bill=proposal["title"], subject=f"transit scoring policy change: {changes}", action="approve rule change")]


def governance_check(name, decision, option=None, proposal=None, policy=None):
    """Return (problems, notes). Problems block the submission; notes are logged in the audit record."""
    policy = policy or APP.PM.current
    g = policy.get("governance_gates", {})
    problems, notes = [], []
    if not g.get("enabled", True) or not name.strip():
        return problems, notes
    # ---- Section C: ethics training
    st = eth_check_official(name)
    if "error" in st:
        msg = f"{name} is not on the ethics-training roster"
        (problems if g.get("unknown_person") == "block" else notes).append(msg)
    elif st["status"] in g.get("training_block_statuses", []) and decision in g.get("training_gate_decisions", []):
        why = f"missing {', '.join(st['missing_courses'])}" if st["missing_courses"] else (
            "refresher overdue" if st["days_until_refresher_due"] is not None else "never trained")
        problems.append(f"{name} is NON_COMPLIANT on ethics training ({why}) and cannot {decision.lower()} — another reviewer must act")
    elif st["status"] == "AT_RISK":
        notes.append(f"{name}: ethics refresher due in {st['days_until_refresher_due']} days")
    else:
        notes.append(f"{name}: ethics training {st['status']}")
    # ---- Section D: conflict of interest
    if decision in g.get("coi_gate_decisions", []) and (option or proposal):
        disc = STAFF_DISCLOSURES.get(name.strip())
        if disc is None:
            notes.append(f"{name}: no conflict-of-interest disclosure on file")
        else:
            decs = decisions_for_option(option) if option else decisions_for_proposal(proposal)
            screen = rule_screen(disc["facts"], decs, g.get("coi_threshold", COI_THRESHOLD))
            name_hits = [h for h in screen["hits"] if h["match"] == "name" and h["score"] >= screen["threshold"]]
            if name_hits:
                h = name_hits[0]
                problems.append(f"possible conflict of interest: {name}'s {h['owner']} holding '{h['holding']}' ({h['kind']}) is a party "
                                f"to this option (score {h['score']} ≥ {screen['threshold']}) — recuse; another reviewer must approve")
            elif screen["hits"]:
                notes.append(f"{name}: conflict screen hits below the recusal rule — " +
                             "; ".join(f"{h['holding']} ({h['match']}, {h['score']})" for h in screen["hits"]))
            else:
                notes.append(f"{name}: conflict screen clear")
    return problems, notes


print("✓ Governance gate ready (training: Section C rules · independence: Section D rule screener)")


# ====================================================================================================
# cell: e3_core
# ====================================================================================================
#@title E3 · Shared case state, Explanation Agent (live LLM), human review, deadlines, safe default, implementation
REVIEW_STATUSES = ("AWAITING_REVIEW", "AWAITING_SECOND_REVIEW")
STATUS_HELP = {
    "MONITORING": "Continue monitoring (unresolved or low-confidence data)",
    "CONFLICT_REVIEW": "Conflicting data quarantined — operator review needed",
    "SAFE_DEFAULT": "Safe default: verified rider information only, keep monitoring",
    "AWAITING_REVIEW": "Waiting for human reviewer",
    "AWAITING_SECOND_REVIEW": "High impact — waiting for independent second reviewer",
    "AWAITING_IMPLEMENTATION": "Approved — operations staff to carry out the action outside the prototype",
    "IMPLEMENTED": "Action implemented — ready for outcome evaluation",
    "EVALUATED": "Outcome evaluated",
    "CLOSED_FALSE_ALARM": "Closed as false alarm",
}


def agent_llm_generate(prompt, system, max_tokens=320, timeout=None):
    """Agent text through the shared live LLM client (OpenAI → OpenRouter → Hugging Face → Ollama if enabled)."""
    try:
        return llm_generate(prompt, system, max_tokens=max_tokens)
    except LLMUnavailable as e:
        raise AgentFailure(f"LLM call failed: {e}")


def rider_alert_text(case):
    """Deterministic rider message built from verified evidence only."""
    A, raw = case["analysis"], case["raw"]
    f = A.get("fields", {})
    acc = [e for e in A.get("evidence", []) if e["status"] == "accepted" and not e["source_id"].startswith("VISION-")]
    if not acc:
        return f"Route {raw['route_name']}: we are checking reports of service problems. Updates will follow."
    src = min(acc, key=lambda e: precedence_rank(get_case_policy(case), e["source_type"]))
    stops = raw.get("stops") or []
    parts = [f"Route {raw['route_name']}"]
    if f.get("cancelled"):
        parts.append("a scheduled trip is cancelled")
    elif f.get("delay_min") is not None:
        parts.append(f"buses are running about {f['delay_min']:.0f} min late")
    if f.get("event_severity", "none") != "none" and raw.get("event_type", "none") != "none":
        parts.append(f"due to {raw.get('event_type').replace('_', ' ')}")
    msg = ": ".join([parts[0], ", ".join(parts[1:]) or "service disruption"])
    if len(stops) > 1:
        msg += f" between {stops[0]['name']} and {stops[-1]['name']}"
    elif stops:
        msg += f" near {stops[0]['name']}"
    msg += "."
    nearby = raw.get("connections", {}).get("connecting_routes") or []
    if nearby:
        net = network_or_none()
        msg += " Alternatives nearby: route(s) " + ", ".join((net.route_label(r) if net else r) for r in nearby[:6]) + "."
    msg += f" Verified by {src['source_id']} at {local_label(parse_iso(src['timestamp']))[11:]}. Accessible-service riders can call the paratransit desk."
    return msg


class CaseStore:
    """Single shared case record that every stage reads and writes (thread-safe; shared by all reviewers via the Gradio link)."""

    def __init__(self, policy_manager):
        self.PM = policy_manager
        self.cases, self.history, self.audit = {}, [], []
        self.active_id = None
        self.lock = threading.RLock()
        self.counter = 0
        self.last_message = "Ready. Pick a scenario and press ▶ Run pipeline."
        self.proposals, self.proposal_log, self.backtest = [], [], None
        self.insight_narrative = ""

    # ---------- audit
    def log(self, case, actor, role, stage, event, detail=""):
        e = {"ts": iso(utcnow()), "case_id": case["case_id"] if case else "—", "actor": actor, "role": role, "stage": stage,
             "event": event, "detail": detail, "policy_version": case["policy_version"] if case else self.PM.current["version"]}
        self.audit.append(e)
        if case is not None:
            case["audit"].append(e)

    @property
    def active(self):
        return self.cases.get(self.active_id)

    # ---------- intake
    def new_case(self, raw, use_llm=True):
        with self.lock:
            self.counter += 1
            policy = self.PM.current
            case = {"case_id": f"CASE-{self.counter:04d}", "created_at": iso(utcnow()), "scenario_name": raw["scenario_name"],
                    "policy_version": policy["version"], "raw": raw, "status": None, "analysis": None,
                    "constraints": {}, "revisions": 0, "operator_checks": 0, "operator_resolution": None,
                    "review": None, "decisions": [], "implementation": [], "outcome": None, "alerts": [], "audit": [],
                    "explanation": None, "rider_alert": None, "use_llm": use_llm, "history": False}
            self.cases[case["case_id"]] = case
            self.active_id = case["case_id"]
            self.log(case, "Signal Scout Agent", "agent", "Intake", "Case created", raw["scenario_name"])
            self._analyze(case)
            return case

    def _analyze(self, case):
        policy = get_case_policy(case)
        A = run_analysis(case["raw"], policy, case["operator_resolution"], case["constraints"])
        case["analysis"] = A
        for t in A["trace"]:
            self.log(case, "pipeline", "agent/rule", t["stage"], t["result"], t["detail"])
        for a in A["alerts"]:
            case["alerts"].append({**a, "ts": iso(utcnow())})
            self.log(case, "Signal Scout Agent", "agent", "Alert", f"Alert → {a['to']}", a["message"])
        g = A["gate"]
        if g == "SAFE_MODE":
            self.safe_default(case, "Agent error path: deterministic safe mode")
        elif g == "BELOW_THRESHOLD":
            self.safe_default(case, "Below significance threshold")
        elif g == "MONITORING":
            case["status"] = "MONITORING"
            self.log(case, "Signal Scout Agent", "agent", "Monitoring", "Continue monitoring", A["gate_reason"])
        elif g == "CONFLICT_REVIEW":
            case["status"] = "CONFLICT_REVIEW"
        elif g == "SIGNIFICANT":
            self.log(case, "Signal Scout Agent", "agent", "Create case + log", "Significant case", A["gate_reason"])
            self.explain(case)
            self.start_review(case)
        self.last_message = f"{case['case_id']}: {STATUS_HELP.get(case['status'], case['status'])}"

    # ---------- exception-path human actions
    def operator_resolve(self, case, operator, resolution):
        if case["status"] != "CONFLICT_REVIEW":
            raise ValueError("This case has no conflict awaiting operator review.")
        if not operator.strip():
            raise ValueError("Enter the operator's name.")
        case["operator_checks"] += 1
        self.log(case, operator, "operator", "Operator review", resolution, f"check {case['operator_checks']} of max {get_case_policy(case)['max_operator_checks']}")
        if resolution == "unresolved" and case["operator_checks"] < get_case_policy(case)["max_operator_checks"]:
            self.last_message = "Conflict still unresolved; one more operator check allowed before monitoring."
            return
        case["operator_resolution"] = "accept_precedence" if resolution == "resolved" else "unresolved"
        self._analyze(case)

    def monitor_action(self, case, action="recheck", actor="planner"):
        """Recheck a monitoring / safe-default case against a fresh pull of the live feeds (same vehicle)."""
        if case["status"] not in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW"):
            raise ValueError("Only monitoring / safe-default / conflict cases can be rechecked.")
        refresh_live_context(census=False)
        vid = case["raw"].get("vehicle_id")
        try:
            raw = raw_for_vehicle(vid, store=self, name=case["raw"]["scenario_name"])
        except LiveDataUnavailable as e:
            self.log(case, actor, "planner", "Monitoring", "Recheck: vehicle no longer in the live feed", str(e))
            self.last_message = f"{case['case_id']}: {e}"
            return
        case["raw"], case["operator_resolution"] = raw, None
        case["policy_version"] = self.PM.current["version"]
        self.log(case, actor, "planner", "Monitoring", "Recheck", "Live feeds pulled again and re-verified")
        self._analyze(case)

    def close_false_alarm(self, case, actor, reason):
        if case["status"] not in ("MONITORING", "SAFE_DEFAULT"):
            raise ValueError("Only monitoring or safe-default cases can be closed as false alarms.")
        if len(reason.strip()) < 5:
            raise ValueError("Enter a reason for closing as false alarm.")
        case["status"] = "CLOSED_FALSE_ALARM"
        case["outcome"] = {"evaluated_at": iso(utcnow()), "false_alarm": True, "note": reason, "action_type": "none"}
        self.log(case, actor, "planner", "Monitoring", "Closed as false alarm", reason)
        self.last_message = f"{case['case_id']} closed as false alarm."

    # ---------- explanation agent
    def explain(self, case):
        case["explanation"] = {"brief_md": explanation_brief(case), "narrative": "", "narrative_source": "template",
                               "generated_at": iso(utcnow())}
        case["rider_alert"] = rider_alert_text(case)
        if case.get("use_llm") and LLM_READY:
            sc = SignalScout()
            try:
                txt = sc.invoke("explanation_agent", llm_narrative, case, retry=True)
                case["explanation"].update(narrative=txt, narrative_source=f"live LLM · {llm_source()}")
            except AgentFailure as e:
                case["explanation"]["narrative_source"] = "template (LLM narrative failed twice; template used)"
                self.log(case, "Explanation Agent", "agent", "Agent error path", "Fallback to template", str(e))
                sc.alert("technical owner", f"Explanation Agent failed twice: {e}")
            for x in sc.exceptions:
                self.log(case, "Explanation Agent", "agent", "Agent error path", x["outcome"], x["detail"])
        if not case["explanation"]["narrative"]:
            case["explanation"]["narrative"] = template_narrative(case)
        self.log(case, "Explanation Agent", "agent", "Explanation", "Options ranking brief published",
                 case["explanation"]["narrative_source"])

    # ---------- human review
    def start_review(self, case, note="Options presented to reviewer"):
        policy = get_case_policy(case)
        tier = case["analysis"]["tier"] or "moderate"
        mins = policy["deadlines_min"][tier]
        now = utcnow()
        case["review"] = {"tier": tier, "deadline_min": mins, "started_at": iso(now),
                          "deadline_at": iso(now + timedelta(seconds=mins * 60 * DEMO_DEADLINE_SCALE)),
                          "first": None, "second": None, "required_second": None,
                          "no_feasible_option": not case["analysis"]["ranking"]}
        case["status"] = "AWAITING_REVIEW"
        self.log(case, "Deadline control", "rule", "Human review", f"Timer started ({tier}: {mins} min, covers both reviewers)", note)

    def seconds_left(self, case):
        if not case or case["status"] not in REVIEW_STATUSES or not case.get("review"):
            return None
        return (parse_iso(case["review"]["deadline_at"]) - utcnow()).total_seconds()

    def enforce_deadline(self, case):
        left = self.seconds_left(case)
        if left is not None and left <= 0:
            self.safe_default(case, "Deadline expired with no complete response")
            return True
        return False

    def _validate_submission(self, case, name, reason, checklist, viewed, decision=None, option_id=None):
        p = get_case_policy(case)["review"]
        errs = []
        # INTEGRATION (Sections C + D): ethics-training and conflict-of-interest gates
        if decision and name.strip():
            opt = next((o for o in case["analysis"].get("options", []) if o["option_id"] == option_id), None)
            gp, gn = governance_check(name, decision, option=opt, policy=get_case_policy(case))
            errs += gp
            for n_ in gn:
                self.log(case, "Governance gate", "rule", "Reviewer gate (C+D)", "note", n_)
        if not name.strip():
            errs.append("reviewer name required")
        if len(reason.strip()) < p["min_reason_chars"]:
            errs.append(f"typed reason of at least {p['min_reason_chars']} characters required")
        if set(checklist or []) != set(REVIEW_CHECKLIST):
            errs.append("every checklist item must be ticked")
        if not viewed:
            errs.append("confirm you viewed the alternatives (rubber-stamp protection)")
        return errs

    def is_high_impact(self, case, option_id):
        p = get_case_policy(case)["high_impact"]
        o = next((o for o in case["analysis"]["options"] if o["option_id"] == option_id), None)
        P = case["analysis"]["scores"]["P"]
        return bool(o and (o["type"] in p["actions"] or o["cost_usd"] >= p["cost_min"] or P >= p["P_min"]))

    def submit_first(self, case, name, decision, option_id, reason, checklist, viewed, revision=None):
        if case["status"] != "AWAITING_REVIEW":
            raise ValueError(f"Case is {case['status']}; not awaiting first review.")
        if self.enforce_deadline(case):
            raise ValueError("Deadline had already expired — case routed to safe default.")
        errs = self._validate_submission(case, name, reason, checklist, viewed, decision, option_id)
        if errs:
            self.log(case, name or "?", "reviewer", "Human review", "Submission blocked", "; ".join(errs))
            raise ValueError("Cannot submit: " + "; ".join(errs))
        now = utcnow()
        rt = (now - parse_iso(case["review"]["started_at"])).total_seconds()
        ranking = case["analysis"]["ranking"]
        rec = ranking[0]["option_id"] if ranking else None
        d = {"stage": "first review", "reviewer": name.strip(), "decision": decision, "option_id": option_id if decision == "Approve" else None,
             "recommended_option_id": rec, "override": decision == "Approve" and option_id != rec, "reason": reason.strip(),
             "checklist_complete": True, "viewed_alternatives": True, "response_s": round(rt, 1), "ts": iso(now),
             "deadline_at": case["review"]["deadline_at"], "tier": case["review"]["tier"], "revision_no": case["revisions"]}
        if decision == "Approve":
            if not ranking:
                raise ValueError("No feasible option: revise constraints or reject.")
            if option_id not in [r["option_id"] for r in ranking]:
                raise ValueError("Pick one of the ranked feasible options.")
            case["decisions"].append(d)
            case["review"]["first"] = d
            hi = self.is_high_impact(case, option_id)
            case["review"]["required_second"] = hi
            self.log(case, name, "reviewer", "Human review", "APPROVE" + (" (override of recommendation)" if d["override"] else ""),
                     f"{option_id}; {reason}")
            if hi:
                case["status"] = "AWAITING_SECOND_REVIEW"
                case["review"]["second_requested_at"] = iso(now)
                self.last_message = "High-impact action — sent to an independent second reviewer (same deadline)."
            else:
                self._approved(case, option_id)
        elif decision == "Reject":
            case["decisions"].append(d)
            self.log(case, name, "reviewer", "Human review", "REJECT", reason)
            self.safe_default(case, f"Rejected by {name}")
        elif decision == "Revise":
            pmax = get_case_policy(case)["max_revisions"]
            case["decisions"].append(d)
            if case["revisions"] >= pmax:
                self.log(case, name, "reviewer", "Revision loop", "Third revision request", "Max two revisions → safe default")
                self.safe_default(case, "Third revision request")
                return
            case["revisions"] += 1
            case["constraints"] = {**case["constraints"], **(revision or {})}
            self.log(case, name, "planner", "Revision loop", f"Revision {case['revisions']} of {pmax}", json.dumps(revision or {}))
            A = run_analysis(case["raw"], get_case_policy(case), case["operator_resolution"], case["constraints"])
            case["analysis"] = A
            self.log(case, "Scenario evaluator", "rule", "Revision loop", "Scenarios rebuilt and rescored with unchanged rules",
                     " > ".join(f"{r['label']} ({r['Si']:.3f})" for r in A["ranking"]) or "no feasible option")
            self.explain(case)
            case["status"] = "AWAITING_REVIEW"
            case["review"]["started_at"] = iso(utcnow())  # response time measured per round
            if get_case_policy(case).get("deadline_resets_on_revision"):   # policy flag
                mins = case["review"]["deadline_min"]
                case["review"]["deadline_at"] = iso(utcnow() + timedelta(seconds=mins * 60 * DEMO_DEADLINE_SCALE))
            case["review"]["no_feasible_option"] = not A["ranking"]
            self.last_message = f"Revision {case['revisions']} rescored. " + (
                "New deadline started." if get_case_policy(case).get("deadline_resets_on_revision") else "Deadline unchanged.")
        else:
            raise ValueError("Decision must be Approve, Revise or Reject.")

    def submit_second(self, case, name, decision, reason, checklist, viewed):
        if case["status"] != "AWAITING_SECOND_REVIEW":
            raise ValueError("Case is not awaiting a second reviewer.")
        if self.enforce_deadline(case):
            raise ValueError("Deadline had already expired — case routed to safe default.")
        errs = self._validate_submission(case, name, reason, checklist, viewed, decision, case["review"]["first"]["option_id"])
        if name.strip() and case["review"]["first"] and name.strip().lower() == case["review"]["first"]["reviewer"].lower():
            errs.append("second reviewer must be a different, independent person")
        if errs:
            self.log(case, name or "?", "reviewer", "Second reviewer", "Submission blocked", "; ".join(errs))
            raise ValueError("Cannot submit: " + "; ".join(errs))
        now = utcnow()
        rt = (now - parse_iso(case["review"]["second_requested_at"])).total_seconds()
        oid = case["review"]["first"]["option_id"]
        d = {"stage": "second review", "reviewer": name.strip(), "decision": decision, "option_id": oid if decision == "Approve" else None,
             "recommended_option_id": case["review"]["first"]["recommended_option_id"], "override": False, "reason": reason.strip(),
             "checklist_complete": True, "viewed_alternatives": True, "response_s": round(rt, 1), "ts": iso(now),
             "deadline_at": case["review"]["deadline_at"], "tier": case["review"]["tier"], "revision_no": case["revisions"]}
        case["decisions"].append(d)
        case["review"]["second"] = d
        self.log(case, name, "second reviewer", "Second reviewer", decision.upper(), reason)
        if decision == "Approve":
            self._approved(case, oid)
        else:
            self.safe_default(case, f"Rejected by second reviewer {name}")

    def _approved(self, case, option_id):
        case["approved_option_id"] = option_id
        case["status"] = "AWAITING_IMPLEMENTATION"
        self.log(case, "system", "rule", "Implement action", "Approved — handed to operations staff", option_id)
        self.last_message = f"Approved {option_id}. Operations staff carry it out outside the prototype and send rider alerts."

    def safe_default(self, case, reason):
        case["status"] = "SAFE_DEFAULT"
        case["approved_option_id"] = None
        case["rider_alert"] = rider_alert_text(case)
        case["safe_default_reason"] = reason
        self.log(case, "Safe default", "rule", "Safe default", reason,
                 "Verified rider information sent; monitoring continues; no dispatch, rerouting or live control")
        self.last_message = f"{case['case_id']} → SAFE DEFAULT ({reason})."

    # ---------- implementation
    def record_implementation(self, case, staff, result, notes):
        if case["status"] != "AWAITING_IMPLEMENTATION":
            raise ValueError("No approved action is awaiting implementation.")
        if not staff.strip():
            raise ValueError("Enter the operations staff name.")
        if result in ("Partial", "Failed") and not notes.strip():          # notes required so the reviewer can decide
            raise ValueError("Describe what went wrong so the reviewer can decide (notes required for partial or failed).")
        rec = {"ts": iso(utcnow()), "staff": staff, "result": result, "notes": notes, "option_id": case["approved_option_id"]}
        case["implementation"].append(rec)
        self.log(case, staff, "operations", "Implement action", result, notes)
        if result == "Completed":
            case["status"] = "IMPLEMENTED"
            self.last_message = "Implementation recorded. Run the outcome evaluator next."
        else:
            self.log(case, "system", "rule", "Implement action", "FAILED OR PARTIAL: return to review", result)
            self.start_review(case, "Returned to review after failed/partial implementation")
            self.last_message = "Failed/partial action returned to human review with a new timer."

    # ---------- outcome evaluator (separate from the recommender; never changes policy)
    def predicted_for(self, case):
        A = case["analysis"]
        oid = case.get("approved_option_id")
        if not oid and A.get("options"):
            oid = next((o["option_id"] for o in A["options"] if o["type"] == "rider_alerts"), None)
        o = next((o for o in A.get("options", []) if o["option_id"] == oid), None)
        if o is None:
            return {"option_id": "rider information only", "type": "rider_alerts", "delay_reduction_min": 0.0,
                    "access_protection": 0.1, "vulnerable_riders_helped": 0.0, "cost_usd": 50.0}
        return {"option_id": o["option_id"], "type": o["type"], "delay_reduction_min": o["pred_delay_reduction_min"],
                "access_protection": o["pred_access_protection"], "vulnerable_riders_helped": o["pred_vulnerable_riders_helped"],
                "cost_usd": o["cost_usd"]}

    def evaluate_outcome(self, case, obs, actor="Outcome Evaluator"):
        """obs: what operations staff observed after the incident (entered by a person; nothing is simulated)."""
        if case["status"] not in ("IMPLEMENTED", "SAFE_DEFAULT", "MONITORING", "EVALUATED"):
            raise ValueError("Evaluate after implementation, or for safe-default / monitoring cases.")
        missing = [k for k in OBSERVED_FIELDS if obs.get(k) is None]
        if missing:
            raise ValueError("Enter every observed value before evaluating: " + ", ".join(missing))
        case["outcome"] = outcome_metrics(self.predicted_for(case), obs)
        if case["status"] == "IMPLEMENTED":
            case["status"] = "EVALUATED"
        self.log(case, actor, "agent", "Outcome evaluator", "Predicted vs observed compared",
                 f"delay err {case['outcome']['delay_error_min']:+.1f} min; false alarm={case['outcome']['false_alarm']}. Policy weights unchanged.")
        self.last_message = f"Outcome evaluated for {case['case_id']}. Policy weights are never changed automatically."

    # ---------- reporting
    def all_cases(self, include_history=True):
        return (self.history if include_history else []) + list(self.cases.values())

    def rubber_stamp_report(self):
        rows = []
        p = self.PM.current["review"]
        decs = [d for c in self.all_cases() for d in c["decisions"]]
        by = {}
        for d in decs:
            by.setdefault(d["reviewer"], []).append(d)
        for r, ds in sorted(by.items()):
            n = len(ds)
            appr = sum(d["decision"] == "Approve" for d in ds) / n
            med = statistics.median(d["response_s"] for d in ds)
            fast = sum(d["response_s"] < p["fast_response_flag_s"] for d in ds)
            over = sum(bool(d.get("override")) for d in ds)
            flags = []
            if n >= p["min_n_for_pattern"] and appr >= p["approval_rate_flag"]:
                flags.append(f"approval rate ≥ {p['approval_rate_flag']:.0%}")
            if fast:
                flags.append(f"{fast} response(s) < {p['fast_response_flag_s']} s")
            rows.append({"reviewer": r, "decisions": n, "approval_rate": round(appr, 2), "median_response_s": round(med, 1),
                         "fast_responses": fast, "overrides": over, "audit_flag": "; ".join(flags) or "—"})
        return pd.DataFrame(rows)


OBSERVED_FIELDS = {"observed_delay_reduction_min": "Observed delay reduction (min)",
                   "observed_access_protection": "Observed access protection (0–1)",
                   "observed_vulnerable_riders_served": "Vulnerable riders served (count)",
                   "actual_cost_usd": "Actual cost (USD)", "rider_reported_impact": "Rider-reported impact (1–5)",
                   "false_alarm": "False alarm (true/false)"}


def get_case_policy(case):
    return APP.PM.get(case["policy_version"])


def outcome_metrics(pr, obs):
    return {"evaluated_at": iso(utcnow()), "action": pr["option_id"], "action_type": pr["type"],
            "pred_delay_reduction_min": pr["delay_reduction_min"], "obs_delay_reduction_min": obs["observed_delay_reduction_min"],
            "delay_error_min": round(pr["delay_reduction_min"] - obs["observed_delay_reduction_min"], 2),
            "pred_access": pr["access_protection"], "obs_access": obs["observed_access_protection"],
            "pred_equity_riders": pr["vulnerable_riders_helped"], "obs_equity_riders": obs["observed_vulnerable_riders_served"],
            "pred_cost_usd": pr["cost_usd"], "actual_cost_usd": obs["actual_cost_usd"],
            "cost_ratio": round(obs["actual_cost_usd"] / pr["cost_usd"], 2) if pr["cost_usd"] else None,
            "rider_reported_impact": obs["rider_reported_impact"], "false_alarm": bool(obs["false_alarm"])}


def explanation_brief(case):
    """Deterministic Options Ranking Brief: every option, formula, component score, rank, assumption, uncertainty, source, timestamp.
    Alternatives are displayed BEFORE the recommendation."""
    A = case["analysis"]
    pol = get_case_policy(case)
    P = A["scores"]["P"]
    w = scenario_weights(P, pol)
    L = [f"#### Options ranking brief — {case['case_id']} · rules v{case['policy_version']}"]
    L.append(f"**Priority P = {P:.3f}** · tier **{A['tier']}** · confidence **C = {A['confidence']['C']:.3f}**"
             + (" (official hard-trigger bypass of C gate)" if A["confidence"]["C"] < pol["confidence_threshold"] else ""))
    L.append(f"\n**Formula** Si = Ci × [{w['DR']:.3f}·DR + {w['EB']:.3f}·EB + {w['AP']:.3f}·AP + {w['OF']:.3f}·OF + {w['CE']:.3f}·CE]"
             f" — weights sum to {sum(w.values()):.2f}; as P rises, weight shifts from OF/CE to EB/AP. Ci = C × option certainty.")
    L.append("\n**All alternatives (shown before the recommendation)**\n")
    L.append("| Rank | Option | DR | EB | AP | OF | CE | Ci | **Si** | Cost | Assumptions |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    opts = {o["option_id"]: o for o in A["options"]}
    for r in A["ranking"]:
        o = opts[r["option_id"]]
        L.append(f"| {r['rank']} | {r['label']} | {r['DR']:.2f} | {r['EB']:.2f} | {r['AP']:.2f} | {r['OF']:.2f} | {r['CE']:.2f} | {r['Ci']:.2f} | **{r['Si']:.3f}** | ${r['cost_usd']:.0f} | {o['notes']}; parties: {', '.join(o.get('parties', []))}{'; ' + o['assumptions'] if o['assumptions'] else ''} |")
    inf = [o for o in A["options"] if not o["feasible"]]
    if inf:
        L.append("\n**Considered but infeasible:** " + "; ".join(f"{o['label']} ({o['infeasible_reason']})" for o in inf))
    L.append("\n**Uncertainty:** " + (f"missing fields not imputed: {', '.join(A['missing'])}; " if A["missing"] else "no missing fields; ")
             + f"Q={A['confidence']['Q']}, F={A['confidence']['F']}, G={A['confidence']['G']}, M={A['confidence']['M']}.")
    raw = case["raw"]
    if raw.get("missing_inputs"):
        L.append("**Inputs with no live source (left missing, never imputed):** " + ", ".join(m.replace("_", " ") for m in raw["missing_inputs"]))
    if raw.get("incident_history"):
        L.append("**Incident history (input):** " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in raw["incident_history"].items() if k != "observed_at"))
    if raw.get("slm_interpretation"):
        L.append(f"**Section A SLM reading of the event report (supporting only, not scored):** {_html.escape(str(raw['slm_interpretation']))}")
    srcs = [f"{e['source_id']} ({e['source_type']}, {e['timestamp'][11:19]} ET, age {e['age_s']} s)" for e in A["evidence"] if e["status"] == "accepted"]
    L.append("**Sources used:** " + "; ".join(srcs))
    if A["ranking"]:
        top = A["ranking"][0]
        tie = len(A["ranking"]) > 1 and abs(A["ranking"][1]["Si"] - top["Si"]) < 0.01
        L.append(f"\n**Recommendation (rank 1):** {top['label']} — Si = {top['Si']:.3f}"
                 + (". ⚠️ Close call (< 0.01 from rank 2) — tie-break order: higher AP, higher OF, lower cost." if tie else "."))
    else:
        L.append("\n**No feasible option — escalated to human review.** Revise constraints or reject (safe default).")
    return "\n".join(L)


def template_narrative(case):
    A = case["analysis"]
    rk = A["ranking"]
    if not rk:
        return "No option satisfies the current resource, road, policy and safety constraints. The reviewer should revise constraints or reject, which routes the case to the safe default."
    alt = "; ".join(f"{r['label']} scores {r['Si']:.3f}" for r in rk[1:]) or "there are no other feasible options"
    trig = ", ".join(t["trigger"] for t in A["hard_triggers"]) or f"R = {A['scores']['R']:.2f}"
    return (f"This case is significant because of {trig}. Priority P = {A['scores']['P']:.2f} (tier {A['tier']}) shifts scenario weight toward equity "
            f"and access. Alternatives: {alt}. The deterministic evaluator ranks **{rk[0]['label']}** first with Si = {rk[0]['Si']:.3f}. "
            f"This is a recommendation only; the human reviewer decides.")


def narrative_names_rank1(text, label):
    """True when the narrative names the rank-1 option's action, e.g. 'dispatch a second bus' (any capitalisation)."""
    return label.split(" —")[0].strip().lower() in (text or "").lower()


def llm_narrative(case, attempt=1):
    A = case["analysis"]
    rk = A["ranking"]
    facts = {"route": f"{case['raw']['route_id']} {case['raw']['route_name']}", "triggers": [t["trigger"] for t in A["hard_triggers"]],
             "R": A["scores"]["R"], "V": A["scores"]["V"], "A": A["scores"]["A"], "P": A["scores"]["P"], "tier": A["tier"],
             "confidence": A["confidence"]["C"], "missing_fields": A["missing"],
             "options_ranked": [{"rank": r["rank"], "option": r["label"], "Si": r["Si"], "DR": r["DR"], "EB": r["EB"], "AP": r["AP"],
                                 "OF": r["OF"], "CE": r["CE"]} for r in rk],
             "essential_destinations": case["raw"]["access"]["essential_names"]}
    system = ("You are the Explanation Agent for a transit operations prototype. Explain deterministic results to a human reviewer in plain language. "
              "Rules: use ONLY the numbers in the facts; never invent or change numbers; describe the alternatives BEFORE the recommendation; "
              "the recommendation is always the rank-1 option; say the human reviewer makes the final decision; 5 sentences maximum; no headings.")
    if not rk:
        return template_narrative(case)
    txt = agent_llm_generate("Facts (JSON):\n" + json.dumps(facts, indent=1), system)
    if not narrative_names_rank1(txt, rk[0]["label"]):
        raise AgentFailure("Narrative did not name the rank-1 option; discarded by guard")
    return txt


# ====================================================================================================
# cell: e3_policy
# ====================================================================================================
#@title E3 · Incident history, Outcome Evaluator metrics, Insight Agent, policy-owner approval, backtest and versioning
def incident_history_df(store):
    """Past cases recorded by this deployment (there is no seeded history)."""
    rows = []
    for c in store.all_cases():
        A = c["analysis"]
        rows.append({"case_id": c["case_id"], "date": c["created_at"][:10], "route": c["raw"]["route_name"],
                     "event": c["raw"]["event_type"], "delay_min": A.get("fields", {}).get("delay_min"),
                     "significant": A["significant"], "status": c["status"],
                     "response": c.get("approved_option_id") or "rider information only",
                     "false_alarm": (c.get("outcome") or {}).get("false_alarm")})
    return pd.DataFrame(rows)


def insight_agent(store):
    """Insight Agent: proposes threshold, bound or weight changes from outcome results. It cannot apply any change."""
    pol = store.PM.current
    ev = [c for c in store.all_cases() if c.get("outcome")]
    sig = [c for c in ev if c["analysis"] and c["analysis"].get("significant") and "pred_delay_reduction_min" in c["outcome"]]
    props = []

    rejected = {(c["path"], json.dumps(c["new"])) for p in store.proposal_log if p["status"] == "rejected" for c in p["changes"]}

    def add(title, changes, rationale, evidence):
        if all((c["path"], json.dumps(c["new"])) in rejected for c in changes):   # a rejected change is not re-proposed
            store.log(None, "Insight Agent", "agent", "Insight agent", "Skipped previously rejected proposal", title)
            return
        props.append({"id": f"PR-{len(store.proposal_log) + len(props) + 1:03d}", "title": title, "changes": changes,
                      "rationale": rationale, "evidence": evidence, "status": "proposed", "proposed_at": iso(utcnow())})

    if sig:
        nohard = [c for c in sig if not c["analysis"]["hard_triggers"]]
        if len(nohard) >= 3:
            fa = sum(c["outcome"]["false_alarm"] for c in nohard) / len(nohard)
            if fa > 0.25 and pol["significance_R"] < 0.70:
                add("Raise R significance threshold", [{"path": "significance_R", "old": pol["significance_R"], "new": round(pol["significance_R"] + 0.05, 2)}],
                    "Too many R-only significant cases turned out to be false alarms.", f"false-alarm rate {fa:.0%} over {len(nohard)} R-only cases (limit 25%)")
        for t in ["dispatch_second_bus", "reroute", "protect_connection"]:
            cs = [c for c in sig if c["outcome"]["action_type"] == t and c["outcome"]["pred_delay_reduction_min"] > 0]
            if len(cs) >= 3:
                ratio = sum(c["outcome"]["obs_delay_reduction_min"] for c in cs) / sum(c["outcome"]["pred_delay_reduction_min"] for c in cs)
                cur = pol["lookups"]["option_certainty"][t]
                if ratio < 0.85:
                    add(f"Lower option certainty for {t}", [{"path": f"lookups.option_certainty.{t}", "old": cur, "new": round(max(0.5, cur - 0.05), 2)}],
                        "Observed delay reduction is consistently below prediction for this action.", f"observed/predicted = {ratio:.2f} over {len(cs)} cases")
            cc = [c for c in cs if c["outcome"].get("cost_ratio")]
            if len(cc) >= 3:
                cr = statistics.mean(c["outcome"]["cost_ratio"] for c in cc)
                if cr > 1.2:
                    U = pol["bounds"]["cost_usd"][1]
                    add("Tighten cost normalization bound", [{"path": "bounds.cost_usd", "old": list(pol["bounds"]["cost_usd"]), "new": [0, int(U * 0.85)]}],
                        f"Actual costs for {t} exceed predictions, so cost efficiency is overstated.", f"mean actual/predicted cost {cr:.2f} over {len(cc)} cases")
        hp = [c for c in sig if c["analysis"]["scores"]["P"] >= 0.5 and c["outcome"]["pred_equity_riders"] > 0]
        if len(hp) >= 3:
            er = sum(c["outcome"]["obs_equity_riders"] for c in hp) / sum(c["outcome"]["pred_equity_riders"] for c in hp)
            sh = pol["scenario_shift"]["EB"]
            if er < 0.85 and sh < 0.15:
                ns = round(sh + 0.02, 2)
                add("Increase priority shift toward equity and access",
                    [{"path": "scenario_shift.EB", "old": sh, "new": ns}, {"path": "scenario_shift.AP", "old": pol["scenario_shift"]["AP"], "new": ns},
                     {"path": "scenario_shift.OF", "old": pol["scenario_shift"]["OF"], "new": -ns}, {"path": "scenario_shift.CE", "old": pol["scenario_shift"]["CE"], "new": -ns}],
                    "High-priority cases deliver less equity benefit than predicted; weight equity/access more as P rises (weights still sum to 1).",
                    f"observed/predicted vulnerable riders served = {er:.2f} over {len(hp)} cases with P ≥ 0.5")
    mon = [c for c in ev if c["outcome"].get("missed_real_disruption") is not None]
    if len(mon) >= 5:
        miss = sum(c["outcome"]["missed_real_disruption"] for c in mon) / len(mon)
        if miss > 0.30 and pol["confidence_threshold"] > 0.6:
            add("Lower confidence threshold", [{"path": "confidence_threshold", "old": pol["confidence_threshold"], "new": round(pol["confidence_threshold"] - 0.05, 2)}],
                "Monitored cases later proved to be real disruptions too often.", f"missed-disruption rate {miss:.0%} over {len(mon)} cases")
    store.proposals = props
    store.log(None, "Insight Agent", "agent", "Insight agent", f"{len(props)} proposal(s) generated",
              "; ".join(p["title"] for p in props) or "No change proposed — current rules kept")
    store.insight_narrative = insight_narrative_text(props, len(ev))
    return props


def insight_narrative_text(props, n):
    if not props:
        return f"Reviewed {n} evaluated cases. No change is warranted; current rules are kept."
    base = f"Reviewed {n} evaluated cases and drafted {len(props)} proposal(s). None is applied until the policy owner approves it and it passes a backtest."
    if LLM_READY:
        try:
            return agent_llm_generate(json.dumps([{k: p[k] for k in ("title", "changes", "evidence")} for p in props]),
                                   "Summarize these proposed transit policy changes for a policy owner in 3 sentences. Use only the given numbers. "
                                   "State clearly that nothing has been applied.", max_tokens=200, timeout=90)
        except AgentFailure:
            pass
    return base


def apply_changes(policy, changes):
    newp = copy.deepcopy(policy)
    for ch in changes:
        set_path(newp, ch["path"], copy.deepcopy(ch["new"]))
    return newp


def run_backtest(store, new_policy):
    """Re-run the deterministic pipeline on past cases under current vs proposed rules."""
    cur = store.PM.current
    rows = []
    for c in store.all_cases():
        a = run_analysis(c["raw"], cur, c.get("operator_resolution"), c.get("constraints"))
        b = run_analysis(c["raw"], new_policy, c.get("operator_resolution"), c.get("constraints"))
        out = c.get("outcome") or {}
        rows.append({"case_id": c["case_id"], "gate_current": a["gate"], "gate_proposed": b["gate"],
                     "top_current": a["ranking"][0]["option_id"] if a["ranking"] else "—",
                     "top_proposed": b["ranking"][0]["option_id"] if b["ranking"] else "—",
                     "Si_current": a["ranking"][0]["Si"] if a["ranking"] else None, "Si_proposed": b["ranking"][0]["Si"] if b["ranking"] else None,
                     "known_false_alarm": out.get("false_alarm")})
    df = pd.DataFrame(rows)
    sig_change = int((df.gate_current != df.gate_proposed).sum())
    top_change = int((df.top_current != df.top_proposed).sum())
    fa_avoided = int(((df.gate_current == "SIGNIFICANT") & (df.gate_proposed != "SIGNIFICANT") & (df.known_false_alarm == True)).sum())
    real_lost = int(((df.gate_current == "SIGNIFICANT") & (df.gate_proposed != "SIGNIFICANT") & (df.known_false_alarm == False)).sum())
    summary = (f"Backtest on {len(df)} past cases: {sig_change} gate change(s), {top_change} top-option change(s), "
               f"{fa_avoided} known false alarm(s) avoided, {real_lost} real disruption(s) that would no longer be escalated.")
    store.backtest = {"df": df, "summary": summary}
    return df, summary


def decide_proposal(store, pid, owner, decision, reason):
    p = next((p for p in store.proposals if p["id"] == pid), None)
    if p is None:
        raise ValueError("Pick a proposal (run the Insight Agent first).")
    if not owner.strip() or len(reason.strip()) < 10:
        raise ValueError("Policy owner name and a reason (≥ 10 characters) are required.")
    gp, gn = governance_check(owner, decision, proposal=p)                # INTEGRATION (Sections C + D)
    for n_ in gn:
        store.log(None, "Governance gate", "rule", "Policy owner gate (C+D)", "note", n_)
    if gp:
        store.log(None, owner, "policy owner", "Policy owner approval", "Submission blocked", "; ".join(gp))
        raise ValueError("Cannot submit: " + "; ".join(gp))
    if decision == "Approve":
        newp = apply_changes(store.PM.current, p["changes"])
        errs = validate_policy(newp)
        if errs:
            raise ValueError("Proposal breaks policy invariants: " + "; ".join(errs))
        _, summary = run_backtest(store, newp)
        v = store.PM.release(newp, owner, reason, p["changes"], summary)
        p.update(status=f"approved → v{v}", decided_by=owner, reason=reason, backtest=summary)
        store.log(None, owner, "policy owner", "Policy owner approval", f"APPROVED {pid} → new rule version {v}", summary)
        store.last_message = f"{pid} approved, backtested and released as rules v{v} (rollback available)."
    else:
        p.update(status="rejected", decided_by=owner, reason=reason)
        store.log(None, owner, "policy owner", "Policy owner approval", f"REJECTED {pid}", f"Current rules kept. Reason: {reason}")
        store.last_message = f"{pid} rejected; reason returned to the Insight Agent and current rules kept."
    store.proposal_log.append(copy.deepcopy(p))
    store.proposals = [x for x in store.proposals if x["id"] != pid]


PM = PolicyManager(DEFAULT_POLICY)
APP = CaseStore(PM)


# ====================================================================================================
# cell: e4_core
# ====================================================================================================
#@title E4 · Outputs: interactive map, tables and charts
import folium
import plotly.graph_objects as go

STATUS_COLOR = {"MONITORING": "#6b7280", "CONFLICT_REVIEW": "#d97706", "SAFE_DEFAULT": "#2563eb", "AWAITING_REVIEW": "#dc2626",
                "AWAITING_SECOND_REVIEW": "#b91c1c", "AWAITING_IMPLEMENTATION": "#7c3aed", "IMPLEMENTED": "#059669",
                "EVALUATED": "#047857", "CLOSED_FALSE_ALARM": "#9ca3af"}
def _tract_color(v):
    # light → dark purple ramp for transit dependence (share of zero-vehicle households in the stop's census tract)
    ramp = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#54278f"]
    return "#9ca3af" if v is None else ramp[min(len(ramp) - 1, int(v / 0.6 * len(ramp)))]


def live_vehicle_layer(m, highlight=None):
    """Every live vehicle near the service area (NJ Transit GTFS-realtime)."""
    fg = folium.FeatureGroup(name="Live vehicles (NJ Transit GTFS-realtime)", show=True)
    for v in live_vehicles():
        d = v["delay_s"]
        col = "#b91c1c" if v["cancelled"] or (d or 0) >= 900 else "#f59e0b" if (d or 0) >= 300 else "#16a34a" if d is not None else "#6b7280"
        hot = str(v["vehicle_id"]) == str(highlight)
        folium.CircleMarker((v["lat"], v["lon"]), radius=9 if hot else 5, color="#111827" if hot else col, weight=3 if hot else 1,
                            fill=True, fill_color=col, fill_opacity=0.9,
                            tooltip=_html.escape(f"Route {v['route_id']} · vehicle {v['vehicle_id']} · "
                                    + ("cancelled" if v["cancelled"] else "delay unknown" if d is None else f"{d / 60:+.1f} min"))).add_to(fg)
    fg.add_to(m)


def build_map(case):
    """Real map: live vehicles, and for the active case its route's remaining stops (static GTFS) coloured by census tract."""
    m = folium.Map(location=list(SERVICE_AREA_LATLON), zoom_start=14, tiles="CartoDB positron", control_scale=True)
    folium.Marker(list(SERVICE_AREA_LATLON), tooltip="Service area centre", icon=folium.Icon(color="gray", icon="home")).add_to(m)
    folium.Circle(list(SERVICE_AREA_LATLON), radius=NJT_NEARBY_KM * 1000, color="#64748b", weight=1, fill=False,
                  tooltip=f"{NJT_NEARBY_KM} km live-vehicle radius").add_to(m)
    live_vehicle_layer(m, highlight=case["raw"].get("vehicle_id") if case else None)
    if case:
        raw, A = case["raw"], case["analysis"]
        stops = raw.get("stops") or []
        tract_by_name = {t["tract_name"]: t for t in raw.get("community", {}).get("tracts", [])}
        fg = folium.FeatureGroup(name=f"{case['case_id']}: route {raw['route_name']} (next stops)", show=True)
        if len(stops) > 1:
            folium.PolyLine([(s["lat"], s["lon"]) for s in stops], color="#b91c1c", weight=6, opacity=0.85,
                            tooltip=_html.escape(f"Route {raw['route_name']}: remaining stops of trip {raw.get('trip_id')}")).add_to(fg)
        for s in stops:
            t = None
            try:
                geo = census_tract_at(s["lat"], s["lon"])          # cached from case intake, so no new request
                t = _CENSUS_TRACT_CACHE.get(geo.get("GEOID")) if geo else None
            except Exception:
                pass
            v = t.get("zero_vehicle_share") if t else None
            tip = f"{_html.escape(str(s['name']))} · {_html.escape(str(s['stop_id']))}"
            if t:
                tip += (f" · {_html.escape(str(t['tract_name']))}: zero-vehicle {v if v is not None else '—'}, poverty {t.get('poverty_rate')}, "
                        f"disability {t.get('disability_rate')}")
            folium.CircleMarker((s["lat"], s["lon"]), radius=7, color="#111827", weight=1, fill=True, fill_color=_tract_color(v),
                                fill_opacity=0.95, tooltip=tip).add_to(fg)
        f = A.get("fields", {})
        sc = A.get("scores") or {}
        if stops:
            pop = (f"<b>{case['case_id']}</b><br>{_html.escape(case['scenario_name'])}<br>Status: <b>{case['status']}</b><br>"
                   f"Delay: {f.get('delay_min', '—')} min · gap {f.get('headway_ratio', '—')}× · load {f.get('load_factor', '—')}<br>"
                   f"C={A['confidence']['C'] if A.get('confidence') else '—'} · R={sc.get('R', '—')} V={sc.get('V', '—')} "
                   f"A={sc.get('A', '—')} P={sc.get('P', '—')}")
            folium.Marker((stops[0]["lat"], stops[0]["lon"]), popup=folium.Popup(pop, max_width=320), tooltip="Incident (next stop)",
                          icon=folium.Icon(color="red" if case["status"] in REVIEW_STATUSES else "orange", icon="warning-sign")).add_to(fg)
            ts = raw.get("connections", {}).get("transfer_stop_id")
            tstop = next((s for s in stops if s["stop_id"] == ts), None)
            if tstop:
                folium.CircleMarker((tstop["lat"], tstop["lon"]), radius=13, color="#0891b2", fill=False, weight=3,
                                    tooltip=_html.escape("Connecting routes here: " + ", ".join(raw["connections"].get("connecting_routes", [])[:8]))).add_to(fg)
            pts = [(s["lat"], s["lon"]) for s in stops]
            m.fit_bounds([[min(p[0] for p in pts) - 0.005, min(p[1] for p in pts) - 0.008],
                          [max(p[0] for p in pts) + 0.005, max(p[1] for p in pts) + 0.008]])
        fg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    legend = ("<div style='position:fixed;bottom:18px;left:18px;z-index:9999;background:white;padding:8px 10px;border-radius:6px;"
              "font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)'><b>Live NJ Transit data</b><br>"
              "Vehicles: <span style='color:#16a34a'>●</span> on time <span style='color:#f59e0b'>●</span> 5+ min "
              "<span style='color:#b91c1c'>●</span> 15+ min / cancelled <span style='color:#6b7280'>●</span> unknown<br>"
              "Stops shaded by tract zero-vehicle share (darker = higher; grey = no Census data)</div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m


def map_html(case):
    return build_map(case).get_root().render()


def map_iframe(case, height=540):
    doc = build_map(case).get_root().render()
    return f'<iframe srcdoc="{_html.escape(doc, quote=True)}" style="width:100%;height:{height}px;border:0;border-radius:8px"></iframe>'


def ph(df, msg="—"):
    return df if (df is not None and len(df)) else pd.DataFrame({"info": [msg]})


def evidence_df(case):
    if not case:
        return ph(None)
    rows = [{"source_id": e["source_id"], "source_type": e["source_type"], "status": e["status"], "quality": e.get("quality"),
             "timestamp_utc": e["timestamp"][:19], "age_s": e["age_s"],
             "fields": ", ".join(f"{k}={v}" for k, v in e["fields"].items()), "note": e["note"]} for e in case["analysis"]["evidence"]]
    return ph(pd.DataFrame(rows))


def exceptions_df(case):
    if not case:
        return ph(None)
    return ph(pd.DataFrame(case["analysis"]["exceptions"]), "No exception paths taken")


def trace_md(case):
    if not case:
        return "_No case yet._"
    icon = lambda r: "🟥" if any(k in r.upper() for k in ("SAFE", "NO", "MONITOR", "CONFLICT", "FAIL")) else "🟩"
    lines = [f"### Pipeline trace — {case['case_id']}", f"**Gate:** `{case['analysis']['gate']}` — {case['analysis']['gate_reason']}", ""]
    lines += [f"{icon(t['result'])} **{t['stage']}** · {t['result']}" + (f"  \n<small>{_html.escape(t['detail'])}</small>" if t["detail"] else "")
              for t in case["analysis"]["trace"]]
    if case.get("safe_default_reason"):
        lines.append(f"\n🔵 **Safe default:** {case['safe_default_reason']}")
    return "\n\n".join(lines)


def conf_df(case):
    if not case or not case["analysis"].get("confidence"):
        return ph(None, "Confidence not computed (agent safe mode)")
    c = case["analysis"]["confidence"]
    w = get_case_policy(case)["confidence_weights"]
    rows = [{"component": k, "meaning": m, "value": c[k], "weight": w[k], "contribution": r3(w[k] * c[k])}
            for k, m in [("Q", "source quality"), ("F", "freshness"), ("G", "cross-source agreement"), ("M", "completeness")]]
    rows.append({"component": "penalty", "meaning": "unresolved conflict", "value": c["penalty"], "weight": -1, "contribution": -c["penalty"]})
    rows.append({"component": "C", "meaning": f"threshold {get_case_policy(case)['confidence_threshold']}", "value": c["C"], "weight": None, "contribution": c["C"]})
    return pd.DataFrame(rows)


def norm_df(case):
    if not case or not case["analysis"].get("normalization"):
        return ph(None, "Impact scores not computed")
    return pd.DataFrame(case["analysis"]["normalization"])[["group", "component", "feature", "raw", "L", "U", "method", "normalized", "weight", "contribution", "note"]]


def rvap_plot(case):
    fig = go.Figure()
    if case and case["analysis"].get("scores"):
        s = case["analysis"]["scores"]
        pol = get_case_policy(case)
        fig.add_bar(x=["R reliability", "V vulnerability", "A access", "P priority"], y=[s["R"], s["V"], s["A"], s["P"]],
                    marker_color=["#2563eb", "#7c3aed", "#0891b2", "#111827"], text=[f"{v:.3f}" for v in (s["R"], s["V"], s["A"], s["P"])],
                    textposition="outside")
        fig.add_hline(y=pol["significance_R"], line_dash="dash", line_color="#dc2626", annotation_text=f"R significance {pol['significance_R']}")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), yaxis_range=[0, 1.1], title="Component scores (0–1)", plot_bgcolor="white")
    return fig


def sig_md(case):
    if not case:
        return ""
    A = case["analysis"]
    trig = "\n".join(f"- **{t['trigger']}** (source: {t['source']}{', official ✔' if t['official'] else ''})" for t in A["hard_triggers"]) or "- none"
    miss = ", ".join(A["missing"]) or "none"
    return (f"### Significance\n**Result:** {'SIGNIFICANT' if A['significant'] else 'not significant'} · tier **{A['tier'] or '—'}**\n\n"
            f"**Hard triggers (bypass R threshold):**\n{trig}\n\n**Missing values (not imputed):** {miss}")


def options_df(case):
    if not case or not case["analysis"].get("ranking"):
        return ph(None, "No ranked options (case not significant, or no feasible option)")
    return pd.DataFrame([{k: r[k] for k in ["rank", "option_id", "label", "DR", "EB", "AP", "OF", "CE", "Ci", "Si", "cost_usd"]} for r in case["analysis"]["ranking"]])


def infeasible_df(case):
    if not case:
        return ph(None)
    inf = [{"option": o["label"], "reason": o["infeasible_reason"]} for o in case["analysis"].get("options", []) if not o["feasible"]]
    return ph(pd.DataFrame(inf), "All generated options are feasible")


def options_plot(case):
    fig = go.Figure()
    if case and case["analysis"].get("ranking"):
        rk = case["analysis"]["ranking"]
        labels = [f"#{r['rank']} {r['label'][:34]}" for r in rk][::-1]
        colors = {"DR": "#2563eb", "EB": "#7c3aed", "AP": "#0891b2", "OF": "#16a34a", "CE": "#f59e0b"}
        for c in ["DR", "EB", "AP", "OF", "CE"]:
            fig.add_bar(y=labels, x=[r["weighted"][c] for r in rk][::-1], name=c, orientation="h", marker_color=colors[c])
        fig.update_layout(barmode="stack")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=20), title="Si = Ci × weighted components (stacked)", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.15))
    return fig


def decisions_df(case):
    if not case or not case["decisions"]:
        return ph(None, "No human decisions yet")
    return pd.DataFrame(case["decisions"])[["ts", "stage", "reviewer", "decision", "option_id", "recommended_option_id", "override",
                                             "response_s", "tier", "deadline_at", "revision_no", "reason"]]


def deadline_md(case):
    if not case:
        return "_No active case._"
    st = case["status"]
    if st in REVIEW_STATUSES:
        left = APP.seconds_left(case)
        mm, ss = divmod(max(0, int(left)), 60)
        r = case["review"]
        who = "second reviewer" if st == "AWAITING_SECOND_REVIEW" else "first reviewer"
        bar = "🟥" if left < 60 * DEMO_DEADLINE_SCALE else "🟧" if left < 180 * DEMO_DEADLINE_SCALE else "🟩"
        extra = " · ⚠️ **No feasible option** — revise or reject" if r.get("no_feasible_option") else ""
        return (f"### {bar} {mm:02d}:{ss:02d} left — waiting for {who}\nTier **{r['tier']}** ({r['deadline_min']} min, covers both reviewers) · "
                f"revisions used {case['revisions']}/{get_case_policy(case)['max_revisions']}{extra}. No response → safe default.")
    return f"**Status:** `{st}` — {STATUS_HELP.get(st, '')}"


def outcome_df(case):
    if not case or not case.get("outcome"):
        return ph(None, "No outcome evaluated yet")
    o = case["outcome"]
    if "pred_delay_reduction_min" not in o:
        return pd.DataFrame([o])
    rows = [("Delay reduction (min)", o["pred_delay_reduction_min"], o["obs_delay_reduction_min"]),
            ("Access protection (0–1)", o["pred_access"], o["obs_access"]),
            ("Equity: vulnerable riders served", o["pred_equity_riders"], o["obs_equity_riders"]),
            ("Cost (USD)", o["pred_cost_usd"], o["actual_cost_usd"])]
    df = pd.DataFrame([{"metric": m, "predicted": p, "observed": ob, "difference": r3((ob or 0) - (p or 0))} for m, p, ob in rows])
    extra = pd.DataFrame([{"metric": "False alarm", "predicted": "—", "observed": o["false_alarm"], "difference": "—"},
                          {"metric": "Rider-reported impact (1 low – 5 high)", "predicted": "—", "observed": o["rider_reported_impact"], "difference": "—"}])
    return pd.concat([df.astype(object), extra], ignore_index=True)


def outcome_plot(case):
    fig = go.Figure()
    o = case.get("outcome") if case else None
    if o and "pred_delay_reduction_min" in o:
        m = ["Delay reduction", "Access", "Equity riders", "Cost"]
        pr = [o["pred_delay_reduction_min"], o["pred_access"], o["pred_equity_riders"], o["pred_cost_usd"]]
        ob = [o["obs_delay_reduction_min"], o["obs_access"], o["obs_equity_riders"], o["actual_cost_usd"]]
        fig.add_bar(x=m, y=[1.0 if p else 0 for p in pr], name="predicted (=1.0)", marker_color="#94a3b8")
        fig.add_bar(x=m, y=[(b / p) if p else 0 for p, b in zip(pr, ob)], name="observed / predicted", marker_color="#2563eb")
    fig.update_layout(height=300, barmode="group", margin=dict(l=20, r=20, t=30, b=20), title="Observed relative to predicted", plot_bgcolor="white")
    return fig


def versions_df():
    return pd.DataFrame([{"version": v["version"], "released_at": v["released_at"], "approved_by": v["approved_by"], "reason": v["reason"],
                          "changes": "; ".join(f"{c['path']}: {c['old']} → {c['new']}" for c in v["changes"]) or "—",
                          "backtest": v["backtest_summary"] or "—", "active": i == APP.PM.current_index}
                         for i, v in enumerate(APP.PM.versions)])


def proposals_df():
    return ph(pd.DataFrame([{"id": p["id"], "title": p["title"], "changes": "; ".join(f"{c['path']}: {c['old']} → {c['new']}" for c in p["changes"]),
                             "evidence": p["evidence"], "rationale": p["rationale"], "status": p["status"]} for p in APP.proposals]),
              "No open proposals — run the Insight Agent")


def proposal_log_df():
    return ph(pd.DataFrame([{"id": p["id"], "title": p["title"], "status": p["status"], "decided_by": p.get("decided_by"), "reason": p.get("reason"),
                             "backtest": p.get("backtest", "—")} for p in APP.proposal_log]), "No decisions yet")


def audit_df(case_only=False):
    rows = APP.audit if not case_only or not APP.active else APP.active["audit"]
    return ph(pd.DataFrame(rows[::-1][:500]), "Audit log empty")


def cases_df():
    rows = [{"case_id": c["case_id"], "created_at": c["created_at"][:19], "scenario": c["scenario_name"], "status": c["status"],
             "gate": c["analysis"]["gate"], "P": (c["analysis"].get("scores") or {}).get("P"), "tier": c["analysis"].get("tier"),
             "rules": c["policy_version"], "approved_option": c.get("approved_option_id")} for c in APP.cases.values()]
    return ph(pd.DataFrame(rows[::-1]), "No live cases yet")


def monitor_df():
    rows = [{"case_id": c["case_id"], "status": c["status"], "route": c["raw"]["route_id"],
             "confidence": (c["analysis"].get("confidence") or {}).get("C"), "priority_P": (c["analysis"].get("scores") or {}).get("P"),
             "reason": c.get("safe_default_reason") or c["analysis"]["gate_reason"]}
            for c in APP.cases.values() if c["status"] in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW")]
    return ph(pd.DataFrame(rows), "Monitoring queue empty")


def alerts_df(case):
    return ph(pd.DataFrame(case["alerts"]) if case else None, "No alerts")


def export_outputs():
    """Write every Colab output (shared case state, evidence, exceptions, scores, scenarios, decisions, outcomes, audit, rules, map)."""
    ts = local_now().strftime("%Y%m%dT%H%M%S%Z")
    path = os.path.join(OUTPUT_DIR, f"transit_dashboard_outputs_{ts}.zip")
    cases = list(APP.cases.values())
    ev, ex, sc, op, dc, oc = [], [], [], [], [], []
    for c in cases:
        A = c["analysis"]
        ev += [{"case_id": c["case_id"], **{k: v for k, v in e.items() if k != "fields"}, "fields": json.dumps(e["fields"])} for e in A["evidence"]]
        ex += [{"case_id": c["case_id"], **e} for e in A["exceptions"]]
        if A.get("scores"):
            sc.append({"case_id": c["case_id"], **{k: A["scores"][k] for k in "RVAP"}, **A["scores"]["components"], **(A["confidence"] or {}), "tier": A["tier"]})
        op += [{"case_id": c["case_id"], **{k: v for k, v in r.items() if k not in ("weights", "weighted")}} for r in A.get("ranking", [])]
        dc += [{"case_id": c["case_id"], **d} for d in c["decisions"]]
        if c.get("outcome"):
            oc.append({"case_id": c["case_id"], **c["outcome"]})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("shared_case_state.json", json.dumps(cases, indent=1, default=str))
        z.writestr("policy_versions.json", json.dumps(APP.PM.versions, indent=1, default=str))
        z.writestr("insight_proposals.json", json.dumps(APP.proposals + APP.proposal_log, indent=1, default=str))
        for name, rows in [("verified_evidence", ev), ("exception_log", ex), ("component_scores", sc), ("scenario_table_and_rank", op),
                           ("human_decisions_and_timing", dc), ("outcome_evaluation", oc), ("audit_record", APP.audit)]:
            z.writestr(f"{name}.csv", pd.DataFrame(rows).to_csv(index=False))
        z.writestr("reviewer_audit_rubber_stamp.csv", APP.rubber_stamp_report().to_csv(index=False))
        z.writestr("incident_history.csv", incident_history_df(APP).to_csv(index=False))
        z.writestr("section_c_ethics_training_status.csv", pd.DataFrame(eth_check_all()).to_csv(index=False))
        z.writestr("section_d_disclosures.csv", disclosures_df().to_csv(index=False))
        try:
            z.writestr("section_b_notebook_form_audit.csv", audit_table().to_csv(index=False))
        except NameError:
            pass
        if APP.active:
            z.writestr(f"map_{APP.active['case_id']}.html", build_map(APP.active).get_root().render())
    APP.log(None, "system", "system", "Export", "Outputs exported", path)
    return path


# ====================================================================================================
# cell: e5_core
# ====================================================================================================
#@title E5 · Gradio dashboard (human-in-the-loop)
try:
    import gradio as gr          # needed only for the Gradio dashboard (the Streamlit app does not import it)
except ImportError:
    gr = None


def banner_md():
    c = APP.active
    llm = f"live LLM: {', '.join(p['label'] for p in LLM_CHAIN)}" if LLM_READY else "LLM off → template narratives"
    head = (f"**Rules v{APP.PM.current['version']}** · {llm} · {len(APP.cases)} case(s) · "
            f"{len(live_vehicles())} live vehicle(s) · feed pulled {(LIVE_CONTEXT['nj_transit'].get('fetched_at') or 'never')[11:19]} ET · "
            f"deadline scale ×{DEMO_DEADLINE_SCALE}")
    if c:
        col = STATUS_COLOR.get(c["status"], "#111")
        head += (f"<br><span style='background:{col};color:white;padding:2px 8px;border-radius:4px'>{c['status']}</span> "
                 f"**{c['case_id']}** · {_html.escape(c['scenario_name'])}")
    return head + f"<br>💬 {_html.escape(APP.last_message)}"


def _notify(msg, warn=False):
    try:
        (gr.Warning if warn else gr.Info)(msg)
    except Exception:
        pass


def refresh():
    c = APP.active
    feas = [(f"#{r['rank']} {r['label']} (Si {r['Si']:.3f})", r["option_id"]) for r in (c["analysis"]["ranking"] if c else [])]
    mon_ids = [x["case_id"] for x in APP.cases.values() if x["status"] in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW")]
    expl = c.get("explanation") if c else None
    second = ""
    if c and c["status"] == "AWAITING_SECOND_REVIEW":
        f = c["review"]["first"]
        second = f"**First reviewer {f['reviewer']}** approved `{f['option_id']}`: “{_html.escape(f['reason'])}”. An independent reviewer must also approve; either can reject."
    impl = "_No approved action awaiting implementation._"
    if c and c["status"] == "AWAITING_IMPLEMENTATION":
        impl = f"### ✅ Approved: `{c['approved_option_id']}`\nOperations staff carry out the action **outside the prototype** and send rider alerts. Record the result below."
    elif c and c["implementation"]:
        impl = "\n".join(f"- {i['ts'][:19]} · {i['staff']} · **{i['result']}** · {i['notes']}" for i in c["implementation"])
    bt = APP.backtest
    prop_choices = [p["id"] for p in APP.proposals]
    vers = [v["version"] for v in APP.PM.versions]
    return [
        banner_md(),
        gr.update(choices=list(APP.cases)[::-1], value=APP.active_id),
        map_iframe(c), trace_md(c), evidence_df(c), exceptions_df(c), alerts_df(c), monitor_df(),
        gr.update(choices=mon_ids, value=(APP.active_id if APP.active_id in mon_ids else (mon_ids[0] if mon_ids else None))),
        conf_df(c), norm_df(c), rvap_plot(c), sig_md(c),
        options_df(c), infeasible_df(c), options_plot(c),
        expl["brief_md"] if expl else "_Options appear for significant cases._",
        (f"**Narrative ({expl['narrative_source']}, non-authoritative):** {expl['narrative']}" if expl else ""),
        deadline_md(c), options_df(c), gr.update(choices=feas, value=feas[0][1] if feas else None), second, decisions_df(c),
        (f"**Rider alert (verified info only):** {c['rider_alert']}" if c and c.get("rider_alert") else ""),
        impl, outcome_df(c), outcome_plot(c),
        json.dumps(APP.PM.current, indent=1), proposals_df(), gr.update(choices=prop_choices, value=prop_choices[0] if prop_choices else None),
        APP.insight_narrative or "_Run the Insight Agent to analyse outcomes._",
        bt["summary"] if bt else "_No backtest yet._", ph(bt["df"] if bt else None, "No backtest yet"),
        versions_df(), gr.update(choices=vers, value=vers[0]), proposal_log_df(),
        audit_df(), APP.rubber_stamp_report() if len(APP.rubber_stamp_report()) else ph(None, "No decisions yet"), cases_df(),
        b_audit_df(), eth_summary_md(), eth_status_df(), disclosures_df(), coi_screen_df(c),
        gr.update(choices=[(o["label"], o["option_id"]) for o in (c["analysis"].get("options", []) if c else []) if o["feasible"]]),
        live_status_md(), live_vehicles_df(), gr.update(choices=vehicle_choices()),
        incident_history_df(APP) if len(APP.cases) else ph(None, "No cases recorded yet in this deployment"),
        gr.update(choices=list(STAFF_DISCLOSURES)),
    ]


def action(fn):
    def wrapper(*args):
        with APP.lock:
            try:
                msg = fn(*args)
                if msg:
                    APP.last_message = msg
                _notify(APP.last_message)
            except (ValueError, KeyError, LiveDataUnavailable) as e:
                APP.last_message = f"⚠️ {e}"
                _notify(str(e), warn=True)
            return refresh()
    return wrapper


def _need_case():
    if not APP.active:
        raise ValueError("Run a scenario first.")
    return APP.active


# ---- handlers: every case comes from a live NJ Transit vehicle
def live_status_md():
    gt = _NETWORK["net"].source if _NETWORK["net"] else (f"not loaded ({_NETWORK['error']})" if _NETWORK["error"] else "not loaded yet")
    return "  \n".join(["**Live sources**"] + [_html.escape(l) for l in live_status_lines()] + [f"Static GTFS: {_html.escape(gt)}"])


def vehicle_choices():
    net = _NETWORK["net"]
    return [(vehicle_label(v, net), str(v["vehicle_id"])) for v in live_vehicles()]


def live_vehicles_df():
    rows = [{"route": v["route_id"], "vehicle": v["vehicle_id"], "trip": v["trip_id"],
             "delay_min": None if v["delay_s"] is None else round(v["delay_s"] / 60, 1), "cancelled": v["cancelled"],
             "distance_km": v["distance_km"],
             "report_age_s": int((utcnow() - datetime.fromtimestamp(v["timestamp"], timezone.utc)).total_seconds()) if v["timestamp"] else None,
             "alert": bool(alert_for_route(v["route_id"], LIVE_CONTEXT["nj_transit"].get("alerts")))} for v in live_vehicles()]
    return ph(pd.DataFrame(rows), "No live vehicles yet — press ↻ Pull live data")


def h_pull_live():
    refresh_live_context()
    network_or_none()
    return f"Live data pulled: {len(live_vehicles())} vehicle(s) within {NJT_NEARBY_KM} km."


def h_run_vehicle(vehicle_id, use_llm):
    if not vehicle_id:
        raise ValueError("Pull live data and pick a vehicle first.")
    APP.new_case(raw_for_vehicle(vehicle_id, store=APP), use_llm=use_llm)


def h_select_case(cid):
    if cid in APP.cases:
        APP.active_id = cid
        return f"Switched to {cid}."


def h_operator(name, resolution):
    APP.operator_resolve(_need_case(), name, resolution)


def h_monitor(cid, act, name, reason=""):
    if cid not in APP.cases:
        raise ValueError("Pick a case from the monitoring queue.")
    APP.active_id = cid
    if act == "false_alarm":
        APP.close_false_alarm(APP.cases[cid], name or "planner", reason)
    else:
        APP.monitor_action(APP.cases[cid], "recheck", name or "planner")


def h_regen(use_llm):
    c = _need_case()
    if not c.get("analysis", {}).get("ranking") and c["status"] not in REVIEW_STATUSES:
        raise ValueError("Narratives are generated for significant cases with options.")
    c["use_llm"] = use_llm
    APP.explain(c)
    return f"Narrative regenerated ({c['explanation']['narrative_source']})."


def h_first(name, decision, option_id, reason, checklist, viewed, excl, budget, hold, alt_depot):
    c = _need_case()
    rev = None
    if decision == "Revise":
        rev = {"excluded_options": list(excl or []), "budget_cap_usd": float(budget), "max_hold_min": float(hold), "allow_alternate_depot": bool(alt_depot)}
    APP.submit_first(c, name, decision, option_id, reason, checklist, viewed, rev)


def h_second(name, decision, reason, checklist, viewed):
    APP.submit_second(_need_case(), name, decision, reason, checklist, viewed)


def h_expire():
    c = _need_case()
    if c["status"] not in REVIEW_STATUSES:
        raise ValueError("No review timer is running.")
    c["review"]["deadline_at"] = iso(utcnow() - timedelta(seconds=1))
    APP.log(c, "demo", "tester", "Deadline control", "Deadline expiry simulated")
    APP.enforce_deadline(c)


def h_impl(staff, result, notes):
    APP.record_implementation(_need_case(), staff, result, notes)


def h_evaluate(dr, acc, eq, cost, impact, fa):
    to_f = lambda x: None if x is None or x == "" else float(x)
    APP.evaluate_outcome(_need_case(), {"observed_delay_reduction_min": to_f(dr), "observed_access_protection": to_f(acc),
                                         "observed_vulnerable_riders_served": to_f(eq), "actual_cost_usd": to_f(cost),
                                         "rider_reported_impact": None if impact in (None, "—") else int(impact),
                                         "false_alarm": {"yes": True, "no": False}.get(fa)})

def h_insight():
    props = insight_agent(APP)
    return f"Insight Agent drafted {len(props)} proposal(s). Nothing is applied without policy-owner approval."


def h_preview(pid):
    p = next((p for p in APP.proposals if p["id"] == pid), None)
    if not p:
        raise ValueError("Pick a proposal.")
    _, s = run_backtest(APP, apply_changes(APP.PM.current, p["changes"]))
    return "Preview only — " + s


def h_decide(pid, owner, decision, reason):
    decide_proposal(APP, pid, owner, decision, reason)


def h_rollback(version, owner, reason):
    if not owner.strip() or len(reason.strip()) < 10:
        raise ValueError("Policy owner name and a reason (≥ 10 characters) are required for rollback.")
    v = APP.PM.rollback(version, owner, reason)
    APP.log(None, owner, "policy owner", "Rollback", f"Rules of {version} re-released as v{v}", reason)
    return f"Rolled back: rules of v{version} are now active as v{v}. Open cases keep the version they were scored with."


def tick():
    with APP.lock:
        c = APP.active
        expired = bool(c and APP.enforce_deadline(c))
        return banner_md() + (" <b>— press ↻ Refresh</b>" if expired else ""), deadline_md(c)


# ---- Sections C + D helpers for tab 8
def eth_summary_md():
    t = eth_team_summary()
    return (f"**Roster:** {t['total']} people · compliance rate **{t['compliance_rate_pct']}%** (COMPLIANT + AT_RISK) · "
            f"fully current {t['fully_current_pct']}% · 🟢 {t['COMPLIANT']} · 🟡 {t['AT_RISK']} · 🔴 {t['NON_COMPLIANT']}")


def eth_status_df():
    return ph(pd.DataFrame(eth_check_all()), "Roster empty")


def disclosures_df():
    rows = [{"person": n, "holdings": "; ".join(f"{h.owner}: {h.entity} ({h.kind})" for h in d["facts"].holdings) or "none",
             "disclosure": d["text"]} for n, d in STAFF_DISCLOSURES.items()]
    return pd.DataFrame(rows)


def coi_screen_df(case):
    if not case or not case["analysis"].get("options"):
        return ph(None, "Run a significant case to screen people against its options")
    rows = []
    for n, d in STAFF_DISCLOSURES.items():
        for o in case["analysis"]["options"]:
            if not o["feasible"]:
                continue
            sc = rule_screen(d["facts"], decisions_for_option(o), APP.PM.current["governance_gates"]["coi_threshold"])
            top = max(sc["hits"], key=lambda h: h["score"]) if sc["hits"] else None
            rows.append({"person": n, "option": o["label"], "parties": ", ".join(o["parties"]), "rule_score": sc["rule_score"],
                         "match": top["match"] if top else "—",
                         "gate": "RECUSE" if (top and top["match"] == "name" and sc["over_threshold"]) else ("note" if top else "clear")})
    return pd.DataFrame(rows)


def b_audit_df():
    try:
        df = audit_table()
    except NameError:
        df = None
    return ph(df, "No Section B notebook-form cases yet")


def h_load_roster(text, save):
    n = load_eth_roster(json.loads(text))
    ETH_ROSTER_SOURCE["source"] = "pasted in the dashboard"
    where = ""
    if save:
        where = f" and saved to {save_roster()}"
    APP.log(None, "system", "system", "Ethics roster", "Roster loaded", f"{n} people{where}")
    return f"Roster loaded ({n} people){where}."


def h_load_disclosures(text, save):
    n = load_disclosures(json.loads(text))
    STAFF_DISCLOSURES_SOURCE["source"] = "pasted in the dashboard"
    where = ""
    if save:
        with open(STAFF_DISCLOSURES_PATH, "w") as f:
            json.dump(disclosures_as_json(), f, indent=1)
        where = f" and saved to {STAFF_DISCLOSURES_PATH}"
    APP.log(None, "system", "system", "Disclosures", "Disclosures loaded", f"{n} people{where}")
    return f"Disclosures loaded ({n} people){where}."


def h_gate(name, decision, option_id):
    c = APP.active
    opt = next((o for o in (c["analysis"].get("options", []) if c else []) if o["option_id"] == option_id), None)
    if not name.strip():
        return "Enter a name."
    probs, notes = governance_check(name, decision, option=opt)
    head = "### ⛔ Blocked" if probs else "### ✅ Allowed"
    return head + "\n" + "\n".join(f"- **{p}**" for p in probs) + ("\n" if probs else "") + "\n".join(f"- {n}" for n in notes)


def h_eth_agent(question):
    try:
        rep_ = eth_agent.run(question, verbose=False)
    except Exception as e:
        return f"⚠️ Agent unavailable: {e}"
    issues = verify_report(rep_)
    fin = reconcile_report(rep_)
    rows = "\n".join(f"| {a.official} | {a.status} | {', '.join(a.missing_courses) or '—'} | {a.priority} | {a.action} |" for a in fin.action_items)
    return (f"**Answer:** {_html.escape(fin.answer)}\n\n**Guardrail:** " + ("; ".join(issues) + " (corrected below)" if issues else "✅ matched the rule engine") +
            f"\n\n| Person | Status | Missing | Priority | Action |\n|---|---|---|---|---|\n{rows}\n\n_Ethics notes:_ {_html.escape(fin.ethics_notes)}")


def h_coi_pick(name):
    return STAFF_DISCLOSURES[name]["text"] if name in STAFF_DISCLOSURES else ""


def h_coi_pipeline(text):
    try:
        r = coi_run_pipeline(text, verbose=False)
    except Exception as e:
        return f"⚠️ Pipeline unavailable: {e}"
    f = r["final"]
    issues = coi_guardrail(r)
    return (f"### {f.recommendation} · risk {f.risk_score:.2f} · {_html.escape(f.official)}\n{_html.escape(f.summary)}\n\n"
            f"**Rule screener:** score {r['screen']['rule_score']} (threshold {r['screen']['threshold']}) · hits: "
            + ("; ".join(f"{h['holding']} ↔ {h['decision']} ({h['match']}, {h['score']})" for h in r["screen"]["hits"]) or "none") +
            "\n\n**Required actions:** " + "; ".join(f.required_actions) + f"\n\n**Due process:** {_html.escape(f.due_process_note)}\n\n"
            "**Guardrail:** " + ("; ".join(issues) if issues else "✅ consistent with flags, mitigations and the rule screen"))


CSS = """
.gradio-container {max-width: 1500px !important}
#banner {border-left: 4px solid #2563eb; padding: 6px 12px; background: rgba(37,99,235,.06); border-radius: 6px}
"""


def build_dashboard():
    if gr is None:
        raise ImportError("Install gradio to build the Gradio dashboard: pip install 'gradio>=5'")
    with gr.Blocks(title="Transit Reliability & Equity — HITL Dashboard") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        gr.Markdown("# 🚌 Transit Reliability & Equity — Integrated Team System\n"
                    "Sections A (data inputs + SLM) · B (agents + workflow) · C (ethics training gate) · D (conflict-of-interest gate) · "
                    "E (engine, dashboard, outputs) — deterministic scoring · human-supervised decisions · **prototype: does not control live transit operations**")
        banner = gr.Markdown(elem_id="banner")
        with gr.Row():
            case_dd = gr.Dropdown(label="Active case", choices=[], interactive=True, scale=3)
            refresh_btn = gr.Button("↻ Refresh", scale=1)
        with gr.Tabs():
            # ---------------- 1
            with gr.Tab("0 · Live data + SLM (Section A)"):
                gr.Markdown("**Section A — live data inputs + small language model.** Every value comes from NJ Transit "
                            "(GTFS-realtime + static GTFS), the US Census ACS or the National Weather Service; inputs with no live "
                            "source are shown as missing and are never filled in. Pick a vehicle, load its seven sources, then hand "
                            "them to the Signal Scout. The SLM text is supporting information only.")
                with gr.Row():
                    pull_btn = gr.Button("↻ Pull live data (NJ Transit · Census · NWS)", variant="secondary", scale=2)
                    veh_dd = gr.Dropdown(label="Live vehicle", choices=[], scale=4)
                live_md = gr.Markdown()
                live_df = gr.Dataframe(label="Live vehicles near the service area (NJ Transit GTFS-realtime)", interactive=False, wrap=True)
                with gr.Row():
                    with gr.Column(scale=3):
                        a_json = gr.Code(value="", language="json", label="Seven input sources for the selected vehicle (read-only)")
                    with gr.Column(scale=2):
                        a_use_slm = gr.Checkbox(value=USE_SLM, label="Use FLAN-T5 small language model (first run downloads ~300 MB)")
                        a_load = gr.Button("Load the seven sources for this vehicle")
                        a_run = gr.Button("▶ Collate, interpret and hand off to Signal Scout", variant="primary")
                        a_slm = gr.Markdown()
                a_ssi = gr.Code(language="json", label="signal_scout_input (structured hand-off)")
            with gr.Tab("1 · Intake & verification"):
                with gr.Row():
                    use_llm = gr.Checkbox(value=LLM_READY, label="Live LLM narrative", scale=1)
                    run_btn = gr.Button("▶ Run pipeline on the selected live vehicle", variant="primary", scale=2)
                with gr.Row():
                    with gr.Column(scale=3):
                        map_html = gr.HTML(label="Interactive map")
                    with gr.Column(scale=2):
                        trace = gr.Markdown()
                evidence = gr.Dataframe(label="Verified evidence (accepted / stale / quarantined / excluded)", interactive=False, wrap=True)
                with gr.Row():
                    exceptions = gr.Dataframe(label="Exception log", interactive=False, wrap=True)
                    alerts = gr.Dataframe(label="Alerts (operator, technical owner, planner)", interactive=False, wrap=True)
                with gr.Accordion("Operator review — conflicting data (max two checks, then monitor)", open=True):
                    with gr.Row():
                        op_name = gr.Textbox(label="Operator name", value="")
                        op_resolve = gr.Button("Resolved: apply source precedence and re-verify")
                        op_unres = gr.Button("Still unresolved")
                gr.Markdown("### Monitoring queue (monitoring, safe default, conflict review)")
                monitor = gr.Dataframe(interactive=False, wrap=True)
                with gr.Row():
                    mon_dd = gr.Dropdown(label="Case", choices=[])
                    mon_name = gr.Textbox(label="Planner name", value="")
                    mon_reason = gr.Textbox(label="Reason (for false alarm)")
                with gr.Row():
                    mon_recheck = gr.Button("Recheck (pull the live feed again)")
                    mon_fa = gr.Button("Close as false alarm")
                with gr.Accordion("Incident history (cases recorded by this deployment)", open=False):
                    history = gr.Dataframe(interactive=False, wrap=True)
            # ---------------- 2
            with gr.Tab("2 · Component scores"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Confidence** C = 0.40Q + 0.25F + 0.20G + 0.15M (proceed when C ≥ 0.70 or an official source verifies a hard trigger)")
                        conf = gr.Dataframe(interactive=False)
                        sig = gr.Markdown()
                    with gr.Column():
                        rvap = gr.Plot()
                gr.Markdown("**Normalization and impact** — n(x) = clip((x−L)/(U−L), 0, 1); ninv(x) = 1 − n(x); "
                            "R = 0.30D + 0.25H + 0.25K + 0.20E · V = 0.35N + 0.25I + 0.20B + 0.20T · A = 0.35S + 0.30L + 0.20X + 0.15U · P = 0.40R + 0.35V + 0.25A")
                norm = gr.Dataframe(interactive=False, wrap=True)
            # ---------------- 3
            with gr.Tab("3 · Options & ranking"):
                with gr.Row():
                    with gr.Column(scale=3):
                        options = gr.Dataframe(label="Scenario table and rank", interactive=False, wrap=True)
                        infeasible = gr.Dataframe(label="Considered but infeasible", interactive=False, wrap=True)
                    with gr.Column(scale=2):
                        options_fig = gr.Plot()
                explain = gr.Markdown()
                narrative = gr.Markdown()
                regen = gr.Button("↻ Regenerate narrative")
            # ---------------- 4
            with gr.Tab("4 · Human review"):
                deadline = gr.Markdown()
                review_opts = gr.Dataframe(label="All alternatives (review before deciding)", interactive=False, wrap=True)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### First reviewer")
                        r1_name = gr.Textbox(label="Reviewer name", placeholder="name as on the tab 8 roster")
                        r1_dec = gr.Radio(["Approve", "Revise", "Reject"], value="Approve", label="Decision")
                        r1_opt = gr.Dropdown(label="Option to approve (defaults to rank 1; other choices are logged as overrides)", choices=[])
                        r1_reason = gr.Textbox(label="Typed reason (required)", lines=2)
                        r1_check = gr.CheckboxGroup(REVIEW_CHECKLIST, label="Checklist (all required)")
                        r1_viewed = gr.Checkbox(label="I viewed all alternatives before the recommendation")
                        with gr.Accordion("Revision: change constraints / request another option (max two)", open=False):
                            rv_excl = gr.CheckboxGroup(list(OPTION_LABELS), label="Exclude option types")
                            rv_budget = gr.Slider(100, 2000, value=1000, step=50, label="Budget cap (USD)")
                            rv_hold = gr.Slider(0, 15, value=5, step=1, label="Max connection hold (min)")
                            rv_alt = gr.Checkbox(label="Request option: dispatch from alternate depot")
                        r1_submit = gr.Button("Submit first review", variant="primary")
                    with gr.Column():
                        gr.Markdown("### Second reviewer (high-impact actions)")
                        second_info = gr.Markdown()
                        r2_name = gr.Textbox(label="Independent reviewer name", placeholder="a different person on the roster")
                        r2_dec = gr.Radio(["Approve", "Reject"], value="Approve", label="Decision")
                        r2_reason = gr.Textbox(label="Typed reason (required)", lines=2)
                        r2_check = gr.CheckboxGroup(REVIEW_CHECKLIST, label="Checklist (all required)")
                        r2_viewed = gr.Checkbox(label="I viewed all alternatives before the recommendation")
                        r2_submit = gr.Button("Submit second review", variant="primary")
                        expire_btn = gr.Button("⏱ Simulate deadline expiry (demo)", variant="secondary")
                rider_alert = gr.Markdown()
                decisions = gr.Dataframe(label="Human decision and timing", interactive=False, wrap=True)
            # ---------------- 5
            with gr.Tab("5 · Implementation & outcome"):
                impl = gr.Markdown()
                with gr.Row():
                    im_staff = gr.Textbox(label="Operations staff")
                    im_res = gr.Radio(["Completed", "Partial", "Failed"], value="Completed", label="Result")
                    im_notes = gr.Textbox(label="Notes")
                    im_btn = gr.Button("Record implementation", variant="primary")
                gr.Markdown("### Outcome evaluator (separate from the recommender; never silently changes policy weights)")
                with gr.Row():
                    o_dr = gr.Number(label="Observed delay reduction (min)", value=None)
                    o_acc = gr.Number(label="Observed access protection (0–1)", value=None)
                    o_eq = gr.Number(label="Vulnerable riders served", value=None)
                    o_cost = gr.Number(label="Actual cost (USD)", value=None)
                    o_imp = gr.Dropdown(["—", "1", "2", "3", "4", "5"], value="—", label="Rider-reported impact (1–5)")
                    o_fa = gr.Dropdown(["—", "no", "yes"], value="—", label="False alarm")
                gr.Markdown("_Enter what was observed after the incident (from operations records). Nothing is simulated._")
                with gr.Row():
                    eval_btn = gr.Button("Evaluate outcome", variant="primary")
                with gr.Row():
                    outcome = gr.Dataframe(label="Outcome evaluation", interactive=False, wrap=True)
                    outcome_fig = gr.Plot()
            # ---------------- 6
            with gr.Tab("6 · Policy governance"):
                with gr.Row():
                    insight_btn = gr.Button("🔍 Run Insight Agent (proposes only)", variant="primary")
                insight_md = gr.Markdown()
                proposals = gr.Dataframe(label="Proposed changes", interactive=False, wrap=True)
                with gr.Row():
                    prop_dd = gr.Dropdown(label="Proposal", choices=[])
                    owner = gr.Textbox(label="Policy owner name", placeholder="policy owner (on the roster)")
                    owner_reason = gr.Textbox(label="Reason (required)")
                with gr.Row():
                    preview_btn = gr.Button("Preview backtest")
                    approve_btn = gr.Button("Approve → backtest → release new version", variant="primary")
                    reject_btn = gr.Button("Reject (keep current rules)", variant="stop")
                backtest_md = gr.Markdown()
                backtest = gr.Dataframe(label="Backtest on past cases", interactive=False, wrap=True)
                proposal_log = gr.Dataframe(label="Decision log", interactive=False, wrap=True)
                versions = gr.Dataframe(label="Rule versions", interactive=False, wrap=True)
                with gr.Row():
                    rb_dd = gr.Dropdown(label="Roll back to version", choices=[])
                    rb_btn = gr.Button("Roll back")
                with gr.Accordion("Active policy (thresholds, bounds, weights, lookups)", open=False):
                    policy_code = gr.Code(language="json")
            # ---------------- 7
            with gr.Tab("7 · Audit & export"):
                cases = gr.Dataframe(label="All live cases (shared case state)", interactive=False, wrap=True)
                rubber = gr.Dataframe(label="Rubber-stamp protection: approval patterns and response times", interactive=False, wrap=True)
                audit = gr.Dataframe(label="Audit record (newest first)", interactive=False, wrap=True)
                b_audit = gr.Dataframe(label="Section B notebook-form cases (audit_table)", interactive=False, wrap=True)
                with gr.Row():
                    export_btn = gr.Button("⬇ Export all outputs (zip: JSON + CSV + map)", variant="primary")
                    export_file = gr.File(label="Download")
                gr.Markdown("**Governance boundary:** human authority is final; thresholds, bounds and weights are versioned policy assumptions; "
                            "the recommender cannot change scoring and the evaluator cannot create options; exception paths never trigger resource action; "
                            "social reports require corroboration; vision data anonymized; community data area-level only.")

            # ---------------- 8
            with gr.Tab("8 · Ethics & independence (Sections C + D)"):
                gr.Markdown("**Section C — Ethics Training and Compliance Monitor** and **Section D — Conflict of Interest Detection** "
                            "protect the review steps. Their deterministic rule engines are the gate (NON_COMPLIANT people cannot approve or revise; "
                            "a name match between a reviewer's disclosure and a party to the option requires recusal). The LLM agents explain; "
                            "their guardrails correct them against the rules.")
                with gr.Row():
                    with gr.Column(scale=3):
                        eth_summary = gr.Markdown()
                        eth_table = gr.Dataframe(label="Ethics training status (Section C rule engine)", interactive=False, wrap=True)
                    with gr.Column(scale=2):
                        eth_json = gr.Code(value=json.dumps([o.model_dump() for o in ETH_ROSTER] or ROSTER_TEMPLATE, indent=1),
                                           language="json", label="Reviewer roster JSON (your real reviewers)")
                        eth_save = gr.Checkbox(value=False, label=f"Also save to {REVIEWER_ROSTER_PATH}")
                        eth_load = gr.Button("Load roster")
                with gr.Row():
                    g_name = gr.Textbox(label="Check a person", placeholder="e.g. Gabe Ortiz")
                    g_dec = gr.Radio(["Approve", "Revise", "Reject"], value="Approve", label="Decision")
                    g_opt = gr.Dropdown(label="Option on the active case", choices=[])
                    g_btn = gr.Button("Run governance gate")
                g_md = gr.Markdown()
                disclosures = gr.Dataframe(label="Conflict-of-interest disclosures on file (Section D facts)", interactive=False, wrap=True)
                with gr.Accordion("Load disclosures (JSON)", open=not STAFF_DISCLOSURES):
                    disc_json = gr.Code(value=json.dumps(disclosures_as_json() or DISCLOSURE_TEMPLATE, indent=1), language="json",
                                        label="Disclosures JSON (facts can be omitted when an LLM key is set)")
                    disc_save = gr.Checkbox(value=False, label=f"Also save to {STAFF_DISCLOSURES_PATH}")
                    disc_load = gr.Button("Load disclosures")
                screen_df = gr.Dataframe(label="Section D rule screen: each person × each option on the active case", interactive=False, wrap=True)
                with gr.Accordion("Section C agent: ask about the roster (LLM + guardrail)", open=False):
                    eth_q = gr.Textbox(value=ETH_DEFAULT_QUESTION, label="Question")
                    eth_run = gr.Button("Run compliance agent")
                    eth_answer = gr.Markdown()
                with gr.Accordion("Section D multi-agent review of a disclosure (LLM + guardrail)", open=False):
                    coi_pick = gr.Dropdown(list(STAFF_DISCLOSURES), label="Disclosure on file")
                    coi_text = gr.Textbox(value="", lines=4, label="Case text (editable)")
                    coi_run = gr.Button("Run 5-step conflict-of-interest pipeline")
                    coi_md = gr.Markdown()

        OUT = [banner, case_dd, map_html, trace, evidence, exceptions, alerts, monitor, mon_dd, conf, norm, rvap, sig,
               options, infeasible, options_fig, explain, narrative, deadline, review_opts, r1_opt, second_info, decisions, rider_alert,
               impl, outcome, outcome_fig, policy_code, proposals, prop_dd, insight_md, backtest_md, backtest, versions, rb_dd, proposal_log,
               audit, rubber, cases, b_audit, eth_summary, eth_table, disclosures, screen_df, g_opt,
               live_md, live_df, veh_dd, history, coi_pick]
        demo.load(refresh, outputs=OUT)
        refresh_btn.click(refresh, outputs=OUT)
        case_dd.input(action(h_select_case), inputs=[case_dd], outputs=OUT)
        run_btn.click(action(h_run_vehicle), inputs=[veh_dd, use_llm], outputs=OUT)
        pull_btn.click(action(h_pull_live), outputs=OUT)
        op_resolve.click(action(lambda n: h_operator(n, "resolved")), inputs=[op_name], outputs=OUT)
        op_unres.click(action(lambda n: h_operator(n, "unresolved")), inputs=[op_name], outputs=OUT)
        mon_recheck.click(action(lambda c, n: h_monitor(c, "recheck", n)), inputs=[mon_dd, mon_name], outputs=OUT)
        mon_fa.click(action(lambda c, n, r: h_monitor(c, "false_alarm", n, r)), inputs=[mon_dd, mon_name, mon_reason], outputs=OUT)
        regen.click(action(h_regen), inputs=[use_llm], outputs=OUT)
        r1_submit.click(action(h_first), inputs=[r1_name, r1_dec, r1_opt, r1_reason, r1_check, r1_viewed, rv_excl, rv_budget, rv_hold, rv_alt], outputs=OUT)
        r2_submit.click(action(h_second), inputs=[r2_name, r2_dec, r2_reason, r2_check, r2_viewed], outputs=OUT)
        expire_btn.click(action(h_expire), outputs=OUT)
        im_btn.click(action(h_impl), inputs=[im_staff, im_res, im_notes], outputs=OUT)

        eval_btn.click(action(h_evaluate), inputs=[o_dr, o_acc, o_eq, o_cost, o_imp, o_fa], outputs=OUT)
        insight_btn.click(action(h_insight), outputs=OUT)
        preview_btn.click(action(h_preview), inputs=[prop_dd], outputs=OUT)
        approve_btn.click(action(lambda p, o, r: h_decide(p, o, "Approve", r)), inputs=[prop_dd, owner, owner_reason], outputs=OUT)
        reject_btn.click(action(lambda p, o, r: h_decide(p, o, "Reject", r)), inputs=[prop_dd, owner, owner_reason], outputs=OUT)
        rb_btn.click(action(h_rollback), inputs=[rb_dd, owner, owner_reason], outputs=OUT)

        def _export():
            with APP.lock:
                return export_outputs()
        export_btn.click(_export, outputs=[export_file])

        # ---- Section A tab
        def _load_a(vid):
            try:
                inp = section_a_live_inputs(vid or None, APP)
                return json.dumps(inp, indent=1, default=str)
            except LiveDataUnavailable as e:
                _notify(str(e), warn=True)
                return ""
        a_load.click(_load_a, inputs=[veh_dd], outputs=[a_json])

        def _run_a(vid, use_slm, use_llm_):
            global USE_SLM
            with APP.lock:
                extra = [gr.update(), gr.update()]
                try:
                    USE_SLM = bool(use_slm)
                    inputs = section_a_live_inputs(vid or None, APP)
                    ssi = build_signal_scout_input(inputs)
                    raw = signal_scout_input_to_raw(ssi)
                    APP.new_case(raw, use_llm=use_llm_)
                    wx = ssi.get("external_feeds", {}).get("weather", {})
                    checks = "".join(f"  \n⚠️ {_html.escape(f)}" for f in ssi["event"].get("slm_cross_check", []))
                    extra = [f"**SLM reading (supporting only):** {_html.escape(ssi['event']['slm_interpretation'])}{checks}  \n"
                             f"**Deterministic fields:** delay in alert text = {ssi['event']['delay_min']} min · cause = {ssi['event']['cause']}  \n"
                             f"**Weather ({_html.escape(str(wx.get('source')))}):** {_html.escape(str(inputs['event_signals'].get('weather')))} · "
                             f"{len(inputs['event_signals'].get('weather_alerts', []))} active alert(s)  \n"
                             f"**Missing inputs:** {', '.join(raw['missing_inputs']) or 'none'}  \n"
                             f"**LLM:** {_html.escape(str(ssi.get('external_feeds', {}).get('llm')))}",
                             json.dumps(ssi, indent=1, default=str)]
                    _notify(APP.last_message)
                except (ValueError, KeyError, TypeError, LiveDataUnavailable) as e:
                    APP.last_message = f"⚠️ Section A: {e}"
                    _notify(APP.last_message, warn=True)
                return refresh() + extra
        a_run.click(_run_a, inputs=[veh_dd, a_use_slm, use_llm], outputs=OUT + [a_slm, a_ssi])

        # ---- Sections C + D tab
        eth_load.click(action(h_load_roster), inputs=[eth_json, eth_save], outputs=OUT)
        disc_load.click(action(h_load_disclosures), inputs=[disc_json, disc_save], outputs=OUT)
        g_btn.click(h_gate, inputs=[g_name, g_dec, g_opt], outputs=[g_md])
        eth_run.click(h_eth_agent, inputs=[eth_q], outputs=[eth_answer])
        coi_pick.change(h_coi_pick, inputs=[coi_pick], outputs=[coi_text])
        coi_run.click(h_coi_pipeline, inputs=[coi_text], outputs=[coi_md])
        if hasattr(gr, "Timer"):
            gr.Timer(2.0).tick(tick, outputs=[banner, deadline])
    return demo
