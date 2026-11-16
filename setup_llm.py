"""SETUP 3 · Live LLM client (OpenAI → OpenRouter → Hugging Face → templates)

Generated from the notebook (cells 6). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# SETUP 3 · Live LLM client (OpenAI by default → OpenRouter → Hugging Face → templates)   [notebook cell 6]
# ----------------------------------------------------------------------------------------------------
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

LLM_PROVIDER = param("LLM_PROVIDER", "openrouter")  # env-var override; notebook form field
LLM_FALLBACKS = param("LLM_FALLBACKS", "openai, openrouter, huggingface")  # env-var override; notebook form field
LLM_CUSTOM_MODEL = param("LLM_CUSTOM_MODEL", "")  # env-var override; notebook form field
LLM_STRUCT_MODE = param("LLM_STRUCT_MODE", "auto")  # env-var override; notebook form field
LLM_MAX_CALLS = param("LLM_MAX_CALLS", 300)  # env-var override; notebook form field
import re, time, warnings
from datetime import date
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Keys are read only from app secrets (environment variables) or environment variables; they are never printed or saved.
CATALOG = {
    "openai": {
        "label": "OpenAI", "url": "https://api.openai.com/v1/chat/completions", "key_name": "OPENAI_API_KEY",
        "key_help": "https://platform.openai.com/api-keys", "token_param": "max_completion_tokens",
        "models": [("gpt-4.1-mini", "Fast, low cost, reliable JSON-schema output")],
    },
    "openrouter": {
        "label": "OpenRouter", "url": "https://openrouter.ai/api/v1/chat/completions", "key_name": "OPENROUTER_API_KEY",
        "key_help": "https://openrouter.ai/keys", "token_param": "max_tokens",
        # Tried in order. Named instruct models first, because the free router sometimes picks a "thinking" model whose
        # reply is its working notes; a model that is retired, rejected or returns notes is skipped for the next one.
        "models": [("meta-llama/llama-3.3-70b-instruct:free", "Free instruct model"),
                   ("mistralai/mistral-small-3.2-24b-instruct:free", "Free instruct model"),
                   ("openrouter/free", "Free-models router: picks an available free model")],
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
    """app secrets (environment variables) first, then environment variables."""
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v.strip()
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
            "alt_models": [m for m, _ in cfg["models"]],
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
        print(f"  {cfg['label']}: no {cfg['key_name']} in the app secrets (skipped)")


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
            if r.status_code == 429 and "insufficient_quota" in r.text:   # no credit: retrying will not help
                raise LLMUnavailable("quota exhausted (HTTP 429 insufficient_quota); add credit or billing for this key")
            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {r.status_code}" + (" (rate limited; try again shortly)" if r.status_code == 429 else "")
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise LLMUnavailable(f"auth failed (HTTP {r.status_code}); check {CATALOG[p['provider']]['key_name']}")
            if r.status_code in (400, 404, 415, 422) and "response_format" in payload and LLM["struct_mode"] == "auto":
                p["native_ok"] = False
                payload.pop("response_format")
                continue
            if r.status_code in (400, 404, 422):
                raise LLMUnavailable(f"request rejected (HTTP {r.status_code}): {_scrub(r.text)}")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if content:
                content = re.sub(r"(?s)<think>.*?</think>", "", content).strip() or content   # drop inline working notes
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


def _call_models(p, messages, temperature, max_tokens, schema):
    """Try the provider's models in order. A model the provider rejects (retired, no longer free) is dropped for the
    session; a model that returns its working notes or nothing is skipped this time only. The provider itself is
    switched off only when its key fails or every model has been rejected."""
    dropped = p.setdefault("dropped", [])
    models = [m for m in dict.fromkeys([p["model"]] + p.get("alt_models", [])) if m not in dropped]
    if p["provider"] == "openrouter" and "openrouter/free" in models:
        models.append("openrouter/free")             # the free router may pick a different model on a second try
    last = LLMUnavailable("no models left")
    for m in models:
        if m in dropped:
            continue
        p["model"] = m
        try:
            out = _call_one(p, messages, temperature, max_tokens, schema)
            if schema is None and _looks_like_reasoning(out):
                raise LLMUnavailable("returned its reasoning instead of an answer (not used)")
            return out
        except LLMUnavailable as e:
            if re.search(r"auth failed|quota exhausted", str(e)):
                raise
            last = e
            if "request rejected" in str(e):
                dropped.append(m)                     # retired or no longer free: not asked for again this session
    left = [m for m in models if m not in dropped]
    p["model"] = left[0] if left else models[0]
    if left:                                          # models remain: a temporary miss, so the provider stays on
        raise LLMUnavailable(f"no usable answer this time ({_scrub(last)[:160]})")
    raise last                                        # every model rejected: the caller switches the provider off


def call_llm(messages, temperature=0.2, max_tokens=1800, schema=None):
    """One chat request. Tries each provider in LLM_CHAIN in order and records which one answered."""
    if not LLM_READY:
        raise LLMUnavailable("No LLM provider configured (add a key in the app secrets). Deterministic rules still work.")
    if LLM_STATS["calls"] >= LLM_MAX_CALLS:
        raise LLMUnavailable(f"LLM call budget reached ({LLM_MAX_CALLS}); raise LLM_MAX_CALLS to continue")
    errors = []
    for i, p in enumerate(LLM_CHAIN):
        if p.get("disabled"):                 # circuit breaker: a provider that failed auth/quota/model is skipped
            errors.append(f"{p['label']}: skipped ({p['disabled']})")
            continue
        try:
            LLM_STATS["calls"] += 1
            out = _call_models(p, messages, temperature, max_tokens, schema)
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




if LLM_READY:
    try:
        reply = call_llm([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=200)
        print(f"✓ Connected: {llm_source()} -> {reply.strip()[:40]!r}")
        for _p in LLM_CHAIN:
            if _p["label"] in LLM_STATS["provider_errors"]:
                print(f"  ✗ {_p['label']} did not answer: {LLM_STATS['provider_errors'][_p['label']]}"
                      + (" → skipped for this session" if _p.get("disabled") else " → will be retried"))
    except LLMUnavailable as e:
        print(f"✗ Connection test failed: {e}")
