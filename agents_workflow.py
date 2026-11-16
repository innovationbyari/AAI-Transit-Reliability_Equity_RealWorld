"""Section B · Typed agents, rule steps, HITL forms, agent-level runner, audit trail

Generated from the notebook (cells 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 67). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

wf = None   # notebook B·8 runner is not used in the web app

# ----------------------------------------------------------------------------------------------------
# B · 1. Setup and policy configuration (rules come from the shared PolicyManager)   [notebook cell 39]
# ----------------------------------------------------------------------------------------------------
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML

class PolicyConfigView(dict):
    """Read-only view of the shared versioned policy (APP.PM). Rules change only through Policy Owner Approval."""
    KEYS = {"rule_version": lambda p: "v" + p["version"], "confidence_threshold": lambda p: p["confidence_threshold"],
            "significance_threshold_R": lambda p: p["significance_R"], "max_reverify_attempts": lambda p: p["max_operator_checks"],
            "max_revisions": lambda p: p["max_revisions"], "min_reason_chars": lambda p: p["review"]["min_reason_chars"],
            "review_deadline_minutes": lambda p: p["deadlines_min"], "high_impact_actions": lambda p: p["high_impact"]["actions"],
            "deadline_resets_on_revision": lambda p: p.get("deadline_resets_on_revision", False),
            "delay_hard_trigger_min": lambda p: p["hard_trigger"]["delay_min"],
            "unboardable_load_factor": lambda p: p["hard_trigger"]["unboardable_load_factor"]}

    def __getitem__(self, k):
        return self.KEYS[k](APP.PM.current)

    def get(self, k, default=None):
        return self[k] if k in self.KEYS else default

    def __contains__(self, k):
        return k in self.KEYS

    def __setitem__(self, k, v):
        raise TypeError("CONFIG is read-only: change rules through Policy Owner Approval (APP.PM.release).")

    def __repr__(self):
        return repr({k: self[k] for k in self.KEYS})


CONFIG = PolicyConfigView()
RULE_VERSION_HISTORY = [{"version": CONFIG["rule_version"], "changed": "initial", "rollback_to": None}]


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone(LOCAL_TZ)


def stamp(dt: Optional[datetime] = None) -> str:
    return (dt or now()).isoformat(timespec="seconds")


print("Setup complete. Rule version:", CONFIG["rule_version"])


# ----------------------------------------------------------------------------------------------------
# B · 2. Data structures (the Shared Case State contracts)   [notebook cell 41]
# ----------------------------------------------------------------------------------------------------
@dataclass
class LoadSignal:
    """Vision Agent output. Counts only: no images or identities are stored."""
    stop_id: str
    route: str
    timestamp: str
    passenger_count: int
    capacity: int
    load_factor: float
    pass_ups: int
    crowding_level: str          # "normal" | "busy" | "crowded" | "unboardable"
    source: str = "vision_agent"


@dataclass
class VerificationResult:
    """Data Verification output."""
    status: str                  # "ready" | "feed_outage" | "agent_error" | "conflict"
    issues: list = field(default_factory=list)


@dataclass
class ImpactScores:
    """Deterministic Impact Analyzer output. All components normalized 0 to 1."""
    R: float                     # reliability impact
    V: float                     # rider vulnerability
    A: float                     # access impact
    P: float                     # priority
    hard_trigger: Optional[str] = None
    severity: str = "moderate"   # "severe" | "significant" | "moderate"


@dataclass
class Option:
    """Planner Copilot output. Options carry NO scores: the planner cannot score."""
    option_id: str
    action: str                  # "do_nothing" | "rider_alert" | "protect_connection" | "reroute" | "dispatch_second_bus"
    description: str
    resources_needed: dict
    constraints_checked: list


@dataclass
class ScoredOption:
    """Scenario Evaluator output: fixed published formula, same inputs give the same rank."""
    option: Option
    DR: float
    EB: float
    AP: float
    OF: float
    CE: float
    Ci: float
    score: float
    rank: int = 0


@dataclass
class Brief:
    """Explanation Agent output: what the human reviewer sees."""
    case_id: str
    summary: str
    options_table: list          # one dict per option, alternatives shown before the recommendation
    recommendation: str
    uncertainty_notes: list
    sources: list


@dataclass
class ReviewDecision:
    """Human Review / Second Reviewer output."""
    reviewer_role: str
    reviewer_name: str
    decision: str               # "approve" | "revise" | "reject" | "timeout"
    selected_option_id: Optional[str]
    reason: str
    new_constraints: str
    checklist_confirmed: bool
    seconds_to_decide: float
    timestamp: str


@dataclass
class ActionResult:
    """Implement Action output, entered by operations staff."""
    option_id: str
    status: str                  # "completed" | "partial" | "failed"
    rider_alert_sent: bool
    notes: str
    timestamp: str


@dataclass
class OutcomeReport:
    """Outcome Evaluator output. Never changes policy weights by itself."""
    case_id: str
    path: str                    # "action" | "safe_default" | "monitoring"
    predicted: dict
    observed: dict
    metrics: dict
    false_alarm: Optional[bool]


@dataclass
class PolicyProposal:
    """Insight Agent output. A proposal only: it cannot be applied without approval.
    parameter is a path to the rule it would change, e.g. "CONFIG.significance_threshold_R"
    or "COMPONENTS.dispatch_second_bus.DR"."""
    proposal_id: str
    parameter: str
    current_value: object
    proposed_value: object
    rationale: str
    evidence_cases: list


@dataclass
class PolicyDecision:
    """Policy Owner Approval output."""
    proposal_id: str
    decision: str                # "approve" | "reject"
    reason: str
    timestamp: str


@dataclass
class BacktestResult:
    """Backtest output: run before an approved rule change is released."""
    proposal_id: str
    passed: bool
    summary: str


@dataclass
class Case:
    """Shared Case State: the single record every stage reads and writes."""
    case_id: str
    route: str
    disruption_type: str
    description: str
    opened_at: str
    rule_version: str
    parent_case_id: Optional[str] = None
    status: str = "open"
    severity: str = "moderate"
    load_signals: list = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    reverify_attempts: int = 0
    confidence: Optional[float] = None
    impact: Optional[ImpactScores] = None
    options: list = field(default_factory=list)
    ranked_options: list = field(default_factory=list)
    brief: Optional[Brief] = None
    revision_count: int = 0
    constraints: list = field(default_factory=list)
    review_deadline: Optional[datetime] = None
    decisions: list = field(default_factory=list)
    selected_option_id: Optional[str] = None
    action_result: Optional[ActionResult] = None
    outcome: Optional[OutcomeReport] = None
    audit: list = field(default_factory=list)
    scout_ticket: dict = field(default_factory=dict)   # the validated copy of every input source the case was opened from

    def log(self, stage: str, actor: str, actor_type: str, output: str, note: str = ""):
        self.audit.append({
            "time": stamp(), "case_id": self.case_id, "stage": stage, "actor": actor,
            "actor_type": actor_type, "output": output, "note": note,
            "rule_version": self.rule_version,   # the case's own rule version, not the current global one
        })

    def option(self, option_id: str) -> Optional[ScoredOption]:
        return next((s for s in self.ranked_options if s.option.option_id == option_id), None)


ALL_CASES: list = []          # every case, for the audit trail
OUTCOME_HISTORY: list = []    # every OutcomeReport, read by the Insight Agent
REJECTED_PROPOSALS: dict = {} # (parameter, proposed value) -> policy owner's reason; never resubmitted
# These persist across reruns of the workflow cell, like a real system would. Rerun this cell to clear them.
print("Data structures defined.")


# ----------------------------------------------------------------------------------------------------
# B · Sample scenario (Route 12 cancellation, Main St crowding, last rail connection)   [notebook cell 43]
# ----------------------------------------------------------------------------------------------------
SAMPLE_INPUTS = {
    "service_feeds": {
        "route": "12", "trip_id": "12-1740", "status": "cancelled",
        "scheduled_departure": "17:40", "next_trip": "17:55", "headway_min": 15, "delay_min": 0,
    },
    "connecting_services": {
        "stop_id": "MAIN_ST", "rail_line": "Northeast Line", "last_departure": "18:05", "transfer_window_min": 8,
    },
    "incident_history": [
        {"route": "12", "type": "cancellation", "avg_recovery_min": 22, "past_response": "rider_alert"},
    ],
    "community_access": {
        "area": "Downtown East (area level aggregate)", "transit_dependence_pct": 62,
        "disability_share_pct": 14, "low_income_share_pct": 41,
        "essential_destinations": ["Regional Hospital", "Community College"],
    },
    "operating_resources": {
        "spare_buses": 1, "spare_operators": 1, "cost_per_bus_hour": 145, "deadhead_min": 12,
    },
    "event_signals": [
        {"type": "official_alert", "source": "agency", "text": "Route 12 17:40 trip cancelled (operator shortage)"},
    ],
    "rider_feedback": [
        {"channel": "app", "text": "Bus never came at Main St", "corroborated": False},
    ],
}

# Anonymized counts from station and onboard sensors (no images, no identities)
RAW_SENSOR_FEED = [
    {"stop_id": "MAIN_ST", "route": "12", "timestamp": "17:46", "person_count": 58, "capacity": 45, "pass_ups": 0},
    {"stop_id": "ELM_AVE", "route": "12", "timestamp": "17:44", "person_count": 41, "capacity": 45, "pass_ups": 6},
]

# Optional demo controls: force a path for a presentation or test. None = use the normal result.
# Used only by the Section B runner (B · 8); the orchestrator and dashboard never apply them.
DEMO_OVERRIDES = {
    "verification_status": None,   # "feed_outage" | "agent_error" | "conflict" on the first check only
    "exception_resolved": None,    # True | False
    "confidence": None,            # e.g. 0.55 to test the confidence gate
    "R": None,                     # e.g. 0.30 to test the significance rule
    "remove_hard_trigger": False,  # True to test a case with no hard trigger
}
print("Sample scenario loaded.")


# ----------------------------------------------------------------------------------------------------
# B · 3. Validation and metrics functions (deterministic; numbers from the E2 engine)   [notebook cell 45]
# ----------------------------------------------------------------------------------------------------
# Every function keeps the typed contract from B · 2 (inputs → VerificationResult, ImpactScores, ScoredOption, ...).
# The numbers come from the E2 engine, so Section B, the orchestrator and the dashboard give identical results.
_B_ENGINE = {}                 # case_id -> {"raw", "analysis", "resolution", "constraints"}
_B_LAST = {}                   # last verification run (inputs, signals, raw, analysis)
_B_RESOLVED_CONFLICTS = set()  # inputs whose conflict was resolved by applying source precedence
B_ACTION_FROM_ENGINE = {"do_nothing": "do_nothing", "rider_alerts": "rider_alert", "protect_connection": "protect_connection",
                        "reroute": "reroute", "dispatch_second_bus": "dispatch_second_bus"}
RULE_PATH_ALIASES = {"CONFIG.significance_threshold_R": "significance_R", "CONFIG.confidence_threshold": "confidence_threshold"}


def _inputs_key(inputs):
    return zlib.crc32(json.dumps(inputs, sort_keys=True, default=str).encode())


def _b_policy(case=None):
    return APP.PM.get(case.rule_version.lstrip("v")) if case is not None else APP.PM.current


def workflow_inputs_to_raw(inputs, load_signals, name="Section B sample · Route 12 cancellation"):
    """Section B's SAMPLE_INPUTS + Vision Agent load signals → shared raw bundle."""
    svc = inputs["service_feeds"]
    conn = inputs.get("connecting_services", {})
    comm = inputs.get("community_access", {})
    res = inputs.get("operating_resources", {})
    ext = str(svc["route"])
    rid = EXTERNAL_ROUTE_MAP.get(ext, "R3")
    stop_range = (2, 6)
    cancelled = svc.get("status") == "cancelled"
    headway = svc.get("headway_min") or NET["routes"][rid]["headway_min"]
    delay = svc.get("delay_min", 0) or 0
    gap = 2.0 if cancelled else round((headway + delay) / headway, 2)
    sigs = load_signals or []
    boardings = sum(s.passenger_count for s in sigs)
    load = max([s.load_factor for s in sigs], default=0.0)
    passups = sum(s.pass_ups for s in sigs)
    official = [e for e in inputs.get("event_signals", []) if e.get("type") == "official_alert"]
    feedback = inputs.get("rider_feedback", [])
    sources = ["vehicle_apc"] + (["official_alert"] if official else []) + (["public_report"] if feedback else [])
    cause = classify_cause(" ".join(e.get("text", "") for e in official))
    dep = nearest_depot_id(rid, stop_range)
    spec = dict(route_id=rid, stop_range=stop_range, time_period="peak", delay_min=delay, headway_ratio=gap,
                cancelled=cancelled, cancellation_risk=1.0 if cancelled else 0.2, boardings=boardings, load_factor=load,
                passups=passups, event_type=cause, event_severity="none" if cause == "none" else SEVERITY_BY_CAUSE.get(cause, "minor"),
                event_verified=bool(official), sources=sources, public_corroborated=any(f.get("corroborated") for f in feedback),
                last_service=bool(conn.get("last_departure")),
                depot_overrides={dep: {"spare_buses": res.get("spare_buses", 0), "operators": res.get("spare_operators", 0),
                                       "deadhead_min": res.get("deadhead_min")}})
    raw = make_raw(spec, name)
    raw["external_route_id"] = ext
    raw["route_name"] = f"Route {ext} (drawn on synthetic {rid})"
    raw["community"].update(transit_dependence=comm.get("transit_dependence_pct", 0) / 100 if comm.get("transit_dependence_pct") is not None else None,
                            income_disadvantage=comm.get("low_income_share_pct", 0) / 100 if comm.get("low_income_share_pct") is not None else None,
                            disability=comm.get("disability_share_pct", 0) / 100 if comm.get("disability_share_pct") is not None else None,
                            areas=[comm.get("area", "area aggregate")])
    names = comm.get("essential_destinations", [])
    types = [essential_type_of(n) for n in names]
    sev = DEFAULT_POLICY["lookups"]["essential_severity"]
    raw["access"].update(essential_type=max(types, key=lambda t: sev[t]) if types else "none", essential_names=names)
    raw["connections"].update(external_connecting_route=conn.get("rail_line"), transfer_window_min=conn.get("transfer_window_min", 4))
    if res.get("cost_per_bus_hour"):
        raw["resources"]["bus_hour_cost_usd"] = res["cost_per_bus_hour"]
    if inputs.get("incident_history"):
        h = inputs["incident_history"][0]
        raw["incident_history"] = {"past_type": h.get("type"), "average_recovery_min": h.get("avg_recovery_min"),
                                   "previous_response": h.get("past_response")}
    raw["raw_sensor_feed"] = [asdict(s) for s in sigs]
    raw["handoff_from"] = "Section B — transit_agents_workflow.ipynb"
    return raw


def _b_analyze(inputs, load_signals, constraints=None, policy=None):
    raw = workflow_inputs_to_raw(inputs, load_signals)
    res = "accept_precedence" if _inputs_key(inputs) in _B_RESOLVED_CONFLICTS else None
    return raw, run_analysis(raw, policy or APP.PM.current, res, constraints), res


# ---------------- data verification and exception paths ----------------
def verify_data(inputs: dict, load_signals: list) -> VerificationResult:
    """Schema, timestamps, refresh, source identity, completeness and agreement → "ready" or the exception path to take."""
    raw, A, _ = _b_analyze(inputs, load_signals)
    _B_LAST.update(inputs=inputs, signals=load_signals, raw=raw, analysis=A)
    issues = [f"{e['path']}: {e['detail']} → {e['outcome']}" for e in A["exceptions"]]   # incl. uncorroborated public reports
    if A["gate"] == "SAFE_MODE":
        status = "agent_error"
    elif A["conflict"] == "pending":
        status = "conflict"
    elif A["outage"] == "no_backup":
        status = "feed_outage"
    else:
        status = "ready"
    return VerificationResult(status=status, issues=issues)


def handle_feed_outage(case: Case, verification: VerificationResult) -> bool:
    """Freeze the last verified snapshot, request a backup, alert the operator. Returns True if resolved."""
    case.log("Feed outage path", "Exception handler", "Rule / code", "snapshot frozen; backup requested; operator alerted",
             "No backup source available → monitoring only")
    return False


def handle_agent_error(case: Case, verification: VerificationResult) -> bool:
    """Retry once with the same inputs; if it fails again use deterministic safe mode. Returns True if resolved."""
    case.log("Agent error path", "Exception handler", "Rule / code", "retried once with same inputs; failed again",
             "Technical owner and planner alerted; evidence preserved")
    return False


def handle_conflict(case: Case, verification: VerificationResult) -> bool:
    """Quarantine the conflict and apply source precedence (vehicle > official > trusted external > public)."""
    _B_RESOLVED_CONFLICTS.add(_inputs_key(_B_LAST["inputs"]))
    case.log("Conflicting data path", "Operator review", "Person", "source precedence applied",
             "Notebook path assumes the operator accepted precedence; the dashboard (tab 1) records an explicit operator decision")
    return True


# ---------------- confidence, hard triggers, impact and scenario ranking ----------------
def calculate_confidence(inputs: dict, load_signals: list, verification: VerificationResult) -> float:
    """C = 0.40Q + 0.25F + 0.20G + 0.15M (source quality, freshness, agreement, completeness)."""
    _, A, _ = _b_analyze(inputs, load_signals)
    return A["confidence"]["C"]


def detect_official_hard_trigger(inputs: dict) -> Optional[str]:
    """Name of the first hard trigger an OFFICIAL source verifies (lets the case bypass the C gate), else None."""
    _, A, _ = _b_analyze(inputs, _B_LAST.get("signals", []))
    return next((t["trigger"] for t in A["hard_triggers"] if t["official"]), None)


def analyze_impact(case: Case, inputs: dict) -> ImpactScores:
    """R = 0.30D + 0.25H + 0.25K + 0.20E · V = 0.35N + 0.25I + 0.20B + 0.20T · A = 0.35S + 0.30L + 0.20X + 0.15U ·
    P = 0.40R + 0.35V + 0.25A, plus hard trigger and severity tier."""
    raw, A, res = _b_analyze(inputs, case.load_signals, policy=_b_policy(case))
    _B_ENGINE[case.case_id] = {"raw": raw, "analysis": A, "resolution": res, "constraints": {}}
    s = A["scores"]
    trig = A["hard_triggers"][0]["trigger"] if A["hard_triggers"] else None
    return ImpactScores(R=s["R"], V=s["V"], A=s["A"], P=s["P"], hard_trigger=trig, severity=A["tier"] or "moderate")


def evaluate_scenarios(case: Case, options: list) -> list:
    """Si = Ci × [0.30DR + (0.15+0.10P)EB + (0.15+0.10P)AP + (0.25−0.10P)OF + (0.15−0.10P)CE]; tie break AP, OF, cost."""
    A = _B_ENGINE[case.case_id]["analysis"]
    by_id = {r["option_id"]: r for r in A["ranking"]}
    scored = []
    for o in options:
        r = by_id[o.option_id.split(f"{case.case_id}-", 1)[1]]
        scored.append(ScoredOption(option=o, DR=r["DR"], EB=r["EB"], AP=r["AP"], OF=r["OF"], CE=r["CE"], Ci=r["Ci"],
                                   score=r["Si"], rank=r["rank"]))
    scored.sort(key=lambda x: x.rank)
    for i, s in enumerate(scored, 1):
        s.rank = i
    return scored


# ---------------- outcome evaluation ----------------
def workflow_option_as_engine_option(case, option_id):
    eng = _B_ENGINE.get(case.case_id)
    if not eng or not option_id:
        return None
    eid = option_id.split(f"{case.case_id}-", 1)[-1]
    return next((o for o in eng["analysis"]["options"] if o["option_id"] == eid), None)


def evaluate_outcome(case: Case, action_result: Optional[ActionResult]) -> OutcomeReport:
    """Predicted vs observed delay, access, equity, cost and false alarm (separate from the recommender)."""
    o = workflow_option_as_engine_option(case, case.selected_option_id) if action_result else None
    if o is None:
        return OutcomeReport(case_id=case.case_id, path=case.status, predicted={}, observed={},
                             metrics={"note": "no action taken; false-alarm status is set when the case is closed"}, false_alarm=None)
    pr = {"option_id": o["option_id"], "type": o["type"], "delay_reduction_min": o["pred_delay_reduction_min"],
          "access_protection": o["pred_access_protection"], "vulnerable_riders_helped": o["pred_vulnerable_riders_helped"],
          "cost_usd": o["cost_usd"]}
    m = outcome_metrics(pr, simulated_observation(pr, case.case_id))
    gap = round((m["pred_equity_riders"] - m["obs_equity_riders"]) / m["pred_equity_riders"], 2) if m["pred_equity_riders"] else 0.0
    return OutcomeReport(case_id=case.case_id, path="action",
                         predicted={"delay_reduction_min": m["pred_delay_reduction_min"], "vulnerable_riders": m["pred_equity_riders"],
                                    "access": m["pred_access"], "cost_usd": m["pred_cost_usd"]},
                         observed={"delay_reduction_min": m["obs_delay_reduction_min"], "vulnerable_riders": m["obs_equity_riders"],
                                   "access": m["obs_access"], "cost_usd": m["actual_cost_usd"], "rider_reported_impact": m["rider_reported_impact"]},
                         metrics={"delay_reduction_error_min": m["delay_error_min"], "equity_benefit_gap": gap,
                                  "cost_usd": m["actual_cost_usd"], "cost_ratio": m["cost_ratio"], "action_type": o["type"]},
                         false_alarm=m["false_alarm"])


# ---------------- versioned rule changes (only reached after Policy Owner approval + backtest) ----------------
def get_rule(path: str):
    return copy.deepcopy(get_path(APP.PM.current, RULE_PATH_ALIASES.get(path, path)))


def _proposal_change(proposal):
    path = RULE_PATH_ALIASES.get(proposal.parameter, proposal.parameter)
    return {"path": path, "old": get_rule(path), "new": proposal.proposed_value}


def backtest_rule_change(proposal: PolicyProposal, past_cases: list) -> BacktestResult:
    """Re-run every past case under current vs proposed rules; fails if a real disruption would stop being escalated."""
    newp = apply_changes(APP.PM.current, [_proposal_change(proposal)])
    errs = validate_policy(newp)
    if errs:
        return BacktestResult(proposal_id=proposal.proposal_id, passed=False, summary="; ".join(errs))
    df, summary = run_backtest(APP, newp)
    lost = int(((df.gate_current == "SIGNIFICANT") & (df.gate_proposed != "SIGNIFICANT") & (df.known_false_alarm == False)).sum())
    return BacktestResult(proposal_id=proposal.proposal_id, passed=lost == 0,
                          summary=summary + ("" if lost == 0 else " → FAILED: real disruptions would no longer be escalated."))


def apply_rule_change(proposal: PolicyProposal) -> str:
    """Release the approved change as a new immutable version in the shared PolicyManager. Returns the version."""
    change = _proposal_change(proposal)
    old_version = CONFIG["rule_version"]
    v = APP.PM.release(apply_changes(APP.PM.current, [change]), "Policy owner (Section B form)", proposal.rationale, [change])
    RULE_VERSION_HISTORY.append({"version": "v" + v, "parameter": change["path"], "old_value": change["old"],
                                 "new_value": change["new"], "rollback_to": old_version, "proposal_id": proposal.proposal_id})
    APP.log(None, "Policy owner (Section B form)", "policy owner", "Rule release", f"v{v}", f"{change['path']}: {change['old']} → {change['new']}")
    return "v" + v


def rollback_last_rule_change() -> Optional[str]:
    """Re-release the previous rules as a new version; the history entry is kept, not deleted."""
    releases = [r for r in RULE_VERSION_HISTORY if r.get("rollback_to")]
    if not releases:
        print("Nothing to roll back.")
        return None
    last = releases[-1]
    v = APP.PM.rollback(last["rollback_to"].lstrip("v"), "Policy owner (Section B)", f"rollback of {last['version']}")
    RULE_VERSION_HISTORY.append({"version": "v" + v, "parameter": "*", "old_value": last["version"],
                                 "new_value": f"rules of {last['rollback_to']}", "rollback_to": None})
    print(f"Rolled back {last['version']} → rules of {last['rollback_to']} re-released as v{v}")
    return "v" + v


print("Validation and metrics functions ready (deterministic E2 engine) · rules", CONFIG["rule_version"])



# ----------------------------------------------------------------------------------------------------
# B · 4. Agent base class (live LLM notes through the shared client)   [notebook cell 47]
# ----------------------------------------------------------------------------------------------------
class Agent:
    name = "Agent"
    actor_type = "AI agent"
    SYSTEM = ("You are an assistant inside a public transit operations prototype. Use only the facts given. Never invent "
              "numbers, never score or rank options, never approve or recommend actions beyond what the facts state. Be brief.")

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.last_note = None
        self.last_error = None

    def call_llm(self, prompt: str, max_tokens: int = 220) -> str:
        return llm_generate(prompt, self.SYSTEM, max_tokens=max_tokens)

    def maybe_llm(self, task: str, facts=None):
        """Live, non-authoritative LLM note. Returns None when the LLM is off, no facts are given, or the call fails.
        An agent's typed output never depends on this note."""
        self.last_note = None
        if not (self.use_llm and LLM_READY) or facts is None:
            return None
        try:
            self.last_note = self.call_llm(f"Task: {task}\nFacts (JSON): {json.dumps(facts, default=str)}")
        except LLMUnavailable as e:
            self.last_error = str(e)
        return self.last_note



# ----------------------------------------------------------------------------------------------------
# B · Vision Agent   [notebook cell 49]
# ----------------------------------------------------------------------------------------------------
class VisionAgent(Agent):
    name = "Vision Agent"

    def run(self, raw_feed: list) -> list:
        signals = []
        for r in raw_feed:
            lf = round(r["person_count"] / r["capacity"], 2)
            # same rule as the policy hard trigger (load ≥ 1.0 with at least one pass-up)
            if lf >= CONFIG["unboardable_load_factor"] and r["pass_ups"] >= 1:
                level = "unboardable"      # riders left behind at or over capacity
            elif lf >= 1.0 or r["pass_ups"] >= 1:
                level = "crowded"
            elif lf >= 0.8:
                level = "busy"
            else:
                level = "normal"
            signals.append(LoadSignal(stop_id=r["stop_id"], route=r["route"], timestamp=r["timestamp"],
                                      passenger_count=r["person_count"], capacity=r["capacity"],
                                      load_factor=lf, pass_ups=r["pass_ups"], crowding_level=level))
        return signals


# ----------------------------------------------------------------------------------------------------
# B · Signal Scout Agent (validates every input source, then opens a case with a source ticket)   [notebook cell 51]
# ----------------------------------------------------------------------------------------------------
import os, json, uuid
from dataclasses import asdict

# The source checks (SOURCE_MODELS, USE_SOURCE_LLMS, check_sources) are defined in E2 so that every case path
# (orchestrator, scenario gallery, dashboard and this agent) runs the same checks before a case is opened.


class SignalScoutAgent(Agent):
    name = "Signal Scout Agent"

    def run(self, inputs: dict, load_signals: list,
            parent_case_id: Optional[str] = None) -> Optional[Case]:

        # 1. Every source must be known, structured and of a sane size before the case is opened
        sources = {**inputs, "load_signals": [asdict(s) for s in load_signals]}
        ticket = check_sources(sources)

        # 2. Possible disruption? (policy values, not hard-coded numbers)
        svc = inputs["service_feeds"]
        reasons = []
        if svc.get("status") == "cancelled":
            reasons.append(f"trip {svc['trip_id']} cancelled")
        if (svc.get("delay_min") or 0) >= CONFIG["delay_hard_trigger_min"]:
            reasons.append(f"delay {svc['delay_min']} min")
        crowded = [s for s in load_signals if s.crowding_level in ("crowded", "unboardable")]
        if crowded:
            reasons.append("crowding at " + ", ".join(
                f"{s.stop_id} ({s.crowding_level})" for s in crowded))
        if any(e.get("type") == "official_alert" for e in inputs.get("event_signals", [])):
            reasons.append("official alert received")
        if not reasons:
            return None

        case = Case(
            case_id="CASE-" + uuid.uuid4().hex[:6].upper(),
            route=svc["route"],
            disruption_type="cancellation" if svc.get("status") == "cancelled" else "delay_or_crowding",
            description=f"Route {svc['route']}: " + "; ".join(reasons),
            opened_at=stamp(),
            rule_version=CONFIG["rule_version"],
            parent_case_id=parent_case_id,
            load_signals=load_signals,
            scout_ticket=ticket
        )
        if ticket["flagged_for_review"]:
            case.log("Signal Scout Agent", "Signal Scout Agent", "AI agent", "sources flagged for human review (non-authoritative)",
                     "model said invalid: " + ", ".join(ticket["flagged_for_review"]) + "; the reviewer decides whether to trust them")
        # 3. Live, non-authoritative case note for the operator (never used for routing)
        self.maybe_llm("Write a two-sentence case note describing the possible disruption for an operator.",
                       {"route": svc["route"], "reasons": reasons, "rule_version": CONFIG["rule_version"]})
        return case



# ----------------------------------------------------------------------------------------------------
# B · Planner Copilot Agent   [notebook cell 53]
# ----------------------------------------------------------------------------------------------------
class PlannerCopilotAgent(Agent):
    """Reads the reviewer's constraints ("no dispatch", "budget $300", "alternate depot", ...) and returns feasible options.
    Feasibility, resources and costs come from the E2 planner (resource, road, policy and safety checks). Assigns no scores."""
    name = "Planner Copilot Agent"

    def run(self, case: Case, inputs: dict, constraints: list) -> list:
        text = " ".join(constraints).lower()
        excluded = set()
        if any(k in text for k in ("no dispatch", "no extra bus", "no second bus")):
            excluded.add("dispatch_second_bus")
        if "no reroute" in text:
            excluded.add("reroute")
        if "no connection" in text:
            excluded.add("protect_connection")
        if "no alert" in text or "no rider alert" in text:
            excluded.add("rider_alerts")
        cons = {"excluded_options": sorted(excluded)}
        m = re.search(r"budget\s*\$?\s*(\d+)", text)
        if m:
            cons["budget_cap_usd"] = float(m.group(1))
        if "alternate depot" in text:
            cons["allow_alternate_depot"] = True
        eng = _B_ENGINE[case.case_id]
        A = run_analysis(eng["raw"], _b_policy(case), eng["resolution"], cons)
        eng.update(analysis=A, constraints=cons)
        options = []
        for o in A["options"]:
            if not o["feasible"]:
                case.log("Planner Copilot Agent", "Planner Copilot Agent", "AI agent", f"infeasible: {o['label']}", o["infeasible_reason"])
                continue
            options.append(Option(option_id=f"{case.case_id}-{o['option_id']}", action=B_ACTION_FROM_ENGINE[o["type"]],
                                  description=f"{o['label']} — {o['notes']}",
                                  resources_needed={"est_cost": o["cost_usd"], "parties": o["parties"]},
                                  constraints_checked=["resources", "roads", "policy", "safety"] +
                                                      [f"planner excluded {x}" for x in sorted(excluded)]))
        return options




# ----------------------------------------------------------------------------------------------------
# B · Explanation Agent   [notebook cell 55]
# ----------------------------------------------------------------------------------------------------
class ExplanationAgent(Agent):
    name = "Explanation Agent"

    def run(self, case: Case, ranked: list) -> Brief:
        table =[{"rank": s.rank, "option_id": s.option.option_id, "action": s.option.action,
                  "description": s.option.description, "score_Si": s.score, "DR": s.DR, "EB": s.EB,
                  "AP": s.AP, "OF": s.OF, "CE": s.CE, "Ci": s.Ci,
                  "est_cost": s.option.resources_needed.get("est_cost", 0)} for s in ranked]
        feasible = [s for s in ranked if s.option.action != "do_nothing"]
        notes = [f"Data confidence C = {case.confidence:.2f}"]
        notes += [f"{s.option.action}: option confidence {s.Ci:.2f}" for s in ranked if s.Ci < 0.8]
        if not feasible:
            notes.append("No feasible option besides doing nothing: escalated for a human decision.")
        top = ranked[0]
        narrative = self.maybe_llm(
            "Explain these deterministic results to a human reviewer in at most 4 sentences. Describe the alternatives "
            "before the recommendation, which is always the rank-1 option, and say the human reviewer decides.",
            {"options_ranked": table, "confidence_C": case.confidence,
             "impact": {"R": case.impact.R, "V": case.impact.V, "A": case.impact.A, "P": case.impact.P}})
        if narrative and top.option.action.replace("_", " ") not in narrative.lower().replace("_", " ") \
                and top.option.description.lower()[:20] not in narrative.lower():
            narrative = None                      # guard: a narrative that does not name the rank-1 option is discarded
        if narrative:
            notes.append(f"Narrative ({llm_source()}, non-authoritative): {narrative}")
        return Brief(
            case_id=case.case_id,
            summary=f"{case.description}. Impact R={case.impact.R:.2f}, V={case.impact.V:.2f}, "
                    f"A={case.impact.A:.2f}, priority P={case.impact.P:.2f}. Severity: {case.severity}.",
            options_table=table,
            recommendation=f"Highest ranked: #{top.rank} {top.option.description} (Si = {top.score})",
            uncertainty_notes=notes,
            sources=["service feed", "vision load signals", "official alert", "community access (area level)"],
        )


# ----------------------------------------------------------------------------------------------------
# B · Insight Agent   [notebook cell 57]
# ----------------------------------------------------------------------------------------------------
class InsightAgent(Agent):
    name = "Insight Agent"

    def receive_rejection(self, proposal: PolicyProposal, reason: str):
        """Input from Policy Owner Approval when a proposal is rejected. The same change is not resubmitted."""
        REJECTED_PROPOSALS[(proposal.parameter, proposal.proposed_value)] = reason

    def run(self, outcome_history: list) -> list:
        candidates = []
        actions = [o for o in outcome_history if o.path == "action"]
        # only dispatch outcomes can justify a dispatch change, and the target is the dispatch option certainty
        dispatch = [o for o in actions if o.metrics.get("action_type") == "dispatch_second_bus"]
        errors = [o.metrics.get("delay_reduction_error_min", 0) for o in dispatch]
        if errors and max(errors) >= 3:
            cur = get_rule("lookups.option_certainty.dispatch_second_bus")
            candidates.append(("lookups.option_certainty.dispatch_second_bus", round(max(0.5, cur - 0.05), 2),
                               f"Observed delay reduction was {max(errors)} min below prediction.",
                               [o.case_id for o in dispatch]))
        known = [o for o in outcome_history if o.false_alarm is not None]
        if known and sum(o.false_alarm for o in known) / len(known) > 0.3:
            candidates.append(("significance_R", round(CONFIG["significance_threshold_R"] + 0.05, 2),   # policy path
                               "False alarm rate above 30 percent.", [o.case_id for o in known]))
        proposals = []
        for param, value, why, evidence in candidates:
            current = get_rule(param)
            if current == value or (param, value) in REJECTED_PROPOSALS:
                continue   # already in force, or the policy owner already said no
            note = self.maybe_llm("Explain this proposed rule change to a policy owner in 2 sentences and state that "
                                  "nothing is applied until they approve it and it passes a backtest.",
                                  {"parameter": param, "current": current, "proposed": value, "reason": why,
                                   "evidence_cases": len(evidence)})
            proposals.append(PolicyProposal(proposal_id="PROP-" + uuid.uuid4().hex[:5].upper(), parameter=param,
                                            current_value=current, proposed_value=value,
                                            rationale=why + (f" — LLM summary ({llm_source()}): {note}" if note else ""),
                                            evidence_cases=evidence))
        return proposals


print("Agents defined.")


# ----------------------------------------------------------------------------------------------------
# B · 5. Rule steps   [notebook cell 59]
# ----------------------------------------------------------------------------------------------------
def passes_confidence_gate(C: float, official_trigger: Optional[str]) -> bool:
    """Proceed when C >= threshold, or when an official source verifies a hard trigger."""
    return C >= CONFIG["confidence_threshold"] or official_trigger is not None


def is_significant(impact: ImpactScores) -> bool:
    """Significant if a hard trigger applies OR R >= threshold."""
    return impact.hard_trigger is not None or impact.R >= CONFIG["significance_threshold_R"]


def is_high_impact(option: Option) -> bool:
    return option.action in CONFIG["high_impact_actions"]


def review_deadline_for(severity: str) -> datetime:
    return now() + timedelta(minutes=CONFIG["review_deadline_minutes"].get(severity, 20))


def safe_default(case: Case, reason: str) -> dict:
    """Send verified rider information and keep monitoring. No dispatch, rerouting or live control."""
    case.status = "safe_default"
    msg = {"rider_message": f"Route {case.route}: service disruption. Check verified service updates before travelling.",
           "monitoring": True, "reason": reason}
    case.log("Safe Default", "Safe Default", "Rule / code", "verified rider information + monitoring", reason)
    return msg


def continue_monitoring(case: Case, reason: str) -> dict:
    """Record confidence, priority and reason. Recheck when a feed changes. No action taken."""
    case.status = "monitoring"
    rec = {"confidence": case.confidence, "priority": case.impact.P if case.impact else None, "reason": reason}
    case.log("Continue Monitoring", "Continue Monitoring", "Rule / code", "monitoring record", reason)
    return rec


EXCEPTION_HANDLERS = {"feed_outage": handle_feed_outage, "agent_error": handle_agent_error, "conflict": handle_conflict}


# ---------- Demo controls for the Section B runner only (see DEMO_OVERRIDES; off by default) ----------
def demo_verification(v: VerificationResult, attempt: int) -> VerificationResult:
    forced = DEMO_OVERRIDES["verification_status"]
    if forced and attempt == 0:
        return VerificationResult(status=forced, issues=[f"demo: forced {forced}"])
    return v


def demo_resolved(resolved: bool) -> bool:
    return resolved if DEMO_OVERRIDES["exception_resolved"] is None else bool(DEMO_OVERRIDES["exception_resolved"])


def demo_confidence(C: float) -> float:
    return C if DEMO_OVERRIDES["confidence"] is None else DEMO_OVERRIDES["confidence"]


def demo_impact(i: ImpactScores) -> ImpactScores:
    if DEMO_OVERRIDES["R"] is not None:
        i.R = DEMO_OVERRIDES["R"]
    if DEMO_OVERRIDES["remove_hard_trigger"]:
        i.hard_trigger = None
    return i


print("Rules defined.")


# ----------------------------------------------------------------------------------------------------
# B · 6. Human in the loop forms (ipywidgets)   [notebook cell 61]
# ----------------------------------------------------------------------------------------------------
def run_safely(fn, msg_widget):
    """Widget callbacks in Colab hide errors. Show them in the form instead."""
    def handler(_):
        try:
            fn()
        except Exception as e:
            msg_widget.value = f"<span style='color:#C4496B'><b>Error:</b> {type(e).__name__}: {e}</span>"
            raise
    return handler


class ReviewForm:
    """Human Review form. Output: ReviewDecision passed to on_submit."""
    role = "Human Review"
    choices = [("Approve", "approve"), ("Revise", "revise"), ("Reject", "reject")]

    def __init__(self, case: Case, on_submit: Callable, preselected: Optional[str] = None,
                 exclude_reviewer: Optional[str] = None):
        self.case, self.on_submit, self.opened_at, self.done = case, on_submit, now(), False
        self.exclude_reviewer = exclude_reviewer
        labels = [(f"#{s.rank}  {s.option.description}  (Si {s.score})", s.option.option_id) for s in case.ranked_options]
        self.name_w = widgets.Text(placeholder="Your name (required)", description="Reviewer",
                                   layout=widgets.Layout(width="60%"))
        self.option_w = widgets.Dropdown(options=labels, value=preselected or labels[0][1],
                                         description="Option", layout=widgets.Layout(width="95%"))
        self.decision_w = widgets.ToggleButtons(options=self.choices, description="Decision")
        self.reason_w = widgets.Textarea(placeholder="Typed reason (required)", description="Reason",
                                         layout=widgets.Layout(width="95%", height="60px"))
        self.constraints_w = widgets.Text(placeholder="e.g. no dispatch (revise only)", description="Constraints",
                                          layout=widgets.Layout(width="95%"))
        self.check_w = widgets.Checkbox(description="I reviewed every alternative, score and uncertainty note",
                                        layout=widgets.Layout(width="95%"))
        self.submit_w = widgets.Button(description="Submit decision", button_style="primary")
        self.timeout_w = widgets.Button(description="Simulate timeout", button_style="warning")
        self.msg_w = widgets.HTML()
        self.submit_w.on_click(run_safely(lambda: self.submit(
            self.decision_w.value, self.option_w.value, self.reason_w.value, self.constraints_w.value,
            self.check_w.value, self.name_w.value), self.msg_w))
        self.timeout_w.on_click(run_safely(lambda: self.submit("timeout"), self.msg_w))

    def widget(self):
        deadline = self.case.review_deadline.astimezone(LOCAL_TZ).strftime("%H:%M:%S %Z") if self.case.review_deadline else "none"
        head = widgets.HTML(f"<h4 style='margin:6px 0'>🟦 {self.role} | case {self.case.case_id} | deadline {deadline}</h4>")
        return widgets.VBox([head, self.name_w, self.option_w, self.decision_w, self.reason_w, self.constraints_w,
                             self.check_w, widgets.HBox([self.submit_w, self.timeout_w]), self.msg_w],
                            layout=widgets.Layout(border="2px solid #0E2E4F", padding="8px", margin="8px 0"))

    def submit(self, decision: str, option_id: Optional[str] = None, reason: str = "",
               constraints: str = "", checklist: bool = False, reviewer: str = "") -> Optional[ReviewDecision]:
        if self.done:
            return None
        if decision != "timeout" and self.case.review_deadline and now() > self.case.review_deadline:
            decision, reason = "timeout", "Deadline passed before submit"
        if decision != "timeout":
            problems = []
            if not reviewer.strip():
                problems.append("enter your name")
            elif self.exclude_reviewer and reviewer.strip().lower() == self.exclude_reviewer.strip().lower():
                problems.append("the second reviewer must be a different person from the first")
            if decision == "approve" and self.case.option(option_id) is None:
                problems.append("choose an option to approve")
            if not checklist:
                problems.append("tick the checklist")
            if len(reason.strip()) < CONFIG["min_reason_chars"]:
                problems.append(f"type a reason of at least {CONFIG['min_reason_chars']} characters")
            if decision == "revise" and not constraints.strip():
                problems.append("enter the new constraints for the planner")
            # INTEGRATION (Sections C + D): ethics-training and conflict-of-interest gate
            if reviewer.strip():
                opt = workflow_option_as_engine_option(self.case, option_id) if decision == "approve" else None
                gp, gn = governance_check(reviewer, decision.capitalize(), option=opt)
                problems += gp
                for n_ in gn:
                    self.case.log("Reviewer gate (C+D)", "Governance gate", "Rule / code", "note", n_)
            if problems:
                self.msg_w.value = "<span style='color:#C4496B'>Cannot submit: " + "; ".join(problems) + ".</span>"
                return None
        self.done = True
        for w in (self.name_w, self.option_w, self.decision_w, self.reason_w, self.constraints_w, self.check_w,
                  self.submit_w, self.timeout_w):
            w.disabled = True
        rd = ReviewDecision(reviewer_role=self.role, reviewer_name=reviewer.strip() or "(no response)",
                            decision=decision, selected_option_id=option_id if decision == "approve" else None,
                            reason=reason or "No decision before deadline", new_constraints=constraints,
                            checklist_confirmed=checklist,
                            seconds_to_decide=round((now() - self.opened_at).total_seconds(), 1), timestamp=stamp())
        self.case.decisions.append(asdict(rd))
        self.case.log(self.role, rd.reviewer_name, "Person", rd.decision,
                      f"{rd.reason} (decided in {rd.seconds_to_decide}s)")
        self.msg_w.value = f"<b>Submitted:</b> {decision}"
        self.on_submit(rd)
        return rd


class SecondReviewerForm(ReviewForm):
    """Second Reviewer for high impact actions. Same deadline. Can approve or reject only."""
    role = "Second Reviewer"
    choices = [("Approve", "approve"), ("Reject", "reject")]

    def __init__(self, case: Case, on_submit: Callable, preselected: str, first_reviewer: str):
        super().__init__(case, on_submit, preselected, exclude_reviewer=first_reviewer)
        self.option_w.disabled = True
        self.constraints_w.layout.display = "none"


class ImplementActionForm:
    """Operations staff confirm what happened. Output: ActionResult passed to on_submit."""
    role = "Implement Action"

    def __init__(self, case: Case, on_submit: Callable):
        self.case, self.on_submit, self.done = case, on_submit, False
        opt = case.option(case.selected_option_id)
        self.head = widgets.HTML(f"<h4 style='margin:6px 0'>🟦 Implement Action | {opt.option.description}</h4>"
                                 "<i>Carried out by operations staff outside this prototype.</i>")
        self.status_w = widgets.ToggleButtons(options=[("Completed", "completed"), ("Partial", "partial"), ("Failed", "failed")],
                                              description="Result")
        self.alert_w = widgets.Checkbox(value=True, description="Rider alert sent")
        self.notes_w = widgets.Textarea(placeholder="Notes (required if partial or failed)", description="Notes",
                                        layout=widgets.Layout(width="95%", height="50px"))
        self.submit_w = widgets.Button(description="Confirm result", button_style="primary")
        self.msg_w = widgets.HTML()
        self.submit_w.on_click(run_safely(lambda: self.submit(self.status_w.value, self.alert_w.value,
                                                              self.notes_w.value), self.msg_w))

    def widget(self):
        return widgets.VBox([self.head, self.status_w, self.alert_w, self.notes_w, self.submit_w, self.msg_w],
                            layout=widgets.Layout(border="2px solid #0E2E4F", padding="8px", margin="8px 0"))

    def submit(self, status: str, rider_alert_sent: bool = True, notes: str = "") -> Optional[ActionResult]:
        if self.done:
            return None
        if status in ("partial", "failed") and not notes.strip():
            self.msg_w.value = "<span style='color:#C4496B'>Describe what went wrong so the reviewer can decide.</span>"
            return None
        self.done = True
        for w in (self.status_w, self.alert_w, self.notes_w, self.submit_w):
            w.disabled = True
        ar = ActionResult(option_id=self.case.selected_option_id, status=status,
                          rider_alert_sent=rider_alert_sent, notes=notes, timestamp=stamp())
        self.case.action_result = ar
        self.case.log(self.role, "Operations staff", "Person", status, notes)
        self.on_submit(ar)
        return ar


class PolicyOwnerForm:
    """Policy owner approves or rejects an Insight Agent proposal. Output: PolicyDecision."""
    role = "Policy Owner Approval"

    def __init__(self, proposal: PolicyProposal, case: Case, on_submit: Callable):
        self.proposal, self.case, self.on_submit, self.done = proposal, case, on_submit, False
        p = proposal
        self.head = widgets.HTML(
            f"<h4 style='margin:6px 0'>🟦 Policy Owner Approval | {p.proposal_id}</h4>"
            f"<b>{p.parameter}</b>: {p.current_value} → {p.proposed_value}<br><i>{p.rationale}</i><br>"
            "Approved changes are backtested on past cases, then released as a new rule version with rollback.")
        self.name_w = widgets.Text(placeholder="Your name (required)", description="Policy owner",
                                   layout=widgets.Layout(width="60%"))
        self.decision_w = widgets.ToggleButtons(options=[("Approve", "approve"), ("Reject", "reject")], description="Decision")
        self.reason_w = widgets.Textarea(placeholder="Typed reason (required)", description="Reason",
                                         layout=widgets.Layout(width="95%", height="50px"))
        self.submit_w = widgets.Button(description="Submit decision", button_style="primary")
        self.msg_w = widgets.HTML()
        self.submit_w.on_click(run_safely(lambda: self.submit(self.decision_w.value, self.reason_w.value,
                                                              self.name_w.value), self.msg_w))

    def widget(self):
        return widgets.VBox([self.head, self.name_w, self.decision_w, self.reason_w, self.submit_w, self.msg_w],
                            layout=widgets.Layout(border="2px solid #0E2E4F", padding="8px", margin="8px 0"))

    def submit(self, decision: str, reason: str = "", owner: str = "") -> Optional[PolicyDecision]:
        if self.done:
            return None
        problems = []
        if not owner.strip():
            problems.append("enter your name")
        if len(reason.strip()) < CONFIG["min_reason_chars"]:
            problems.append(f"type a reason of at least {CONFIG['min_reason_chars']} characters")
        if owner.strip():   # INTEGRATION (Sections C + D)
            gp, _ = governance_check(owner, decision.capitalize(), proposal={"title": self.proposal.parameter,
                                     "changes": [{"path": self.proposal.parameter, "old": self.proposal.current_value, "new": self.proposal.proposed_value}]})
            problems += gp
        if problems:
            self.msg_w.value = "<span style='color:#C4496B'>Cannot submit: " + "; ".join(problems) + ".</span>"
            return None
        self.done = True
        for w in (self.name_w, self.decision_w, self.reason_w, self.submit_w):
            w.disabled = True
        pd_ = PolicyDecision(proposal_id=self.proposal.proposal_id, decision=decision, reason=reason, timestamp=stamp())
        self.case.log(self.role, owner.strip(), "Person", decision, f"{self.proposal.parameter}: {reason}")
        self.on_submit(pd_)
        return pd_


print("Human in the loop forms defined.")


# ----------------------------------------------------------------------------------------------------
# B · 7. Agent-level runner (Section B's typed agents, Signal Scout coordination)   [notebook cell 63]
# ----------------------------------------------------------------------------------------------------
class Workflow:
    def __init__(self, inputs: dict, raw_feed: list, use_llm: bool = False):
        self.inputs, self.raw_feed = inputs, raw_feed
        self.vision, self.scout = VisionAgent(use_llm), SignalScoutAgent(use_llm)
        self.planner, self.explainer, self.insight = PlannerCopilotAgent(use_llm), ExplanationAgent(use_llm), InsightAgent(use_llm)
        self.case: Optional[Case] = None
        self.active_form = None
        self.out = widgets.Output()

    # ---------- display helpers ----------
    def say(self, kind: str, title: str, body: str = ""):
        colors = {"ai": "#DCEFEC", "code": "#FFF3DC", "human": "#DCE3EC", "data": "#EFE9F8", "decision": "#FDE6EC"}
        with self.out:
            display(HTML(f"<div style='background:{colors[kind]};padding:6px 10px;margin:4px 0;border-radius:6px'>"
                         f"<b>{title}</b><br>{body}</div>"))

    def show(self, obj):
        with self.out:
            display(obj)

    # ---------- Section 1 and 2: detect and check ----------
    def start(self, parent_case_id: Optional[str] = None):
        display(self.out)
        self.run_detection(parent_case_id)

    def run_detection(self, parent_case_id: Optional[str] = None):
        signals = self.vision.run(self.raw_feed)
        self.say("ai", "Vision Agent → load signals", "<br>".join(
            f"{s.stop_id}: load {s.load_factor}, pass ups {s.pass_ups}, {s.crowding_level}" for s in signals))

        case = self.scout.run(self.inputs, signals, parent_case_id)
        if case is None:
            self.say("ai", "Signal Scout Agent → no disruption detected")
            return
        self.case = case
        ALL_CASES.append(case)
        case.log("Vision Agent", "Vision Agent", "AI agent", f"{len(signals)} load signals")
        case.log("Signal Scout Agent", "Signal Scout Agent", "AI agent", "case created", case.description)
        self.say("ai", f"Signal Scout Agent → case {case.case_id} created", case.description)
        if self.scout.last_note:
            case.log("Signal Scout Agent", "Signal Scout Agent", "AI agent", "LLM case note (non-authoritative)",
                     f"{llm_source()}: {self.scout.last_note}")
            self.say("ai", f"Signal Scout note ({llm_source()}, non-authoritative)", self.scout.last_note)
        self.verify_and_assess()

    def verify_and_assess(self):
        case = self.case
        while True:
            v = demo_verification(verify_data(self.inputs, case.load_signals), case.reverify_attempts)
            case.verification = v
            case.log("Data Verification", "Data Verification", "Rule / code", v.status, "; ".join(v.issues))
            self.say("code", f"Data Verification → {v.status}", "; ".join(v.issues))
            if v.status == "ready":
                break
            resolved = demo_resolved(EXCEPTION_HANDLERS[v.status](case, v))
            case.log(f"Exception path: {v.status}", "Exception handler", "Rule / code", f"resolved={resolved}")
            self.say("code", f"Exception path ({v.status}) → resolved: {resolved}")
            if not resolved and v.status == "agent_error":
                # the flowchart sends a repeated agent failure to deterministic safe mode → Safe Default
                return self.to_safe_default("Agent error: deterministic safe mode")
            if not resolved:
                return self.to_monitoring("Unresolved data problem")
            case.reverify_attempts += 1
            if case.reverify_attempts > CONFIG["max_reverify_attempts"]:
                return self.to_monitoring("Reverify limit reached")

        case.confidence = demo_confidence(calculate_confidence(self.inputs, case.load_signals, case.verification))
        official = detect_official_hard_trigger(self.inputs)
        gate = passes_confidence_gate(case.confidence, official)
        case.log("Confidence Calculation", "Confidence Calculation", "Rule / code",
                 f"C={case.confidence:.2f}", f"official hard trigger: {official}")
        self.say("code", f"Confidence Calculation → C = {case.confidence:.2f}",
                 f"Official hard trigger: {official}. Gate passed: {gate}")
        if not gate:
            return self.to_monitoring("Confidence below threshold")

        case.impact = demo_impact(analyze_impact(case, self.inputs))
        case.severity = case.impact.severity
        i = case.impact
        case.log("Impact Analyzer", "Impact Analyzer", "Rule / code", f"R={i.R} V={i.V} A={i.A} P={i.P}",
                 f"hard trigger: {i.hard_trigger}")
        self.say("code", "Impact Analyzer → scores", f"R={i.R}, V={i.V}, A={i.A}, P={i.P}, hard trigger: {i.hard_trigger}")

        sig = is_significant(i)
        case.log("Significant?", "Significance rule", "Decision", "yes" if sig else "no")
        self.say("decision", f"Significant? → {'yes' if sig else 'no'}")
        if not sig:
            return self.to_safe_default("Below significance threshold")
        self.plan()

    # ---------- Section 3: plan and decide ----------
    def plan(self):
        case = self.case
        case.options = self.planner.run(case, self.inputs, case.constraints)
        case.log("Planner Copilot Agent", "Planner Copilot Agent", "AI agent", f"{len(case.options)} options (no scores)")
        self.say("ai", f"Planner Copilot Agent → {len(case.options)} options",
                 "<br>".join(o.description for o in case.options))
        if all(o.action == "do_nothing" for o in case.options):
            self.say("ai", "Planner Copilot Agent → no feasible option: escalating to review")
            case.log("Planner Copilot Agent", "Planner Copilot Agent", "AI agent", "no feasible option: escalate")

        case.ranked_options = evaluate_scenarios(case, case.options)
        case.log("Scenario Evaluator", "Scenario Evaluator", "Rule / code",
                 f"ranked {len(case.ranked_options)}; top {case.ranked_options[0].option.action}")

        case.brief = self.explainer.run(case, case.ranked_options)
        case.log("Explanation Agent", "Explanation Agent", "AI agent", "brief ready")
        b = case.brief
        self.say("ai", "Explanation Agent → brief for the reviewer", b.summary)
        self.show(pd.DataFrame(b.options_table).set_index("rank"))
        self.say("ai", "Recommendation", b.recommendation + "<br><i>" + "<br>".join(b.uncertainty_notes) + "</i>")
        self.open_review()

    def open_review(self):
        """Every new brief starts a new review deadline. The second reviewer shares it."""
        # a revision keeps the original deadline unless the policy flag says otherwise (policy flag)
        if self.case.review_deadline is None or self.case.revision_count == 0 or CONFIG["deadline_resets_on_revision"]:
            self.case.review_deadline = review_deadline_for(self.case.severity)
        self.active_form = ReviewForm(self.case, self.on_review)
        self.show(self.active_form.widget())

    def on_review(self, rd: ReviewDecision):
        case = self.case
        if rd.decision in ("reject", "timeout"):
            return self.to_safe_default(f"Human Review: {rd.decision}")
        if rd.decision == "revise":
            case.revision_count += 1
            if case.revision_count > CONFIG["max_revisions"]:
                return self.to_safe_default("Revision limit reached")
            case.constraints.append(rd.new_constraints)
            self.say("human", f"Revision {case.revision_count} of {CONFIG['max_revisions']}",
                     f"New constraints: {rd.new_constraints}. Rebuilding options; rules unchanged.")
            return self.plan()
        case.selected_option_id = rd.selected_option_id
        opt = case.option(rd.selected_option_id).option
        if is_high_impact(opt):
            self.say("human", "High impact action → second reviewer required")
            self.active_form = SecondReviewerForm(case, self.on_second_review, preselected=opt.option_id,
                                                  first_reviewer=rd.reviewer_name)
            return self.show(self.active_form.widget())
        self.open_implement()

    def on_second_review(self, rd: ReviewDecision):
        if rd.decision != "approve":
            return self.to_safe_default(f"Second Reviewer: {rd.decision}")
        self.open_implement()

    def open_implement(self):
        self.case.status = "approved"
        self.active_form = ImplementActionForm(self.case, self.on_action_result)
        self.show(self.active_form.widget())

    def on_action_result(self, ar: ActionResult):
        if ar.status in ("failed", "partial"):
            self.say("human", f"Action {ar.status} → returned to Human Review", ar.notes)
            self.case.review_deadline = review_deadline_for(self.case.severity)
            self.active_form = ReviewForm(self.case, self.on_review)
            return self.show(self.active_form.widget())
        self.case.status = "implemented"
        self.learn(ar)

    # ---------- Fallbacks ----------
    def to_safe_default(self, reason: str):
        msg = safe_default(self.case, reason)
        self.say("code", "Safe Default", f"{reason}<br>Rider message: {msg['rider_message']}")
        self.learn(None)

    def to_monitoring(self, reason: str):
        continue_monitoring(self.case, reason)
        self.say("code", "Continue Monitoring", reason)
        self.learn(None)

    def re_escalate(self, updated_inputs: Optional[dict] = None, updated_feed: Optional[list] = None):
        """Safe default / monitoring re-escalate to the Signal Scout when conditions change (connector A)."""
        if self.case is None or self.case.status not in ("safe_default", "monitoring"):
            print("Re-escalation applies only to cases in safe default or monitoring.")
            return
        self.case.log("Re-escalate", "Signal Scout Agent", "AI agent", "re-escalated")
        self.inputs = updated_inputs or self.inputs
        self.raw_feed = updated_feed or self.raw_feed
        self.run_detection(parent_case_id=self.case.case_id)

    # ---------- Learn and improve ----------
    def learn(self, ar: Optional[ActionResult]):
        case = self.case
        case.outcome = evaluate_outcome(case, ar)
        OUTCOME_HISTORY.append(case.outcome)
        case.log("Outcome Evaluator", "Outcome Evaluator", "Rule / code", case.outcome.path, json.dumps(case.outcome.metrics))
        self.say("code", "Outcome Evaluator → report", f"Path: {case.outcome.path}. Metrics: {case.outcome.metrics}")

        proposals = self.insight.run(OUTCOME_HISTORY)
        case.log("Insight Agent", "Insight Agent", "AI agent", f"{len(proposals)} proposal(s)")
        if not proposals:
            self.say("ai", "Insight Agent → no rule changes proposed")
            return self.finish()
        self.pending = list(proposals)
        self.next_proposal()

    def next_proposal(self):
        if not self.pending:
            return self.finish()
        p = self.pending.pop(0)
        self.say("ai", f"Insight Agent → proposal {p.proposal_id}", f"{p.parameter}: {p.current_value} → {p.proposed_value}")
        self.active_form = PolicyOwnerForm(p, self.case, lambda d, p=p: self.on_policy_decision(p, d))
        self.show(self.active_form.widget())

    def on_policy_decision(self, p: PolicyProposal, d: PolicyDecision):
        if d.decision == "reject":
            self.insight.receive_rejection(p, d.reason)
            self.say("human", "Policy owner rejected → reason returned to Insight Agent, current rules kept", d.reason)
            self.case.log("Insight Agent", "Insight Agent", "AI agent", "received rejection reason", d.reason)
            return self.next_proposal()
        bt = backtest_rule_change(p, ALL_CASES)
        self.case.log("Backtest", "Backtest", "Rule / code", f"passed={bt.passed}", bt.summary)
        if not bt.passed:
            self.say("code", "Backtest failed → current rules kept", bt.summary)
            return self.next_proposal()
        old = CONFIG["rule_version"]
        new = apply_rule_change(p)
        self.case.log("Rule release", "Scoring rules", "Rule / code", new,
                      f"{p.parameter}: {p.current_value} → {p.proposed_value}; rollback to {old}")
        self.say("code", f"Backtest passed → released rule version {new}",
                 f"{p.parameter}: {p.current_value} → {p.proposed_value}. Rollback available to {old}. {bt.summary}")
        self.next_proposal()

    def finish(self):
        self.active_form = None
        self.say("data", f"Case {self.case.case_id} complete → status: {self.case.status}",
                 "Run the <b>Audit trail and outputs</b> cell to see every step.")


print("Workflow defined.")


# ----------------------------------------------------------------------------------------------------
# B · 9. Audit trail and outputs   [notebook cell 67]
# ----------------------------------------------------------------------------------------------------
AUDIT_COLUMNS = ["time", "case_id", "stage", "actor_type", "actor", "output", "note", "rule_version"]


def audit_table(cases=None) -> pd.DataFrame:
    rows = [r for c in (cases if cases is not None else ALL_CASES) for r in c.audit]
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def decisions_table(cases=None) -> pd.DataFrame:
    rows = [dict(case_id=c.case_id, **d) for c in (cases if cases is not None else ALL_CASES) for d in c.decisions]
    return pd.DataFrame(rows)


def export_case(case: Case, folder: str = OUTPUT_DIR) -> str:
    path = f"{folder}/{case.case_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(case), f, indent=2, default=str)
    return path
