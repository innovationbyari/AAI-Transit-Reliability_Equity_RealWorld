"""Section E3 · Shared case state, explanation, human review, outcomes, policy governance

Generated from the notebook (cells 35, 36). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# E3 · Shared case state, Explanation Agent (live LLM), human review, deadlines, safe default, implementation   [notebook cell 35]
# ----------------------------------------------------------------------------------------------------
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
        return f"Route {raw['route_id']} {raw['route_name']}: we are checking reports of service problems. Updates will follow."
    src = min(acc, key=lambda e: precedence_rank(get_case_policy(case), e["source_type"]))
    stops = raw["affected_stop_ids"]
    parts = [f"Route {raw['route_id']} {raw['route_name']}"]
    if f.get("cancelled"):
        parts.append("a scheduled trip is cancelled")
    elif f.get("delay_min") is not None:
        parts.append(f"buses are running about {f['delay_min']:.0f} min late")
    if f.get("event_severity", "none") != "none":
        parts.append(f"due to {raw.get('event_type', 'an incident').replace('_', ' ')}")
    msg = ": ".join([parts[0], ", ".join(parts[1:]) or "service disruption"]) + f" between {NET['stops'][stops[0]]['name']} and {NET['stops'][stops[-1]]['name']}."
    nearby = sorted({r for s in stops for r in NET["stops"][s]["nearby_routes"]})
    if nearby:
        msg += f" Alternatives: route(s) {', '.join(nearby)}."
    msg += f" Verified by {src['source_id']} at {src['timestamp'][11:16]} ET. Accessible-service riders can call the paratransit desk."
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
            ticket = check_sources(raw_sources(raw), use_llm=use_llm)   # raises before any case exists if a source fails
            self.counter += 1
            policy = self.PM.current
            case = {"case_id": f"CASE-{self.counter:04d}", "created_at": iso(utcnow()), "scenario_name": raw["scenario_name"],
                    "policy_version": policy["version"], "raw": raw, "status": None, "analysis": None,
                    "constraints": {}, "revisions": 0, "operator_checks": 0, "operator_resolution": None,
                    "review": None, "decisions": [], "implementation": [], "outcome": None, "alerts": [], "audit": [],
                    "explanation": None, "rider_alert": None, "use_llm": use_llm, "history": False,
                    "scout_ticket": ticket}
            self.cases[case["case_id"]] = case
            self.active_id = case["case_id"]
            self.log(case, "Signal Scout Agent", "agent", "Intake", "Case created", raw["scenario_name"])
            self.log(case, "Signal Scout Agent", "rule", "Source checks", "source ticket issued", scout_ticket_summary(ticket))
            if ticket["flagged_for_review"]:
                self.log(case, "Signal Scout Agent", "agent", "Source checks", "sources flagged for human review (non-authoritative)",
                         "model said invalid: " + ", ".join(ticket["flagged_for_review"]) + "; the reviewer decides whether to trust them")
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

    def monitor_action(self, case, action, actor="planner"):
        if case["status"] not in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW"):
            raise ValueError("Only monitoring / safe-default cases can be rechecked or re-escalated.")
        raw = copy.deepcopy(case["raw"])
        if action == "recheck":
            for o in raw["observations"]:
                o["age_s"] = min(o["age_s"], 30)
                o["timestamp"] = iso(utcnow() - timedelta(seconds=o["age_s"]))
            note = "Feed changed: fresh snapshot re-verified"
        else:
            s = {**raw["spec"], "delay_min": (raw["spec"]["delay_min"] or 0) + 8,
                 "headway_ratio": raw["spec"]["headway_ratio"] + 0.6, "passups": raw["spec"]["passups"] + 8,
                 "feed_age_s": 20, "sources": sorted(set(raw["spec"]["sources"]) | {"vehicle_apc"}),
                 "vision_agent_failures": 0, "public_corroborated": raw["spec"]["public_corroborated"]}
            raw = make_raw(s, raw["scenario_name"] + " (worsened)")
            note = "Conditions worsened: delay +8 min, gap +0.6×, pass-ups +8"
        case["raw"], case["operator_resolution"] = raw, None
        case["policy_version"] = self.PM.current["version"]
        self.log(case, actor, "planner", "Monitoring", "Re-escalate" if action != "recheck" else "Recheck", note)
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

    def simulate_observed(self, case):
        return simulated_observation(self.predicted_for(case), case["case_id"])

    def evaluate_outcome(self, case, obs, actor="Outcome Evaluator"):
        if case["status"] not in ("IMPLEMENTED", "SAFE_DEFAULT", "MONITORING", "EVALUATED"):
            raise ValueError("Evaluate after implementation, or for safe-default / monitoring cases.")
        case["outcome"] = outcome_metrics(self.predicted_for(case), obs)
        if case["status"] == "IMPLEMENTED":
            case["status"] = "EVALUATED"
        self.log(case, actor, "agent", "Outcome evaluator", "Predicted vs observed compared",
                 f"observed data: {case['outcome']['observed_source']}; delay err {case['outcome']['delay_error_min']:+.1f} min; "
                 f"false alarm={case['outcome']['false_alarm']}. Policy weights unchanged.")
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


SIMULATED_SOURCE = "SIMULATED: synthetic, no post-incident data yet"


def simulated_observation(pr, key):
    """Deterministic synthetic 'observed' outcome for a predicted action (replace with real post-incident data)."""
    rng = stable_rng("observed", key)
    k = {"dispatch_second_bus": 0.75, "reroute": 0.9}.get(pr["type"], 0.95)
    return {"observed_delay_reduction_min": round(pr["delay_reduction_min"] * k * rng.uniform(0.85, 1.1), 1),
            "observed_access_protection": round(min(1, pr["access_protection"] * rng.uniform(0.85, 1.05)), 2),
            "observed_vulnerable_riders_served": round(pr["vulnerable_riders_helped"] * rng.uniform(0.75, 1.0)),
            "actual_cost_usd": round(pr["cost_usd"] * (1.2 if pr["type"] == "dispatch_second_bus" else 1.0) * rng.uniform(0.95, 1.1)),
            "rider_reported_impact": rng.choice([1, 2, 2, 3, 3, 4]), "false_alarm": rng.random() < 0.15,
            "observed_source": SIMULATED_SOURCE}


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
            "rider_reported_impact": obs["rider_reported_impact"], "false_alarm": bool(obs["false_alarm"]),
            "observed_source": obs.get("observed_source", "entered by a person")}


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


# ----------------------------------------------------------------------------------------------------
# E3 · Incident history, Outcome Evaluator metrics, Insight Agent, policy-owner approval, backtest and versioning   [notebook cell 36]
# ----------------------------------------------------------------------------------------------------
def random_spec(rng):
    rid = rng.choice(["R1", "R2", "R3", "R4", "R5"])
    n = len(NET["routes"][rid]["stops"])
    i0 = rng.randint(0, n - 5)
    canc = rng.random() < 0.12
    ev = rng.choice(["none", "none", "none", "weather", "obstruction", "road_closure", "mechanical"])
    sev = "none" if ev == "none" else rng.choice(["minor", "major", "major", "emergency" if ev == "road_closure" else "major"])
    load = round(rng.uniform(0.4, 1.12), 2)
    return dict(route_id=rid, stop_range=(i0, i0 + rng.randint(3, 5)), time_period=rng.choice(["peak", "midday", "off_peak", "shift_change"]),
                delay_min=round(rng.uniform(2, 28), 1), headway_ratio=round(rng.uniform(1.0, 2.8), 2), cancelled=canc,
                cancellation_risk=round(rng.uniform(0.05, 0.7), 2), boardings=rng.randint(40, 220), load_factor=load,
                passups=rng.randint(1, 25) if load >= 1 else rng.randint(0, 3), event_type=ev, event_severity=sev,
                event_verified=ev != "none", sources=rng.choice([["vehicle_apc"], ["vehicle_apc", "official_alert"],
                                                                 ["vehicle_apc", "trusted_external"], ["vehicle_apc", "official_alert", "trusted_external"]]),
                feed_age_s=rng.randint(10, 120), detour_extra_min=rng.randint(4, 10))


def seed_history(store, n=30):
    """Simulated past cases: disruptions, responses and recovery outcomes (incident history input)."""
    rng = stable_rng("history")
    policy = store.PM.current
    base = utcnow() - timedelta(days=90)
    for i in range(n):
        spec = random_spec(rng)
        raw = make_raw(spec, f"Historical incident {i + 1:02d}")
        when = base + timedelta(days=i * 3, hours=rng.randint(6, 20))
        raw["generated_at"] = iso(when)
        A = run_analysis(raw, policy)
        case = {"case_id": f"HIST-{i + 1:03d}", "created_at": iso(when), "scenario_name": raw["scenario_name"], "policy_version": policy["version"],
                "raw": raw, "analysis": A, "constraints": {}, "operator_resolution": None, "decisions": [], "implementation": [],
                "outcome": None, "audit": [], "history": True, "revisions": 0, "approved_option_id": None}
        if A["gate"] == "SIGNIFICANT" and A["ranking"]:
            top = A["ranking"][0]["option_id"]
            rev = rng.choice(["Supervisor Lee", "Supervisor Ortiz", "Planner Chen"])
            case["decisions"].append({"stage": "first review", "reviewer": rev, "decision": "Approve", "option_id": top, "recommended_option_id": top,
                                      "override": False, "reason": "historical approval", "response_s": round(rng.uniform(20, 400), 1),
                                      "ts": iso(when + timedelta(minutes=4)), "tier": A["tier"]})
            case["approved_option_id"] = top
            case["status"] = "EVALUATED"
            pr = store.predicted_for(case)
            hard = bool(A["hard_triggers"])
            fa = (rng.random() < 0.6) if (not hard and A["scores"]["R"] < 0.62) else (rng.random() < 0.08)
            k = 0.72 if pr["type"] == "dispatch_second_bus" else 0.95
            eq = 0.78 if A["scores"]["P"] >= 0.5 else 0.95
            obs = {"observed_delay_reduction_min": round(pr["delay_reduction_min"] * k * rng.uniform(0.9, 1.1), 1),
                   "observed_access_protection": round(pr["access_protection"] * rng.uniform(0.85, 1.0), 2),
                   "observed_vulnerable_riders_served": round(pr["vulnerable_riders_helped"] * eq * rng.uniform(0.92, 1.05)),
                   "actual_cost_usd": round(pr["cost_usd"] * (1.28 if pr["type"] == "dispatch_second_bus" else 1.03)),
                   "rider_reported_impact": rng.randint(1, 4), "false_alarm": fa}
            case["outcome"] = outcome_metrics(pr, obs)
            case["recovery_min"] = round(max(5, (A["fields"].get("delay_min") or 10) * rng.uniform(1.2, 2.5)))
        else:
            case["status"] = "SAFE_DEFAULT" if A["gate"] in ("BELOW_THRESHOLD", "SAFE_MODE") else "MONITORING"
            real = rng.random() < 0.15
            case["outcome"] = {"evaluated_at": iso(when), "action": "rider information only", "action_type": "none",
                               "false_alarm": not real, "missed_real_disruption": real}
            case["recovery_min"] = round(rng.uniform(5, 25))
        store.history.append(case)


def incident_history_df(store):
    rows = []
    for c in store.history:
        A = c["analysis"]
        rows.append({"case_id": c["case_id"], "date": c["created_at"][:10], "route": c["raw"]["route_id"],
                     "event": c["raw"]["event_type"], "delay_min": A["fields"].get("delay_min") if A.get("fields") else None,
                     "significant": A["significant"], "response": c.get("approved_option_id") or "rider information only",
                     "recovery_min": c.get("recovery_min"), "false_alarm": (c.get("outcome") or {}).get("false_alarm")})
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
seed_history(APP)
print(f"Seeded {len(APP.history)} historical cases; rules v{PM.current['version']} active.")
