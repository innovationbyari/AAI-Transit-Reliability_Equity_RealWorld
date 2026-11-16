"""Section E5 · Dashboard actions (UI-free; the Streamlit layout is in streamlit_app.py)

Generated from the notebook (cells 101). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# E5 · Gradio dashboard (human-in-the-loop)   [notebook cell 101]
# ----------------------------------------------------------------------------------------------------


def banner_md():
    c = APP.active
    llm = f"live LLM: {', '.join(p['label'] for p in LLM_CHAIN)}" if LLM_READY else "LLM off → template narratives"
    head = (f"**Rules v{APP.PM.current['version']}** · {llm} · {len(APP.cases)} live case(s) · {len(APP.history)} historical · "
            f"deadline scale ×{DEMO_DEADLINE_SCALE}")
    if c:
        col = STATUS_COLOR.get(c["status"], "#111")
        head += (f"<br><span style='background:{col};color:white;padding:2px 8px;border-radius:4px'>{c['status']}</span> "
                 f"**{c['case_id']}** · {_html.escape(c['scenario_name'])}")
    return head + f"<br>💬 {_html.escape(APP.last_message)}"


def _need_case():
    if not APP.active:
        raise ValueError("Run a scenario first.")
    return APP.active


# ---- handlers
# Section A and B sample scenarios run through the same pipeline
RAW_PRESETS = {
    "Section A · NJ-001 road closure (Inputs_SLM hand-off)":
        lambda: signal_scout_input_to_raw(build_signal_scout_input(section_a_sample_inputs())),
    "Section B · Route 12 cancellation + last rail connection (transit_agents_workflow)":
        lambda: workflow_inputs_to_raw(SAMPLE_INPUTS, VisionAgent().run(RAW_SENSOR_FEED)),
}
ALL_PRESETS = list(RAW_PRESETS) + list(PRESETS)


def h_run_preset(name, use_llm):
    raw = RAW_PRESETS[name]() if name in RAW_PRESETS else make_raw(PRESETS[name], name)
    APP.new_case(raw, use_llm=use_llm)


def h_run_custom(use_llm, route, i0, n_stops, period, delay, gap, cancelled, risk, boardings, load, passups, ev_type, ev_sev,
                 sources, feed_age, backup, vfail, detour, public_delay, public_corr, missing):
    spec = dict(route_id=route, stop_range=(int(i0), int(i0) + int(n_stops) - 1), time_period=period, delay_min=float(delay),
                headway_ratio=float(gap), cancelled=bool(cancelled), cancellation_risk=float(risk), boardings=int(boardings),
                load_factor=float(load), passups=int(passups), event_type=ev_type, event_severity=ev_sev,
                event_verified=ev_sev != "none", sources=list(sources) or ["vehicle_apc"], feed_age_s=int(feed_age),
                backup_available=bool(backup), vision_agent_failures=int(vfail), detour_open=bool(detour),
                public_delay=float(public_delay), public_corroborated=bool(public_corr), missing_fields=list(missing or []))
    APP.new_case(make_raw(spec, f"Custom · {route} · delay {delay} min"), use_llm=use_llm)


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
        APP.monitor_action(APP.cases[cid], act, name or "planner")


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


def h_prefill():
    c = _need_case()
    return APP.simulate_observed(c)


def h_evaluate(dr, acc, eq, cost, impact, fa):
    APP.evaluate_outcome(_need_case(), {"observed_delay_reduction_min": float(dr), "observed_access_protection": float(acc),
                                         "observed_vulnerable_riders_served": float(eq), "actual_cost_usd": float(cost),
                                         "rider_reported_impact": int(impact), "false_alarm": bool(fa)})


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


def h_load_roster(text):
    n = load_eth_roster(json.loads(text))
    APP.log(None, "system", "system", "Ethics roster", "Roster loaded", f"{n} people")
    return f"Roster loaded ({n} people)."


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


def h_coi_pick(choice):
    kind, name = choice.split(" · ", 1)
    return STAFF_DISCLOSURES[name]["text"] if kind == "Staff" else COI_SAMPLES[name].strip()


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
