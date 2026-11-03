"""Streamlit app — Agentic AI for Public Transit Reliability and Equity.

Run locally:  streamlit run app_streamlit.py
Streamlit Community Cloud: set the main file to app_streamlit.py and add the keys under App settings → Secrets:
    NJ_TRANSIT_USERNAME = "..."     NJ_TRANSIT_PASSWORD = "..."
    CENSUS_API_KEY = "..."          (optional)   NWS_CONTACT_EMAIL = "..."   (optional)
    OPENAI_API_KEY = "..."          (optional LLM narratives)
    DASHBOARD_PASSWORD = "..."      (recommended: protects the public URL)
All cases come from live NJ Transit vehicles; there is no sample data. Case state is shared by everyone using the app
(so a second reviewer can open the same case), and resets when the app restarts.
"""
import json
import os
import time
from datetime import timedelta

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Transit Reliability & Equity", page_icon="🚌", layout="wide")


def _dashboard_password():
    try:
        v = st.secrets.get("DASHBOARD_PASSWORD")
    except Exception:
        v = None
    return v or os.environ.get("DASHBOARD_PASSWORD")


def password_gate():
    """A public URL spends the app's API keys, so DASHBOARD_PASSWORD (when set) is required before anything loads."""
    import hmac
    pw = _dashboard_password()
    if not pw or st.session_state.get("authenticated"):
        return
    st.title("🚌 Transit Reliability & Equity")
    with st.form("login"):
        typed = st.text_input("Dashboard password", type="password")
        if st.form_submit_button("Sign in"):
            if hmac.compare_digest(typed.encode(), str(pw).encode()):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


password_gate()
if not _dashboard_password():
    st.warning("No DASHBOARD_PASSWORD secret is set, so anyone with this URL can use the app and its API keys.")


@st.cache_resource(show_spinner="Loading the transit engine …")
def load_core():
    for k in ("TRANSIT_USE_SLM", "TRANSIT_USE_CV"):          # heavy local models are off unless the host turns them on
        os.environ.setdefault(k, "0")
    import threading
    import transit_core as tc
    if tc.read_secret("NJ_TRANSIT_USERNAME"):         # warm up: first live pull + static GTFS in the background
        def warm():
            try:
                tc.refresh_live_context()
                tc.network_or_none()
            except Exception as e:
                tc.APP.last_message = f"Start-up live pull failed: {e}"
        threading.Thread(target=warm, daemon=True).start()
    return tc


tc = load_core()
APP = tc.APP


def run(fn, *a, ok=None, **k):
    """Run a store action; show refusals (gates, rubber-stamp checks, deadlines, missing live data) instead of crashing."""
    with APP.lock:
        try:
            out = fn(*a, **k)
            st.session_state["flash"] = ("ok", ok or APP.last_message)
            return out
        except (ValueError, KeyError, tc.LiveDataUnavailable) as e:
            st.session_state["flash"] = ("err", str(e))
            return None


def active():
    cid = st.session_state.get("case_id") or APP.active_id
    c = APP.cases.get(cid)
    if c and c["status"] in tc.REVIEW_STATUSES:
        APP.enforce_deadline(c)
    return c


def show_df(df, **k):
    df = df.copy()
    for col in df.columns:                      # mixed-type columns (numbers and labels) display as text
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x if x is None or isinstance(x, str) else str(x))
    st.dataframe(df, width="stretch", hide_index=True, **k)


# ---------------------------------------------------------------- sidebar: live data and case selection
with st.sidebar:
    st.header("🚌 Live data")
    if st.button("↻ Pull live data (NJ Transit · Census · NWS)", width="stretch"):
        with st.spinner("Pulling live feeds …"):
            run(tc.refresh_live_context, ok="Live data pulled.")
            with st.spinner("Loading NJ Transit static GTFS (first time can take a few minutes) …"):
                tc.network_or_none()
    st.markdown(tc.live_status_md())
    choices = tc.vehicle_choices()
    labels = {vid: lab for lab, vid in choices}
    vid = st.selectbox("Live vehicle", [vid for _, vid in choices], format_func=lambda v: labels.get(v, v),
                       index=None, placeholder="pull live data first" if not choices else "pick a vehicle")
    use_llm = st.checkbox("Live LLM narrative", value=tc.LLM_READY, disabled=not tc.LLM_READY)
    if st.button("▶ Run pipeline on this vehicle", type="primary", width="stretch", disabled=vid is None):
        with st.spinner("Signal Scout → verification → scoring → planner → explanation …"):
            c = run(lambda: APP.new_case(tc.raw_for_vehicle(vid, store=APP), use_llm=use_llm))
        if c:
            st.session_state["case_id"] = c["case_id"]
    st.divider()
    ids = list(APP.cases)[::-1]
    if ids:
        cur = st.session_state.get("case_id") or APP.active_id
        pick = st.selectbox("Active case", ids, index=ids.index(cur) if cur in ids else 0)
        st.session_state["case_id"] = pick
        APP.active_id = pick
    st.caption(f"Rules v{APP.PM.current['version']} · {'LLM: ' + ', '.join(p['label'] for p in tc.LLM_CHAIN) if tc.LLM_READY else 'LLM off → templates'}"
               f" · prototype: does not control live operations")

st.title("Transit Reliability & Equity — human-in-the-loop dashboard")
flash = st.session_state.pop("flash", None)
if flash:
    (st.success if flash[0] == "ok" else st.error)(flash[1])
case = active()
if case:
    col = tc.STATUS_COLOR.get(case["status"], "#111")
    st.markdown(f"<span style='background:{col};color:white;padding:2px 8px;border-radius:4px'>{case['status']}</span> "
                f"**{case['case_id']}** · {tc._html.escape(case['scenario_name'])} · "
                f"{tc.STATUS_HELP.get(case['status'], '')}", unsafe_allow_html=True)

tabs = st.tabs(["1 · Intake & verification", "2 · Scores", "3 · Options", "4 · Human review", "5 · Implementation & outcome",
                "6 · Policy governance", "7 · Ethics & independence", "8 · Audit & export"])

# ---------------------------------------------------------------- 1 intake
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        if hasattr(st, "iframe"):
            st.iframe(tc.map_html(case), height=540)
        else:
            components.html(tc.map_html(case), height=540)
    with right:
        st.markdown(tc.trace_md(case), unsafe_allow_html=True)
    st.subheader("Live vehicles near the service area")
    show_df(tc.live_vehicles_df())
    if case:
        if case["raw"].get("missing_inputs"):
            st.info("Inputs with no live source (left missing, never imputed): " + ", ".join(case["raw"]["missing_inputs"]))
        st.subheader("Verified evidence")
        show_df(tc.evidence_df(case))
        a, b = st.columns(2)
        with a:
            st.caption("Exception log"); show_df(tc.exceptions_df(case))
        with b:
            st.caption("Alerts"); show_df(tc.alerts_df(case))
        if case["status"] == "CONFLICT_REVIEW":
            with st.form("operator"):
                st.markdown("**Operator review — conflicting data** (max two checks, then monitoring)")
                op = st.text_input("Operator name")
                res = st.radio("Result", ["resolved", "unresolved"], horizontal=True,
                               captions=["apply source precedence (vehicle data wins)", "cannot tell which source is right"])
                if st.form_submit_button("Submit operator check"):
                    run(APP.operator_resolve, case, op, res); st.rerun()
    st.subheader("Monitoring queue")
    show_df(tc.monitor_df())
    mon_ids = [x["case_id"] for x in APP.cases.values() if x["status"] in ("MONITORING", "SAFE_DEFAULT", "CONFLICT_REVIEW")]
    if mon_ids:
        with st.form("monitor"):
            m_id = st.selectbox("Case", mon_ids)
            m_name = st.text_input("Planner name")
            m_reason = st.text_input("Reason (for closing as a false alarm)")
            c1, c2 = st.columns(2)
            if c1.form_submit_button("Recheck (pull the live feed again)"):
                run(APP.monitor_action, APP.cases[m_id], "recheck", m_name or "planner"); st.rerun()
            if c2.form_submit_button("Close as false alarm"):
                run(APP.close_false_alarm, APP.cases[m_id], m_name, m_reason); st.rerun()

# ---------------------------------------------------------------- 2 scores
with tabs[1]:
    if not case:
        st.info("Run the pipeline on a live vehicle first.")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown("**Confidence** C = 0.40Q + 0.25F + 0.20G + 0.15M")
            show_df(tc.conf_df(case))
            st.markdown(tc.sig_md(case))
        with b:
            st.plotly_chart(tc.rvap_plot(case), width="stretch")
        st.markdown("**Normalization and impact** — R = 0.30D + 0.25H + 0.25K + 0.20E · V = 0.35N + 0.25I + 0.20B + 0.20T · "
                    "A = 0.35S + 0.30L + 0.20X + 0.15U · P = 0.40R + 0.35V + 0.25A")
        show_df(tc.norm_df(case))

# ---------------------------------------------------------------- 3 options
with tabs[2]:
    if not case:
        st.info("Run the pipeline on a live vehicle first.")
    else:
        a, b = st.columns([3, 2])
        with a:
            st.caption("Scenario table and rank"); show_df(tc.options_df(case))
            st.caption("Considered but infeasible"); show_df(tc.infeasible_df(case))
        with b:
            st.plotly_chart(tc.options_plot(case), width="stretch")
        ex = case.get("explanation")
        if ex:
            st.markdown(ex["brief_md"])
            st.markdown(f"**Narrative ({ex['narrative_source']}, non-authoritative):** {ex['narrative']}")

# ---------------------------------------------------------------- 4 human review
with tabs[3]:
    if not case:
        st.info("Run the pipeline on a live vehicle first.")
    else:
        @st.fragment(run_every=5)
        def clock():
            c = active()
            if c:
                st.markdown(tc.deadline_md(c))
        clock()
        show_df(tc.options_df(case))
        ranking = case["analysis"].get("ranking", [])
        opt_labels = {r["option_id"]: f"#{r['rank']} {r['label']} (Si {r['Si']:.3f})" for r in ranking}
        names = [o.name for o in tc.ETH_ROSTER]
        a, b = st.columns(2)
        with a:
            st.subheader("First reviewer")
            with st.form("first", clear_on_submit=False):
                r_name = st.selectbox("Reviewer (roster)", names, index=None, accept_new_options=True,
                                      placeholder="type a name") if names else st.text_input("Reviewer name")
                r_dec = st.radio("Decision", ["Approve", "Revise", "Reject"], horizontal=True)
                r_opt = st.selectbox("Option to approve (rank 1 by default; others are logged as overrides)", list(opt_labels),
                                     format_func=opt_labels.get) if opt_labels else None
                r_reason = st.text_area("Typed reason (required)")
                r_check = [t for t in tc.REVIEW_CHECKLIST if st.checkbox(t, key="r1" + t)]
                r_viewed = st.checkbox("I viewed all alternatives before the recommendation", key="r1v")
                with st.expander("Revision: change constraints (max two)"):
                    rv_excl = st.multiselect("Exclude option types", list(tc.OPTION_LABELS))
                    rv_budget = st.slider("Budget cap (USD)", 100, 2000, 1000, 50)
                    rv_hold = st.slider("Max connection hold (min)", 0, 15, 5)
                    rv_alt = st.checkbox("Allow dispatch from an alternate depot")
                if st.form_submit_button("Submit first review", type="primary"):
                    rev = ({"excluded_options": rv_excl, "budget_cap_usd": float(rv_budget), "max_hold_min": float(rv_hold),
                            "allow_alternate_depot": rv_alt} if r_dec == "Revise" else None)
                    run(APP.submit_first, case, r_name or "", r_dec, r_opt, r_reason, r_check, r_viewed, rev); st.rerun()
        with b:
            st.subheader("Second reviewer (high-impact actions)")
            if case["status"] == "AWAITING_SECOND_REVIEW":
                f = case["review"]["first"]
                st.info(f"First reviewer {f['reviewer']} approved `{f['option_id']}`: “{f['reason']}”. An independent reviewer must "
                        "also approve; either can reject.")
            with st.form("second"):
                s_name = st.selectbox("Independent reviewer (roster)", names, index=None, accept_new_options=True,
                                      placeholder="a different person") if names else st.text_input("Independent reviewer name")
                s_dec = st.radio("Decision", ["Approve", "Reject"], horizontal=True)
                s_reason = st.text_area("Typed reason (required)", key="s_reason")
                s_check = [t for t in tc.REVIEW_CHECKLIST if st.checkbox(t, key="r2" + t)]
                s_viewed = st.checkbox("I viewed all alternatives before the recommendation", key="r2v")
                if st.form_submit_button("Submit second review", type="primary"):
                    run(APP.submit_second, case, s_name or "", s_dec, s_reason, s_check, s_viewed); st.rerun()
        if case.get("rider_alert"):
            st.markdown(f"**Rider alert (verified info only):** {case['rider_alert']}")
        st.caption("Human decisions and timing"); show_df(tc.decisions_df(case))

# ---------------------------------------------------------------- 5 implementation and outcome
with tabs[4]:
    if not case:
        st.info("Run the pipeline on a live vehicle first.")
    else:
        if case["status"] == "AWAITING_IMPLEMENTATION":
            st.markdown(f"### ✅ Approved: `{case['approved_option_id']}`\nOperations staff carry out the action outside the "
                        "prototype and send rider alerts. Record the result below.")
            with st.form("impl"):
                i_staff = st.text_input("Operations staff")
                i_res = st.radio("Result", ["Completed", "Partial", "Failed"], horizontal=True)
                i_notes = st.text_input("Notes (required for Partial or Failed)")
                if st.form_submit_button("Record implementation", type="primary"):
                    run(APP.record_implementation, case, i_staff, i_res, i_notes); st.rerun()
        for i in case["implementation"]:
            st.markdown(f"- {i['ts'][:19]} · {i['staff']} · **{i['result']}** · {i['notes']}")
        st.subheader("Outcome evaluator")
        st.caption("Enter what was observed after the incident (from operations records). Nothing is simulated; "
                   "policy weights are never changed automatically.")
        if case["status"] in ("IMPLEMENTED", "SAFE_DEFAULT", "MONITORING", "EVALUATED"):
            pr = APP.predicted_for(case)
            st.caption(f"Predicted for {pr['option_id']}: delay reduction {pr['delay_reduction_min']} min, access "
                       f"{pr['access_protection']}, vulnerable riders {pr['vulnerable_riders_helped']}, cost ${pr['cost_usd']:.0f}")
            with st.form("outcome"):
                cols = st.columns(3)
                obs = {}
                for n, (k, lab) in enumerate(list(tc.OBSERVED_FIELDS.items())[:4]):
                    obs[k] = cols[n % 3].number_input(lab, value=None, placeholder="enter value")
                obs["rider_reported_impact"] = cols[1].selectbox(tc.OBSERVED_FIELDS["rider_reported_impact"], [1, 2, 3, 4, 5], index=None)
                fa = cols[2].selectbox(tc.OBSERVED_FIELDS["false_alarm"], ["no", "yes"], index=None)
                obs["false_alarm"] = {"yes": True, "no": False}.get(fa)
                if st.form_submit_button("Evaluate outcome", type="primary"):
                    run(APP.evaluate_outcome, case, obs); st.rerun()
        a, b = st.columns(2)
        with a:
            show_df(tc.outcome_df(case))
        with b:
            st.plotly_chart(tc.outcome_plot(case), width="stretch")

# ---------------------------------------------------------------- 6 policy governance
with tabs[5]:
    if st.button("🔍 Run Insight Agent (proposes only)"):
        props = run(tc.insight_agent, APP)
        if props is not None:
            st.session_state["flash"] = ("ok", f"Insight Agent drafted {len(props)} proposal(s). Nothing changes until the "
                                               "policy owner approves."); st.rerun()
    st.markdown(APP.insight_narrative or "_Run the Insight Agent to analyse evaluated outcomes._")
    show_df(tc.proposals_df())
    if APP.proposals:
        with st.form("policy"):
            pid = st.selectbox("Proposal", [p["id"] for p in APP.proposals])
            owner = st.text_input("Policy owner name")
            why = st.text_input("Reason (required)")
            c1, c2, c3 = st.columns(3)
            if c1.form_submit_button("Preview backtest"):
                p = next(p for p in APP.proposals if p["id"] == pid)
                _, summ = tc.run_backtest(APP, tc.apply_changes(APP.PM.current, p["changes"]))
                st.session_state["flash"] = ("ok", "Preview only — " + summ); st.rerun()
            if c2.form_submit_button("Approve → backtest → release", type="primary"):
                run(tc.decide_proposal, APP, pid, owner, "Approve", why); st.rerun()
            if c3.form_submit_button("Reject (keep current rules)"):
                run(tc.decide_proposal, APP, pid, owner, "Reject", why); st.rerun()
    if APP.backtest:
        st.markdown(APP.backtest["summary"]); show_df(APP.backtest["df"])
    st.caption("Decision log"); show_df(tc.proposal_log_df())
    st.caption("Rule versions"); show_df(tc.versions_df())
    with st.form("rollback"):
        rb_v = st.selectbox("Roll back to version", [v["version"] for v in APP.PM.versions])
        rb_o = st.text_input("Policy owner name", key="rb_o")
        rb_r = st.text_input("Reason (required, ≥ 10 characters)", key="rb_r")
        if st.form_submit_button("Roll back"):
            def _rb():
                if not rb_o.strip() or len(rb_r.strip()) < 10:
                    raise ValueError("Policy owner name and a reason (≥ 10 characters) are required for rollback.")
                v = APP.PM.rollback(rb_v, rb_o, rb_r)
                APP.log(None, rb_o, "policy owner", "Rollback", f"Rules of {rb_v} re-released as v{v}", rb_r)
                APP.last_message = f"Rules of v{rb_v} are now active as v{v}."
            run(_rb); st.rerun()
    with st.expander("Active policy (thresholds, bounds, weights, lookups)"):
        st.json(APP.PM.current)

# ---------------------------------------------------------------- 7 ethics and independence
with tabs[6]:
    st.markdown(tc.eth_summary_md())
    st.caption(f"Roster source: {tc.ETH_ROSTER_SOURCE['source']}")
    show_df(tc.eth_status_df())
    with st.expander("Load the reviewer roster (JSON)", expanded=not tc.ETH_ROSTER):
        txt = st.text_area("Roster JSON", json.dumps([o.model_dump() for o in tc.ETH_ROSTER] or tc.ROSTER_TEMPLATE, indent=1),
                           height=220)
        save = st.checkbox(f"Also save to {tc.REVIEWER_ROSTER_PATH} (lost on Streamlit Cloud restarts; use the "
                           "REVIEWER_ROSTER_JSON secret to persist)")
        if st.button("Load roster"):
            def _load():
                n = tc.load_eth_roster(json.loads(txt))
                tc.ETH_ROSTER_SOURCE["source"] = "pasted in the app"
                if save:
                    tc.save_roster()
                APP.last_message = f"Roster loaded ({n} people)."
            run(_load); st.rerun()
    st.subheader("Conflict-of-interest disclosures")
    st.caption(f"Source: {tc.STAFF_DISCLOSURES_SOURCE['source']}")
    show_df(tc.disclosures_df())
    with st.expander("Load disclosures (JSON)", expanded=not tc.STAFF_DISCLOSURES):
        dtxt = st.text_area("Disclosures JSON", json.dumps(tc.disclosures_as_json() or tc.DISCLOSURE_TEMPLATE, indent=1), height=220)
        if st.button("Load disclosures"):
            def _ld():
                n = tc.load_disclosures(json.loads(dtxt))
                tc.STAFF_DISCLOSURES_SOURCE["source"] = "pasted in the app"
                APP.last_message = f"Disclosures loaded ({n} people)."
            run(_ld); st.rerun()
    st.caption("Rule screen: each person × each option on the active case"); show_df(tc.coi_screen_df(case))
    with st.form("gate"):
        st.markdown("**Run the governance gate for a person**")
        g_name = st.text_input("Name")
        g_dec = st.radio("Decision", ["Approve", "Revise", "Reject"], horizontal=True)
        feas = [o for o in (case["analysis"].get("options", []) if case else []) if o["feasible"]]
        g_opt = st.selectbox("Option on the active case", [o["option_id"] for o in feas], index=None,
                             format_func=lambda i: next(o["label"] for o in feas if o["option_id"] == i))
        if st.form_submit_button("Check"):
            st.markdown(tc.h_gate(g_name, g_dec, g_opt))
    if tc.LLM_READY:
        with st.expander("Section C agent: ask about the roster (LLM + guardrail)"):
            q = st.text_input("Question", tc.ETH_DEFAULT_QUESTION)
            if st.button("Run compliance agent"):
                with st.spinner("Agent working …"):
                    st.markdown(tc.h_eth_agent(q))
        with st.expander("Section D multi-agent review of a disclosure (LLM + guardrail)"):
            who = st.selectbox("Disclosure on file", list(tc.STAFF_DISCLOSURES), index=None)
            ctext = st.text_area("Case text", tc.STAFF_DISCLOSURES[who]["text"] if who else "")
            if st.button("Run 5-step conflict-of-interest pipeline") and ctext.strip():
                with st.spinner("Pipeline running …"):
                    st.markdown(tc.h_coi_pipeline(ctext))

# ---------------------------------------------------------------- 8 audit and export
with tabs[7]:
    st.caption("All cases (shared case state)"); show_df(tc.cases_df())
    rs = APP.rubber_stamp_report()
    st.caption("Rubber-stamp protection: approval patterns and response times"); show_df(rs if len(rs) else tc.ph(None, "No decisions yet"))
    st.caption("Audit record (newest first)"); show_df(tc.audit_df())
    if st.button("Prepare export (zip: JSON + CSV + map)"):
        st.session_state["export"] = tc.export_outputs()
    if st.session_state.get("export") and os.path.exists(st.session_state["export"]):
        p = st.session_state["export"]
        st.download_button("⬇ Download outputs", open(p, "rb").read(), file_name=os.path.basename(p), mime="application/zip")
    st.caption("Governance boundary: human authority is final; thresholds, bounds and weights are versioned policy assumptions; "
               "exception paths never trigger resource action; vision data anonymized; community data tract-level only.")
