"""Transit Reliability & Equity — live, human-in-the-loop dashboard on the v7 engine (Streamlit).

Run locally:  streamlit run app_streamlit.py
Streamlit Community Cloud: main file app_streamlit.py; keys go in App settings → Secrets
(see .streamlit/secrets.toml.example and README.md).

How it flows
  Step 1  Live buses: pull the live NJ Transit feed and open a case for a delayed or cancelled bus.
  Step 2  Who are you? Pick your role: Dispatcher (1st reviewer), Supervisor (2nd reviewer),
          Operations Officer (carries out the action, alerts the driver) or Bus Driver (acknowledges).
  Then    "My task" shows exactly what your role needs to do on the selected case.
Every case comes from a live NJ Transit bus. The v7 notebook engine (transit_ai/) scores it; inputs with no live source
use v7's documented samples and are labelled. Case state is shared by everyone using the app.
"""
import hmac
import json
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Transit Reliability & Equity", page_icon="🚌", layout="wide")
ss = st.session_state


# ================================================================================================ access
def _dashboard_password():
    try:
        v = st.secrets.get("DASHBOARD_PASSWORD")
    except Exception:
        v = None
    return v or os.environ.get("DASHBOARD_PASSWORD")


def password_gate():
    pw = _dashboard_password()
    if not pw or ss.get("authenticated"):
        return
    st.markdown("## 🚌 Transit Reliability & Equity")
    st.caption("Live bus disruption response · Newark Penn Station area")
    with st.form("login"):
        typed = st.text_input("Dashboard password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(typed.encode(), str(pw).encode()):
                ss["authenticated"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


password_gate()


def _export_secrets():
    """v7 reads keys and settings from environment variables: copy Streamlit secrets there first."""
    try:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ.setdefault(k, str(v))
    except Exception:
        pass


def _import_engine(app_dir):
    """The v7 engine normally lives in the transit_ai/ folder. GitHub's web upload can flatten folders, putting its files
    (__init__.py, _modules.json, live_ops.py, …) next to this app instead. Both layouts work."""
    import importlib.util
    try:
        from transit_ai import load
        return load
    except ModuleNotFoundError as e:
        if e.name != "transit_ai":
            raise
    init = os.path.join(app_dir, "__init__.py")
    if os.path.exists(init) and os.path.exists(os.path.join(app_dir, "_modules.json")):
        spec = importlib.util.spec_from_file_location("transit_ai", init, submodule_search_locations=[app_dir])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["transit_ai"] = mod
        spec.loader.exec_module(mod)
        return mod.load
    st.error("The v7 engine files are missing. Upload the transit_ai folder (or its files) next to app_streamlit.py: "
             "__init__.py, _modules.json, live_ops.py and the other section files.")
    st.stop()


@st.cache_resource(show_spinner="Starting up: loading the v7 engine and connecting to NJ Transit, Census and the weather service …")
def load_core():
    _export_secrets()
    for k in ("USE_CV", "USE_SLM"):            # camera model and FLAN-T5 need ~1.5 GB: off unless turned on in Secrets
        os.environ.setdefault(k, "false")
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    import threading
    load = _import_engine(app_dir)
    import team_ops
    tc = load(verbose=False)
    team_ops.setup(tc)
    if tc.read_secret("NJ_TRANSIT_USERNAME"):         # live pull + NJ Transit route data in the background
        def warm():
            try:
                tc.refresh_live_context()
                tc.network_or_none()
            except Exception as e:
                tc.APP.last_message = f"Start-up live pull failed: {e}"
        threading.Thread(target=warm, daemon=True).start()
    return tc, team_ops


tc, T = load_core()
T.setup(tc)
APP = tc.APP
TEAM = T.team(tc)

# ================================================================================================ style
st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1400px}
h1, h2, h3 {letter-spacing: -0.01em}
.hero {display:flex; align-items:center; gap:14px; margin-bottom: 4px}
.hero .t {font-size: 1.65rem; font-weight: 700}
.hero .s {color:#6b7280; font-size: .95rem}
.card {border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; background:#fff; margin-bottom:10px}
.case-card {border-left:6px solid var(--c, #2563eb)}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600; color:#fff;
       background: var(--c, #2563eb)}
.muted {color:#6b7280; font-size:.9rem}
.steps {display:flex; gap:8px; flex-wrap:wrap; margin: 8px 0 14px 0}
.step {flex:1; min-width:130px; padding:9px 11px; border-radius:12px; border:1px solid #e5e7eb; background:#f9fafb;
       font-size:.85rem; line-height:1.25}
.step b {display:block; font-size:.75rem; color:#6b7280; font-weight:600; text-transform:uppercase; letter-spacing:.04em}
.step.done {background:#ecfdf5; border-color:#a7f3d0}
.step.current {background:#eff6ff; border-color:#2563eb; box-shadow:0 0 0 3px rgba(37,99,235,.12)}
.step.skipped, .step.stopped {opacity:.5}
.alert-sms {border:1px solid #fde68a; background:#fffbeb; border-radius:14px; padding:14px 16px; margin-bottom:10px}
.alert-sms.ack {border-color:#a7f3d0; background:#ecfdf5}
.big {font-size:1.05rem; line-height:1.45}
</style>
""", unsafe_allow_html=True)

STATUS_WORDS = {
    "AWAITING_REVIEW": ("Waiting for the dispatcher", "#dc2626"),
    "AWAITING_SECOND_REVIEW": ("Waiting for the supervisor", "#b91c1c"),
    "AWAITING_IMPLEMENTATION": ("Approved: waiting for the operations officer", "#7c3aed"),
    "IMPLEMENTED": ("Carried out: outcome to be recorded", "#059669"),
    "EVALUATED": ("Closed: outcome recorded", "#047857"),
    "SAFE_DEFAULT": ("Safe default: rider information only, monitoring", "#2563eb"),
    "MONITORING": ("Monitoring: data not strong enough to act yet", "#6b7280"),
    "CONFLICT_REVIEW": ("Data sources disagree: dispatcher check needed", "#d97706"),
    "CLOSED_FALSE_ALARM": ("Closed as a false alarm", "#9ca3af"),
}
STEP_ICON = {"done": "✅", "current": "👉", "waiting": "⏳", "skipped": "➖", "stopped": "⏸️"}


# ================================================================================================ helpers
def run(fn, *a, ok=None, **k):
    """Run an action; show refusals (training gate, missing reason, deadline …) as a message instead of crashing."""
    with APP.lock:
        try:
            out = fn(*a, **k)
            ss["flash"] = ("ok", ok or APP.last_message)
            return out
        except (ValueError, KeyError, tc.LiveDataUnavailable) as e:
            ss["flash"] = ("err", str(e))
            return None


def show_df(df, **k):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x if x is None or isinstance(x, str) else str(x))
    st.dataframe(df, width="stretch", hide_index=True, **k)


def case_label(c):
    words = STATUS_WORDS.get(c["status"], (c["status"], ""))[0]
    route = c["raw"].get("route_name") or c["raw"].get("route_id")
    return f"{c['case_id']} · Route {route} · {words}"


def pick_case():
    """The case shown to this person: their own pick, else the newest case waiting for their role, else the newest."""
    ids = list(APP.cases)[::-1]
    if not ids:
        return None
    if ss.get("case_id") in APP.cases:
        cid = ss["case_id"]
    else:
        mine = [i for i in ids if T.whose_turn(tc, APP.cases[i]) == ss.get("role")]
        cid = (mine or ids)[0]
    c = APP.cases[cid]
    if c["status"] in tc.REVIEW_STATUSES:
        with APP.lock:
            APP.enforce_deadline(c)
    return c


def fmt_delay(c):
    f = c["analysis"].get("fields", {})
    if f.get("cancelled"):
        return "trip cancelled"
    d = f.get("delay_min")
    return "delay unknown" if d is None else f"{d:.0f} min late"


def ranked_options(c):
    return c["analysis"].get("ranking", [])


def option_text(c, oid):
    o = next((o for o in c["analysis"].get("options", []) if o["option_id"] == oid), None)
    return o["label"] if o else str(oid)


def simple_options_df(c):
    import pandas as pd
    rows = [{"Rank": r["rank"], "Action": r["label"] + (" (v7 model depot)" if r["option_id"].startswith("dispatch_second_bus") else "")
             + ("  ← recommended" if r["rank"] == 1 else ""),
             "Estimated cost": f"${next((o['cost_usd'] for o in c['analysis']['options'] if o['option_id'] == r['option_id']), 0):,.0f}"}
            for r in ranked_options(c)]
    return pd.DataFrame(rows) if rows else tc.ph(None, "No action is possible right now")


def plain_summary(c):
    """Two or three sentences a non-specialist can follow. The technical explanation stays one click away."""
    raw, A = c["raw"], c["analysis"]
    f = A.get("fields", {})
    route = raw.get("route_name") or raw.get("route_id")
    stop = ((raw.get("real") or {}).get("stops") or [{}])[0].get("name")
    what = "a trip is cancelled" if f.get("cancelled") else (
        f"the bus is running about {f['delay_min']:.0f} minutes late" if f.get("delay_min") is not None else "there is a service problem")
    parts = [f"On **Route {route}**, {what}" + (f" near **{stop}**." if stop else ".")]
    td = f.get("transit_dependence")
    if td is not None:
        parts.append(f"In the neighborhoods along the next stops, about **{td:.0%} of households have no car**, so riders "
                     "depend on this bus.")
    rk = ranked_options(c)
    if rk:
        model = " (this uses a v7 model depot, not a real one)" if rk[0]["option_id"].startswith("dispatch_second_bus") else ""
        parts.append(f"The system recommends **{rk[0]['label']}**{model}" +
                     (f" (best of {len(rk)} possible actions)." if len(rk) > 1 else ".") + " You make the final decision.")
    else:
        parts.append("No action is possible right now, so the safe default applies: rider information only.")
    return " ".join(parts)


def explain_more(c):
    ex = c.get("explanation") or {}
    with st.expander("Why? (detailed explanation)"):
        if ex.get("narrative"):
            st.markdown(ex["narrative"])
        st.markdown(ex.get("brief_md") or "")


def countdown(c):
    if c["status"] not in tc.REVIEW_STATUSES:
        return
    left = max(0, int(APP.seconds_left(c)))
    mm, s = divmod(left, 60)
    who = "supervisor" if c["status"] == "AWAITING_SECOND_REVIEW" else "dispatcher"
    color = "#dc2626" if left < 60 else "#d97706" if left < 180 else "#059669"
    st.markdown(f"<div class='card' style='border-color:{color}'>⏱️ <b style='color:{color}'>{mm:02d}:{s:02d}</b> left for the "
                f"{who}. If nobody decides in time, the system falls back to the safe default (rider information only).</div>",
                unsafe_allow_html=True)


@st.fragment(run_every=5)
def live_countdown(cid):
    """Ticks every 5 s; when the deadline passes, the case goes to the safe default and the whole page refreshes."""
    c = APP.cases.get(cid)
    if not c or c["status"] not in tc.REVIEW_STATUSES:
        return
    with APP.lock:
        expired = APP.enforce_deadline(c)
    if expired:
        st.rerun(scope="app")
    countdown(c)


def checklist(prefix):
    st.markdown("**Before you decide, confirm:**")
    done = [t for t in tc.REVIEW_CHECKLIST if st.checkbox(t, key=f"{prefix}_{t}")]
    viewed = any("alternative" in t.lower() for t in done)        # the checklist item covers reviewing all alternatives
    return done, viewed


# ================================================================================================ sidebar
with st.sidebar:
    st.markdown("### 🚌 Transit Response")
    st.caption("Newark Penn Station area · live NJ Transit data")

    st.markdown("#### Step 1 · Live buses")
    if st.button("↻ Refresh live data", width="stretch", type="primary"):
        with st.spinner("Pulling NJ Transit, Census and weather …"):
            run(tc.refresh_live_context, ok="Live data refreshed.")
            tc.network_or_none()
        st.rerun()
    nj = tc.LIVE_CONTEXT["nj_transit"]
    n_bus = len(tc.live_vehicles())
    net_ok = tc._NETWORK.get("net") is not None
    st.markdown(
        f"{'🟢' if n_bus else '⚪'} **{n_bus}** live buses nearby  \n"
        f"{'🟢' if net_ok else '🟡'} Route map {'ready' if net_ok else 'loading'}  \n"
        f"{'🟢' if tc.LIVE_CONTEXT['weather'].get('source') == 'NWS live' else '⚪'} Weather "
        f"{tc.LIVE_CONTEXT['weather'].get('current_conditions') or ''}  \n"
        f"<span class='muted'>Updated {(nj.get('fetched_at') or 'not yet')[11:16]} UTC</span>", unsafe_allow_html=True)

    st.markdown("#### Step 2 · Who are you?")
    role_keys = list(T.ROLE_KEYS)
    role = st.radio("Your role", role_keys, format_func=lambda k: T.label(TEAM[k]),
                    index=role_keys.index(ss.get("role", "dispatcher")), label_visibility="collapsed")
    if role != ss.get("role"):
        ss["role"] = role
        ss.pop("case_id", None)          # jump to the case that is waiting for this role
    me = TEAM[role]
    st.caption(me["does"])

    st.markdown("#### Case")
    ids = list(APP.cases)[::-1]
    if ids:
        current = pick_case()
        sel = st.selectbox("Case", ids, index=ids.index(current["case_id"]),
                           format_func=lambda i: case_label(APP.cases[i]), label_visibility="collapsed")
        ss["case_id"] = sel
    else:
        st.caption("No cases yet. Open one from a live bus (Live map tab).")
    use_llm = st.toggle("AI-written explanations", value=tc.LLM_READY, disabled=not tc.LLM_READY,
                        help="Uses your LLM key to write the plain-language explanation. The decision rules never use AI.")

case = pick_case()

# ================================================================================================ header
st.markdown(f"<div class='hero'><div class='t'>Transit Reliability & Equity</div></div>"
            f"<div class='s muted'>Live bus disruption response · signed in as <b>{me['name']}</b> ({me['title']})</div>",
            unsafe_allow_html=True)
flash = ss.pop("flash", None)
if flash:
    (st.success if flash[0] == "ok" else st.error)(flash[1])

if case:
    words, color = STATUS_WORDS.get(case["status"], (case["status"], "#111827"))
    raw = case["raw"]
    st.markdown(
        f"<div class='card case-card' style='--c:{color}'><span class='pill' style='--c:{color}'>{words}</span>"
        f"<div class='big' style='margin-top:6px'><b>{case['case_id']}</b> · Route <b>{raw.get('route_name') or raw.get('route_id')}</b> · "
        f"Bus {raw.get('vehicle_id', '—')} · {fmt_delay(case)}</div>"
        f"<div class='muted'>Next stop: {((raw.get('real') or {}).get('stops') or [{}])[0].get('name', '—')}</div></div>",
        unsafe_allow_html=True)
    steps = T.progress(tc, case)
    who = {"detect": "Live feed", "dispatcher": TEAM["dispatcher"]["name"], "supervisor": TEAM["supervisor"]["name"],
           "officer": TEAM["officer"]["name"], "driver": TEAM["driver"]["name"], "done": TEAM["officer"]["name"]}
    st.markdown("<div class='steps'>" + "".join(
        f"<div class='step {s}'><b>{who[k]}</b>{STEP_ICON[s]} {lab}</div>" for k, lab, s in steps) + "</div>",
        unsafe_allow_html=True)
    nxt = T.whose_turn(tc, case)
    if nxt and nxt != role:
        st.info(f"Next: **{TEAM[nxt]['name']} ({TEAM[nxt]['title']})**. Switch your role in Step 2 to act as them.")

tab_task, tab_map, tab_case, tab_records, tab_check, tab_help = st.tabs(
    ["✅ My task", "🗺️ Live map", "📄 Case summary", "🗂️ Records", "🔌 System check", "❓ How to use"])

# ================================================================================================ My task
with tab_task:
    if role == "dispatcher":
        st.subheader(f"{me['icon']} Dispatcher: {me['name']}")
        if case and case["status"] == "AWAITING_REVIEW":
            live_countdown(case["case_id"])
            st.markdown("#### What's happening")
            st.markdown(plain_summary(case))
            show_df(simple_options_df(case))
            explain_more(case)
            opts = {r["option_id"]: f"#{r['rank']} {r['label']}" for r in ranked_options(case)}
            with st.form("dispatcher_review"):
                dec = st.radio("Your decision", ["Approve", "Revise", "Reject"], horizontal=True,
                               captions=["send it to the supervisor", "ask for other options", "safe default instead"])
                oid = st.selectbox("Action to approve (the top-ranked one is recommended)", list(opts), format_func=opts.get) \
                    if opts else None
                reason = st.text_area("Your reason (required)", placeholder="Why this action? At least 15 characters, e.g. 'Riders need to know about the 25-minute delay now.'")
                done, viewed = checklist("d")
                with st.expander("If revising: change the constraints (up to two revisions)"):
                    rv_excl = st.multiselect("Leave out these action types", list(tc.OPTION_LABELS),
                                             format_func=tc.OPTION_LABELS.get)
                    rv_budget = st.slider("Budget cap (USD)", 100, 2000, 1000, 50)
                    rv_hold = st.slider("Longest connection hold (min)", 0, 15, 5)
                    rv_alt = st.checkbox("Allow a bus from another depot")
                if st.form_submit_button(f"Submit as {me['name']}", type="primary"):
                    rev = ({"excluded_options": rv_excl, "budget_cap_usd": float(rv_budget), "max_hold_min": float(rv_hold),
                            "allow_alternate_depot": rv_alt} if dec == "Revise" else None)
                    run(APP.submit_first, case, me["name"], dec, oid, reason, done, viewed, rev)
                    st.rerun()
        elif case and case["status"] == "CONFLICT_REVIEW":
            st.warning("Two data sources disagree about this bus. Check it, then choose:")
            with st.form("operator"):
                res = st.radio("Result", ["resolved", "unresolved"], horizontal=True,
                               captions=["trust the vehicle data and continue", "can't tell which is right: keep monitoring"])
                if st.form_submit_button(f"Submit as {me['name']}", type="primary"):
                    run(APP.operator_resolve, case, me["name"], res); st.rerun()
        elif case and case["status"] in ("MONITORING", "SAFE_DEFAULT"):
            st.info(STATUS_WORDS[case["status"]][0] + (f". Reason: {case.get('safe_default_reason')}" if case.get("safe_default_reason") else ""))
            c1, c2 = st.columns(2)
            if c1.button("Recheck with fresh live data"):
                run(APP.monitor_action, case, "recheck", me["name"]); st.rerun()
            with c2.popover("Close as a false alarm"):
                why = st.text_input("Why is it a false alarm?")
                if st.button("Close case"):
                    run(APP.close_false_alarm, case, me["name"], why); st.rerun()
        elif case:
            st.success("Nothing is waiting for you on this case.")
        pending_mine = bool(case and case["status"] in ("AWAITING_REVIEW", "CONFLICT_REVIEW"))
        with st.expander("➕ Open a new case from a live bus", expanded=not pending_mine):
            choices = tc.vehicle_choices()
            if not choices:
                st.caption("No live buses yet. Press **↻ Refresh live data** in the sidebar (Step 1).")
            else:
                labels = {vid: lab for lab, vid in choices}
                vid = st.selectbox("Live bus (most delayed first)", [v for _, v in choices], format_func=labels.get)
                if st.button("Open case for this bus", type="primary"):
                    with st.spinner("Checking the data, scoring the options, writing the explanation …"):
                        c = run(lambda: APP.new_case(tc.raw_for_vehicle(vid), use_llm=use_llm))
                    if c:
                        ss["case_id"] = c["case_id"]
                    st.rerun()

    elif role == "supervisor":
        st.subheader(f"{me['icon']} Supervisor: {me['name']}")
        if case and case["status"] == "AWAITING_SECOND_REVIEW":
            live_countdown(case["case_id"])
            f = case["review"]["first"]
            st.markdown(f"<div class='card'><b>{f['reviewer']}</b> (Dispatcher) approved "
                        f"<b>{option_text(case, f['option_id'])}</b><br><span class='muted'>“{f['reason']}”</span></div>",
                        unsafe_allow_html=True)
            st.markdown(plain_summary(case))
            show_df(simple_options_df(case))
            explain_more(case)
            with st.form("supervisor_review"):
                dec = st.radio("Your decision", ["Approve", "Reject"], horizontal=True,
                               captions=["send it to the operations officer", "safe default instead"])
                reason = st.text_area("Your reason (required)", placeholder="At least 15 characters")
                done, viewed = checklist("s")
                if st.form_submit_button(f"Submit as {me['name']}", type="primary"):
                    run(APP.submit_second, case, me["name"], dec, reason, done, viewed); st.rerun()
        else:
            st.success("Nothing is waiting for your approval on this case." if case else
                       "No cases yet. The dispatcher opens cases from live buses.")

    elif role == "officer":
        st.subheader(f"{me['icon']} Operations Officer: {me['name']}")
        if case and case["status"] == "AWAITING_IMPLEMENTATION":
            st.markdown(f"<div class='card'>✅ Approved action: <b>{option_text(case, case['approved_option_id'])}</b><br>"
                        f"<span class='muted'>Carry it out through normal dispatch, then send the alert to the driver.</span></div>",
                        unsafe_allow_html=True)
            sms_on = T.twilio_ready(tc)
            st.caption(("📱 A text message will go to the driver's phone " + T._mask(T.driver_phone(tc)) + ", and the alert "
                        "appears on the driver screen.") if sms_on else
                       "📱 Text messaging isn't set up yet (see System check). The alert will still appear on the driver screen.")
            with st.form("officer_impl"):
                body = st.text_area("Message to the driver", T.compose_driver_message(tc, case), height=110, max_chars=320)
                res = st.radio("Result of the action", ["Completed", "Partial", "Failed"], horizontal=True)
                notes = st.text_input("Notes (required if Partial or Failed)")
                if st.form_submit_button("Send alert to driver and record", type="primary"):
                    def _do():
                        if res in ("Partial", "Failed") and not notes.strip():
                            raise ValueError("Describe what went wrong (notes are required for Partial or Failed).")
                        rec = T.send_driver_alert(tc, case, body, me["name"])
                        APP.record_implementation(case, me["name"], res, notes)
                        sms = {"sent": "Text message sent.", "failed": "The text message failed: " + rec["sms_detail"],
                               "not sent": "Posted to the driver screen (text messaging not set up)."}[rec["sms"]]
                        APP.last_message = f"Recorded as {res}. {sms}"
                    run(_do); st.rerun()
        elif case and case["status"] == "IMPLEMENTED":
            alerts = case.get("driver_alerts") or []
            if alerts:
                a = alerts[-1]
                st.markdown(f"Driver alert: text **{a['sms']}** · " +
                            (f"acknowledged by {a['acknowledged_by']} at {a['acknowledged_at'][11:16]} UTC" if a["acknowledged_at"]
                             else "**not acknowledged yet**"))
            st.markdown("#### Record the outcome")
            st.caption("Enter what actually happened, from operations records. Nothing here is estimated by the system.")
            pr = APP.predicted_for(case)
            st.caption(f"The system predicted: {pr['delay_reduction_min']} min of delay saved, about "
                       f"{pr['vulnerable_riders_helped']} riders who depend on transit helped, cost ${pr['cost_usd']:.0f}.")
            with st.form("outcome"):
                c1, c2, c3 = st.columns(3)
                obs = {}
                fields = list(tc.OBSERVED_FIELDS.items())
                for n, (k, lab) in enumerate(fields[:4]):
                    obs[k] = (c1, c2, c3)[n % 3].number_input(lab, value=None, placeholder="enter value")
                obs["rider_reported_impact"] = c2.selectbox(tc.OBSERVED_FIELDS["rider_reported_impact"], [1, 2, 3, 4, 5], index=None)
                fa = c3.selectbox("Was it a false alarm?", ["no", "yes"], index=None)
                obs["false_alarm"] = {"yes": True, "no": False}.get(fa)
                if st.form_submit_button(f"Save outcome as {me['name']}", type="primary"):
                    run(APP.evaluate_outcome, case, obs); st.rerun()
        elif case and case.get("outcome"):
            st.success("Outcome recorded. This case is closed.")
            show_df(tc.outcome_df(case))
        else:
            st.success("Nothing is waiting for you on this case." if case else "No cases yet.")
        if case and case["status"] in ("IMPLEMENTED", "EVALUATED") and case.get("driver_alerts"):
            with st.expander("Send another alert to the driver"):
                with st.form("resend"):
                    body2 = st.text_area("Message", T.compose_driver_message(tc, case), max_chars=320)
                    if st.form_submit_button("Send"):
                        run(lambda: T.send_driver_alert(tc, case, body2, me["name"]), ok="Alert sent again."); st.rerun()

    else:  # driver
        st.subheader(f"{me['icon']} Driver screen: {me['name']}")
        inbox = T.driver_inbox(tc)
        if not inbox:
            st.info("No alerts yet. Alerts from the operations officer appear here (and by text message, if set up).")
        for c, i, a in inbox:
            cls = "alert-sms ack" if a["acknowledged_at"] else "alert-sms"
            st.markdown(f"<div class='{cls}'><div class='muted'>{a['ts'][11:16]} UTC · from {a['from']} · {c['case_id']} · "
                        f"text {a['sms']}</div><div class='big' style='margin-top:4px'>{tc._html.escape(a['body'])}</div></div>",
                        unsafe_allow_html=True)
            if a["acknowledged_at"]:
                st.caption(f"✅ Acknowledged at {a['acknowledged_at'][11:16]} UTC")
            elif st.button("👍 Acknowledge", key=f"ack_{c['case_id']}_{i}", type="primary"):
                run(T.acknowledge, tc, c, i, me["name"]); st.rerun()

# ================================================================================================ Live map
with tab_map:
    @st.fragment
    def live_map():
        html = tc.map_html(case)
        if hasattr(st, "iframe"):
            st.iframe(html, height=560)
        else:
            components.html(html, height=560)
        st.caption(tc.map_status_md())
    live_map()
    with st.expander(f"All live buses near Newark Penn ({len(tc.live_vehicles())})"):
        show_df(tc.live_vehicles_df())
    if role != "dispatcher":
        st.caption("To open a case for one of these buses, switch to the Dispatcher role (Step 2).")

# ================================================================================================ Case summary
with tab_case:
    if not case:
        st.info("No case selected yet.")
    else:
        raw, A = case["raw"], case["analysis"]
        f = A.get("fields", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delay", fmt_delay(case))
        c2.metric("Households without a car", "—" if f.get("transit_dependence") is None else f"{f['transit_dependence']:.0%}")
        c3.metric("Live weather alerts", len(tc.LIVE_CONTEXT["weather"].get("active_alerts", [])))
        c4.metric("Priority (0–1)", "—" if not A.get("scores") else f"{A['scores']['P']:.2f}")
        st.markdown("#### What's happening")
        st.markdown(plain_summary(case))
        explain_more(case)
        st.markdown("#### Possible actions")
        show_df(simple_options_df(case))
        inf = [o for o in A.get("options", []) if not o["feasible"]]
        if inf:
            st.caption("Not possible right now: " + "; ".join(f"{o['label']} ({o['infeasible_reason']})" for o in inf))
        if case.get("rider_alert"):
            st.markdown(f"**Rider alert (verified information only):** {case['rider_alert']}")
        if case.get("driver_alerts"):
            st.markdown("#### Driver alerts")
            for a in case["driver_alerts"]:
                st.markdown(f"- {a['ts'][11:16]} UTC · {a['from']} → driver · text {a['sms']} · "
                            + ("✅ acknowledged" if a["acknowledged_at"] else "not acknowledged yet"))
        st.markdown("#### Timeline")
        people = {m["name"] for m in TEAM.values()}
        rows = [{"Time (UTC)": e["ts"][11:19], "Who": e["actor"], "What": e["event"],
                 "Details": (e["detail"] or "")[:140]} for e in case["audit"]
                if e["role"] in ("reviewer", "second reviewer", "operations", "driver", "rule") or e["actor"] in people]
        import pandas as pd
        show_df(pd.DataFrame(rows) if rows else tc.ph(None, "No events yet"))
        names = {"vision_ridership": "rider counts", "connecting_services": "connecting service", "incident_history": "incident history",
                 "operating_resources": "spare buses and depots", "community_access": "neighborhood data", "event_signals": "event report"}
        if raw.get("sample_inputs"):
            st.caption("Live: the bus, its delay, NJ Transit alerts, route and stops" +
                       ("" if "community_access" in raw["sample_inputs"] else ", Census") + ", weather. From v7's sample data "
                       "(no live source connected): " + ", ".join(names.get(x, x) for x in raw["sample_inputs"]) +
                       f". Scored on v7 model route {(raw.get('real') or {}).get('model_route_id', raw.get('route_id'))}.")

# ================================================================================================ Records (analysts)
with tab_records:
    st.caption("Technical details for analysts and the policy owner. Not needed to run a case.")
    with st.expander("📈 Scores and formulas (active case)"):
        if case:
            a, b = st.columns(2)
            with a:
                st.markdown("**Confidence** C = 0.40Q + 0.25F + 0.20G + 0.15M"); show_df(tc.conf_df(case)); st.markdown(tc.sig_md(case))
            with b:
                st.plotly_chart(tc.rvap_plot(case), width="stretch")
            st.markdown("**Normalization and impact**"); show_df(tc.norm_df(case))
            st.markdown("**Scenario ranking**"); show_df(tc.options_df(case)); st.plotly_chart(tc.options_plot(case), width="stretch")
    with st.expander("🔎 Evidence, exceptions and alerts (active case)"):
        if case:
            show_df(tc.evidence_df(case)); show_df(tc.exceptions_df(case)); show_df(tc.alerts_df(case))
    with st.expander("📚 All cases, audit record and reviewer patterns"):
        show_df(tc.cases_df())
        rs = APP.rubber_stamp_report()
        st.caption("Rubber-stamp protection: approval patterns and response times")
        show_df(rs if len(rs) else tc.ph(None, "No decisions yet"))
        st.caption("Audit record (newest first)"); show_df(tc.audit_df())
        st.caption("Monitoring queue"); show_df(tc.monitor_df())
    with st.expander("🛡️ Team training and conflict of interest"):
        st.markdown(tc.eth_summary_md())
        st.caption(f"Roster: {T.ROSTER_SOURCE['roster']}")
        show_df(tc.eth_status_df())
        st.caption(f"Conflict-of-interest disclosures: {T.ROSTER_SOURCE['disclosures']}")
        show_df(tc.disclosures_df())
        with st.form("load_roster"):
            txt = st.text_area("Replace the roster (JSON)", json.dumps([o.model_dump() for o in tc.ETH_ROSTER], indent=1), height=180)
            if st.form_submit_button("Load roster"):
                def _load():
                    n = tc.load_eth_roster(json.loads(txt)); T.ROSTER_SOURCE["roster"] = "pasted in the app"
                    APP.last_message = f"Roster loaded ({n} people)."
                run(_load); st.rerun()
    with st.expander("⚖️ Policy governance (Insight Agent, rule versions)"):
        if st.button("Run Insight Agent (proposes only)"):
            props = run(tc.insight_agent, APP)
            if props is not None:
                ss["flash"] = ("ok", f"Insight Agent drafted {len(props)} proposal(s). Nothing changes until the policy owner approves.")
            st.rerun()
        st.markdown(APP.insight_narrative or "_Run the Insight Agent after some outcomes are recorded._")
        show_df(tc.proposals_df())
        if APP.proposals:
            with st.form("policy"):
                pid = st.selectbox("Proposal", [p["id"] for p in APP.proposals])
                owner = st.text_input("Policy owner name", value=TEAM["supervisor"]["name"])
                why = st.text_input("Reason (required)")
                c1, c2 = st.columns(2)
                if c1.form_submit_button("Approve → backtest → release"):
                    run(tc.decide_proposal, APP, pid, owner, "Approve", why); st.rerun()
                if c2.form_submit_button("Reject"):
                    run(tc.decide_proposal, APP, pid, owner, "Reject", why); st.rerun()
        if APP.backtest:
            st.markdown(APP.backtest["summary"]); show_df(APP.backtest["df"])
        st.caption("Rule versions"); show_df(tc.versions_df())
    with st.expander("⬇️ Export everything (zip: JSON, CSV and map)"):
        if st.button("Prepare export"):
            ss["export"] = tc.export_outputs()
        if ss.get("export") and os.path.exists(ss["export"]):
            p = ss["export"]
            st.download_button("Download", open(p, "rb").read(), file_name=os.path.basename(p), mime="application/zip")

# ================================================================================================ System check
with tab_check:
    st.markdown("Tests every connection with your keys. It sends no text message and opens no case.")
    test_llm = st.checkbox("Also test the AI key (makes one very small paid call)")
    if st.button("Run system check", type="primary"):
        with st.spinner("Testing NJ Transit, Census, weather, map, AI and text messaging …"):
            ss["check"] = T.system_check(tc, test_llm=test_llm)
    for r in ss.get("check", []):
        st.markdown(f"{'✅' if r['ok'] else '❌'} **{r['service']}**: {r['detail']}" + (f"  \n↳ *Fix:* {r['fix']}" if r["fix"] else ""))
    st.caption(tc.map_status_md())

# ================================================================================================ Help
with tab_help:
    t = TEAM
    st.markdown(f"""
### How to run a case

1. **Step 1 · Live buses.** Press **↻ Refresh live data** in the sidebar. The map shows live NJ Transit buses near
   Newark Penn Station and the real bus routes.
2. **Step 2 · Who are you?** Pick your role in the sidebar. Each person sees only their own task on **✅ My task**,
   and the step tracker at the top shows whose turn it is.
3. **{t['dispatcher']['name']} (Dispatcher)** picks a delayed bus and presses **Open case for this bus**, reads the
   recommendation, ticks the checklist, writes a reason and presses **Submit**.
4. **{t['supervisor']['name']} (Supervisor)** sees the dispatcher's decision, then approves or rejects it with a reason.
5. **{t['officer']['name']} (Operations Officer)** carries out the approved action, edits the message if needed and
   presses **Send alert to driver and record**. The driver gets a text message and the alert appears on the driver screen.
6. **{t['driver']['name']}** opens the driver screen and presses **👍 Acknowledge**.
7. **{t['officer']['name']}** records the outcome when the incident is over. The case is then closed.

### Good to know
- **Deadlines are real:** 5, 10 or 20 minutes depending on how serious the case is. If nobody decides in time, the
  system falls back to the safe default: verified rider information only.
- **Approvals need training:** only people on the training roster can approve, and nobody can approve their own decision twice.
- **Missing data stays missing:** if a live source has no data, the system says so instead of guessing.
- **Nothing is dispatched automatically:** approved actions are recorded for operations staff to carry out.
- **Something not working?** Open **🔌 System check**.
""")
