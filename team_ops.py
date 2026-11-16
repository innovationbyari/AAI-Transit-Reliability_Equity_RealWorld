"""Team workflow for the live dashboard: the four people (Step 2), their training records, the driver alert
(text message through Twilio + the driver's screen) and a connection check for every live service.

The v7 engine (transit_ai/) is unchanged in what it scores and decides. This module only:
  * makes the team the ethics-training roster (plus anyone in REVIEWER_ROSTER_JSON), replacing the notebook's
    fictional sample roster, and loads real disclosures from STAFF_DISCLOSURES_JSON (the fictional ones are removed),
  * makes the supervisor's approval required for every action (REQUIRE_SUPERVISOR_APPROVAL, on by default),
  * sends and records the driver alert, and lets the driver acknowledge it.
"""
from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone

import requests

# ------------------------------------------------------------------------------------------------ the team
# Names can be changed with secrets (DISPATCHER_NAME, SUPERVISOR_NAME, OFFICER_NAME, DRIVER_NAME).
# Training: all four required courses completed on TEAM_TRAINING_DATE (recorded as instructed on 2026-11-16).
TEAM_TRAINING_DATE_DEFAULT = "2026-11-16"

ROLE_KEYS = ("dispatcher", "supervisor", "officer", "driver")


def _secret(tc, name, default=""):
    return tc.read_secret(name) or default


def team(tc):
    """The four roles, in workflow order."""
    return {
        "dispatcher": {"name": _secret(tc, "DISPATCHER_NAME", "Ariyan"), "title": "Dispatcher", "icon": "🎧",
                       "does": "Opens a case from a live bus and makes the first decision (1st reviewer)."},
        "supervisor": {"name": _secret(tc, "SUPERVISOR_NAME", "Terri-Ann"), "title": "Supervisor", "icon": "🧭",
                       "does": "Independently approves or rejects the dispatcher's decision (2nd reviewer)."},
        "officer": {"name": _secret(tc, "OFFICER_NAME", "Grant"), "title": "Operations Officer", "icon": "🛠️",
                    "does": "Carries out the approved action and sends the alert to the bus driver."},
        "driver": {"name": _secret(tc, "DRIVER_NAME", "Bus Driver"), "title": "Bus Driver", "icon": "🚌",
                   "does": "Receives the alert by text message and on the driver screen, and acknowledges it."},
    }


def label(member):
    if member["name"].strip().lower() == member["title"].lower():
        return f"{member['icon']} {member['title']}"
    return f"{member['icon']} {member['name']} — {member['title']}"


def require_supervisor(tc):
    return (tc.read_secret("REQUIRE_SUPERVISOR_APPROVAL") or "1").strip().lower() not in ("0", "false", "no", "off")


_SETUP_DONE = {"done": False}
_LOCK = threading.Lock()


def setup(tc):
    """Run once per server: roster + supervisor rule. Safe to call on every page load."""
    with _LOCK:
        if _SETUP_DONE["done"]:
            return
        _ensure_roster(tc)
        _install_supervisor_rule(tc)
        _SETUP_DONE["done"] = True


ROSTER_SOURCE = {"roster": "team (Step 2)", "disclosures": "none on file"}


def _ensure_roster(tc):
    """Roster = the team (+ REVIEWER_ROSTER_JSON). The notebook's fictional sample roster and disclosures are not used."""
    import json
    rows = []
    extra = tc.read_secret("REVIEWER_ROSTER_JSON")
    if extra:
        try:
            rows = [r for r in json.loads(extra) if isinstance(r, dict) and r.get("name")]
            ROSTER_SOURCE["roster"] = "REVIEWER_ROSTER_JSON secret + team (Step 2)"
        except ValueError as e:
            tc.APP.last_message = f"REVIEWER_ROSTER_JSON could not be read ({e}); using the team roster."
    date = _secret(tc, "TEAM_TRAINING_DATE", TEAM_TRAINING_DATE_DEFAULT)
    have = {r["name"].strip().lower() for r in rows}
    t = team(tc)
    depts = {"dispatcher": "Bus Dispatch", "supervisor": "Bus Operations Supervision", "officer": "Bus Operations"}
    for key in ("dispatcher", "supervisor", "officer"):
        if t[key]["name"].strip().lower() not in have:
            rows.append({"name": t[key]["name"], "department": depts[key], "completed": list(tc.ETH_REQUIRED),
                         "last_training": date})
    tc.load_eth_roster(rows)
    # conflict-of-interest disclosures: only real ones from the secret
    tc.STAFF_DISCLOSURES.clear()
    disc = tc.read_secret("STAFF_DISCLOSURES_JSON")
    if disc:
        try:
            for name, d in json.loads(disc).items():
                facts = d.get("facts") or {"official_name": name}
                tc.STAFF_DISCLOSURES[name.strip()] = {"text": d.get("text", ""), "facts": tc.ExtractedFacts(**facts)}
            ROSTER_SOURCE["disclosures"] = "STAFF_DISCLOSURES_JSON secret"
        except (ValueError, TypeError, AttributeError) as e:
            tc.APP.last_message = f"STAFF_DISCLOSURES_JSON could not be read ({e})."


def _install_supervisor_rule(tc):
    """Every approval goes to the supervisor when REQUIRE_SUPERVISOR_APPROVAL is on (default)."""
    app = tc.APP
    original = type(app).is_high_impact

    def is_high_impact(case, option_id):
        return True if require_supervisor(tc) else original(app, case, option_id)

    app.is_high_impact = is_high_impact


# ------------------------------------------------------------------------------------------------ driver alert
def _mask(phone):
    digits = re.sub(r"\D", "", phone or "")
    return f"•••-•••-{digits[-4:]}" if len(digits) >= 4 else "(no number)"


def driver_phone(tc):
    return _secret(tc, "DRIVER_PHONE_NUMBER")


def twilio_ready(tc):
    sid, tok = _secret(tc, "TWILIO_ACCOUNT_SID"), _secret(tc, "TWILIO_AUTH_TOKEN")
    sender = _secret(tc, "TWILIO_FROM_NUMBER") or _secret(tc, "TWILIO_MESSAGING_SERVICE_SID")
    return bool(sid and tok and sender and driver_phone(tc))


def option_for(case):
    oid = case.get("approved_option_id")
    return next((o for o in case["analysis"].get("options", []) if o["option_id"] == oid), None)


INSTRUCTIONS = {
    "rider_alerts": "Rider alerts have been sent. Let riders know about the delay at each stop and continue your route.",
    "protect_connection": "Hold at the connection stop for transferring riders, then continue.",
    "reroute": "Follow the detour around the blockage. Dispatch will confirm when the normal route reopens.",
    "dispatch_second_bus": "A second bus is being sent to help. Continue service and expect it behind you.",
    "do_nothing": "No change to your trip. Continue service.",
}


def compose_driver_message(tc, case):
    """A short operational text for the driver of the affected bus (kept under ~300 characters)."""
    raw = case["raw"]
    o = option_for(case)
    kind = o["type"] if o else "rider_alerts"
    action = (o["label"] if o else "Rider information only").split(" — ")[0]
    stop = ((raw.get("real") or {}).get("stops") or raw.get("stops") or [{}])[0].get("name")
    lines = [f"NJ TRANSIT OPS · Route {raw.get('route_name') or raw.get('route_id')} · Bus {raw.get('vehicle_id') or '—'}",
             f"Approved: {action}.", INSTRUCTIONS.get(kind, "Follow dispatch instructions.")]
    if stop:
        lines.append(f"Next stop: {stop}.")
    lines.append(f"Ref {case['case_id']}. Please acknowledge.")
    return " ".join(lines)[:320]


def send_driver_alert(tc, case, body, sender_name):
    """Send the text (if Twilio is set up) and always post the alert to the driver screen."""
    body = (body or "").strip()
    if not body:
        raise ValueError("The driver message is empty.")
    phone = driver_phone(tc)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from": sender_name, "body": body,
           "to": _mask(phone), "sms": "not sent", "sms_detail": "", "acknowledged_at": None, "acknowledged_by": None}
    if twilio_ready(tc):
        ok, detail = _twilio_send(tc, phone, body)
        rec["sms"], rec["sms_detail"] = ("sent" if ok else "failed"), detail
    else:
        rec["sms_detail"] = "Text messaging is not set up (Twilio secrets or DRIVER_PHONE_NUMBER missing)."
    case.setdefault("driver_alerts", []).append(rec)
    tc.APP.log(case, sender_name, "operations", "Driver alert",
               f"Alert to driver · text {rec['sms']}", f"{body} [{rec['to']}] {rec['sms_detail']}")
    return rec


def _twilio_send(tc, to, body):
    sid, tok = _secret(tc, "TWILIO_ACCOUNT_SID"), _secret(tc, "TWILIO_AUTH_TOKEN")
    data = {"To": to, "Body": body}
    if _secret(tc, "TWILIO_MESSAGING_SERVICE_SID"):
        data["MessagingServiceSid"] = _secret(tc, "TWILIO_MESSAGING_SERVICE_SID")
    else:
        data["From"] = _secret(tc, "TWILIO_FROM_NUMBER")
    try:
        r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", data=data,
                          auth=(sid, tok), timeout=20)
        js = r.json() if r.content else {}
    except (requests.RequestException, ValueError) as e:
        return False, f"could not reach Twilio ({type(e).__name__})"
    if r.status_code in (200, 201):
        return True, f"Twilio message {js.get('sid', '')} ({js.get('status', 'queued')})"
    return False, f"Twilio error {js.get('code', r.status_code)}: {js.get('message', r.text[:160])}"


def acknowledge(tc, case, index, driver_name):
    alerts = case.get("driver_alerts") or []
    if not 0 <= index < len(alerts):
        raise ValueError("That alert no longer exists.")
    a = alerts[index]
    if a["acknowledged_at"]:
        return a
    a["acknowledged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    a["acknowledged_by"] = driver_name
    tc.APP.log(case, driver_name, "driver", "Driver alert", "Acknowledged", a["body"][:120])
    tc.APP.last_message = f"{driver_name} acknowledged the alert for {case['case_id']}."
    return a


def driver_inbox(tc):
    """All alerts sent to the driver, newest first: (case, index, alert)."""
    out = []
    for c in tc.APP.cases.values():
        for i, a in enumerate(c.get("driver_alerts") or []):
            out.append((c, i, a))
    return sorted(out, key=lambda x: x[2]["ts"], reverse=True)


# ------------------------------------------------------------------------------------------------ workflow progress
STEPS = [("detect", "Bus problem detected"), ("dispatcher", "Dispatcher decides"), ("supervisor", "Supervisor approves"),
         ("officer", "Officer carries it out"), ("driver", "Driver acknowledges"), ("done", "Outcome recorded")]


def progress(tc, case):
    """For the step tracker: list of (key, label, state) with state in done / current / waiting / skipped / stopped."""
    st = case["status"] if case else None
    rv = (case or {}).get("review") or {}
    alerts = (case or {}).get("driver_alerts") or []
    acked = any(a["acknowledged_at"] for a in alerts)
    implemented = bool((case or {}).get("implementation"))
    if not case:
        return [(k, l, "current" if k == "detect" else "waiting") for k, l in STEPS]
    if st in ("MONITORING", "CONFLICT_REVIEW", "SAFE_DEFAULT", "CLOSED_FALSE_ALARM"):
        state = {"detect": "done"}
        return [(k, l, state.get(k, "stopped")) for k, l in STEPS]
    first_done = bool(rv.get("first"))
    second_needed = rv.get("required_second", True)
    second_done = bool(rv.get("second")) or (first_done and not second_needed)
    states = {
        "detect": "done",
        "dispatcher": "done" if first_done else ("current" if st == "AWAITING_REVIEW" else "waiting"),
        "supervisor": ("skipped" if first_done and not second_needed else "done" if second_done else
                       "current" if st == "AWAITING_SECOND_REVIEW" else "waiting"),
        "officer": "done" if implemented and st != "AWAITING_IMPLEMENTATION" else (
            "current" if st == "AWAITING_IMPLEMENTATION" else "waiting"),
        "driver": "done" if acked else ("current" if alerts else "waiting"),
        "done": "done" if st == "EVALUATED" else ("current" if st == "IMPLEMENTED" else "waiting"),
    }
    return [(k, l, states[k]) for k, l in STEPS]


def whose_turn(tc, case):
    """The role that should act next on this case (or None)."""
    if not case:
        return "dispatcher"
    st = case["status"]
    if st == "AWAITING_REVIEW":
        return "dispatcher"
    if st == "AWAITING_SECOND_REVIEW":
        return "supervisor"
    if st == "AWAITING_IMPLEMENTATION":
        return "officer"
    alerts = case.get("driver_alerts") or []
    if alerts and not all(a["acknowledged_at"] for a in alerts):
        return "driver"
    if st == "IMPLEMENTED":
        return "officer"
    return None


# ------------------------------------------------------------------------------------------------ system check
def _ok(name, ok, detail, fix=""):
    return {"service": name, "ok": ok, "detail": detail, "fix": "" if ok else fix}


def system_check(tc, test_llm=False):
    """Test every live service with the deployment's keys. No text message is sent and no case is created."""
    out = []
    # NJ Transit login + realtime feed
    if not (tc.read_secret("NJ_TRANSIT_USERNAME") and tc.read_secret("NJ_TRANSIT_PASSWORD")):
        out.append(_ok("NJ Transit login", False, "no NJ_TRANSIT_USERNAME / NJ_TRANSIT_PASSWORD secret",
                       "Add both secrets (developer.njtransit.com account)."))
    else:
        tc._NJT["token"] = None
        status = tc.njt_authenticate()
        ok = bool(tc._NJT["token"])
        out.append(_ok("NJ Transit login", ok, status, "Check the username/password; the account must have GTFS access."))
        if ok:
            try:
                feed = tc.njt_feed("getVehiclePositions")
                out.append(_ok("NJ Transit live buses", True, f"{len(feed.entity)} buses in the statewide feed"))
            except Exception as e:
                out.append(_ok("NJ Transit live buses", False, f"{type(e).__name__}: {str(e)[:140]}",
                               "Try again in a minute; if it persists, check the account's API access."))
    # static GTFS (routes, stops, route shapes)
    net = tc._NETWORK.get("net")
    if net is not None:
        try:
            shapes = net.q("SELECT COUNT(DISTINCT shape_id) AS n FROM shapes")[0]["n"]
        except Exception:
            shapes = 0
        out.append(_ok("NJ Transit routes & stops", True,
                       f"{net.counts['routes']} routes, {net.counts['stops']} stops, {shapes} route shapes ({net.source})"))
    else:
        err = tc._NETWORK.get("error")
        out.append(_ok("NJ Transit routes & stops", False, err or "still loading in the background (first start takes a few minutes)",
                       "Wait a few minutes and run the check again; if it fails, set NJT_GTFS_ZIP."))
    # Census
    try:
        c = tc.fetch_census(tc.LAT, tc.LON)
        ok = str(c.get("status", "")).startswith("ok")
        out.append(_ok("US Census", ok, c.get("tract_name") or c.get("status", "no answer"),
                       "Check CENSUS_API_KEY, or remove it to use the keyless limit."))
        if ok:
            tc.LIVE_CONTEXT["census"] = c
    except Exception as e:
        out.append(_ok("US Census", False, f"{type(e).__name__}: {str(e)[:140]}", "Check CENSUS_API_KEY."))
    # Weather
    try:
        w = tc.fetch_weather(tc.LAT, tc.LON)
        ok = w.get("source") == "NWS live"
        n_ok = sum(r["status"] == "ok" for r in w.get("connectors", []))
        out.append(_ok("National Weather Service", ok, f"{n_ok}/{len(w.get('connectors', []))} connectors answered",
                       "Set NWS_CONTACT_EMAIL; the service asks callers to identify themselves."))
        if ok:
            tc.LIVE_CONTEXT["weather"] = w
    except Exception as e:
        out.append(_ok("National Weather Service", False, type(e).__name__, "Try again later."))
    # CARTO map tiles
    key = tc.read_secret("CARTO_API_KEY")
    if not key:
        out.append(_ok("CARTO street map", False, "no CARTO_API_KEY secret: the map uses OpenStreetMap instead",
                       "Get a free key at carto.com/basemaps/apikey and add CARTO_API_KEY."))
    else:
        try:
            r = requests.get(f"https://basemaps.cartocdn.com/rastertiles/voyager/13/2408/3079.png?key={key}", timeout=15)
            ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image")
            out.append(_ok("CARTO street map", ok, "map tiles load with your key" if ok else f"HTTP {r.status_code}",
                           "Check the key; if you restricted it to websites, allow your *.streamlit.app address."))
        except requests.RequestException as e:
            out.append(_ok("CARTO street map", False, type(e).__name__, "Try again later."))
    # LLM (one tiny call, only when asked: it uses API credit)
    if not tc.LLM_READY:
        out.append(_ok("AI narratives (LLM)", False, "no LLM key: explanations use built-in templates",
                       "Optional: add OPENAI_API_KEY (or OPENROUTER_API_KEY / HF_API_KEY)."))
    elif test_llm:
        try:
            txt = tc.llm_generate("Reply with the single word OK.", "You are a connectivity test.", max_tokens=20)
            out.append(_ok("AI narratives (LLM)", bool(txt), f"replied: {str(txt)[:40]}" if txt else "no reply",
                           "Check the key and the account's credit."))
        except Exception as e:
            out.append(_ok("AI narratives (LLM)", False, f"{type(e).__name__}: {str(e)[:140]}", "Check the key and credit."))
    else:
        out.append(_ok("AI narratives (LLM)", True, "key found: " + ", ".join(p["label"] for p in tc.LLM_CHAIN) +
                       " (tick 'test the AI key' to make one test call)"))
    # Twilio (checks the account; sends nothing)
    sid, tok = _secret(tc, "TWILIO_ACCOUNT_SID"), _secret(tc, "TWILIO_AUTH_TOKEN")
    if not (sid and tok):
        out.append(_ok("Driver text messages (Twilio)", False, "no TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN",
                       "Add the Twilio secrets; the driver screen still works without them."))
    else:
        try:
            r = requests.get(f"https://api.twilio.com/2010-04-01/Accounts/{sid}.json", auth=(sid, tok), timeout=15)
            js = r.json() if r.content else {}
            ok = r.status_code == 200
            missing = [n for n in ("TWILIO_FROM_NUMBER or TWILIO_MESSAGING_SERVICE_SID", "DRIVER_PHONE_NUMBER")
                       if (n.startswith("TWILIO") and not (_secret(tc, "TWILIO_FROM_NUMBER") or _secret(tc, "TWILIO_MESSAGING_SERVICE_SID")))
                       or (n == "DRIVER_PHONE_NUMBER" and not driver_phone(tc))]
            detail = (f"account {js.get('friendly_name', '')} ({js.get('type', '')}, {js.get('status', '')})" if ok
                      else f"HTTP {r.status_code}: {js.get('message', '')[:120]}")
            if ok and missing:
                ok, detail = False, detail + " · missing " + ", ".join(missing)
            if ok and js.get("type") == "Trial":
                detail += " · trial account: the driver's number must be a Verified Caller ID"
            out.append(_ok("Driver text messages (Twilio)", ok, detail,
                           "Check the Account SID / Auth Token, and add the sender number and DRIVER_PHONE_NUMBER."))
        except (requests.RequestException, ValueError) as e:
            out.append(_ok("Driver text messages (Twilio)", False, type(e).__name__, "Try again later."))
    return out
