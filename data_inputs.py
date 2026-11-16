"""Section A · Live connectors (NJ Transit, Census ACS, NWS), data inputs, SLM, hand-off adapter

Generated from the notebook (cells 16, 17, 18). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# A1 · Live data connectors: NJ Transit GTFS-realtime, US Census ACS, National Weather Service   [notebook cell 16]
# ----------------------------------------------------------------------------------------------------
DATA_MODE = param("DATA_MODE", "live")  # env-var override; notebook form field
NJT_GTFS_SYSTEM = param("NJT_GTFS_SYSTEM", "GTFSG2")  # env-var override; notebook form field
NJT_NEARBY_KM = param("NJT_NEARBY_KM", 2.0)  # env-var override; notebook form field
ACS_YEAR = param("ACS_YEAR", 2023)  # env-var override; notebook form field
import re
from datetime import datetime, timezone, timedelta

LAT, LON = SERVICE_AREA_LATLON           # Newark Penn Station
LIVE = DATA_MODE == "live"


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
        return "skipped (no NJ_TRANSIT_USERNAME / NJ_TRANSIT_PASSWORD in the app secrets)"
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


def njt_feed(endpoint):
    from google.transit import gtfs_realtime_pb2
    url = NJT_BASE.format(system=_NJT["system"]) + endpoint
    r = requests.post(url, files={"token": (None, _NJT["token"])}, timeout=30)
    r.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)
    return feed


def _text(ts):
    return ts.translation[0].text.strip() if ts and len(ts.translation) else ""


def fetch_njt_service():
    """Live service feed near Newark Penn Station: the most-delayed nearby vehicle and any alert on its route."""
    out = {"status": njt_authenticate(), "feeds": {}, "service": None, "alert": None}
    if not _NJT["token"]:
        return out
    feeds = {}
    for ep in ("getVehiclePositions", "getTripUpdates", "getAlerts"):
        try:
            feeds[ep] = njt_feed(ep)
            out["feeds"][ep] = f"ok ({len(feeds[ep].entity)} entities)"
        except Exception as e:
            out["feeds"][ep] = f"failed ({type(e).__name__})"
    vp, tu, al = feeds.get("getVehiclePositions"), feeds.get("getTripUpdates"), feeds.get("getAlerts")
    if vp is None:
        return out
    updates = {}
    if tu is not None:
        for ent in tu.entity:
            if ent.HasField("trip_update"):
                t = ent.trip_update
                delays = [s.arrival.delay if s.HasField("arrival") else s.departure.delay
                          for s in t.stop_time_update if s.HasField("arrival") or s.HasField("departure")]
                updates[t.trip.trip_id] = {"delay_s": delays[0] if delays else (t.delay if t.HasField("delay") else None),
                                           "cancelled": t.trip.schedule_relationship == t.trip.CANCELED,
                                           "route_id": t.trip.route_id}
    nearby = []
    for ent in vp.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        d = km_between((v.position.latitude, v.position.longitude), (LAT, LON))
        if d <= NJT_NEARBY_KM:
            u = updates.get(v.trip.trip_id, {})
            nearby.append({"route_id": v.trip.route_id or u.get("route_id"), "trip_id": v.trip.trip_id,
                           "vehicle_id": v.vehicle.id, "lat": v.position.latitude, "lon": v.position.longitude,
                           "distance_km": round(d, 2), "delay_s": u.get("delay_s"), "cancelled": u.get("cancelled", False),
                           "timestamp": v.timestamp or vp.header.timestamp})
    out["nearby_vehicles"] = len(nearby)
    if not nearby:
        return out
    pick = max(nearby, key=lambda n: (n["cancelled"], n["delay_s"] or 0))
    out["service"] = pick
    if al is not None:
        for ent in al.entity:
            if ent.HasField("alert") and any(ie.route_id == pick["route_id"] for ie in ent.alert.informed_entity):
                a = ent.alert
                out["alert"] = {"header": _text(a.header_text), "description": _text(a.description_text)[:400],
                                "effect": a.Effect.Name(a.effect) if a.effect else None}
                break
    return out


# ---------------------------------------------------------------- US Census ACS (tract-level equity inputs)
ACS_DETAILED = {"B08201_001E": "households", "B08201_002E": "households_no_vehicle",
                "B08301_001E": "workers", "B08301_010E": "workers_public_transit", "B19013_001E": "median_household_income"}
ACS_SUBJECT = {"S1701_C03_001E": "poverty_rate_pct", "S1810_C03_001E": "disability_rate_pct"}


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


def fetch_census(lat, lon):
    key = read_secret("CENSUS_API_KEY")
    res = {"status": "not attempted", "tract": None}
    try:
        r = requests.get("https://geocoding.geo.census.gov/geocoder/geographies/coordinates",
                         params={"x": lon, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current",
                                 "layers": "Census Tracts", "format": "json"}, timeout=30)
        r.raise_for_status()
        tracts = _find_key(r.json(), "Census Tracts")
        geo = tracts[0] if tracts else None
        if not geo:
            res["status"] = "geocoder returned no tract"
            return res
        d = _acs("acs/acs5", ACS_DETAILED, geo, key)
        s = _acs("acs/acs5/subject", ACS_SUBJECT, geo, key)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as e:
        res["status"] = f"failed ({type(e).__name__})"
        return res
    share = lambda a, b: round(a / b, 3) if (a is not None and b) else None
    res.update({
        "status": f"ok (ACS {ACS_YEAR} 5-year{'' if key else ', no key'})",
        "tract": geo.get("GEOID") or f"{geo['STATE']}{geo['COUNTY']}{geo['TRACT']}", "tract_name": d["NAME"],
        "zero_vehicle_share": share(d["households_no_vehicle"], d["households"]),
        "transit_commute_share": share(d["workers_public_transit"], d["workers"]),
        "poverty_rate": None if s["poverty_rate_pct"] is None else round(s["poverty_rate_pct"] / 100, 3),
        "disability_rate": None if s["disability_rate_pct"] is None else round(s["disability_rate_pct"] / 100, 3),
        "median_household_income": d["median_household_income"],
    })
    return res


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


_fetch_njt_once = fetch_njt_service


def fetch_njt_service():
    out = _fetch_njt_once()
    out["fetched_at"] = iso(utcnow())
    return out


# Live connectors run ONCE here (not on every dashboard click). Re-run this cell to refresh.
LIVE_CONTEXT = {
    "nj_transit": fetch_njt_service() if LIVE else {"status": "sample mode", "service": None, "alert": None, "feeds": {}},
    "census": fetch_census(LAT, LON) if LIVE else {"status": "sample mode", "tract": None},
    "weather": fetch_weather(LAT, LON) if LIVE else {"source": "sample mode", "active_alerts": []},
}
_n, _c, _w = LIVE_CONTEXT["nj_transit"], LIVE_CONTEXT["census"], LIVE_CONTEXT["weather"]
print(f"NJ Transit GTFS-RT: {_n['status']} · feeds: {_n.get('feeds') or '-'} · nearby vehicles: {_n.get('nearby_vehicles', '-')}")
if _n.get("service"):
    s = _n["service"]
    print(f"  selected route {s['route_id']} (vehicle {s['vehicle_id']}, {s['distance_km']} km away): "
          f"delay {None if s['delay_s'] is None else round(s['delay_s'] / 60, 1)} min, cancelled={s['cancelled']}, "
          f"alert: {(_n.get('alert') or {}).get('header') or 'none'}")
print(f"US Census ACS: {_c['status']}" + (f" · {_c['tract_name']} · zero-vehicle {_c['zero_vehicle_share']}, "
      f"poverty {_c['poverty_rate']}, disability {_c['disability_rate']}, transit commute {_c['transit_commute_share']}"
      if _c.get("tract") else ""))
print(f"NWS weather: {_w.get('source')} · {_w.get('current_conditions') or 'conditions not reported'} · active alerts: {len(_w.get('active_alerts', []))} · "
      f"connectors ok: {sum(r['status'] == 'ok' for r in _w.get('connectors', []))}/{len(_w.get('connectors', []))}")



# ----------------------------------------------------------------------------------------------------
# A2 · Data inputs (live where available, sample otherwise) + small language model + hand-off to the Signal Scout   [notebook cell 17]
# ----------------------------------------------------------------------------------------------------
USE_SLM = param("USE_SLM", True)  # env-var override; notebook form field
USE_LLM_ORGANIZER = param("USE_LLM_ORGANIZER", True)  # env-var override; notebook form field
SLM_MODEL_ID = "google/flan-t5-small"

# 1. DATA INPUTS — seven sources. Each value is live when its connector succeeded, otherwise the documented sample;
# every source says which ("source"), and every source carries "observed_at" so freshness can be measured.
def _cv_stop_camera():
    cams = [c for c in CV_RESULTS.get("cameras", []) if c["role"] == "bus stop curb camera"] or CV_RESULTS.get("cameras", [])
    return cams[0] if cams else None


def section_a_sample_inputs():
    """Name kept for the dashboard. Returns live inputs where available (DATA_MODE='live'), samples otherwise."""
    n, c, w = LIVE_CONTEXT["nj_transit"], LIVE_CONTEXT["census"], LIVE_CONTEXT["weather"]
    svc_live, alert = n.get("service"), n.get("alert")
    cam = _cv_stop_camera()
    if svc_live:
        # keep one decimal: rounding 14.7 → 15 would cross the 15-minute hard trigger on its own
        delay = None if svc_live["delay_s"] is None else max(0.0, round(svc_live["delay_s"] / 60, 1))
        service = {"route": svc_live["route_id"], "scheduled_headway_min": 15, "headway_source": "assumed (no static GTFS loaded)",
                   "delay_min": delay, "cancelled": bool(svc_live["cancelled"]), "trip_id": svc_live["trip_id"],
                   "vehicle_position": f"{svc_live['lat']:.5f},{svc_live['lon']:.5f} ({svc_live['distance_km']} km from Newark Penn)",
                   "source": f"NJ Transit {_NJT['system']} GTFS-realtime (live)",
                   "observed_at": iso(datetime.fromtimestamp(svc_live["timestamp"], timezone.utc))
                   if svc_live.get("timestamp") else _ago(0)}
    else:
        service = {"route": "NJ-001", "scheduled_headway_min": 15, "delay_min": 18, "cancelled": False,
                   "vehicle_position": "Newark Penn Station", "source": "sample_data", "observed_at": _ago(30)}
    if alert:
        text = f"{alert['header']}. {alert['description']}".strip(". ")
        events = {"source_type": "official_alert", "report": text, "report_source": "NJ Transit GTFS-realtime alert (live)",
                  "source": "NJ Transit GTFS-realtime alert (live)",
                  "road_closure": any(k in text.lower() for k in ("detour", "road closure", "closed"))}
    elif svc_live:
        events = {"source_type": "none", "report_source": "generated from the live vehicle feed (no active alert)",
                  "source": "NJ Transit GTFS-realtime vehicle feed (live; no alert)",
                  "report": f"Route {service['route']} is running {service['delay_min'] or 0:g} minutes late near Newark Penn "
                            f"Station. No active NJ Transit service alert for this route.", "road_closure": False}
    else:
        events = {"source_type": "official_alert", "report_source": "sample_data", "source": "sample_data", "road_closure": True,
                  "report": "Route NJ-001 has an 18 minute delay near Newark Penn Station due to a road closure."}
    w_live = w.get("source") == "NWS live"
    events.update(weather=(w.get("current_conditions") or "not reported") if w_live else "Clear",
                  weather_alerts=w.get("active_alerts", []) if w_live else [],
                  weather_source="NWS live" if w_live else "sample_data", observed_at=_ago(90) if not alert else _ago(0))
    if w_live:
        events["source"] += " + NWS weather (live)"
    if cam:
        # The camera sees one stop: it measures crowding there (load factor), not total riders on the route,
        # so route boardings stay the documented sample and are labeled as such.
        vision = {"boardings": 45, "boardings_source": "sample_data (route boardings are not visible from one camera)",
                  "waiting_at_stop": cam["person_count"], "load_factor": cam["load_factor"], "alightings": None,
                  "capacity": cam["capacity"], "pass_ups": 0, "pass_ups_source": "not observable in a still image (0 assumed)",
                  "obstruction": False, "identities_stored": False, "camera_id": cam["camera_id"],
                  "detector": CV_RESULTS["detector"], "source": "computer vision (anonymized counts)", "observed_at": cam["observed_at"]}
    else:
        vision = {"boardings": 45, "alightings": 12, "capacity": 50, "pass_ups": 5, "obstruction": False,
                  "identities_stored": False, "source": "sample_data", "observed_at": _ago(20)}
    if c.get("tract") and None not in (c["zero_vehicle_share"], c["poverty_rate"], c["disability_rate"]):
        community = {"transit_dependence": c["zero_vehicle_share"], "income_disadvantage": c["poverty_rate"],
                     "disability_barrier": c["disability_rate"], "transit_commute_share": c["transit_commute_share"],
                     "median_household_income": c["median_household_income"], "essential_destination": "Hospital",
                     "essential_destination_source": "sample_data",
                     "data_level": f"{c['tract_name']} (tract {c['tract']}, ACS {ACS_YEAR} 5-year)",
                     "source": "US Census ACS (live)", "observed_at": _ago(0)}
    else:
        community = {"transit_dependence": 0.75, "income_disadvantage": 0.65, "disability_barrier": 0.40,
                     "essential_destination": "Hospital", "data_level": "Area aggregate", "source": "sample_data",
                     "observed_at": _ago(30 * 86400)}
    return {
        "service_feeds": service,
        "vision_ridership": vision,
        "connecting_services": {"connecting_route": "NJ-002", "arrival_delay_min": 8, "transfer_window_min": 10,
                                "last_service": False, "source": "sample_data", "observed_at": _ago(60)},
        "incident_history": {"past_disruptions": 3, "average_recovery_min": 25, "previous_response": "Rider alert",
                             "source": "sample_data", "observed_at": _ago(86400)},
        "community_access": community,
        "operating_resources": {"available_buses": 2, "available_operators": 2, "estimated_dispatch_cost": 150,
                                "estimated_travel_min": 12, "source": "sample_data", "observed_at": _ago(120)},
        "event_signals": events,
    }


NJT_MAX_SNAPSHOT_AGE_S = 240   # refetch the vehicle feed if A1 ran more than 4 minutes ago (freshness limit is 300 s)


def refresh_live_service(max_age_s=NJT_MAX_SNAPSHOT_AGE_S):
    """A1 fetches once; if this cell runs later the snapshot may be stale, so take a new one before the hand-off."""
    n = LIVE_CONTEXT.get("nj_transit", {})
    fetched = n.get("fetched_at") or iso(utcnow())
    if LIVE and n.get("status", "").startswith("authenticated") and (utcnow() - parse_iso(fetched)).total_seconds() > max_age_s:
        LIVE_CONTEXT["nj_transit"] = fetch_njt_service()
        print(f"NJ Transit snapshot was older than {max_age_s} s → refetched: {LIVE_CONTEXT['nj_transit']['status']}")


refresh_live_service()
raw_inputs = section_a_sample_inputs()


# 2. STANDARDIZE AND COLLATE INPUT DATA
def collate_inputs(data):
    combined = []
    collated_at = datetime.now(timezone.utc)
    for source_name, source_data in data.items():
        observed = source_data.get("observed_at")
        age_s = int((collated_at - parse_iso(observed)).total_seconds()) if observed else None
        combined.append({
            "source": source_name,
            "timestamp": observed or iso(collated_at),
            "collated_at": iso(collated_at),
            "age_s": age_s,
            "data": source_data,
            "status": "unverified",
        })
    return combined


# 3. SMALL LANGUAGE MODEL - FLAN-T5
# loaded as a seq2seq model with explicit generation arguments.
# one whole-report prompt made FLAN-T5 small echo the report back (observed in an earlier run).
# It now answers one short question per field, the task format FLAN-T5 was trained on.
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
        print("Loading small language model...")
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
    """Kept for the dashboard: one readable line built from the per-field answers."""
    a = slm_answers(report)
    if a is None:
        return f"[SLM unavailable — {_SLM['error'] or 'disabled'}; deterministic fields only]"
    return "; ".join(f"{k}: {v}" for k, v in a.items())


# 4. CREATE STRUCTURED EVENT FIELDS (deterministic; numbers never come from a language model)
# broader delay patterns
DELAY_PATTERNS = [
    # a range ("10 to 15 minutes") reports the upper end
    r"(\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)",
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
    """Flags where the SLM reading disagrees with the deterministic fields (for the reviewer, not for scoring)."""
    if not answers:
        return []
    flags = []
    if delay_min is not None and str(delay_min) not in answers.get("delay", ""):
        flags.append(f"SLM delay '{answers.get('delay')}' differs from extracted {delay_min} min")
    if route and route.lower() not in answers.get("route", "").lower():
        flags.append(f"SLM route '{answers.get('route')}' differs from feed route {route}")
    return flags


# 5. LLM ORGANIZER (shared live client; organizes the record for the Signal Scout — does not score, verify or recommend)
def llm_organize(report, slm_text, weather, sources):
    if not USE_LLM_ORGANIZER:
        return None, "disabled"
    if not LLM_READY:
        return None, "skipped (no LLM key in the app secrets)"
    try:
        txt = llm_generate(json.dumps({"report": report, "slm_interpretation": slm_text, "weather": weather,
                                       "data_sources": sources}),
                           "Organize transit incident data for the Signal Scout agent in at most 6 short bullet points. "
                           "Separate live information from sample information, using the data_sources labels. "
                           "The incident is NOT verified yet: describe it as reported, never as confirmed. "
                           "Do not calculate scores, verify incidents, or recommend operational actions.",
                           max_tokens=300)
        bullets = [l.strip() for l in txt.splitlines() if l.strip().startswith(("-", "•", "*")) and len(l.strip()) > 2][:6]
        # Verification is a fixed-rule step that has not run yet, so the organizer may not call anything confirmed
        # ("confirmed" is replaced only when nothing already negates it: not / never / unconfirmed / yet to be)
        bullets = [__import__("re").sub(r"(?i)(?<!not )(?<!never )(?<!be )(?<!yet )(?<!un)\bconfirmed\b",
                                        "reported (not yet verified)", b) for b in bullets]
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
    delay = extract_delay(event_report)
    route = inputs["service_feeds"]["route"]
    weather_ctx = {"conditions": ev.get("weather"), "alerts": ev.get("weather_alerts", []),
                   "source": ev.get("weather_source", "sample_data")}
    llm_text, llm_status = llm_organize(event_report, slm_text, weather_ctx,
                                        {k: v.get("source", "sample_data") for k, v in inputs.items()})
    structured_event = {
        "route": route,
        "delay_min": delay,
        "cause": classify_cause(event_report),
        "report_text": event_report,
        "slm_interpretation": slm_text,
        "slm_answers": answers,
        "slm_cross_check": slm_cross_check(answers, delay, route),
        "llm_interpretation": llm_text,
        "source_type": ev.get("source_type", "public_report"),
        "verification_status": "pending",
    }
    return {
        "case_status": "new",
        "status": "awaiting_verification",
        "created_at": iso(utcnow()),
        "route": route,
        "event": structured_event,
        "all_inputs": collated_data,
        "external_feeds": {
            "data_mode": DATA_MODE,
            "nj_transit": {k: v for k, v in LIVE_CONTEXT["nj_transit"].items() if k in ("status", "feeds", "nearby_vehicles")},
            "census": LIVE_CONTEXT["census"],
            "weather": {k: v for k, v in LIVE_CONTEXT["weather"].items() if k != "connectors"},
            "weather_connectors": LIVE_CONTEXT["weather"].get("connectors", []),
            "computer_vision": {"detector": CV_RESULTS.get("detector"), "cameras": CV_RESULTS.get("cameras", [])},
            "source_of_each_input": {k: v.get("source", "sample_data") for k, v in inputs.items()},
            "llm": llm_status,
        },
        "verification_required": True,
        "confidence_score": None,
        "priority_score": None,
        "audit_log": [{"step": "input_preparation", "status": "completed",
                       "message": "Data collated and prepared for Signal Scout"}],
    }


signal_scout_input = build_signal_scout_input(raw_inputs)

# 6. EXPORT AND DISPLAY
with open(os.path.join(OUTPUT_DIR, "signal_scout_input.json"), "w") as file:
    json.dump(signal_scout_input, file, indent=4, default=str)

print("===== SOURCE OF EACH INPUT =====")
for k, v in signal_scout_input["external_feeds"]["source_of_each_input"].items():
    print(f"  {k:22s} {v}")
print("LLM organizer:", signal_scout_input["external_feeds"]["llm"])
if signal_scout_input["event"]["llm_interpretation"]:
    print(signal_scout_input["event"]["llm_interpretation"])
print("===== SMALL LANGUAGE MODEL OUTPUT (supporting only) =====")
print(signal_scout_input["event"]["slm_interpretation"])
for flag in signal_scout_input["event"]["slm_cross_check"]:
    print("  check:", flag)
print("===== STRUCTURED EVENT (deterministic) =====")
print({k: v for k, v in signal_scout_input["event"].items()
       if k not in ("slm_interpretation", "slm_answers", "slm_cross_check", "llm_interpretation")})
print("JSON file saved:", os.path.join(OUTPUT_DIR, "signal_scout_input.json"))



# ----------------------------------------------------------------------------------------------------
# A · INTEGRATION — adapter from Section A's hand-off to the shared raw bundle   [notebook cell 18]
# ----------------------------------------------------------------------------------------------------
# Routes that are not in the synthetic network are drawn on a stand-in route; A's own values are kept.
EXTERNAL_ROUTE_MAP = {"NJ-001": "R2", "12": "R3"}
ESSENTIAL_NAME_TO_TYPE = {"hospital": "hospital", "dialysis": "dialysis", "clinic": "clinic", "school": "school",
                          "college": "school", "university": "school", "employment": "employment_center",
                          "job": "employment_center", "market": "retail", "grocery": "retail"}
NWS_SEVERITY_RANK = {"Unknown": 0, "Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
NWS_TO_EVENT_SEVERITY = {"Unknown": "minor", "Minor": "minor", "Moderate": "minor", "Severe": "major", "Extreme": "emergency"}
SEVERITY_BY_CAUSE = {"road_closure": "major", "obstruction": "major", "weather": "minor", "mechanical": "minor",
                     "medical": "minor", "none": "none"}


def essential_type_of(name):
    n = (name or "").lower()
    return next((t for k, t in ESSENTIAL_NAME_TO_TYPE.items() if k in n), "none")


def nearest_depot_id(route_id, stop_range):
    stops = NET["routes"][route_id]["stops"][stop_range[0]:stop_range[1] + 1]
    mid = (statistics.mean(s["lat"] for s in stops), statistics.mean(s["lon"] for s in stops))
    return min(NET["depots"].values(), key=lambda d: km_between((d["lat"], d["lon"]), mid))["id"]


def transit_time_period(ts=None):
    """Service period from the local (New Jersey) time of the observation, instead of always assuming peak."""
    from zoneinfo import ZoneInfo
    t = (parse_iso(ts) if ts else utcnow()).astimezone(ZoneInfo("America/New_York"))
    h = t.hour + t.minute / 60
    if t.weekday() >= 5:
        return "midday" if 9 <= h < 20 else "off_peak"
    if 6 <= h < 9.5 or 15.5 <= h < 19:
        return "peak"
    if 9.5 <= h < 15.5:
        return "midday"
    return "off_peak"


def signal_scout_input_to_raw(ssi, name=None):
    """Section A record → raw bundle used by the Signal Scout / E2 engine. Numbers come only from structured fields
    or the deterministic regex; the SLM interpretation is attached as a note."""
    src = {r["source"]: r for r in ssi["all_inputs"]}
    svc, vis = src["service_feeds"]["data"], src["vision_ridership"]["data"]
    conn, hist = src["connecting_services"]["data"], src["incident_history"]["data"]
    comm, res = src["community_access"]["data"], src["operating_resources"]["data"]
    ev = src["event_signals"]["data"]
    ext_route = svc["route"]
    rid = EXTERNAL_ROUTE_MAP.get(ext_route, "R2")
    stop_range = (1, 5)
    headway = svc.get("scheduled_headway_min") or NET["routes"][rid]["headway_min"]
    delay = svc.get("delay_min")
    cancelled = bool(svc.get("cancelled"))
    # Gap between consecutive buses when one runs `delay` minutes late (or one trip is missing) ÷ scheduled headway
    gap_ratio = 2.0 if cancelled else (round((headway + (delay or 0)) / headway, 2) if headway else None)
    cause = ssi["event"]["cause"] if ssi["event"]["cause"] != "none" else ("road_closure" if ev.get("road_closure") else "none")
    stype = ev.get("source_type", "public_report")
    sources = ["vehicle_apc"] + ([stype] if stype in ("official_alert", "trusted_external", "public_report") else [])
    # weather no longer adds a generic trusted_external source. That path created a synthetic "511-TRAFFIC"
    # observation repeating the bus delay + 1 min, which inflated agreement (G) and confidence. Real NWS alerts are
    # added below as their own observation, with no delay value.
    cap = vis.get("capacity") or 60
    dep = nearest_depot_id(rid, stop_range)
    spec = dict(
        route_id=rid, stop_range=stop_range, time_period=transit_time_period(src["service_feeds"].get("observed_at")),
        delay_min=delay, headway_ratio=gap_ratio, headway_from_delay=True,   # gap is computed from the delay, not measured
        backup_available=False,   # no second live feed is connected: a stale live snapshot must go to monitoring, not to a simulated mirror
        cancelled=cancelled, cancellation_risk=1.0 if cancelled else 0.2, boardings=vis.get("boardings", 0),
        load_factor=vis.get("load_factor", round(vis.get("boardings", 0) / cap, 2)), passups=vis.get("pass_ups", 0),
        obstruction=bool(vis.get("obstruction")), event_type=cause, event_severity=SEVERITY_BY_CAUSE.get(cause, "minor"),
        event_verified=stype == "official_alert", sources=sources, official_delay=ssi["event"]["delay_min"],
        public_delay=ssi["event"]["delay_min"], public_corroborated=False,
        feed_age_s=src["service_feeds"]["age_s"] or 0, alert_age_s=src["event_signals"]["age_s"] or 0,
        last_service=bool(conn.get("last_service")),
        depot_overrides={dep: {"spare_buses": res.get("available_buses", 0), "operators": res.get("available_operators", 0),
                               "deadhead_min": res.get("estimated_travel_min"), "dispatch_cost": res.get("estimated_dispatch_cost")}},
    )
    raw = make_raw(spec, name or f"Section A hand-off · {ext_route} ({cause.replace('_', ' ')})")
    raw["external_route_id"] = ext_route                               # keep the real route label
    raw["route_name"] = f"{ext_route} (drawn on synthetic {rid})"
    raw["community"].update(transit_dependence=comm.get("transit_dependence"), income_disadvantage=comm.get("income_disadvantage"),
                            disability=comm.get("disability_barrier"), areas=[comm.get("data_level", "area aggregate")])
    et = essential_type_of(comm.get("essential_destination"))
    raw["access"].update(essential_type=et, essential_names=[comm.get("essential_destination")] if et != "none" else [])
    raw["connections"].update(external_connecting_route=conn.get("connecting_route"),
                              transfer_window_min=conn.get("transfer_window_min", 4),
                              connecting_arrival_delay_min=conn.get("arrival_delay_min"))
    raw["incident_history"] = hist
    raw["slm_interpretation"] = ssi["event"]["slm_interpretation"]
    raw["slm_cross_check"] = ssi["event"].get("slm_cross_check", [])
    raw["llm_interpretation"] = ssi["event"].get("llm_interpretation")       # supporting text only, never scored
    raw["external_feeds"] = ssi.get("external_feeds", {})
    # active NWS alerts become one trusted_external observation (event only, no delay).
    # Field precedence still applies: an agency alert's event fields outrank NWS.
    w_alerts = ev.get("weather_alerts") or []
    if w_alerts:
        top = max(w_alerts, key=lambda a: NWS_SEVERITY_RANK.get(a.get("severity"), 0))
        fetched = ssi.get("external_feeds", {}).get("weather", {}).get("fetched_at")
        age = int((utcnow() - parse_iso(fetched)).total_seconds()) if fetched else src["event_signals"]["age_s"] or 0
        raw["observations"].append({
            "source_id": "NWS-ALERTS", "source_type": "trusted_external", "timestamp": iso(utcnow() - timedelta(seconds=age)),
            "age_s": age, "corroborated": True,
            "fields": {"event_type": "weather", "event_severity": NWS_TO_EVENT_SEVERITY.get(top.get("severity"), "minor")},
            "nws_event": top.get("event"),
        })
    if vis.get("camera_id"):                                           # computer vision provenance (counts only)
        raw["vision"].update(camera_feeds=[vis["camera_id"]], detector=vis.get("detector"), source=vis.get("source"),
                             waiting_at_stop=vis.get("waiting_at_stop"), pass_ups_source=vis.get("pass_ups_source"))
    raw["weather_context"] = {"conditions": ev.get("weather"), "alerts": w_alerts, "source": ev.get("weather_source")}
    raw["handoff_from"] = "Section A — Inputs_SLM.ipynb"
    return raw


_raw_a = signal_scout_input_to_raw(signal_scout_input)
print("Section A hand-off converted:", _raw_a["scenario_name"], "| time period:", _raw_a["time_period"],
      "| delay:", _raw_a["spec"]["delay_min"], "min | observations (source, age s):",
      [(o["source_id"], o["age_s"]) for o in _raw_a["observations"]],
      "| weather:", _raw_a["weather_context"]["source"], _raw_a["weather_context"]["conditions"])
