"""Section E1 · Shared foundations: helpers, versioned policy rules, synthetic network, scenarios

Generated from the notebook (cells 8, 9, 10, 12). Modules in this package run in ONE shared
namespace, in the order listed in transit_ai/__init__.py, exactly like running the
notebook top to bottom. Names used here may therefore be defined in an earlier module.
"""
# ruff: noqa

# ----------------------------------------------------------------------------------------------------
# E1 · Imports, runtime settings and helpers   [notebook cell 8]
# ----------------------------------------------------------------------------------------------------
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

# Service area: Newark Penn Station. Section A's weather connectors use this point, and the synthetic (fictional)
# network is drawn over a real basemap around it. Change freely.
SERVICE_AREA_LATLON = (40.7347, -74.1642)
CITY_CENTER = SERVICE_AREA_LATLON
# 1.0 = real review deadlines (5/10/20 min). Set e.g. 0.1 to make deadlines 10x shorter for demos.
DEMO_DEADLINE_SCALE = param("DEMO_DEADLINE_SCALE", 1.0)  # env-var override; notebook form field
# Every timestamp the notebook shows or saves is in this time zone (EST/EDT for New Jersey).
LOCAL_TZ_NAME = param("LOCAL_TZ_NAME", "America/New_York")  # env-var override; notebook form field
from zoneinfo import ZoneInfo
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)
OUTPUT_DIR = "transit_outputs"
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


def stable_rng(*keys):
    """Seeded RNG so synthetic values are reproducible (Python's hash() is randomized)."""
    return random.Random(zlib.crc32("|".join(map(str, keys)).encode()))


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


# ----------------------------------------------------------------------------------------------------
# E1 · Policy rules (flowchart section 4) — versioned assumptions   [notebook cell 9]
# ----------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------
# E1 · Synthetic transit network (routes, stops, community areas, essential destinations, depots)   [notebook cell 10]
# ----------------------------------------------------------------------------------------------------
KM_LAT = 110.574


def _pt(x_km, y_km):
    lat0, lon0 = CITY_CENTER
    return (lat0 + y_km / KM_LAT, lon0 + x_km / (111.320 * math.cos(math.radians(lat0))))


def km_between(a, b):
    lat0 = math.radians((a[0] + b[0]) / 2)
    dx = (b[1] - a[1]) * 111.320 * math.cos(lat0)
    dy = (b[0] - a[0]) * KM_LAT
    return math.hypot(dx, dy)


def _interp(waypoints, n):
    pts = [_pt(*w) for w in waypoints]
    segs = [km_between(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(segs)
    out = []
    for k in range(n):
        d = total * k / (n - 1)
        i = 0
        while i < len(segs) - 1 and d > segs[i]:
            d -= segs[i]
            i += 1
        f = 0 if segs[i] == 0 else min(1.0, d / segs[i])
        out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f))
    return out


STREETS = ["Oak", "Maple", "Pine", "Cedar", "Elm", "Harbor", "Union", "Market", "Lake", "Park", "River", "Hill",
           "Mill", "Grove", "Spring", "Bay", "Main", "Ash", "Birch", "Linden"]

ROUTE_SPECS = [
    dict(id="R1", name="Crosstown", mode="bus", headway_min=10, waypoints=[(-7, 0.2), (7, 0.2)], n_stops=14, color="#1f77b4"),
    dict(id="R2", name="Hospital Line", mode="bus", headway_min=15, waypoints=[(-1.2, -7), (-1.0, 0), (1.5, 6.5)], n_stops=14, color="#9467bd"),
    dict(id="R3", name="Industrial Connector", mode="bus", headway_min=20, waypoints=[(-6.5, -5.5), (5.5, 4.5)], n_stops=12, color="#8c564b"),
    dict(id="R4", name="Northside", mode="bus", headway_min=12, waypoints=[(-6, 4.2), (0, 5.2), (6, 4.2)], n_stops=12, color="#2ca02c"),
    dict(id="R5", name="Southside", mode="bus", headway_min=15, waypoints=[(-6.5, -4.2), (6.5, -4.0)], n_stops=13, color="#ff7f0e"),
    dict(id="L1", name="Rail Line", mode="rail", headway_min=8, waypoints=[(3, -7.5), (3, 7.5)], n_stops=8, color="#d62728"),
]

ESSENTIAL_DESTINATIONS = [
    dict(id="D1", name="General Hospital", type="hospital", xy=(-1.6, -4.9)),
    dict(id="D2", name="Kidney Dialysis Center", type="dialysis", xy=(-0.5, -2.0)),
    dict(id="D3", name="Westfield High School", type="school", xy=(2.0, 0.6)),
    dict(id="D4", name="Riverside Employment Center", type="employment_center", xy=(4.4, 3.4)),
    dict(id="D5", name="Southside Community Clinic", type="clinic", xy=(-4.0, -4.6)),
    dict(id="D6", name="Northgate Market", type="retail", xy=(-2.5, 5.2)),
]

DEPOTS = [
    dict(id="DEPOT-A", name="Depot A (southwest)", xy=(-7.0, -6.5), spare_buses=2, operators=2, operated_by="Agency (in-house)"),
    dict(id="DEPOT-B", name="Depot B (northeast)", xy=(6.5, 6.5), spare_buses=1, operators=1, operated_by="Metro Coach Services LLC"),
]
# Outside parties involved in actions (shown in the brief for transparency and screened by the Section D conflict-of-interest gate)
CONNECTION_PARTNERS = {"L1": "Northeast Rail Partners", "R1": "Agency (in-house)", "R2": "Agency (in-house)",
                       "R3": "Agency (in-house)", "R4": "Agency (in-house)", "R5": "Agency (in-house)"}


def build_network():
    rng = stable_rng("network")
    routes = {}
    for spec in ROUTE_SPECS:
        coords = _interp(spec["waypoints"], spec["n_stops"])
        stops = []
        for i, (lat, lon) in enumerate(coords):
            stops.append({"stop_id": f"{spec['id']}-S{i:02d}", "route_id": spec["id"], "seq": i,
                          "name": f"{STREETS[(i * 3 + len(spec['id']) * 7 + ord(spec['id'][-1])) % len(STREETS)]} St "
                                  f"({spec['id']} #{i})", "lat": lat, "lon": lon})
        routes[spec["id"]] = {**{k: v for k, v in spec.items() if k != "waypoints"}, "stops": stops,
                              "path": [(s["lat"], s["lon"]) for s in stops]}
    # Community areas: 4 x 4 grid of 4 km cells; area-level aggregates only (no individual data)
    areas = []
    for gx in range(4):
        for gy in range(4):
            x0, y0 = -8 + gx * 4, -8 + gy * 4
            cx, cy = x0 + 2, y0 + 2
            sw01 = (1 + (-(cx / 6) - (cy / 6)) / 2) / 2  # 0 (northeast) .. 1 (southwest)
            sw01 = max(0.0, min(1.0, sw01))
            areas.append({
                "area_id": f"A{gx + 1}{gy + 1}", "name": f"Area {'ABCD'[gx]}{gy + 1}",
                "bounds": [_pt(x0, y0), _pt(x0 + 4, y0 + 4)],
                "transit_dependence": round(0.08 + 0.42 * sw01 + rng.uniform(-0.03, 0.03), 3),
                "income_disadvantage": round(0.06 + 0.30 * sw01 + rng.uniform(-0.02, 0.02), 3),
                "disability": round(0.07 + 0.12 * sw01 + rng.uniform(-0.01, 0.01), 3),
                "population": int(rng.uniform(8000, 26000)),
            })
    dests = [{**d, "lat": _pt(*d["xy"])[0], "lon": _pt(*d["xy"])[1]} for d in ESSENTIAL_DESTINATIONS]
    depots = [{**d, "lat": _pt(*d["xy"])[0], "lon": _pt(*d["xy"])[1]} for d in DEPOTS]
    all_stops = [s for r in routes.values() for s in r["stops"]]
    for s in all_stops:
        p = (s["lat"], s["lon"])
        s["area_id"] = next((a["area_id"] for a in areas
                             if a["bounds"][0][0] <= p[0] <= a["bounds"][1][0] and a["bounds"][0][1] <= p[1] <= a["bounds"][1][1]),
                            areas[0]["area_id"])
        s["essential"] = [d["id"] for d in dests if km_between(p, (d["lat"], d["lon"])) <= 0.9]
        s["nearby_routes"] = sorted({o["route_id"] for o in all_stops if o["route_id"] != s["route_id"]
                                     and km_between(p, (o["lat"], o["lon"])) <= 0.45})
        s["transfer_to"] = sorted({o["route_id"] for o in all_stops if o["route_id"] != s["route_id"]
                                   and km_between(p, (o["lat"], o["lon"])) <= 0.6})
    return {"routes": routes, "areas": {a["area_id"]: a for a in areas}, "destinations": {d["id"]: d for d in dests},
            "depots": {d["id"]: d for d in depots}, "stops": {s["stop_id"]: s for s in all_stops}}


NET = build_network()
print(f"Synthetic network: {len(NET['routes'])} routes, {len(NET['stops'])} stops, {len(NET['areas'])} community areas, "
      f"{len(NET['destinations'])} essential destinations, {len(NET['depots'])} depots")


# ----------------------------------------------------------------------------------------------------
# E1 · Scenario builder and preset incidents (synthetic data inputs)   [notebook cell 12]
# ----------------------------------------------------------------------------------------------------
SPEC_DEFAULTS = dict(
    route_id="R1", stop_range=(3, 7), time_period="peak", delay_min=10.0, headway_ratio=1.5, cancelled=False,
    cancellation_risk=0.2, boardings=150, load_factor=0.8, passups=0, obstruction=False,
    event_type="none", event_severity="none", event_verified=False,
    sources=["vehicle_apc"], feed_age_s=20, alert_age_s=40, external_age_s=60,
    official_delay=None, external_delay=None, public_delay=None, public_corroborated=False,
    backup_available=True, backup_delay=None, vision_agent_failures=0, detour_open=True, detour_extra_min=6,
    reroute_allowed=True, segment_open=True, missed_transfers=None, last_service=False, missing_fields=[],
    depot_overrides={},
)


def _crowding(load):
    return "unboardable" if load >= 1.0 else "high" if load >= 0.85 else "medium" if load >= 0.6 else "low"


def make_raw(spec, name="Custom scenario"):
    s = {**SPEC_DEFAULTS, **spec}
    rid = s["route_id"]
    route = NET["routes"][rid]
    i0, i1 = s["stop_range"]
    i1 = min(i1, len(route["stops"]) - 1)
    affected = route["stops"][i0:i1 + 1]
    now = utcnow()
    risk = 1.0 if s["cancelled"] else s["cancellation_risk"]

    def obs(source_id, stype, age_s, fields, corroborated=True):
        return {"source_id": source_id, "source_type": stype, "timestamp": iso(now - timedelta(seconds=age_s)),
                "age_s": age_s, "corroborated": corroborated, "fields": fields}

    observations = []
    if "vehicle_apc" in s["sources"]:
        observations.append(obs(f"AVL-{rid}", "vehicle_apc", s["feed_age_s"],
                                {"delay_min": s["delay_min"], "headway_ratio": s["headway_ratio"],
                                 "cancelled": s["cancelled"], "cancellation_risk": risk}))
    if "official_alert" in s["sources"]:
        f = {"cancelled": s["cancelled"], "event_type": s["event_type"], "event_severity": s["event_severity"],
             "event_verified": s["event_verified"]}
        if s["official_delay"] is not None:
            f["delay_min"] = s["official_delay"]
        observations.append(obs("AGENCY-ALERTS", "official_alert", s["alert_age_s"], f))
    if "trusted_external" in s["sources"]:
        sid = "NWS-WEATHER" if s["event_type"] == "weather" else "511-TRAFFIC"
        f = {"event_type": s["event_type"], "event_severity": s["event_severity"]}
        ext = s["external_delay"] if s["external_delay"] is not None else (s["delay_min"] + 1 if ("vehicle_apc" in s["sources"] and s["delay_min"] is not None) else None)
        if ext is not None:
            f["delay_min"] = ext
        observations.append(obs(sid, "trusted_external", s["external_age_s"], f))
    if "public_report" in s["sources"]:
        observations.append(obs("SOCIAL-REPORTS", "public_report", 90,
                                {"delay_min": s["public_delay"] if s["public_delay"] is not None else s["delay_min"],
                                 "event_type": s["event_type"]}, corroborated=s["public_corroborated"]))
    for o in observations:
        for m in s["missing_fields"]:
            o["fields"].pop(m, None)

    area_ids = sorted({st["area_id"] for st in affected})
    areas = [NET["areas"][a] for a in area_ids]
    community = {k: round(statistics.mean(a[k] for a in areas), 3) for k in ["transit_dependence", "income_disadvantage", "disability"]}
    community["areas"] = area_ids
    for m in s["missing_fields"]:
        community.pop(m, None)
    sev = DEFAULT_POLICY["lookups"]["essential_severity"]
    ess_ids = sorted({d for st in affected for d in st["essential"]})
    ess_types = [NET["destinations"][d]["type"] for d in ess_ids]
    essential_type = max(ess_types, key=lambda t: sev[t]) if ess_types else "none"
    alternatives = min(len(st["nearby_routes"]) for st in affected)

    # Connecting service: first affected/downstream stop with a transfer (rail preferred)
    conn = {"connecting_route": None, "partner": None, "transfer_stop_id": None, "transfer_window_min": 4, "connecting_headway_min": None,
            "last_service": s["last_service"], "missed_transfers": 0}
    downstream = route["stops"][i0:]
    for pref in (["L1"], None):
        for st in downstream:
            cands = [r for r in st["transfer_to"] if (pref is None or r in pref)]
            if cands:
                conn.update(connecting_route=cands[0], transfer_stop_id=st["stop_id"], partner=CONNECTION_PARTNERS.get(cands[0]),
                            connecting_headway_min=NET["routes"][cands[0]]["headway_min"])
                break
        if conn["connecting_route"]:
            break
    if conn["connecting_route"]:
        conn["missed_transfers"] = s["missed_transfers"] if s["missed_transfers"] is not None else int(round(s["boardings"] * 0.25))

    vision = {"camera_feeds": [f"VISION-{st['stop_id']}" for st in affected[:3]] + [f"VISION-ONBOARD-{rid}"],
              "boardings": s["boardings"], "alightings": int(s["boardings"] * 0.8), "capacity": 60,
              "load_factor": s["load_factor"], "passups": s["passups"], "obstruction": s["obstruction"],
              "crowding_level": _crowding(s["load_factor"]), "anonymized": True, "identities_stored": False}
    for m in s["missing_fields"]:
        vision.pop(m, None)

    depots = []
    for d in NET["depots"].values():
        o = s["depot_overrides"].get(d["id"], {})
        depots.append({"id": d["id"], "name": d["name"], "lat": d["lat"], "lon": d["lon"],
                       "spare_buses": o.get("spare_buses", d["spare_buses"]), "operators": o.get("operators", d["operators"]),
                       "operated_by": d["operated_by"], "deadhead_min_override": o.get("deadhead_min"),
                       "dispatch_cost_override": o.get("dispatch_cost")})

    raw = {
        "scenario_name": name, "route_id": rid, "route_name": route["name"], "headway_min": route["headway_min"],
        "affected_stop_ids": [st["stop_id"] for st in affected], "time_period": "last_service" if s["last_service"] else s["time_period"],
        "event_type": s["event_type"], "observations": observations, "vision": vision, "connections": conn,
        "community": community, "access": {"essential_type": essential_type, "essential_names": [NET["destinations"][d]["name"] for d in ess_ids],
                                            "alternatives": alternatives},
        "resources": {"depots": depots, "bus_hour_cost_usd": DEFAULT_POLICY["bus_hour_cost_usd"]},
        "roads": {"detour_open": s["detour_open"], "detour_extra_min": s["detour_extra_min"], "segment_open": s["segment_open"],
                  "reroute_allowed": s["reroute_allowed"]},
        "faults": {"vision_agent_failures": s["vision_agent_failures"], "backup_available": s["backup_available"]},
        "generated_at": iso(now), "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
    }
    if s["backup_available"]:
        bd = s["backup_delay"] if s["backup_delay"] is not None else s["delay_min"]
        raw["backup_observation"] = obs(f"GTFSRT-MIRROR-{rid}", "trusted_external", 45,
                                        {"delay_min": bd, "headway_ratio": s["headway_ratio"], "cancelled": s["cancelled"],
                                         "cancellation_risk": risk})
    return raw


PRESETS = {
    "Severe · R2 Hospital Line trip cancelled (verified)": dict(
        route_id="R2", stop_range=(1, 6), time_period="peak", delay_min=24, headway_ratio=2.6, cancelled=True,
        boardings=140, load_factor=0.95, passups=6, event_type="mechanical", event_severity="major", event_verified=True,
        sources=["vehicle_apc", "official_alert"], official_delay=25, missed_transfers=34),
    "Significant · R1 Crosstown unboardable crowding at peak": dict(
        route_id="R1", stop_range=(3, 8), time_period="peak", delay_min=8, headway_ratio=1.6, boardings=210,
        load_factor=1.08, passups=22, sources=["vehicle_apc"]),
    "Moderate · R4 Northside winter-weather delays": dict(
        route_id="R4", stop_range=(4, 9), time_period="midday", delay_min=14, headway_ratio=1.95, cancellation_risk=0.5,
        boardings=90, load_factor=0.7, event_type="weather", event_severity="major", event_verified=True,
        sources=["vehicle_apc", "trusted_external", "official_alert"]),
    "Below threshold · R5 Southside 4-minute delay": dict(
        route_id="R5", stop_range=(2, 6), time_period="off_peak", delay_min=4, headway_ratio=1.2, cancellation_risk=0.05,
        boardings=60, load_factor=0.5, sources=["vehicle_apc", "trusted_external"]),
    "Conflicting data · R3 vehicle feed vs. public reports": dict(
        route_id="R3", stop_range=(3, 7), time_period="peak", delay_min=3, headway_ratio=1.1, cancellation_risk=0.05,
        boardings=80, load_factor=0.55, event_type="obstruction", sources=["vehicle_apc", "public_report"],
        public_delay=25, public_corroborated=True),
    "Feed outage · R1 vehicle feed stale (backup available)": dict(
        route_id="R1", stop_range=(6, 11), time_period="shift_change", delay_min=17, headway_ratio=2.1, cancellation_risk=0.4,
        boardings=120, load_factor=0.9, passups=4, sources=["vehicle_apc"], feed_age_s=900, backup_available=True, backup_delay=17),
    "Feed outage · R5 vehicle feed stale (no backup)": dict(
        route_id="R5", stop_range=(1, 5), delay_min=16, headway_ratio=2.0, sources=["vehicle_apc"], feed_age_s=1200,
        backup_available=False),
    "Uncorroborated social report only · R4": dict(
        route_id="R4", stop_range=(2, 6), sources=["public_report"], public_delay=30, public_corroborated=False,
        event_type="obstruction"),
    "Agent error · Vision Agent fails twice on R2": dict(
        route_id="R2", stop_range=(2, 7), delay_min=18, headway_ratio=2.2, sources=["vehicle_apc", "official_alert"],
        vision_agent_failures=2),
    "Hard-trigger bypass · Verified emergency closure on R3, sparse data": dict(
        route_id="R3", stop_range=(2, 6), time_period="peak", boardings=110, load_factor=0.75, event_type="road_closure",
        event_severity="emergency", event_verified=True, sources=["official_alert"], alert_age_s=240,
        missing_fields=["cancellation_risk"], detour_open=True, detour_extra_min=7),
}
print(f"{len(PRESETS)} preset scenarios ready")
