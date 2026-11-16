"""Live operations layer for the v7 engine (hand-written; loaded after the notebook modules, same shared namespace).

What it adds to the v7 notebook code, for the live dashboard:
  * every live NJ Transit bus near Newark Penn (not only the most delayed one), so the dispatcher can choose;
  * NJ Transit's static GTFS (routes, stops, timetable, route shapes): real route names, the real next stops,
    the real scheduled gap between buses, and the real streets each route runs on;
  * a robust NJ Transit login (token reused; one automatic re-login when NJ Transit rejects it);
  * live-only cases: a case is opened only from a live bus (no "NJ-001" sample bus). The v7 engine still scores the
    bus on its model network (synthetic routes R1-R5, depots, neighbourhoods); the dashboard labels this clearly;
  * the map: street basemap (CARTO with CARTO_API_KEY, else OpenStreetMap), real NJ Transit routes, live buses, the
    case bus's real path, and v7's model network as a switchable layer (hidden by default);
  * no seeded history: the 30 simulated past incidents of the notebook are removed, so records show real cases only.
"""
# ruff: noqa
import json, math, os, re, sqlite3, zlib, statistics, time, zipfile
import html as _html
from datetime import datetime, timezone, timedelta

import folium
import requests


class LiveDataUnavailable(Exception):
    """A required live source (NJ Transit, Census, NWS) gave no data. There is no sample fallback for the bus."""


# ================================================================================================ NJ Transit (robust)
class NJTError(Exception):
    pass


def _njt_error_text(content):
    """NJ Transit answers some failures (bad or expired token, rate limit) with HTTP 200 and a JSON/text message."""
    head = content[:400].lstrip()
    if head[:1] in (b"{", b"[", b"<") or b"errorMessage" in head or b"Invalid" in head:
        try:
            return head.decode("utf-8", "replace").strip()[:200]
        except Exception:
            return "unexpected response"
    return None


def njt_post(endpoint, system=None, timeout=30, _retry=True):
    """POST an authenticated request and return the raw bytes (protobuf for feeds, a zip for getGTFS).
    If the token is rejected or expired, log in again once and retry."""
    if not _NJT["token"]:
        njt_authenticate()
        if not _NJT["token"]:
            raise NJTError("not logged in to NJ Transit (check NJ_TRANSIT_USERNAME / NJ_TRANSIT_PASSWORD)")
    url = NJT_BASE.format(system=system or _NJT["system"]) + endpoint
    try:
        r = requests.post(url, files={"token": (None, _NJT["token"])}, timeout=timeout)
        r.raise_for_status()
        err = _njt_error_text(r.content) if not r.content.startswith(b"PK") else None
    except requests.HTTPError as e:
        err = f"HTTP {e.response.status_code if e.response is not None else '?'}"
    if err:
        if _retry:
            _NJT["token"] = None                       # token expired or rejected: log in again once
            return njt_post(endpoint, system=system, timeout=timeout, _retry=False)
        raise NJTError(f"{endpoint}: {err}")
    return r.content




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




def njt_feed(endpoint):
    from google.transit import gtfs_realtime_pb2
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(njt_post(endpoint))
    return feed


def fetch_njt_live():
    """Every live bus within NJT_NEARBY_KM of Newark Penn (cancelled and most delayed first) plus all alerts.
    'service' / 'alert' keep v7's meaning (the most delayed bus and an alert on its route)."""
    status = f"authenticated ({_NJT['system']}, token reused)" if _NJT["token"] else njt_authenticate()
    out = {"status": status, "feeds": {}, "vehicles": [], "alerts": [], "service": None, "alert": None,
           "nearby_vehicles": 0, "fetched_at": iso(utcnow())}
    if not _NJT["token"]:
        return out
    feeds = {}
    for ep in ("getVehiclePositions", "getTripUpdates", "getAlerts"):
        try:
            feeds[ep] = njt_feed(ep)
            out["feeds"][ep] = f"ok ({len(feeds[ep].entity)} entities)"
        except Exception as e:
            out["feeds"][ep] = f"failed ({type(e).__name__}: {str(e)[:120]})"
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
                       "stop_id": v.stop_id or u.get("next_stop_id"),
                       "stop_seq": (v.current_stop_sequence or None) if v.HasField("current_stop_sequence") else u.get("next_stop_seq"),
                       "timestamp": int(v.timestamp or vp.header.timestamp or 0)})
    nearby.sort(key=lambda n: (not n["cancelled"], -(n["delay_s"] or 0)))
    out["vehicles"], out["nearby_vehicles"] = nearby, len(nearby)
    if nearby:
        out["service"] = nearby[0]
        out["alert"] = alert_for_route(nearby[0]["route_id"], out["alerts"])
    return out


def alert_for_route(route_id, alerts):
    for a in alerts or []:
        if route_id and route_id in a["routes"]:
            return a
    return None


fetch_njt_service = fetch_njt_live        # v7's refresh_live_service() and the dashboard use the live-only fetch
_fetch_njt_once = fetch_njt_live


def refresh_live_context():
    """Pull NJ Transit, Census and the weather service again (Step 1 in the dashboard)."""
    LIVE_CONTEXT["nj_transit"] = fetch_njt_live()
    LIVE_CONTEXT["census"] = fetch_census(LAT, LON)
    LIVE_CONTEXT["weather"] = fetch_weather(LAT, LON)
    return LIVE_CONTEXT


def live_vehicles():
    return LIVE_CONTEXT.get("nj_transit", {}).get("vehicles") or []


def vehicle_label(v):
    net = _NETWORK.get("net")
    d = "delay unknown" if v["delay_s"] is None else f"{v['delay_s'] / 60:+.0f} min"
    route = net.route_label(v["route_id"]) if net else v["route_id"]
    return f"Route {route} · bus {v['vehicle_id']} · {'CANCELLED · ' if v['cancelled'] else ''}{d} · {v['distance_km']} km away"


def vehicle_choices():
    return [(vehicle_label(v), str(v["vehicle_id"])) for v in live_vehicles()]


def live_vehicles_df():
    net = _NETWORK.get("net")
    rows = [{"Route": net.route_label(v["route_id"]) if net else v["route_id"], "Bus": v["vehicle_id"],
             "Delay (min)": None if v["delay_s"] is None else round(v["delay_s"] / 60, 1), "Cancelled": v["cancelled"],
             "Distance (km)": v["distance_km"]} for v in live_vehicles()]
    return pd.DataFrame(rows) if rows else ph(None, "No live buses yet: press Refresh live data")


# ================================================================================================ NJ Transit static GTFS
# ====================================================================================================
NJT_GTFS_ZIP = read_secret("NJT_GTFS_ZIP") or param("NJT_GTFS_ZIP", "")   # optional: a local path or URL to an NJ Transit GTFS zip
GTFS_MAX_AGE_DAYS = param("GTFS_MAX_AGE_DAYS", 7)
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
    "shapes": ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],   # real street paths of the routes
}
GTFS_SCHEMA_VERSION = "2"   # bump when GTFS_TABLES changes, so cached indexes from older versions are rebuilt
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
        except (requests.RequestException, NJTError) as e:
            errors.append(f"{system}: {type(e).__name__} {str(e)[:100]}")
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
                    if table == "shapes":
                        for c in ("shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"):
                            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
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
                "CREATE INDEX IF NOT EXISTS sr_stop ON stop_routes(stop_id)",
                "CREATE TABLE IF NOT EXISTS shapes (shape_id TEXT, shape_pt_lat REAL, shape_pt_lon REAL, shape_pt_sequence REAL)",
                "CREATE INDEX IF NOT EXISTS sh_id ON shapes(shape_id, shape_pt_sequence)",
                "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)",
                f"INSERT OR REPLACE INTO meta VALUES ('schema', '{GTFS_SCHEMA_VERSION}')"]:
        cur.execute(sql)
    for t in ("calendar", "calendar_dates", "agency"):       # optional files: make sure the tables exist
        cur.execute(f"CREATE TABLE IF NOT EXISTS {t} ({', '.join(c + ' TEXT' for c in GTFS_TABLES[t])})")
    con.commit()
    con.close()
    os.replace(tmp, db_path)


def _simplify(pts, tol=0.00004):
    """Douglas–Peucker line simplification (tolerance in degrees, ~4 m) so the map page stays light."""
    if len(pts) < 3:
        return pts
    keep, stack = {0, len(pts) - 1}, [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (y1, x1), (y2, x2) = pts[a], pts[b]
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy) or 1e-12
        best, idx = 0.0, None
        for k in range(a + 1, b):
            y, x = pts[k]
            d = abs(dy * (x - x1) - dx * (y - y1)) / norm
            if d > best:
                best, idx = d, k
        if idx is not None and best > tol:
            keep.add(idx)
            stack += [(a, idx), (idx, b)]
    return [pts[k] for k in sorted(keep)]


def _clip_path(pts, centre, clip_km):
    """Split a path into the runs of points within clip_km of the centre (routes leave Newark), then simplify each run."""
    segs, cur = [], []
    for p in pts:
        if km_between(p, centre) <= clip_km:
            cur.append((round(p[0], 6), round(p[1], 6)))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return [_simplify(sg) for sg in segs if len(sg) > 1]


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

    # ---- real route geometry (shapes.txt)
    def shape_points(self, shape_id):
        if not shape_id:
            return []
        rows = self.q("SELECT shape_pt_lat AS lat, shape_pt_lon AS lon FROM shapes WHERE shape_id = ? "
                      "ORDER BY shape_pt_sequence", (shape_id,))
        return [(r["lat"], r["lon"]) for r in rows if r["lat"] is not None and r["lon"] is not None]

    def trip_path(self, trip_id, stops=None):
        """The street path a trip follows (its GTFS shape), trimmed to run from the first to the last of `stops`.
        Empty when the GTFS has no shape for the trip (the map then joins the stops with straight lines)."""
        t = self.trip(trip_id) if trip_id else None
        pts = self.shape_points(t.get("shape_id")) if t else []
        if len(pts) < 2 or not stops:
            return pts
        near = lambda s: min(range(len(pts)), key=lambda k: (pts[k][0] - s["lat"]) ** 2 + (pts[k][1] - s["lon"]) ** 2)
        i, j = near(stops[0]), near(stops[-1])
        return pts[min(i, j):max(i, j) + 1] if j != i else pts

    def route_paths_near(self, lat, lon, radius_km, clip_km=None, max_routes=80):
        """Every route that serves a stop within radius_km, drawn along its most-used shape in each direction and
        clipped to clip_km around the centre. Cached per network, because the map is redrawn on every page refresh."""
        clip_km = clip_km or max(3.0, radius_km * 2)
        key = (round(lat, 4), round(lon, 4), radius_km, clip_km, max_routes)
        cache = self.__dict__.setdefault("_path_cache", {})
        if key in cache:
            return cache[key]
        routes = self.routes_near(lat, lon, radius_km)[:max_routes]
        out = []
        if routes:
            marks = ",".join("?" * len(routes))
            rows = self.q(f"SELECT route_id, direction_id, shape_id, COUNT(*) AS n FROM trips WHERE route_id IN ({marks}) "
                          "AND shape_id != '' GROUP BY route_id, direction_id, shape_id", tuple(routes))
            best = {}
            for r in rows:
                k = (r["route_id"], r["direction_id"])
                if k not in best or r["n"] > best[k]["n"]:
                    best[k] = r
            info = {r["route_id"]: r for r in self.q(f"SELECT * FROM routes WHERE route_id IN ({marks})", tuple(routes))}
            for (rid, direction), r in sorted(best.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
                segs = _clip_path(self.shape_points(r["shape_id"]), (lat, lon), clip_km)
                if not segs:
                    continue
                ri = info.get(rid, {})
                color = (ri.get("route_color") or "").strip().lstrip("#")
                out.append({"route_id": rid, "direction_id": direction, "short_name": ri.get("route_short_name") or rid,
                            "long_name": ri.get("route_long_name") or "", "segments": segs,
                            "color": f"#{color}" if len(color) == 6 and color.upper() not in ("FFFFFF", "000000") else None})
        cache[key] = out
        return out

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


def _gtfs_schema(db):
    try:
        con = sqlite3.connect(db)
        try:
            r = con.execute("SELECT v FROM meta WHERE k = 'schema'").fetchone()
        finally:
            con.close()
        return r[0] if r else None
    except sqlite3.Error:
        return None


def get_network(force=False):
    """The GTFS network (built on first use and refreshed after GTFS_MAX_AGE_DAYS). Raises LiveDataUnavailable."""
    if _NETWORK["net"] is not None and not force:
        return _NETWORK["net"]
    os.makedirs(GTFS_DIR, exist_ok=True)
    db = os.path.join(GTFS_DIR, "njt_gtfs.sqlite")
    fresh = os.path.exists(db) and (time.time() - os.path.getmtime(db)) < GTFS_MAX_AGE_DAYS * 86400 \
        and _gtfs_schema(db) == GTFS_SCHEMA_VERSION
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


# ================================================================================================ live-only cases
def _real_stops(v, n=6):
    net = _NETWORK.get("net")
    if net is None or not v.get("trip_id"):
        return []
    try:
        return net.affected_stops(v["trip_id"], v.get("stop_id"), v.get("stop_seq"), v["lat"], v["lon"], n)
    except sqlite3.Error:
        return []


def raw_for_vehicle(vehicle_id, name=None):
    """One live bus → v7 Section A inputs → Signal Scout hand-off → raw bundle for the v7 engine.
    The bus, its delay, alert, Census and weather inputs are live; v7's documented samples fill the inputs that have no
    live source here (camera counts, connections, incident history, depots), and each input says which it is."""
    v = next((x for x in live_vehicles() if str(x["vehicle_id"]) == str(vehicle_id)), None)
    if v is None:
        raise LiveDataUnavailable(f"Bus {vehicle_id} is no longer in the live feed. Refresh live data and pick again.")
    net = _NETWORK.get("net")
    stops = _real_stops(v)
    nj = LIVE_CONTEXT["nj_transit"]
    saved = (nj.get("service"), nj.get("alert"))
    nj["service"], nj["alert"] = v, alert_for_route(v["route_id"], nj.get("alerts"))
    try:
        inputs = section_a_sample_inputs()
    finally:
        nj["service"], nj["alert"] = saved
    svc = inputs["service_feeds"]
    if not str(svc.get("source", "")).startswith("NJ Transit"):
        raise LiveDataUnavailable("The live bus feed is not available, so no case can be opened (no sample bus is used).")
    headway = net.scheduled_headway_min(v["route_id"], stops[0]["stop_id"]) if (net and stops) else None
    if headway:
        svc["scheduled_headway_min"], svc["headway_source"] = headway, "NJ Transit timetable (static GTFS, median gap ±90 min)"
    label = net.route_label(v["route_id"]) if net else v["route_id"]
    ssi = build_signal_scout_input(inputs)
    raw = signal_scout_input_to_raw(ssi, name or f"Live · Route {label} · bus {v['vehicle_id']}")
    raw["route_name"] = label
    raw["vehicle_id"] = v["vehicle_id"]
    raw["real"] = {"route_id": v["route_id"], "route_label": label, "trip_id": v["trip_id"], "vehicle_id": v["vehicle_id"],
                   "lat": v["lat"], "lon": v["lon"], "delay_s": v["delay_s"], "cancelled": v["cancelled"],
                   "stops": [{"stop_id": s["stop_id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"]} for s in stops],
                   "model_route_id": raw["route_id"]}
    raw["sample_inputs"] = sorted(r["source"] for r in ssi["all_inputs"] if "sample" in str(r["data"].get("source", "")))
    return raw


_v7_rider_alert_text = rider_alert_text


def rider_alert_text(case):
    """Rider message for live cases built from the real route, real next stops and real connecting routes."""
    real = case["raw"].get("real")
    if not real:
        return _v7_rider_alert_text(case)
    A = case["analysis"]
    f = A.get("fields", {})
    parts = [f"Route {real['route_label']}"]
    if f.get("cancelled"):
        parts.append("a scheduled trip is cancelled")
    elif f.get("delay_min") is not None:
        parts.append(f"buses are running about {f['delay_min']:.0f} min late")
    msg = ": ".join([parts[0], ", ".join(parts[1:]) or "service disruption"])
    st = real.get("stops") or []
    if len(st) > 1:
        msg += f" between {st[0]['name']} and {st[-1]['name']}."
    elif st:
        msg += f" near {st[0]['name']}."
    else:
        msg += "."
    net = _NETWORK.get("net")
    if net and st:
        try:
            alts = sorted({net.route_label(r).split(" ")[0] for s in st for r in net.routes_at(s["stop_id"]) if r != real["route_id"]})
        except sqlite3.Error:
            alts = []
        if alts:
            msg += f" Other routes at these stops: {', '.join(alts[:6])}."
    return msg + " Updates will follow. Accessible-service riders can call the paratransit desk."


OBSERVED_FIELDS = {"observed_delay_reduction_min": "Observed delay reduction (min)",
                   "observed_access_protection": "Observed access protection (0–1)",
                   "observed_vulnerable_riders_served": "Vulnerable riders served (count)",
                   "actual_cost_usd": "Actual cost (USD)", "rider_reported_impact": "Rider-reported impact (1–5)",
                   "false_alarm": "False alarm"}



# ================================================================================================ map
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


CARTO_STYLES = {"voyager": "voyager", "positron": "light_all", "dark-matter": "dark_all"}
ROUTE_PALETTE = ["#2563eb", "#0891b2", "#7c3aed", "#db2777", "#ea580c", "#16a34a", "#0d9488", "#9333ea", "#c2410c", "#4f46e5"]


def add_basemap(m):
    """Street basemap with street names. CARTO (Voyager by default) needs a free key since 2025: set the CARTO_API_KEY
    secret (carto.com/basemaps/apikey). Without a key the map uses OpenStreetMap tiles, which also show street names."""
    key = read_secret("CARTO_API_KEY")
    style = CARTO_STYLES.get((read_secret("CARTO_STYLE") or "voyager").lower(), "voyager")
    if key:
        folium.TileLayer(
            tiles=f"https://basemaps.cartocdn.com/rastertiles/{style}/{{z}}/{{x}}/{{y}}{{r}}.png?key={key}",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors '
                 '&copy; <a href="https://carto.com/attributions">CARTO</a>',
            name="Streets (CARTO)", max_zoom=20).add_to(m)
    else:
        folium.TileLayer("OpenStreetMap", name="Streets (OpenStreetMap)", max_zoom=19).add_to(m)


def route_network_layer(m, highlight_route=None):
    """Every NJ Transit bus route serving the service area, drawn along its real street path (GTFS shapes.txt)."""
    net = _NETWORK.get("net")          # only what is already loaded: drawing the map never starts the GTFS download
    if net is None:
        return 0
    try:
        paths = net.route_paths_near(*SERVICE_AREA_LATLON, NJT_NEARBY_KM)
    except sqlite3.Error:
        return 0
    fg = folium.FeatureGroup(name=f"NJ Transit bus routes within {NJT_NEARBY_KM} km (GTFS shapes)", show=True)
    for k, p in enumerate(paths):
        hot = highlight_route is not None and str(p["route_id"]) == str(highlight_route)
        col = p["color"] or ROUTE_PALETTE[zlib.crc32(str(p["route_id"]).encode()) % len(ROUTE_PALETTE)]
        tip = _html.escape(f"Route {p['short_name']}" + (f" · {p['long_name']}" if p["long_name"] else ""))
        for seg in p["segments"]:
            folium.PolyLine(seg, color=col, weight=5 if hot else 3, opacity=0.9 if hot else 0.55, tooltip=tip).add_to(fg)
    fg.add_to(m)
    return len({p["route_id"] for p in paths})


def map_status_md():
    """One line under the map: which basemap is in use and whether the real route paths are loaded."""
    base = ("CARTO streets (CARTO_API_KEY set)" if read_secret("CARTO_API_KEY")
            else "OpenStreetMap streets (no CARTO_API_KEY secret found)")
    net = _NETWORK.get("net")
    if net is not None:
        try:
            n_shapes = net.q("SELECT COUNT(DISTINCT shape_id) AS n FROM shapes")[0]["n"]
        except sqlite3.Error:
            n_shapes = 0
        n_routes = len({p["route_id"] for p in net.route_paths_near(*SERVICE_AREA_LATLON, NJT_NEARBY_KM)}) if n_shapes else 0
        routes = (f"{n_routes} NJ Transit bus routes drawn on their real streets" if n_routes
                  else "NJ Transit GTFS loaded, but it has no route shapes (shapes.txt): routes cannot be drawn")
    elif _NETWORK.get("error"):
        routes = f"route data unavailable: {_NETWORK['error'][:160]}"
    else:
        routes = "route data still loading (first start takes a few minutes; press ↻ Pull live data)"
    return f"🗺️ Map: {base} · {routes} · map code v2 (real routes)"


def _keep_fitted(m, bounds):
    """Dashboards often draw the map while it is hidden (in a tab), with zero size; Leaflet then fits the bounds to an
    empty box and ends up zoomed out to the whole world. Re-fit whenever the map gets a real size, until the user moves it."""
    name = m.get_name()
    js = ("window.addEventListener('load',function(){var B=%s,moved=false,map=%s;"
          "function fit(){map.invalidateSize();var s=map.getSize();if(!moved&&s.x>0&&s.y>0){map.fitBounds(B);}}"
          "var c=map.getContainer();['mousedown','touchstart','wheel','keydown'].forEach(function(ev){"
          "c.addEventListener(ev,function(){moved=true;},{passive:true});});"
          "window.addEventListener('resize',fit);"
          "if(window.ResizeObserver){new ResizeObserver(fit).observe(c);}"
          "fit();[300,1500].forEach(function(t){setTimeout(fit,t);});});") % (json.dumps(bounds), name)
    m.get_root().script.add_child(folium.Element(js))




def _model_layer(m, case):
    """v7's model network (synthetic routes R1-R5, depots, neighbourhoods) that the engine scores against."""
    pol = APP.PM.current
    fg = folium.FeatureGroup(name="v7 model network (synthetic, used for scoring)", show=False)
    for a in NET["areas"].values():
        folium.Rectangle(bounds=a["bounds"], color="#ffffff", weight=1, fill=True, fill_color=_vcolor(area_vulnerability(a, pol)),
                         fill_opacity=0.3, tooltip=f"Model area: {a['name']}").add_to(fg)
    rid = case["raw"]["route_id"] if case else None
    for r in NET["routes"].values():
        folium.PolyLine(r["path"], color=r["color"], weight=5 if r["id"] == rid else 3, opacity=0.7, dash_array="2 6",
                        tooltip=f"Model route {r['id']} {r['name']} (synthetic)").add_to(fg)
    for d in NET["depots"].values():
        folium.CircleMarker((d["lat"], d["lon"]), radius=6, color="#475569", fill=True, fill_opacity=0.8,
                            tooltip=f"Model depot {d['name']} (synthetic)").add_to(fg)
    fg.add_to(m)


def build_map(case):
    """Real map: street basemap, NJ Transit routes on their real streets, live buses, the case bus's real path to its
    next stops; v7's model network as a layer that can be switched on."""
    m = folium.Map(location=list(SERVICE_AREA_LATLON), zoom_start=14, tiles=None, control_scale=True)
    add_basemap(m)
    dlat = NJT_NEARBY_KM / KM_LAT
    dlon = NJT_NEARBY_KM / (111.320 * math.cos(math.radians(SERVICE_AREA_LATLON[0])))
    bounds = [[SERVICE_AREA_LATLON[0] - dlat, SERVICE_AREA_LATLON[1] - dlon],
              [SERVICE_AREA_LATLON[0] + dlat, SERVICE_AREA_LATLON[1] + dlon]]
    real = (case or {}).get("raw", {}).get("real") or {}
    n_routes = route_network_layer(m, highlight_route=real.get("route_id"))
    _model_layer(m, case)
    live_vehicle_layer(m, highlight=real.get("vehicle_id"))
    if real:
        fg = folium.FeatureGroup(name=f"{case['case_id']}: bus {real['vehicle_id']} and its next stops", show=True)
        stops = real.get("stops") or []
        net = _NETWORK.get("net")
        path = []
        if net is not None and stops:
            try:
                path = net.trip_path(real["trip_id"], stops)
            except sqlite3.Error:
                path = []
        line = path if len(path) > 1 else [(s["lat"], s["lon"]) for s in stops]
        if len(line) > 1:
            folium.PolyLine(line, color="#b91c1c", weight=7, opacity=0.9, dash_array=None if len(path) > 1 else "8 6",
                            tooltip=_html.escape(f"Route {real['route_label']}: path to the next stops")).add_to(fg)
        for s in stops:
            folium.CircleMarker((s["lat"], s["lon"]), radius=6, color="#111827", weight=1, fill=True, fill_color="#fde68a",
                                fill_opacity=0.95, tooltip=_html.escape(s["name"])).add_to(fg)
        folium.CircleMarker((real["lat"], real["lon"]), radius=10, color="#111827", weight=3, fill=True, fill_color="#facc15",
                            fill_opacity=1, tooltip=_html.escape(f"Bus {real['vehicle_id']} (route {real['route_label']})")).add_to(fg)
        fg.add_to(m)
        pts = [(s["lat"], s["lon"]) for s in stops] + [(real["lat"], real["lon"])]
        bounds = [[min(p[0] for p in pts) - 0.005, min(p[1] for p in pts) - 0.008],
                  [max(p[0] for p in pts) + 0.005, max(p[1] for p in pts) + 0.008]]
    folium.LayerControl(collapsed=True).add_to(m)
    legend = ("<div style='position:fixed;bottom:22px;left:12px;z-index:9999;background:rgba(255,255,255,.92);padding:6px 9px;"
              "border-radius:8px;font:11px/1.45 sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.25);max-width:330px'>"
              "<b>Buses</b> <span style='color:#16a34a'>●</span> on time <span style='color:#f59e0b'>●</span> 5+ min late "
              "<span style='color:#b91c1c'>●</span> 15+ min / cancelled <span style='color:#6b7280'>●</span> unknown<br>"
              + (f"<b>Lines</b> {n_routes} bus routes on their real streets (hover for name)<br>" if n_routes
                 else "<b>Lines</b> bus routes appear once route data has loaded<br>")
              + "<span style='color:#b91c1c'>━</span> this case's bus and next stops</div>")
    m.get_root().html.add_child(folium.Element(legend))
    _keep_fitted(m, bounds)
    return m


def map_html(case):
    return build_map(case).get_root().render()


# ================================================================================================ start-up
APP.history.clear()        # live input only: the notebook's 30 simulated past incidents are not used
APP.last_message = "Ready. Refresh live data, choose your role, and open a case from a live bus."


# ================================================================================================ wording for the live dashboard
STATUS_HELP.update({
    "AWAITING_REVIEW": "Waiting for the dispatcher",
    "AWAITING_SECOND_REVIEW": "Waiting for the supervisor (independent second reviewer)",
    "AWAITING_IMPLEMENTATION": "Approved: the operations officer carries it out through normal dispatch channels",
})
_MESSAGES = {
    "High-impact action — sent to an independent second reviewer (same deadline).":
        "Sent to the supervisor (independent second reviewer) for approval. Same deadline.",
    "Implementation recorded. Run the outcome evaluator next.": "Implementation recorded. Record the outcome when the incident is over.",
}


def _reword(method):
    def wrapper(self, *a, **k):
        out = method(self, *a, **k)
        self.last_message = _MESSAGES.get(self.last_message, self.last_message)
        return out
    wrapper.__name__ = method.__name__
    return wrapper


for _m in ("submit_first", "record_implementation"):
    setattr(CaseStore, _m, _reword(getattr(CaseStore, _m)))


def _approved(self, case, option_id):
    case["approved_option_id"] = option_id
    case["status"] = "AWAITING_IMPLEMENTATION"
    self.log(case, "system", "rule", "Implement action", "Approved — handed to operations staff", option_id)
    o = next((o for o in case["analysis"].get("options", []) if o["option_id"] == option_id), None)
    self.last_message = (f"Approved: {o['label'] if o else option_id}. The operations officer carries it out through normal "
                         "dispatch channels and alerts the driver.")


CaseStore._approved = _approved
