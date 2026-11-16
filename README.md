# Agentic AI for Public Transit Reliability and Equity

A human-in-the-loop dashboard for NJ Transit bus disruptions around Newark Penn Station. AI agents sense, organize and
explain; fixed, published rules score; people decide.

**Live data only.** Every case starts from a real NJ Transit bus (GTFS-realtime, plus the static GTFS for routes, stops
and schedules). Equity inputs come from the US Census ACS for the census tracts of the affected stops, and weather
comes from the National Weather Service. Any input with no live source is marked **missing**; it is never filled in.

## Data sources (real-world data only)

The app contains no sample scenarios, seeded history, sample people or stand-in images. Every value comes from one of
these sources, or is shown as **missing**:

| Input | Source |
|---|---|
| Bus positions, delays, cancellations | NJ Transit GTFS-realtime (live, your developer account) |
| Routes, stops, schedules, headways, route paths | NJ Transit static GTFS (`getGTFS`) |
| Service alerts and detours | NJ Transit GTFS-realtime alerts |
| Transit dependence, income, disability | US Census ACS 5-year, for the census tracts of the affected stops |
| Weather alerts | National Weather Service API |
| Rider counts at the stop | Your own camera feeds (`CV_CAMERAS_JSON`), otherwise **missing** |
| Spare buses, operators, depots | **Missing** (no depot feed is connected), so "dispatch a second bus" is listed as infeasible |
| Reviewers and their ethics training | Your roster (`REVIEWER_ROSTER_JSON` or the Ethics tab) |
| Conflict-of-interest disclosures | Your disclosures (`STAFF_DISCLOSURES_JSON` or the Ethics tab) |
| Outcomes after an incident | Entered by operations staff from their records |

Thresholds, weights and the bus-hour cost are **policy settings**, not data. They are versioned, shown on the Policy
governance tab, and change only with a policy owner's approval. Review deadlines are the real 5, 10 or 20 minutes.

## The map

- **Street basemap with street names:** CARTO Voyager when the `CARTO_API_KEY` secret is set (the key is free at
  <https://carto.com/basemaps/apikey>). Without a key, the map uses OpenStreetMap, which also shows street names.
- **Real NJ Transit bus routes:** every bus route serving a stop within `NJT_NEARBY_KM` (2 km) of Newark Penn
  Station, drawn along the streets it actually runs on. These come from `shapes.txt` in NJ Transit's static GTFS.
  Hover over a line to see the route number and name.
- **Live buses:** colored by delay.
- **The active case:** its route's real path to the next stops (red), with each stop shaded by its census tract's share
  of households without a car.

The route lines appear once the static GTFS has loaded. The app downloads it in the background at startup, which takes
a few minutes the first time.

## Files

| File | Purpose |
|---|---|
| `app_streamlit.py` | The Streamlit dashboard (main file) |
| `transit_core.py` | The engine: live connectors, GTFS index, rules, agents, review workflow, map |
| `requirements.txt` | Packages Streamlit Community Cloud installs |
| `requirements-ml.txt` | Optional: FLAN-T5 and the person detector (large; off by default) |
| `.streamlit/config.toml` | Theme and server settings |
| `.streamlit/secrets.toml.example` | Every secret and setting, with explanations |
| `.gitignore` | Keeps secrets and generated data out of Git |

## Create the repository

The two folder-level files start with a dot (`.streamlit/` and `.gitignore`). Some computers hide dot-files, and
browsers sometimes rename them when downloading. After unzipping, check that you can see both. On a Mac, press
**Cmd + Shift + .** in Finder to show hidden files.

**Option A: GitHub website.**
1. Create a new, empty repository on GitHub. Don't add a README.
2. Choose **uploading an existing file**.
3. Open the unzipped folder and select **everything inside it**, including the `.streamlit` folder and `.gitignore`.
   Drag it all onto the page. Drag the contents, not the outer folder.
4. Commit.
5. Check the repo page: `app_streamlit.py` and `requirements.txt` must be at the top level, and `.streamlit/` must be
   a folder containing `config.toml`.

**Option B: command line.**
```bash
cd transit-equity-live
git init && git add . && git commit -m "Transit equity dashboard"
git branch -M main
git remote add origin https://github.com/<you>/<new-repo>.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and choose **Create app → Deploy a public app from GitHub**. Pick the repo and
   the `main` branch, and set the main file path to **`app_streamlit.py`**.
2. Under **Advanced settings**:
   - Python version: 3.11 or 3.12.
   - Secrets: paste the lines from `.streamlit/secrets.toml.example` with your values. You need at least
     `NJ_TRANSIT_USERNAME` and `NJ_TRANSIT_PASSWORD`. Set `DASHBOARD_PASSWORD` and `CARTO_API_KEY` too.
3. Deploy. On the first start, the app pulls the live feed and downloads NJ Transit's static GTFS in the background,
   which takes a few minutes. Then press **↻ Pull live data** in the sidebar.

You can change secrets later under **App → Settings → Secrets**. The app restarts by itself.

**If the map still says "API KEY REQUIRED":** check that the secret is named exactly `CARTO_API_KEY`, then
force-refresh the page. Browsers and CARTO's servers cache map tiles for a while.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in your values
streamlit run app_streamlit.py                               # http://localhost:8501
```

## Things to know

- **Shared case state.** Everyone using one running app shares the cases, so a second reviewer can open the same case.
  Cases are lost when the app restarts or sleeps (after 12 hours without visitors). Use **Export** to keep a record.
- **Approvals need a roster.** Nobody can approve until your reviewers and their ethics training are loaded. Use the
  `REVIEWER_ROSTER_JSON` secret, or **Load roster** on the Ethics tab.
- **Memory.** The optional models in `requirements-ml.txt` need about 1.5 GB of memory. Community Cloud allows about
  2.7 GB in total, so leave them off unless you need them. The person detector only runs on cameras listed in
  `CV_CAMERAS_JSON`.
- **Route data refresh.** The route index is rebuilt every `GTFS_MAX_AGE_DAYS` (7). It is also rebuilt automatically
  when an index from an older version of this app (without route shapes) is found.
- **Not connected to operations.** Approved actions are recorded, not dispatched.
