# Deploying the dashboard (Gradio and Streamlit)

Both apps import the same `transit_core.py`, so they give the same scores and gates for the same live data.

## 1. Secrets

Set these on the host. Never commit them; `.gitignore` already excludes `.streamlit/secrets.toml`.

| Name | Required? | What it does |
|---|---|---|
| `NJ_TRANSIT_USERNAME`, `NJ_TRANSIT_PASSWORD` | **Required** | NJ Transit developer account (developer.njtransit.com). With no NJ Transit login there are no cases |
| `CENSUS_API_KEY` | Optional | ACS 5-year data. It works without a key, with lower rate limits |
| `NWS_CONTACT_EMAIL` | Optional | Identifies the app to the National Weather Service (it goes in the User-Agent header) |
| `DASHBOARD_PASSWORD` | **Strongly recommended** | Required to open the app. Gradio user name: `reviewer` |
| `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `HF_API_KEY` | Optional | LLM narratives; deterministic templates are used without a key |
| `REVIEWER_ROSTER_JSON` | Needed for approvals | Your real reviewers and their ethics training. Until it is set, nobody can Approve (the Reviewers tab can also load it) |
| `STAFF_DISCLOSURES_JSON` | Optional | Real conflict-of-interest disclosures |
| `NJT_GTFS_ZIP` | Optional | A URL or path to the static GTFS zip, if the `getGTFS` call is not enabled on your account |

Optional settings (environment variables): `NJT_NEARBY_KM`, `SERVICE_AREA_LAT` / `SERVICE_AREA_LON` (defaults to Newark Penn
Station), `AFFECTED_STOPS`, `GTFS_MAX_AGE_DAYS`, `LLM_PROVIDER`, `LLM_MAX_CALLS`, `TRANSIT_USE_CV=1` / `TRANSIT_USE_SLM=1`
(need `requirements-ml.txt`).

## 2. Hugging Face Spaces (Gradio)

1. Create a Space with the **Gradio** SDK.
2. Upload `app_gradio.py`, `transit_core.py`, `requirements.txt` and `README.md`. The README's header points the Space at
   `app_gradio.py`.
3. Under **Settings → Variables and secrets**, add the secrets above as *Secrets*.
4. The Space builds and starts. The first live pull and the static GTFS download run in the background (a few minutes the
   first time). Press **↻ Refresh** on tab 0.

## 3. Streamlit Community Cloud

1. Push the `deploy/` folder to a GitHub repository (without a real `secrets.toml`).
2. On share.streamlit.io choose **Create app**, pick the repository, and set the main file to `app_streamlit.py`.
3. Under **Advanced settings → Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your values filled in.
4. Deploy. Use **↻ Pull live data** in the sidebar.

## 4. Running locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in, used by Streamlit
export NJ_TRANSIT_USERNAME=... NJ_TRANSIT_PASSWORD=...        # used by Gradio
python app_gradio.py            # http://127.0.0.1:7860
streamlit run app_streamlit.py  # http://localhost:8501
```

## 5. Things to know

- **Case state lives in memory.** All users of one running app share the cases, so a second reviewer can open the same case.
  The cases are lost when the app restarts or sleeps. Use **Export** to keep a record.
- **Free hosting tiers** have little memory. Keep `TRANSIT_USE_CV=0` and `TRANSIT_USE_SLM=0` unless you add
  `requirements-ml.txt`. The static GTFS index is kept in `TRANSIT_DATA_DIR`, and it is rebuilt after a restart on hosts
  with temporary disks.
- **Keeping the core in sync.** `transit_core.py` is built from the notebook's cells tagged `core`. After you change one of
  those cells, regenerate the file (copy the tagged cells in order, without the `!pip` lines) and redeploy both apps.
- **Not connected to operations.** Approved actions are recorded, not dispatched.
