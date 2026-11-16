# Transit Reliability & Equity: live bus disruption response

A human-in-the-loop dashboard for NJ Transit bus disruptions around Newark Penn Station, built on the **v7 notebook**
(`notebooks/Transit_Agentic_AI_Submission_v7_final.ipynb`). AI agents sense, organize and explain; fixed, published
rules score the options; **people decide**.

Every case starts from a **real NJ Transit bus**. The v7 engine then scores it on its **model network**: synthetic
routes R1–R5, depots and neighborhoods. Inputs that have no live source use v7's documented sample data, and the
dashboard says which.

## The team (Step 2)

| Role | Person | What they do in the dashboard |
|---|---|---|
| 🎧 Dispatcher (1st reviewer) | **Ariyan** | Opens a case from a live bus and approves, revises or rejects the recommended action |
| 🧭 Supervisor (2nd reviewer) | **Terri-Ann** | Independently approves or rejects the dispatcher's decision |
| 🛠️ Operations Officer | **Grant** | Carries out the approved action, sends the driver alert, and records the outcome |
| 🚌 Bus Driver | **Bus Driver** | Gets the alert by **text message** and on the **driver screen**, and acknowledges it |

Ariyan, Terri-Ann and Grant are on the training roster, with all four required ethics trainings (conflicts,
transparency, procurement, data protection) recorded as completed on 2026-11-16. Names, the date and the phone
number can be changed in Secrets (see below).

## How to navigate the dashboard

The **sidebar** is where you set things up:
- **Step 1 · Live buses:** **↻ Refresh live data** pulls NJ Transit, Census and weather. The dots show what's connected.
- **Step 2 · Who are you?** Pick your role. The page changes to show only your task.
- **Case:** switch between cases. When you change role, the dashboard jumps to the case waiting for you.

The **top of the page** always shows the selected case: route, bus, delay, its status in plain words, and a
**tracker** with six stages, each labeled with the person responsible:

`Live feed → Ariyan decides → Terri-Ann approves → Grant carries it out → Driver acknowledges → Grant records the outcome`

✅ means done, 👉 means waiting on that person now, and ⏳ means not yet.

The **tabs**:

| Tab | Use it for |
|---|---|
| ✅ **My task** | The only tab you need during a case. It shows your form or your alerts |
| 🗺️ **Live map** | Street map with every bus route near Newark Penn on its real streets, live buses colored by delay, and the active case's route in red |
| 📄 **Case summary** | What's happening in plain language, the possible actions, driver alerts and a timeline of who did what |
| 🗂️ **Records** | For analysts: scores and formulas, evidence, audit record, training and conflict-of-interest records, policy versions, export |
| 🔌 **System check** | Tests every connection with your keys (sends no text message) |
| ❓ **How to use** | This guide, inside the app |

## Running the live demo (about 5 minutes)

1. **Before the audience arrives:** open the app, press **↻ Refresh live data**, then open **🔌 System check** and
   press **Run system check**. Everything you need should show ✅.
2. **Ariyan (Dispatcher):** in **Step 2**, pick *Ariyan — Dispatcher*. On **✅ My task**, choose a delayed bus and
   press **Open case for this bus**. Read *What's happening*, tick the four checklist boxes, write a reason and
   press **Submit as Ariyan**.
3. **Terri-Ann (Supervisor):** pick *Terri-Ann — Supervisor*. Her case opens automatically. She reads Ariyan's
   decision, ticks the checklist, writes a reason and presses **Submit as Terri-Ann**.
4. **Grant (Officer):** pick *Grant — Operations Officer*. He checks the driver message (it can be edited) and presses
   **Send alert to driver and record**. The driver's phone gets the text message.
5. **Bus Driver:** pick *Bus Driver*. The alert is on the driver screen; press **👍 Acknowledge**.
6. **Grant:** switch back to Grant and fill in **Record the outcome** with what actually happened. The tracker turns
   all green.

Show the **🗺️ Live map** and **📄 Case summary** tabs at any point. The timeline lists every action by name.

**Tips:**
- **Several devices:** everyone can use the same link on their own laptop or phone, each picking their own role.
  Everyone sees the same cases, live.
- **Deadlines are real** (5, 10 or 20 minutes). If time runs out, the case falls back to the safe default (rider
  information only). That's also worth showing.
- **No bus delayed right now?** The dashboard only uses real buses. Pick the most delayed one, or refresh in a few
  minutes. Cases for small delays may end in *Monitoring*, which is the correct result.
- **v7 may recommend "Dispatch a second bus".** That option uses v7's model depots (it's marked as such). The
  dispatcher can pick another option, such as rider alerts; the choice is logged as an override with the reason.

## Setup

### 1. Create the repository

Upload **everything inside this folder** to the top level of your GitHub repository, including the hidden
`.streamlit` folder and `.gitignore` (on a Mac, press **Cmd + Shift + .** in Finder to see them). The top level of the
repo must show `app_streamlit.py`, `team_ops.py`, `requirements.txt` and the **`transit_ai`** folder (15 files, including
`__init__.py`, `_modules.json` and `live_ops.py`).

### 2. Deploy on Streamlit Community Cloud

On <https://share.streamlit.io>, choose **Create app**, pick the repo and the `main` branch, and set the main file to
**`app_streamlit.py`**. Under **Advanced settings**, choose Python 3.11 or 3.12 and paste your Secrets (next step).

### 3. Secrets

Paste them under **App → Settings → Secrets**. The full template is in `.streamlit/secrets.toml.example`.

```toml
# Required
NJ_TRANSIT_USERNAME = "..."
NJ_TRANSIT_PASSWORD = "..."

# Strongly recommended
DASHBOARD_PASSWORD = "..."
CARTO_API_KEY = "..."            # street map with street names (free: carto.com/basemaps/apikey)

# Driver text message (Twilio)
TWILIO_ACCOUNT_SID = "AC..."
TWILIO_AUTH_TOKEN = "..."
TWILIO_FROM_NUMBER = "+1..."     # your Twilio number
DRIVER_PHONE_NUMBER = "+1..."    # the phone that plays the bus driver

# Optional
CENSUS_API_KEY = "..."
NWS_CONTACT_EMAIL = "you@example.com"
OPENAI_API_KEY = "sk-..."        # AI-written explanations (otherwise built-in text)
```

Format: `NAME = "value"`, with straight double quotes and one setting per line.

### 4. Text messaging with Twilio

1. Sign up at twilio.com. Copy the **Account SID** and **Auth Token** from the Console.
2. Get a Twilio phone number. US carriers only deliver app texts from registered numbers:
   - **Free trial:** buy a **toll-free** number and submit it for **toll-free verification** (free). A trial
     account can only text **Verified Caller IDs**, so add the driver's phone under *Phone Numbers → Verified
     Caller IDs*.
   - **Paid account:** a local number with **A2P 10DLC** registration, which has fees.
3. Add the four Twilio secrets above. **🔌 System check** confirms the account and warns you if it's a trial.

Without Twilio, everything still works: the alert appears on the driver screen, and the case notes that no text was sent.

### Optional settings

| Secret | Default | What it does |
|---|---|---|
| `DISPATCHER_NAME`, `SUPERVISOR_NAME`, `OFFICER_NAME`, `DRIVER_NAME` | Ariyan, Terri-Ann, Grant, Bus Driver | Names shown in Step 2 |
| `TEAM_TRAINING_DATE` | 2026-11-16 | Date the team's four ethics trainings are recorded as completed |
| `REQUIRE_SUPERVISOR_APPROVAL` | "1" | "1": every action needs Terri-Ann's approval. "0": only high-impact actions do |
| `REVIEWER_ROSTER_JSON` | — | Add more reviewers (the team is always added) |
| `NJT_NEARBY_KM` | "2.0" | Radius around Newark Penn for buses and routes |
| `USE_CV`, `USE_SLM` | "false" | v7's camera model and FLAN-T5 (add `requirements-ml.txt`; about 1.5 GB of memory) |

## What is live and what comes from the v7 model

| Input | Source |
|---|---|
| The bus, its delay or cancellation, alerts | **Live:** NJ Transit GTFS-realtime (your developer account) |
| Route name, next stops, scheduled gap between buses, route paths on the map | **Live:** NJ Transit static GTFS (`getGTFS`) |
| Households without a car, poverty, disability | **Live:** US Census ACS 5-year (v7 sample values only if the Census is unreachable) |
| Weather and weather alerts | **Live:** National Weather Service |
| Street map | **Live:** CARTO (with `CARTO_API_KEY`) or OpenStreetMap |
| Driver text message | **Live:** Twilio |
| Rider counts, connecting service, incident history | **v7 sample data** (no live source connected) |
| Spare buses, depots, dispatch cost | **v7 model** (synthetic depots), so "Dispatch a second bus" is marked *(v7 model depot)* |
| Where the bus sits in the scoring model | **v7 model network:** the real route is scored on a synthetic route (usually R2) |
| Outcomes | Entered by the operations officer from real records |

**Changes from the notebook:**
- **No preset scenarios:** the 12 preset scenarios and the custom scenario builder are not in the dashboard.
- **No made-up history:** the notebook's 30 simulated past incidents aren't loaded, so records and the Insight Agent use real cases only.
- **Your team only:** the fictional sample reviewers (Ana Lopez, etc.) and disclosures (Gabe Ortiz, etc.) are replaced by your team.
- **On the map:** v7's model network is a layer you can switch on (☰ layer button, top right). It's labeled *synthetic*.

Thresholds, weights and the bus-hour cost are **policy settings**. They're versioned on the Records tab and change
only with a policy owner's approval. Approved actions are recorded for staff to carry out; nothing is dispatched
automatically.

## Files

| File | Purpose |
|---|---|
| `app_streamlit.py` | The dashboard (Streamlit main file) |
| `team_ops.py` | The team (Step 2), driver alerts (Twilio + driver screen), progress tracker, system check |
| `transit_ai/` | The v7 notebook's code, one file per section, run in notebook order (`__init__.py` → `load()`) |
| `transit_ai/live_ops.py` | Live layer added for the dashboard: every live bus, NJ Transit route data, robust login, live-only cases, the map |
| `notebooks/` | The v7 notebook (source of truth) · `docs/architecture.png`: architecture diagram |
| `tools/build_from_notebook.py` | Regenerates `transit_ai/*.py` from the notebook (keeps `live_ops.py`) |
| `requirements.txt` | Packages for Streamlit Cloud · `requirements-ml.txt`: optional camera model and FLAN-T5 |
| `.streamlit/config.toml` | Theme and settings · `.streamlit/secrets.toml.example`: secrets template |

## Updating from the notebook

After editing the v7 notebook, regenerate the engine and redeploy:

```bash
python tools/build_from_notebook.py notebooks/Transit_Agentic_AI_Submission_v7_final.ipynb .
```

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in
streamlit run app_streamlit.py
```
