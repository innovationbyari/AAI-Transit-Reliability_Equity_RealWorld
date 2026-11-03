---
title: Transit Reliability and Equity
emoji: 🚌
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app_gradio.py
pinned: false
---

# Agentic AI for Public Transit Reliability and Equity

A human-in-the-loop dashboard for NJ Transit bus disruptions. AI agents sense, organize and explain; fixed, published
rules score; people decide.

**Live data only.** Every case starts from a real NJ Transit vehicle (GTFS-realtime + static GTFS). Equity inputs come from
the US Census ACS for the census tracts of the affected stops, and weather comes from the National Weather Service. Any
input with no live source is marked **missing** and is never filled in, so options that need it are listed as infeasible.

| File | Purpose |
|---|---|
| `app_gradio.py` | Gradio app (Hugging Face Spaces, or `python app_gradio.py`) |
| `app_streamlit.py` | Streamlit app (Streamlit Community Cloud, or `streamlit run app_streamlit.py`) |
| `transit_core.py` | Shared engine, generated from the notebook's core cells. Edit the notebook, not this file |
| `requirements.txt` | Packages for both apps |
| `requirements-ml.txt` | Optional: the small language model and the person detector (large downloads) |
| `.streamlit/secrets.toml.example` | Template for the secrets |

See `DEPLOY.md` for step-by-step deployment on both platforms.
