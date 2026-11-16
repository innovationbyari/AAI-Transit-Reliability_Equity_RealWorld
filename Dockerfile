# Optional: run the dashboard on any container host (Render, Railway, Fly.io, Google Cloud Run, Azure, a VM).
#   docker build -t transit-dashboard .
#   docker run -p 8501:8501 -e DASHBOARD_PASSWORD=... -e OPENROUTER_API_KEY=... transit-dashboard
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

# libgl/glib are needed by OpenCV even in its headless build on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health')"
CMD streamlit run app_streamlit.py --server.port=${PORT} --server.address=0.0.0.0
