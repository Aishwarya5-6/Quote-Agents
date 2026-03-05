# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile — InsurTech 4-Agent LangGraph Backend
#
#  Build:   docker build -t quote-agents-backend ./backend
#  Run:     docker run --env-file backend/.env -p 8001:8001 quote-agents-backend
#
#  ⚠  backend/models/*.pkl are excluded by .gitignore.
#     Before building this image make sure the trained model artifacts are
#     present locally in backend/models/.  If building from CI, either:
#       1. Remove the "backend/models/*.pkl" line from .gitignore and commit
#          the artifacts, or
#       2. Use Git LFS for the .pkl files, or
#       3. Add a training step to the CI pipeline that regenerates them.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.9-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 2: dependencies ────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 3: application code ────────────────────────────────────────────────
#  Copy the entire backend directory structure:
#    agents/   → agent1…agent4 Python modules
#    models/   → trained .pkl artifacts + manifest.json
#    data/     → reference datasets (if any)
#    main.py   → LangGraph orchestrator + FastAPI app
#    app.py    → legacy single-agent API (optional)
COPY . .

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/health')" || exit 1

# ── Run ──────────────────────────────────────────────────────────────────────
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", \
    "--workers", "1", \
    "--timeout-keep-alive", "75"]
