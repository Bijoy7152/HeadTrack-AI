# HeadTrack AI — Dashboard

FastAPI + Bootstrap + Chart.js dashboard for "Automated Soccer Header Exposure
Tracking and LLM-Based Player Reporting". Wired to the **real** trained
YOLOv8m detector and the real `scripts/*.py` pipeline (Phases 1-8) — every
number shown comes from an actual pipeline output file for the uploaded
video. No hardcoded or fake metrics.

## Run

From a fresh clone, on any machine/OS:

```bash
git clone <repo-url> && cd HeadTrack-AI   # any folder name/location works
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt           # installs scripts/ + webapp/ deps
cp .env.example .env                      # optional: only edit if a default doesn't fit
cd webapp
uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000

All paths (jobs, uploads, database, model weights) are resolved relative to
the project root automatically and created on first run — no source edits
needed. To override any of them (e.g. store uploads on another drive, force
a specific inference device, point at a different Ollama host/model), set
the corresponding variable in `.env` — see [.env.example](../.env.example).

## Requirements

- The project's virtualenv with `requirements.txt` (root) installed — this
  covers both `scripts/requirements.txt` and this folder's
  `requirements.txt` (fastapi, uvicorn, python-multipart, jinja2, reportlab,
  aiofiles, python-dotenv).
- A trained model at `models/checkpoints/yolov8m/best.pt` (from
  `scripts/training/train_yolo.py`), or set `MODEL_PATH` in `.env` to point elsewhere.
- [Ollama](https://ollama.com) running locally with `llama3.1:8b` pulled, for
  the LLM report generation step (`scripts/llm_reporting/generate_reports.py`).
  Override the host/model via `OLLAMA_HOST`/`OLLAMA_MODEL` in `.env`.

## How it works

1. Upload a `.mp4` on the Upload page.
2. The video is saved to `webapp/uploads/<job_id>.mp4` and a background task
   runs the real pipeline, scoped to `webapp/jobs/<job_id>/`:
   `inference/process_video.py` → `header_analysis/detect_headers.py` →
   `header_analysis/analyze_exposure.py` →
   `header_analysis/export_structured_data.py` →
   `llm_reporting/generate_reports.py` (all under `scripts/`).
3. Progress is tracked in `webapp/headtrack.db` (SQLite) and polled from the
   Processing Status page.
4. Once done, every dashboard page reads directly from that job's
   `structured/` and `reports/` output files — nothing is cached/mocked.
5. Ambiguous-event accept/reject decisions are recorded in a separate SQLite
   table (`ambiguous_reviews`) rather than mutating the pipeline's own CSVs,
   keeping the raw pipeline output reproducible.

## Pages

Dashboard · Upload · Processing Status · Detection Results · Player
Tracking · Ball Tracking · Header Events (+ video viewer with clickable
event markers) · Player Exposure Summary · Player Detail (timeline + LLM
report) · Ambiguous Event Review · LLM Reports · Export (CSV/JSON/PDF).
