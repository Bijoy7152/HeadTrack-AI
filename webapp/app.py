"""HeadTrack AI — Automated Soccer Header Exposure Tracking and LLM-Based
Player Reporting. FastAPI dashboard wired to the REAL trained YOLOv8m model
and the REAL scripts/*.py pipeline (Phases 1-8) - no hardcoded/fake metrics,
every figure shown comes from an actual pipeline output file for the
uploaded video.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas

import database as db
import pipeline_runner as pr

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR.parent / ".env")
db.init_db()

app = FastAPI(title="HeadTrack AI")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")

pr.JOBS_DIR.mkdir(parents=True, exist_ok=True)
pr.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def job_or_404(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    if job.get("stats_json"):
        job["stats"] = json.loads(job["stats_json"])
    return job


def structured_dir(job_id: str) -> Path:
    return pr.JOBS_DIR / job_id / "structured"


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def job_players(job_id: str) -> list:
    p = structured_dir(job_id) / "player_summary.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def job_events(job_id: str) -> pd.DataFrame:
    return read_csv_safe(structured_dir(job_id) / "header_events.csv")


def job_video_fps(job_id: str) -> float:
    job = db.get_job(job_id)
    if not job or not job.get("video_id"):
        return 30.0
    meta_path = pr.JOBS_DIR / job_id / "video_processing" / job["video_id"] / "video_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text()).get("fps", 30.0) or 30.0
    return 30.0


def apply_reviews(job_id: str, events: pd.DataFrame) -> pd.DataFrame:
    """Overlay human ambiguous-event decisions onto the (immutable) pipeline output."""
    if events.empty:
        return events
    reviews = db.get_reviews_for_job(job_id)
    events = events.copy()
    events["human_decision"] = events["event_id"].map(
        lambda eid: reviews[eid]["decision"] if eid in reviews else "")
    return events


# --------------------------------------------------------------------------- #
# 1. Dashboard
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    jobs = db.list_jobs()
    for j in jobs:
        if j.get("stats_json"):
            j["stats"] = json.loads(j["stats_json"])
    return templates.TemplateResponse(request, "dashboard.html", {"jobs": jobs})


# --------------------------------------------------------------------------- #
# 2. Upload Match Video
# --------------------------------------------------------------------------- #

@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {})


@app.post("/upload")
def upload_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Only .mp4 files are supported.")

    job_id = uuid.uuid4().hex[:12]
    dest = pr.UPLOADS_DIR / f"{job_id}.mp4"
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    db.create_job(job_id, video.filename, datetime.now(timezone.utc).isoformat())

    def _update(**fields):
        db.update_job(job_id, **fields)

    background_tasks.add_task(pr.run_job, job_id, dest, _update)

    return JSONResponse({"job_id": job_id, "redirect": f"/jobs/{job_id}/processing"})


# --------------------------------------------------------------------------- #
# 3. Processing Status
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/processing", response_class=HTMLResponse)
def processing_status(request: Request, job_id: str):
    job = job_or_404(job_id)
    return templates.TemplateResponse(request, "processing.html", {"job": job})


@app.get("/jobs/{job_id}/api/status")
def processing_status_api(job_id: str):
    job = job_or_404(job_id)
    return {"status": job["status"], "stage": job["stage"], "error": job.get("error")}


# --------------------------------------------------------------------------- #
# 4. Detection Results
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/detections", response_class=HTMLResponse)
def detections_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    dets = read_csv_safe(structured_dir(job_id) / "detections.csv")
    counts = dets["class_name"].value_counts().to_dict() if not dets.empty else {}
    sample = dets.head(200).to_dict(orient="records") if not dets.empty else []
    return templates.TemplateResponse(request, "detections.html", {
        "job": job, "counts": counts, "sample": sample, "total": len(dets),
    })


# --------------------------------------------------------------------------- #
# 5. Player Tracking View
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/tracking", response_class=HTMLResponse)
def tracking_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    tracks = read_csv_safe(structured_dir(job_id) / "player_tracks.csv")
    if tracks.empty:
        summary = []
    else:
        g = tracks.groupby("track_id").agg(
            frames=("frame_number", "count"),
            first_frame=("frame_number", "min"),
            last_frame=("frame_number", "max"),
            avg_confidence=("detection_confidence", "mean"),
        ).reset_index().sort_values("frames", ascending=False)
        g["avg_confidence"] = g["avg_confidence"].round(3)
        summary = g.to_dict(orient="records")
    return templates.TemplateResponse(request, "tracking.html", {
        "job": job, "tracks": summary, "total_points": len(tracks),
    })


# --------------------------------------------------------------------------- #
# 6. Ball Tracking View
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/ball", response_class=HTMLResponse)
def ball_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    ball = read_csv_safe(structured_dir(job_id) / "ball_tracks.csv")
    detected = int((ball["position_source"] == "detected").sum()) if not ball.empty else 0
    predicted = int((ball["position_source"] == "predicted").sum()) if not ball.empty else 0
    trajectory = ball[["frame_number", "ball_x", "ball_y", "position_source"]].to_dict(orient="records") if not ball.empty else []
    return templates.TemplateResponse(request, "ball.html", {
        "job": job, "detected": detected, "predicted": predicted,
        "trajectory": trajectory,
    })


# --------------------------------------------------------------------------- #
# 7. Header Event Timeline (+ video viewer)
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/events", response_class=HTMLResponse)
def events_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    events = apply_reviews(job_id, job_events(job_id))
    rows = events.to_dict(orient="records") if not events.empty else []
    return templates.TemplateResponse(request, "events.html", {
        "job": job, "events": rows,
        "video_url": f"/jobs/{job_id}/video",
    })


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    path = pr.UPLOADS_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


# --------------------------------------------------------------------------- #
# 8. Player Exposure Summary + detail
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/players", response_class=HTMLResponse)
def players_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    players = job_players(job_id)
    return templates.TemplateResponse(request, "players.html", {"job": job, "players": players})


@app.get("/jobs/{job_id}/players/{track_id}", response_class=HTMLResponse)
def player_detail_page(request: Request, job_id: str, track_id: int):
    job = job_or_404(job_id)
    players = job_players(job_id)
    player = next((p for p in players if int(p["track_id"]) == track_id), None)
    if player is None:
        raise HTTPException(404, "Player not found in this job")

    events = job_events(job_id)
    player_events = []
    if not events.empty:
        pe = events[events["track_id"] == track_id]
        for _, r in pe.sort_values("timestamp_sec").iterrows():
            t_sec = r["timestamp_sec"]
            player_events.append({"timestamp_mmss": f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}",
                                   "timestamp_sec": round(t_sec, 2), "event_id": r["event_id"],
                                   "confidence": r["attribution_confidence"], "ambiguous": bool(r["ambiguous"])})

    report_path = pr.JOBS_DIR / job_id / "reports" / f"{job['video_id']}__track{track_id}.txt"
    report_text = report_path.read_text() if report_path.exists() else None

    return templates.TemplateResponse(request, "player_detail.html", {
        "job": job, "player": player, "events": player_events, "report_text": report_text,
    })


# --------------------------------------------------------------------------- #
# 9. Ambiguous Event Review
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/ambiguous", response_class=HTMLResponse)
def ambiguous_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    events = job_events(job_id)
    reviews = db.get_reviews_for_job(job_id)
    ambiguous = []
    if not events.empty:
        amb = events[events["ambiguous"] == True]  # noqa: E712
        for _, r in amb.iterrows():
            row = r.to_dict()
            row["review"] = reviews.get(r["event_id"])
            ambiguous.append(row)
    return templates.TemplateResponse(request, "ambiguous.html", {"job": job, "events": ambiguous})


@app.post("/jobs/{job_id}/ambiguous/{event_id}/decide")
def ambiguous_decide(job_id: str, event_id: str, decision: str = Form(...), chosen_track_id: str = Form(None)):
    job_or_404(job_id)
    if decision not in ("accept_primary", "accept_second", "reject"):
        raise HTTPException(400, "Invalid decision")
    db.upsert_review(job_id, event_id, decision, chosen_track_id, datetime.now(timezone.utc).isoformat())
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# 10. LLM Report
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/reports", response_class=HTMLResponse)
def reports_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    all_reports_path = pr.JOBS_DIR / job_id / "reports" / "all_reports.json"
    reports = json.loads(all_reports_path.read_text()) if all_reports_path.exists() else []
    return templates.TemplateResponse(request, "reports.html", {"job": job, "reports": reports})


# --------------------------------------------------------------------------- #
# 11. Export Results
# --------------------------------------------------------------------------- #

@app.get("/jobs/{job_id}/export", response_class=HTMLResponse)
def export_page(request: Request, job_id: str):
    job = job_or_404(job_id)
    return templates.TemplateResponse(request, "export.html", {"job": job})


@app.get("/jobs/{job_id}/export/csv/{name}")
def export_csv(job_id: str, name: str):
    job_or_404(job_id)
    allowed = {"detections", "player_tracks", "ball_tracks", "head_player_associations", "header_events", "player_summary"}
    if name not in allowed:
        raise HTTPException(404, "Unknown export file")
    path = structured_dir(job_id) / f"{name}.csv"
    if not path.exists():
        raise HTTPException(404, "File not generated for this job")
    return FileResponse(path, filename=f"{job_id}_{name}.csv", media_type="text/csv")


@app.get("/jobs/{job_id}/export/json")
def export_json(job_id: str):
    job = job_or_404(job_id)
    meta_path = structured_dir(job_id) / "analysis_metadata.json"
    summary_path = structured_dir(job_id) / "player_summary.json"
    payload = {
        "job": {k: job[k] for k in ("id", "filename", "video_id", "status") if k in job},
        "analysis_metadata": json.loads(meta_path.read_text()) if meta_path.exists() else None,
        "player_summary": json.loads(summary_path.read_text()) if summary_path.exists() else [],
    }
    return JSONResponse(payload)


@app.get("/jobs/{job_id}/export/pdf")
def export_pdf(job_id: str):
    job = job_or_404(job_id)
    players = job_players(job_id)
    stats = json.loads(job["stats_json"]) if job.get("stats_json") else {}

    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 60

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "HeadTrack AI — Header Exposure Report")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Video: {job['filename']}   Job: {job_id}")
    y -= 14
    c.drawString(50, y, f"Players tracked: {stats.get('total_players_tracked', 0)}   "
                        f"Total headers: {stats.get('total_headers_detected', 0)}   "
                        f"Ambiguous: {stats.get('ambiguous_events', 0)}")
    y -= 30

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, y, "Track ID   Headers   1st Half   2nd Half   Shortest Interval (s)   Ambiguous   Priority")
    y -= 16
    c.setFont("Helvetica", 10)
    for p in players:
        if y < 60:
            c.showPage()
            y = height - 60
            c.setFont("Helvetica", 10)
        c.drawString(50, y, f"{p['track_id']:<10} {p['total_headers']:<9} {p['first_half_headers']:<10} "
                             f"{p['second_half_headers']:<10} {str(p.get('shortest_interval_sec') or '-'):<22} "
                             f"{p['ambiguous_events']:<11} {p['review_priority']}")
        y -= 14

    c.showPage()
    c.save()
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
                              headers={"Content-Disposition": f"attachment; filename={job_id}_report.pdf"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.environ.get("WEBAPP_HOST", "127.0.0.1"),
        port=int(os.environ.get("WEBAPP_PORT", "8000")),
        reload=os.environ.get("WEBAPP_RELOAD", "false").lower() == "true",
    )
