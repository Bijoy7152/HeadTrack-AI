"""Orchestrates the REAL computer-vision pipeline (scripts/*.py) for one
uploaded video, scoped to its own job directory. No fake/hardcoded results -
every number the dashboard shows comes from these subprocess runs reading
and writing actual files.

Stages: process_video -> detect_headers -> analyze_exposure ->
        export_structured_data -> generate_reports
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

WEBAPP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _env_path(var: str, default: Path) -> Path:
    """Resolves an env-configurable path relative to PROJECT_ROOT, falling
    back to `default` (itself already absolute) if unset."""
    value = os.environ.get(var)
    if not value:
        return default
    value_path = Path(value).expanduser()
    return value_path if value_path.is_absolute() else PROJECT_ROOT / value_path


MODEL_PATH = _env_path("MODEL_PATH", PROJECT_ROOT / "models" / "checkpoints" / "yolov8m" / "best.pt")
JOBS_DIR = _env_path("JOBS_DIR", WEBAPP_DIR / "jobs")
UPLOADS_DIR = _env_path("UPLOADS_DIR", WEBAPP_DIR / "uploads")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list, cwd=PROJECT_ROOT):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(cmd)}):\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout


def video_id_for(video_path: Path) -> str:
    return video_path.stem


def run_job(job_id: str, video_path: Path, update_status):
    """update_status(stage=..., status=...) is called to report progress -
    the caller (app.py) wires this to database.update_job."""
    job_dir = JOBS_DIR / job_id
    vp_dir = job_dir / "video_processing"
    exposure_dir = job_dir / "exposure_analysis"
    structured_dir = job_dir / "structured"
    reports_dir = job_dir / "reports"
    for d in (vp_dir, exposure_dir, structured_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    started = time.time()
    update_status(status="processing", started_at=_now())

    try:
        update_status(stage="Detecting player/ball/head + tracking (YOLOv8m + ByteTrack + Kalman)")
        _run([
            sys.executable, str(SCRIPTS_DIR / "inference" / "process_video.py"),
            "--video", str(video_path),
            "--model", str(MODEL_PATH),
            "--output_dir", str(vp_dir),
        ])

        video_id = video_id_for(video_path)
        # process_video.py may prefix video_id with its parent folder name;
        # for a single uploaded file with no meaningful parent grouping it
        # keeps the bare stem, so the output subfolder name equals video_id.
        found = [d.name for d in vp_dir.iterdir() if d.is_dir()]
        if found:
            video_id = found[0]

        update_status(stage="Header contact detection + performer attribution")
        _run([
            sys.executable, str(SCRIPTS_DIR / "header_analysis" / "detect_headers.py"),
            "--video_processing_dir", str(vp_dir),
            "--video_id", video_id,
        ])

        update_status(stage="Player exposure aggregation")
        _run([
            sys.executable, str(SCRIPTS_DIR / "header_analysis" / "analyze_exposure.py"),
            "--video_processing_dir", str(vp_dir),
            "--output_dir", str(exposure_dir),
        ])

        update_status(stage="Exporting structured CSV/JSON")
        _run([
            sys.executable, str(SCRIPTS_DIR / "header_analysis" / "export_structured_data.py"),
            "--video_processing_dir", str(vp_dir),
            "--exposure_dir", str(exposure_dir),
            "--output_dir", str(structured_dir),
            "--model_path", str(MODEL_PATH),
        ])

        summary_path = structured_dir / "player_summary.json"
        n_players = 0
        if summary_path.exists():
            n_players = len(json.loads(summary_path.read_text()))

        if n_players > 0:
            update_status(stage="Generating LLM player exposure reports (local model)")
            report_cmd = [
                sys.executable, str(SCRIPTS_DIR / "llm_reporting" / "generate_reports.py"),
                "--player_summary", str(summary_path),
                "--output_dir", str(reports_dir),
            ]
            ollama_model = os.environ.get("OLLAMA_MODEL")
            if ollama_model:
                report_cmd += ["--model", ollama_model]
            _run(report_cmd)

        elapsed = round(time.time() - started, 1)
        stats = _compute_job_stats(structured_dir, elapsed)
        update_status(status="done", stage="Complete", finished_at=_now(),
                      processing_seconds=elapsed, video_id=video_id, stats=stats)

    except Exception as exc:  # noqa: BLE001
        update_status(status="error", error=str(exc)[:4000], finished_at=_now())
        raise


def _compute_job_stats(structured_dir: Path, elapsed_sec: float) -> dict:
    summary_path = structured_dir / "player_summary.json"
    events_path = structured_dir / "header_events.csv"

    players = json.loads(summary_path.read_text()) if summary_path.exists() else []
    events = pd.DataFrame()
    if events_path.exists() and events_path.stat().st_size > 0:
        try:
            events = pd.read_csv(events_path)
        except pd.errors.EmptyDataError:
            events = pd.DataFrame()

    total_headers = sum(p["total_headers"] for p in players)
    ambiguous = int(events["ambiguous"].sum()) if not events.empty else 0
    highest = max((p["total_headers"] for p in players), default=0)
    avg = round(total_headers / len(players), 2) if players else 0.0

    return {
        "total_players_tracked": len(players),
        "total_headers_detected": total_headers,
        "ambiguous_events": ambiguous,
        "highest_header_count": highest,
        "avg_headers_per_player": avg,
        "processing_seconds": elapsed_sec,
    }
