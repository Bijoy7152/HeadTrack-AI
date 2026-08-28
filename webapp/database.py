"""SQLite persistence for the HeadTrack AI dashboard.

Two tables:
  jobs               - one row per uploaded video processing job
  ambiguous_reviews  - human accept/reject/reassign decisions on ambiguous
                       header events (kept separate from the pipeline's own
                       CSV outputs, which stay immutable/reproducible)
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_db_path_env = os.environ.get("DB_PATH")
if _db_path_env:
    _db_path = Path(_db_path_env).expanduser()
    DB_PATH = _db_path if _db_path.is_absolute() else Path(__file__).resolve().parent.parent / _db_path
else:
    DB_PATH = Path(__file__).resolve().parent / "headtrack.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    video_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | processing | done | error
    stage TEXT,                              -- current pipeline stage, for progress UI
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    processing_seconds REAL,
    stats_json TEXT                          -- cached summary stats once done
);

CREATE TABLE IF NOT EXISTS ambiguous_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    event_id TEXT NOT NULL,                  -- header_events.csv event_id (H0001, ...)
    decision TEXT NOT NULL,                  -- accept_primary | accept_second | reject
    chosen_track_id TEXT,
    reviewed_at TEXT NOT NULL,
    UNIQUE(job_id, event_id)
);
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_job(job_id: str, filename: str, created_at: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, status, created_at) VALUES (?, ?, 'queued', ?)",
            (job_id, filename, created_at),
        )


def update_job(job_id: str, **fields):
    if not fields:
        return
    if "stats" in fields:
        fields["stats_json"] = json.dumps(fields.pop("stats"))
    cols = ", ".join(f"{k} = ?" for k in fields)
    with get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def get_job(job_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def upsert_review(job_id: str, event_id: str, decision: str, chosen_track_id, reviewed_at: str):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ambiguous_reviews (job_id, event_id, decision, chosen_track_id, reviewed_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(job_id, event_id) DO UPDATE SET
                 decision=excluded.decision, chosen_track_id=excluded.chosen_track_id,
                 reviewed_at=excluded.reviewed_at""",
            (job_id, event_id, decision, str(chosen_track_id) if chosen_track_id is not None else None, reviewed_at),
        )


def get_reviews_for_job(job_id: str) -> dict:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM ambiguous_reviews WHERE job_id = ?", (job_id,)).fetchall()
        return {r["event_id"]: dict(r) for r in rows}
