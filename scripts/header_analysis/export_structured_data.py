#!/usr/bin/env python3
"""
Structured Data Export (Phase 6)

Consolidates every video's per-video CSVs (Phases 3-5) into the flat,
LLM-ready structure requested for the next stage:

outputs/
├── detections.csv               - every raw detection, all videos (3.2)
├── player_tracks.csv            - ByteTrack player tracks, all videos (3.4)
├── ball_tracks.csv              - Kalman-filtered ball trajectories, all videos (3.6)
├── head_player_associations.csv - head-to-player associations, all videos (3.7)
├── header_events.csv            - CONFIRMED header events only, schema 6.1
├── player_summary.csv           - per-player exposure summary, schema 6.2
├── player_summary.json          - same content as JSON
└── analysis_metadata.json       - pipeline configuration & run statistics

Must be run after scripts/inference/process_video.py, scripts/header_analysis/detect_headers.py, and
scripts/header_analysis/analyze_exposure.py have produced their per-video/aggregate outputs.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PRIORITY_MAP = {
    "Low Review Priority": "low",
    "Moderate Review Priority": "moderate",
    "High Review Priority": "high",
}


def concat_all(video_processing_dir: Path, filename: str) -> pd.DataFrame:
    rows = []
    for f in sorted(video_processing_dir.glob(f"*/{filename}")):
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue  # video with zero rows for this file (e.g. no ball ever detected)
        if not df.empty:
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_video_metadata(video_processing_dir: Path) -> dict:
    meta = {}
    for meta_path in video_processing_dir.glob("*/video_metadata.json"):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta[meta_path.parent.name] = json.load(f)
    return meta


HEADER_EVENT_COLUMNS = [
    "event_id", "video_id", "frame_number", "timestamp_sec", "track_id",
    "ball_head_distance_px", "head_width_px", "adaptive_threshold_px",
    "player_confidence", "head_confidence", "ball_confidence",
    "attribution_confidence", "ambiguous",
]


def build_header_events(video_processing_dir: Path, video_meta: dict) -> pd.DataFrame:
    """Schema 6.1 - confirmed events only, one global sequential event_id."""
    all_events = concat_all(video_processing_dir, "header_events.csv")
    if all_events.empty:
        return pd.DataFrame(columns=HEADER_EVENT_COLUMNS)

    confirmed = all_events[all_events["is_header_candidate"] == True].copy()  # noqa: E712
    if confirmed.empty:
        return pd.DataFrame(columns=HEADER_EVENT_COLUMNS)

    confirmed = confirmed.sort_values(["video_id", "event_frame"]).reset_index(drop=True)
    confirmed["event_id"] = [f"H{i + 1:04d}" for i in range(len(confirmed))]
    confirmed["fps"] = confirmed["video_id"].map(lambda v: video_meta.get(v, {}).get("fps", 30.0))
    confirmed["timestamp_sec"] = (confirmed["event_frame"] / confirmed["fps"]).round(3)

    out = confirmed.rename(columns={
        "event_frame": "frame_number",
        "primary_track_id": "track_id",
        "primary_distance_px": "ball_head_distance_px",
    })

    cols = ["event_id", "video_id", "frame_number", "timestamp_sec", "track_id",
            "ball_head_distance_px", "head_width_px", "adaptive_threshold_px",
            "player_confidence", "head_confidence", "ball_confidence",
            "attribution_confidence", "ambiguous"]
    return out[cols]


PLAYER_SUMMARY_COLUMNS = [
    "video_id", "track_id", "total_headers", "first_half_headers", "second_half_headers",
    "shortest_interval_sec", "headers_last_5_min", "headers_last_10_min", "headers_last_15_min",
    "ambiguous_events", "review_priority",
]


def build_player_summary(exposure_dir: Path) -> pd.DataFrame:
    """Schema 6.2."""
    summary_path = exposure_dir / "player_exposure_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(columns=PLAYER_SUMMARY_COLUMNS)
    try:
        summary = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=PLAYER_SUMMARY_COLUMNS)
    if summary.empty:
        return pd.DataFrame(columns=PLAYER_SUMMARY_COLUMNS)

    # min_interval_sec is undefined (NaN) for players with a single header
    # (no interval to compute) - use None so it serializes as JSON `null`,
    # not the invalid `NaN` token, and as an empty CSV cell.
    shortest_interval = summary["min_interval_sec"].astype(object).where(summary["min_interval_sec"].notna(), None)

    return pd.DataFrame({
        "video_id": summary["video_id"],
        "track_id": summary["track_id"],
        "total_headers": summary["total_headers"],
        "first_half_headers": summary["first_half_headers"],
        "second_half_headers": summary["second_half_headers"],
        "shortest_interval_sec": shortest_interval,
        "headers_last_5_min": summary["max_rolling_5min_count"],
        "headers_last_10_min": summary["max_rolling_10min_count"],
        "headers_last_15_min": summary["max_rolling_15min_count"],
        "ambiguous_events": summary["ambiguous_events"],
        "review_priority": summary["review_priority"].map(PRIORITY_MAP).fillna(summary["review_priority"]),
    })


def main():
    parser = argparse.ArgumentParser(description="Export consolidated, LLM-ready structured data (Phase 6).")
    parser.add_argument("--video_processing_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--exposure_dir", type=str, default="outputs/exposure_analysis")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_path", type=str, default="models/checkpoints/yolov8m/best.pt", help="Recorded in analysis_metadata.json for provenance")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    video_processing_dir = project_root / args.video_processing_dir
    exposure_dir = project_root / args.exposure_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_processing_dir.exists():
        print(f"Error: '{video_processing_dir}' does not exist. Run scripts/inference/process_video.py first.")
        sys.exit(1)

    video_meta = load_video_metadata(video_processing_dir)

    print("Consolidating detections.csv ...")
    detections = concat_all(video_processing_dir, "frame_detections.csv")
    detections.to_csv(output_dir / "detections.csv", index=False)

    print("Consolidating player_tracks.csv ...")
    player_tracks = concat_all(video_processing_dir, "player_tracks.csv")
    player_tracks.to_csv(output_dir / "player_tracks.csv", index=False)

    print("Consolidating ball_tracks.csv ...")
    ball_tracks = concat_all(video_processing_dir, "ball_track.csv")
    ball_tracks.to_csv(output_dir / "ball_tracks.csv", index=False)

    print("Consolidating head_player_associations.csv ...")
    head_assoc = concat_all(video_processing_dir, "head_associations.csv")
    head_assoc.to_csv(output_dir / "head_player_associations.csv", index=False)

    print("Building header_events.csv (confirmed events, schema 6.1) ...")
    header_events = build_header_events(video_processing_dir, video_meta)
    header_events.to_csv(output_dir / "header_events.csv", index=False)

    print("Building player_summary.csv / .json (schema 6.2) ...")
    player_summary = build_player_summary(exposure_dir)
    player_summary.to_csv(output_dir / "player_summary.csv", index=False)
    with open(output_dir / "player_summary.json", "w", encoding="utf-8") as f:
        json.dump(player_summary.to_dict(orient="records"), f, indent=2)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classes": {"0": "player", "1": "ball", "2": "head"},
        "detector_model": args.model_path,
        "num_videos_processed": len(video_meta),
        "num_detections": int(len(detections)),
        "num_player_track_points": int(len(player_tracks)),
        "num_ball_track_points": int(len(ball_tracks)),
        "num_head_player_associations": int(len(head_assoc)),
        "num_confirmed_header_events": int(len(header_events)),
        "num_ambiguous_header_events": int(header_events["ambiguous"].sum()) if not header_events.empty else 0,
        "num_players_summarized": int(len(player_summary)),
        "review_priority_disclaimer": (
            "review_priority is a research-oriented exposure summary derived from image-plane "
            "detection/tracking signals. It is NOT a medical assessment and NOT a concussion "
            "diagnosis."
        ),
        "motion_proxy_disclaimer": (
            "Any ball/head displacement figures in this dataset are image-plane pixel "
            "measurements, not calibrated physical units, unless camera calibration is "
            "explicitly noted elsewhere."
        ),
        "notes": (
            "player track_id is only unique within a single video_id - there is no cross-video "
            "player re-identification in this pipeline."
        ),
    }
    with open(output_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Structured outputs written to '{output_dir}':")
    for name in ["detections.csv", "player_tracks.csv", "ball_tracks.csv", "head_player_associations.csv",
                 "header_events.csv", "player_summary.csv", "player_summary.json", "analysis_metadata.json"]:
        p = output_dir / name
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {name:<32} {size_kb:>10.1f} KB")


if __name__ == "__main__":
    main()
