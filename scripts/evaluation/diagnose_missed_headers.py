#!/usr/bin/env python3
"""
Diagnose Missed Header Detections

For clips with zero confirmed header candidates, reports the closest the
ball ever got to any player's head, and whether that's because:
  - no ball was ever detected/predicted in the clip at all
  - no head was ever associated with any player track
  - both existed, but the minimum distance never crossed the adaptive
    threshold (near-miss - candidate for tuning --k, or a genuine
    detection/association misalignment worth spot-checking visually)

Usage:
  python scripts/evaluation/diagnose_missed_headers.py --video_glob "Header_*"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "header_analysis"))
from detect_headers import attach_head_width, build_distance_series, _read_csv_safe  # noqa: E402


def diagnose_video(video_dir: Path, k: float):
    ball = _read_csv_safe(video_dir / "ball_track.csv",
                           ["frame_number", "ball_x", "ball_y", "position_source"])
    heads = _read_csv_safe(video_dir / "head_associations.csv",
                            ["frame", "head_x", "head_y", "player_track_id"])
    dets = _read_csv_safe(video_dir / "frame_detections.csv",
                           ["frame_number", "class_name", "x1", "x2", "center_x", "center_y", "confidence"])

    if ball.empty:
        return {"reason": "no_ball_detected_or_predicted", "min_distance_px": "", "threshold_at_min_px": "", "margin_px": ""}
    if heads.empty:
        return {"reason": "no_head_associations", "min_distance_px": "", "threshold_at_min_px": "", "margin_px": ""}

    heads = attach_head_width(heads, dets).dropna(subset=["head_w"])
    if heads.empty:
        return {"reason": "head_associations_missing_box_width", "min_distance_px": "", "threshold_at_min_px": "", "margin_px": ""}

    series, _ = build_distance_series(ball, heads)
    if not series:
        return {"reason": "no_overlapping_ball_and_head_frames", "min_distance_px": "", "threshold_at_min_px": "", "margin_px": ""}

    best_dist, best_thresh, best_track = None, None, None
    for track_id, df in series.items():
        dist = df["distance"]
        thresh = k * df["head_w"]
        idx = dist.idxmin()
        if best_dist is None or dist[idx] < best_dist:
            best_dist, best_thresh, best_track = float(dist[idx]), float(thresh[idx]), track_id

    margin = best_dist - best_thresh  # negative would mean it should have qualified
    return {
        "reason": "near_miss" if margin > 0 else "should_have_qualified_check_local_min_or_temporal_gate",
        "closest_track_id": best_track,
        "min_distance_px": round(best_dist, 2),
        "threshold_at_min_px": round(best_thresh, 2),
        "margin_px": round(margin, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose why clips have zero confirmed header candidates.")
    parser.add_argument("--video_processing_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--video_glob", type=str, default="Header_*")
    parser.add_argument("--k", type=float, default=1.5, help="Must match the --k used in detect_headers.py")
    parser.add_argument("--output_csv", type=str, default="outputs/evaluation/missed_header_diagnostics.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    root_dir = project_root / args.video_processing_dir
    out_path = project_root / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for video_dir in sorted(root_dir.glob(args.video_glob)):
        if not video_dir.is_dir():
            continue
        events_path = video_dir / "header_events.csv"
        if not events_path.exists():
            continue
        events = pd.read_csv(events_path)
        n_confirmed = int(events["is_header_candidate"].sum()) if not events.empty else 0
        if n_confirmed > 0:
            continue  # only diagnosing zero-confirmed clips

        diag = diagnose_video(video_dir, args.k)
        diag["video_id"] = video_dir.name
        rows.append(diag)

    if not rows:
        print("No zero-confirmed clips found matching the glob.")
        return

    df = pd.DataFrame(rows)
    cols = ["video_id", "reason", "closest_track_id", "min_distance_px", "threshold_at_min_px", "margin_px"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_path, index=False)

    print(f"Diagnosed {len(df)} zero-confirmed clip(s). Saved: {out_path}\n")
    print("--- Reason breakdown ---")
    print(df["reason"].value_counts().to_string())

    near_miss = df[df["reason"] == "near_miss"].copy()
    if not near_miss.empty:
        near_miss["margin_px"] = pd.to_numeric(near_miss["margin_px"])
        print(f"\n--- Near-miss margin stats (n={len(near_miss)}) ---")
        print(near_miss["margin_px"].describe().to_string())
        print("\nSmallest margins (closest to qualifying - best candidates to inspect or lower --k for):")
        print(near_miss.sort_values("margin_px").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
