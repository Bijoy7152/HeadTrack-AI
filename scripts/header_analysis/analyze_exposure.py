#!/usr/bin/env python3
"""
Player-Wise Header Exposure Analysis (Phase 5)

Aggregates confirmed header events (from scripts/header_analysis/detect_headers.py) by
player track. NOTE: ByteTrack track IDs are only unique within a single
video/clip - there is no cross-video player re-identification in this
pipeline - so "player" here means (video_id, track_id), not a person
tracked across the whole dataset.

  5.1 Total header exposure per (video, track).
  5.2 First-half / second-half split (meaningful for full-match footage;
      our source clips are short highlights, so this mostly degenerates to
      "first half" - included for when longer footage is processed).
  5.3 Inter-header interval statistics (mean/min/max/median); undefined
      (NaN) for players with fewer than 2 confirmed headers.
  5.4 Rolling exposure counts in trailing 5/10/15-minute windows.
  5.5 Exposure timeline (event timestamps in MM:SS).
  5.6 Motion-based proxies: ball/head frame-to-frame image-plane
      displacement and ball trajectory-change angle. These are PIXEL
      measurements, not physical units - never speed in m/s, acceleration,
      g-force, or impact force, without camera calibration.
  5.7 Review priority: a rule-based Low/Moderate/High category from total
      exposure, clustering, ambiguous-event rate, and confidence signals.
      This is a RESEARCH-ORIENTED SUMMARY ONLY. It is NOT a concussion
      diagnosis and must never be presented as one.

Input: outputs/video_processing/<video_id>/{header_events.csv, ball_track.csv,
       head_associations.csv, video_metadata.json}
Output:
  outputs/exposure_analysis/header_events_enriched.csv
  outputs/exposure_analysis/player_exposure_summary.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DISCLAIMER = (
    "Review priority is a research-oriented exposure summary derived from image-plane "
    "detection/tracking signals. It is NOT a medical assessment and NOT a concussion "
    "diagnosis. Motion proxies (ball/head displacement) are pixel measurements, not "
    "calibrated physical units (not m/s, not g-force, not impact force)."
)


def format_mmss(seconds: float) -> str:
    if pd.isna(seconds):
        return ""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def load_all_confirmed_events(video_processing_dir: Path) -> pd.DataFrame:
    rows = []
    for events_path in sorted(video_processing_dir.glob("*/header_events.csv")):
        video_id = events_path.parent.name
        df = pd.read_csv(events_path)
        if df.empty:
            continue
        df = df[df["is_header_candidate"] == True].copy()  # noqa: E712
        if df.empty:
            continue
        df["video_id"] = video_id
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_video_fps(video_dir: Path) -> float:
    meta_path = video_dir / "video_metadata.json"
    if not meta_path.exists():
        return 30.0
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return float(meta.get("fps", 30.0)) or 30.0


def compute_motion_proxies(video_dir: Path, event_frame: int, track_id) -> dict:
    """5.6: image-plane ball/head displacement around the event frame.
    Explicitly pixel-based - not physical units."""
    ball_disp, head_disp = None, None
    ball_path = video_dir / "ball_track.csv"
    if ball_path.exists():
        ball = pd.read_csv(ball_path)
        if not ball.empty:
            ball = ball.set_index("frame_number")
            if event_frame in ball.index and (event_frame - 1) in ball.index:
                b1 = ball.loc[event_frame, ["ball_x", "ball_y"]].to_numpy(dtype=float)
                b0 = ball.loc[event_frame - 1, ["ball_x", "ball_y"]].to_numpy(dtype=float)
                ball_disp = float(np.linalg.norm(b1 - b0))

    heads_path = video_dir / "head_associations.csv"
    if heads_path.exists():
        heads = pd.read_csv(heads_path)
        if not heads.empty:
            h = heads[heads["player_track_id"] == track_id].set_index("frame")
            if event_frame in h.index and (event_frame - 1) in h.index:
                h1 = h.loc[event_frame, ["head_x", "head_y"]].to_numpy(dtype=float)
                h0 = h.loc[event_frame - 1, ["head_x", "head_y"]].to_numpy(dtype=float)
                head_disp = float(np.linalg.norm(h1 - h0))

    return {"ball_displacement_px": ball_disp, "head_displacement_px": head_disp}


def enrich_events(events: pd.DataFrame, video_processing_dir: Path, half_split_sec: float) -> pd.DataFrame:
    enriched_rows = []
    fps_cache = {}
    for _, row in events.iterrows():
        video_id = row["video_id"]
        video_dir = video_processing_dir / video_id
        if video_id not in fps_cache:
            fps_cache[video_id] = load_video_fps(video_dir)
        fps = fps_cache[video_id]

        t_sec = row["event_frame"] / fps if fps else 0.0
        motion = compute_motion_proxies(video_dir, int(row["event_frame"]), row["primary_track_id"])

        enriched_rows.append({
            "video_id": video_id,
            "track_id": row["primary_track_id"],
            "event_frame": row["event_frame"],
            "timestamp_sec": round(t_sec, 3),
            "timestamp_mmss": format_mmss(t_sec),
            "half": "first" if t_sec < half_split_sec else "second",
            "ball_displacement_px": motion["ball_displacement_px"],
            "head_displacement_px": motion["head_displacement_px"],
            "trajectory_change_deg": row.get("trajectory_change_deg", ""),
            "header_confidence": row["header_confidence"],
            "attribution_confidence": row["attribution_confidence"],
            "attribution_confidence_label": row["attribution_confidence_label"],
            "ambiguous": row["ambiguous"],
        })

    return pd.DataFrame(enriched_rows).sort_values(["video_id", "track_id", "event_frame"])


def rolling_window_counts(group_events_sec: np.ndarray, windows_sec: list) -> pd.DataFrame:
    """5.4: for each event at time t, count events in (t - window, t]."""
    out = {f"rolling_{int(w // 60)}min_count": [] for w in windows_sec}
    for t in group_events_sec:
        for w in windows_sec:
            count = int(np.sum((group_events_sec > t - w) & (group_events_sec <= t)))
            out[f"rolling_{int(w // 60)}min_count"].append(count)
    return pd.DataFrame(out)


def review_priority(total_headers, max_rolling_15min, min_interval_sec, ambiguous_ratio,
                     avg_header_confidence, high_attrib_ratio, args) -> str:
    """5.7: rule-based Low/Moderate/High category. Thresholds are explicit
    CLI defaults meant to be replaced with validated/literature-backed
    values - NOT a diagnosis of any kind (see DISCLAIMER)."""
    score = 0
    if total_headers >= args.pr_total_high:
        score += 2
    elif total_headers >= args.pr_total_moderate:
        score += 1

    if max_rolling_15min >= args.pr_cluster_high:
        score += 2
    elif max_rolling_15min >= args.pr_cluster_moderate:
        score += 1

    if not pd.isna(min_interval_sec) and min_interval_sec <= args.pr_short_interval_sec:
        score += 1

    if avg_header_confidence < args.pr_low_confidence_cutoff:
        score += 1  # weak detection confidence adds uncertainty, not certainty
    if ambiguous_ratio > args.pr_high_ambiguous_ratio:
        score += 1

    if score >= 4:
        return "High Review Priority"
    if score >= 2:
        return "Moderate Review Priority"
    return "Low Review Priority"


def summarize_players(enriched: pd.DataFrame, args) -> pd.DataFrame:
    summary_rows = []
    for (video_id, track_id), g in enriched.groupby(["video_id", "track_id"]):
        g = g.sort_values("timestamp_sec")
        times = g["timestamp_sec"].to_numpy()

        total = len(g)
        first_half = int((g["half"] == "first").sum())
        second_half = int((g["half"] == "second").sum())

        if total >= 2:
            intervals = np.diff(times)
            mean_iv, min_iv, max_iv, median_iv = (
                float(np.mean(intervals)), float(np.min(intervals)),
                float(np.max(intervals)), float(np.median(intervals)),
            )
        else:
            mean_iv = min_iv = max_iv = median_iv = float("nan")

        rolling = rolling_window_counts(times, [300, 600, 900])
        max_r5 = int(rolling["rolling_5min_count"].max()) if not rolling.empty else 0
        max_r10 = int(rolling["rolling_10min_count"].max()) if not rolling.empty else 0
        max_r15 = int(rolling["rolling_15min_count"].max()) if not rolling.empty else 0

        ambiguous_events = int(g["ambiguous"].sum())
        ambiguous_ratio = float(g["ambiguous"].mean())
        avg_confidence = float(g["header_confidence"].mean())
        high_attrib_ratio = float((g["attribution_confidence_label"] == "high").mean())
        avg_ball_disp = g["ball_displacement_px"].mean()
        avg_head_disp = g["head_displacement_px"].mean()

        priority = review_priority(total, max_r15, min_iv, ambiguous_ratio, avg_confidence, high_attrib_ratio, args)

        summary_rows.append({
            "video_id": video_id,
            "track_id": track_id,
            "total_headers": total,
            "first_half_headers": first_half,
            "second_half_headers": second_half,
            "mean_interval_sec": round(mean_iv, 2) if not np.isnan(mean_iv) else "",
            "min_interval_sec": round(min_iv, 2) if not np.isnan(min_iv) else "",
            "max_interval_sec": round(max_iv, 2) if not np.isnan(max_iv) else "",
            "median_interval_sec": round(median_iv, 2) if not np.isnan(median_iv) else "",
            "max_rolling_5min_count": max_r5,
            "max_rolling_10min_count": max_r10,
            "max_rolling_15min_count": max_r15,
            "ambiguous_events": ambiguous_events,
            "ambiguous_ratio": round(ambiguous_ratio, 3),
            "avg_header_confidence": round(avg_confidence, 4),
            "high_attribution_ratio": round(high_attrib_ratio, 3),
            "avg_ball_displacement_px": round(avg_ball_disp, 2) if pd.notna(avg_ball_disp) else "",
            "avg_head_displacement_px": round(avg_head_disp, 2) if pd.notna(avg_head_disp) else "",
            "timeline_mmss": ", ".join(g["timestamp_mmss"]),
            "review_priority": priority,
        })

    return pd.DataFrame(summary_rows).sort_values("total_headers", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Aggregate confirmed header events into player-wise exposure statistics.")
    parser.add_argument("--video_processing_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--output_dir", type=str, default="outputs/exposure_analysis")
    parser.add_argument("--half_split_sec", type=float, default=2700.0, help="Seconds marking first/second half boundary (default 45 min; mostly moot for short clips)")
    parser.add_argument("--pr_total_moderate", type=int, default=3)
    parser.add_argument("--pr_total_high", type=int, default=6)
    parser.add_argument("--pr_cluster_moderate", type=int, default=2)
    parser.add_argument("--pr_cluster_high", type=int, default=4)
    parser.add_argument("--pr_short_interval_sec", type=float, default=60.0)
    parser.add_argument("--pr_low_confidence_cutoff", type=float, default=0.6)
    parser.add_argument("--pr_high_ambiguous_ratio", type=float, default=0.3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    video_processing_dir = project_root / args.video_processing_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_processing_dir.exists():
        print(f"Error: '{video_processing_dir}' does not exist. Run scripts/inference/process_video.py and scripts/header_analysis/detect_headers.py first.")
        sys.exit(1)

    events = load_all_confirmed_events(video_processing_dir)
    if events.empty:
        print("No confirmed header events found.")
        sys.exit(0)

    enriched = enrich_events(events, video_processing_dir, args.half_split_sec)
    summary = summarize_players(enriched, args)

    enriched_path = output_dir / "header_events_enriched.csv"
    summary_path = output_dir / "player_exposure_summary.csv"
    enriched.to_csv(enriched_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Enriched events   : {len(enriched)} rows -> {enriched_path}")
    print(f"Player summaries  : {len(summary)} (video_id, track_id) groups -> {summary_path}")
    print("\nReview priority distribution:")
    print(summary["review_priority"].value_counts().to_string())
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
