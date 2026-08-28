#!/usr/bin/env python3
"""
Tracking Evaluation (Phase 8, section 8.2)

No dense multi-object-tracking ground truth exists for this dataset (that
would require frame-by-frame manually annotated player identities), so
true HOTA/IDF1 cannot be computed. Per the spec's own fallback, this
reports an AUTOMATED HEURISTIC PROXY instead:

  - track count and lifespan statistics per video
  - candidate ID-switch events: a track disappears, and shortly after
    (within --gap_frames) a NEW track_id appears close to where the old
    one was last seen (within --spatial_radius_px) - a classic ByteTrack
    failure mode (occlusion breaks the track, a new ID picks it back up)
  - candidate "lost track" / "recovered track" counts derived from the same
    heuristic

These are ESTIMATES, not ground-truth-verified ID switches. Manually
reviewing a sample of flagged candidates (see --output_dir) against the
source video is the recommended way to calibrate confidence in this proxy.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_video(tracks: pd.DataFrame, gap_frames: int, spatial_radius_px: float):
    if tracks.empty:
        return None

    per_track = tracks.groupby("track_id").agg(
        first_frame=("frame_number", "min"),
        last_frame=("frame_number", "max"),
        n_detections=("frame_number", "count"),
        last_x=("center_x", "last"),
        last_y=("center_y", "last"),
    ).reset_index()
    # first_x/first_y need the row at first_frame, not just "first" (groupby
    # preserves original order, which is frame-ascending here, so first/last
    # aggregation on x/y already corresponds to first_frame/last_frame).
    first_pos = tracks.sort_values("frame_number").groupby("track_id").first()[["center_x", "center_y"]]
    per_track = per_track.merge(first_pos.rename(columns={"center_x": "first_x", "center_y": "first_y"}),
                                 on="track_id")

    n_tracks = len(per_track)
    avg_lifespan = float(per_track["n_detections"].mean())
    median_lifespan = float(per_track["n_detections"].median())

    # Candidate ID switches: track A ends at frame f_end near (x,y); track B
    # starts within gap_frames afterward, near (x,y).
    ended = per_track[["track_id", "last_frame", "last_x", "last_y"]].rename(
        columns={"track_id": "ended_track", "last_frame": "end_frame"})
    started = per_track[["track_id", "first_frame", "first_x", "first_y"]].rename(
        columns={"track_id": "started_track", "first_frame": "start_frame"})

    candidates = []
    for _, e in ended.iterrows():
        window = started[(started["start_frame"] > e["end_frame"]) &
                          (started["start_frame"] <= e["end_frame"] + gap_frames) &
                          (started["started_track"] != e["ended_track"])]
        if window.empty:
            continue
        dist = np.sqrt((window["first_x"] - e["last_x"]) ** 2 + (window["first_y"] - e["last_y"]) ** 2)
        close = window[dist <= spatial_radius_px]
        for _, c in close.iterrows():
            candidates.append({
                "ended_track": int(e["ended_track"]), "end_frame": int(e["end_frame"]),
                "started_track": int(c["started_track"]), "start_frame": int(c["start_frame"]),
                "gap_frames": int(c["start_frame"] - e["end_frame"]),
                "spatial_distance_px": round(float(dist[dist.index == c.name].iloc[0]), 2),
            })

    return {
        "n_tracks": n_tracks,
        "avg_track_lifespan_frames": round(avg_lifespan, 1),
        "median_track_lifespan_frames": round(median_lifespan, 1),
        "candidate_id_switches": len(candidates),
    }, candidates


def main():
    parser = argparse.ArgumentParser(description="Automated tracking-quality proxy evaluation (no MOT ground truth required).")
    parser.add_argument("--video_processing_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--gap_frames", type=int, default=10, help="Max frame gap to consider a track handoff a candidate ID switch")
    parser.add_argument("--spatial_radius_px", type=float, default=50.0, help="Max pixel distance between the old track's last position and the new track's first position")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    video_processing_dir = project_root / args.video_processing_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_processing_dir.exists():
        print(f"Error: '{video_processing_dir}' does not exist.")
        sys.exit(1)

    summary_rows, all_candidates = [], []
    for video_dir in sorted(video_processing_dir.iterdir()):
        tracks_path = video_dir / "player_tracks.csv"
        if not video_dir.is_dir() or not tracks_path.exists():
            continue
        try:
            tracks = pd.read_csv(tracks_path)
        except pd.errors.EmptyDataError:
            continue
        if tracks.empty:
            continue

        result = analyze_video(tracks, args.gap_frames, args.spatial_radius_px)
        if result is None:
            continue
        stats, candidates = result
        stats["video_id"] = video_dir.name
        summary_rows.append(stats)
        for c in candidates:
            c["video_id"] = video_dir.name
        all_candidates.extend(candidates)

    if not summary_rows:
        print("No player_tracks.csv data found.")
        sys.exit(0)

    summary_df = pd.DataFrame(summary_rows)[["video_id", "n_tracks", "avg_track_lifespan_frames",
                                              "median_track_lifespan_frames", "candidate_id_switches"]]
    summary_path = output_dir / "tracking_evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    candidates_df = pd.DataFrame(all_candidates)
    candidates_path = output_dir / "tracking_id_switch_candidates.csv"
    if not candidates_df.empty:
        candidates_df = candidates_df[["video_id", "ended_track", "end_frame", "started_track",
                                        "start_frame", "gap_frames", "spatial_distance_px"]]
    candidates_df.to_csv(candidates_path, index=False)

    print(f"--- Tracking Evaluation Proxy (n={len(summary_df)} videos) ---")
    print(f"Total tracks across all videos       : {int(summary_df['n_tracks'].sum())}")
    print(f"Avg tracks per video                 : {summary_df['n_tracks'].mean():.1f}")
    print(f"Avg track lifespan (frames)           : {summary_df['avg_track_lifespan_frames'].mean():.1f}")
    print(f"Total candidate ID switches           : {int(summary_df['candidate_id_switches'].sum())}")
    print(f"Videos with >=1 candidate ID switch    : {int((summary_df['candidate_id_switches'] > 0).sum())} / {len(summary_df)}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {candidates_path}")
    print("\nNOTE: these are heuristic proxy estimates (spatial-temporal handoff detection), "
          "not ground-truth-verified ID switches. No dense MOT annotations exist for this "
          "dataset, so true HOTA/IDF1 cannot be computed - manually spot-check a sample of "
          "candidates against the source video before citing this as a hard number.")


if __name__ == "__main__":
    main()
