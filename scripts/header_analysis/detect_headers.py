#!/usr/bin/env python3
"""
Header Event Detection & Performer Attribution (Phase 4)

No separate header/non-header action classifier is used. Instead a header
event is inferred from ball trajectory + head trajectory + spatial proximity
+ temporal behavior, combining several signals so no single one is decisive:

  4.1 Ball-to-head Euclidean distance, per player track, per frame.
  4.2-4.3 Adaptive contact threshold T = k * head_box_width (not a fixed
          pixel value, since apparent player/head size varies with zoom,
          camera distance, and resolution).
  4.4 Temporal verification: distance should approach a local minimum then
          recede over a +/-2 frame window, with tolerance for noise.
  4.5 Ball trajectory-change angle across the candidate contact, as
          supporting (not required) evidence.
  4.6 A weighted composite confidence combines spatial proximity, the
          approach/departure pattern, player-track continuity across the
          window, and trajectory-change evidence into a header candidate
          decision.
  4.2 Performer attribution: at the representative contact frame, rank every
          candidate head by distance; the closest is the primary performer.
  4.7 Ambiguity handling: a small margin between the closest and second
          closest candidate means the attribution should not be forced -
          the event is flagged ambiguous with both candidates reported.
  4.8 Duplicate suppression: candidate frames that are temporally adjacent
          (regardless of which track triggered them - a crowded duel can
          flag multiple tracks across neighboring frames) are merged into
          one event window; the window's representative contact frame is
          the one with the minimum distance.

Input: outputs/video_processing/<video_id>/{ball_track.csv,
       head_associations.csv, frame_detections.csv, player_tracks.csv}
Output: outputs/video_processing/<video_id>/header_events.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 2  # frames before/after the candidate (t-2..t+2), per spec 4.4


# --------------------------------------------------------------------------- #
# Data loading (4.1)
# --------------------------------------------------------------------------- #

def _read_csv_safe(path: Path, columns: list) -> pd.DataFrame:
    """A video with e.g. zero ball detections/predictions writes a fully
    empty CSV (no header row at all), which pandas otherwise rejects."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def load_video_data(video_dir: Path):
    ball = _read_csv_safe(video_dir / "ball_track.csv",
                           ["video_id", "frame_number", "timestamp", "ball_x", "ball_y", "confidence", "position_source"])
    heads = _read_csv_safe(video_dir / "head_associations.csv",
                            ["video_id", "frame", "head_id", "head_x", "head_y", "player_track_id", "association_confidence"])
    dets = _read_csv_safe(video_dir / "frame_detections.csv",
                           ["video_id", "frame_number", "timestamp_sec", "class_id", "class_name", "confidence",
                            "x1", "y1", "x2", "y2", "center_x", "center_y"])
    tracks = _read_csv_safe(video_dir / "player_tracks.csv",
                             ["video_id", "frame_number", "timestamp", "track_id", "x1", "y1", "x2", "y2",
                              "center_x", "center_y", "detection_confidence"])
    return ball, heads, dets, tracks


def attach_head_width(heads: pd.DataFrame, dets: pd.DataFrame) -> pd.DataFrame:
    """Recover each head detection's box width and detection confidence by
    joining head_associations (center only) back to frame_detections (which
    has the full box + confidence) on the exact center coordinates both
    were derived from."""
    head_dets = dets[dets["class_name"] == "head"][["frame_number", "x1", "x2", "center_x", "center_y", "confidence"]].copy()
    head_dets["head_w"] = head_dets["x2"] - head_dets["x1"]
    head_dets = head_dets.rename(columns={"frame_number": "frame", "center_x": "head_x", "center_y": "head_y",
                                           "confidence": "head_confidence"})
    merged = heads.merge(head_dets[["frame", "head_x", "head_y", "head_w", "head_confidence"]],
                          on=["frame", "head_x", "head_y"], how="left")
    return merged


def build_distance_series(ball: pd.DataFrame, heads: pd.DataFrame):
    """Per player_track_id, a frame-indexed DataFrame of ball-to-head
    distance (4.1), head_w (for the adaptive threshold, 4.3), and the raw
    head detection confidence (for Phase 6 export)."""
    ball_pos = ball.set_index("frame_number")[["ball_x", "ball_y"]]

    series = {}
    for track_id, g in heads.groupby("player_track_id"):
        g = g.set_index("frame")
        common = g.index.intersection(ball_pos.index)
        if len(common) == 0:
            continue
        dx = ball_pos.loc[common, "ball_x"] - g.loc[common, "head_x"]
        dy = ball_pos.loc[common, "ball_y"] - g.loc[common, "head_y"]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        series[track_id] = pd.DataFrame({
            "distance": dist,
            "head_w": g.loc[common, "head_w"],
            "head_confidence": g.loc[common, "head_confidence"],
        }).sort_index()
    return series, ball_pos


# --------------------------------------------------------------------------- #
# Per-signal scoring (4.4-4.6)
# --------------------------------------------------------------------------- #

def temporal_pattern_score(dist: pd.Series, t: int) -> float:
    """4.4: fraction of the 4 pairwise approach/departure comparisons that
    hold (D[t-2]>D[t-1]>D[t] before contact, D[t]<D[t+1]<D[t+2] after),
    among comparisons whose frames actually exist. Tolerant of noise -
    partial credit, not all-or-nothing."""
    pairs = [(t - 2, t - 1), (t - 1, t), (t, t + 1), (t + 1, t + 2)]
    expect_decrease = [True, True, False, False]
    hits, total = 0, 0
    for (a, b), decreasing in zip(pairs, expect_decrease):
        if a in dist.index and b in dist.index:
            total += 1
            if decreasing and dist[a] > dist[b]:
                hits += 1
            elif not decreasing and dist[a] < dist[b]:
                hits += 1
    return hits / total if total > 0 else 0.0


def track_consistency_score(dist: pd.Series, t: int) -> float:
    frames = [t - 2, t - 1, t, t + 1, t + 2]
    present = sum(1 for f in frames if f in dist.index)
    return present / len(frames)


def trajectory_change_degrees(ball_pos: pd.DataFrame, t: int, window: int = WINDOW):
    """Angle between the ball's pre-contact and post-contact velocity
    vectors (4.5). Returns None if not enough ball position data around t."""
    before = [f for f in (t - window, t - window + 1) if f in ball_pos.index]
    after = [f for f in (t + window - 1, t + window) if f in ball_pos.index]
    if len(before) < 2 or len(after) < 2:
        return None

    v_pre = ball_pos.loc[before[-1]].to_numpy(dtype=float) - ball_pos.loc[before[0]].to_numpy(dtype=float)
    v_post = ball_pos.loc[after[-1]].to_numpy(dtype=float) - ball_pos.loc[after[0]].to_numpy(dtype=float)
    n_pre, n_post = np.linalg.norm(v_pre), np.linalg.norm(v_post)
    if n_pre < 1e-6 or n_post < 1e-6:
        return None

    cos_theta = np.clip(np.dot(v_pre, v_post) / (n_pre * n_post), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def _lookup_player_confidence(tracks: pd.DataFrame, t: int, track_id) -> float:
    match = tracks[(tracks["frame_number"] == t) & (tracks["track_id"] == track_id)]
    if match.empty:
        return None
    return float(match["detection_confidence"].iloc[0])


def _lookup_ball_confidence(ball: pd.DataFrame, t: int):
    match = ball[ball["frame_number"] == t]
    if match.empty or pd.isna(match["confidence"].iloc[0]) or match["confidence"].iloc[0] == "":
        return None
    try:
        return float(match["confidence"].iloc[0])
    except (TypeError, ValueError):
        return None


def score_candidate(track_id, t, series, ball_pos, ball_raw, tracks, args):
    dist = series[track_id]["distance"]
    head_w = float(series[track_id]["head_w"][t])
    head_conf = series[track_id]["head_confidence"][t]
    threshold = float(args.k * head_w)
    spatial_score = float(np.clip(1.0 - dist[t] / threshold, 0.0, 1.0))
    temporal_score = temporal_pattern_score(dist, t)
    consistency_score = track_consistency_score(dist, t)
    angle = trajectory_change_degrees(ball_pos, t)
    trajectory_score = 0.5 if angle is None else float(np.clip(angle / 90.0, 0.0, 1.0))

    confidence = (
        args.w_spatial * spatial_score
        + args.w_temporal * temporal_score
        + args.w_consistency * consistency_score
        + args.w_trajectory * trajectory_score
    )

    player_conf = _lookup_player_confidence(tracks, t, track_id)
    ball_conf = _lookup_ball_confidence(ball_raw, t)

    return {
        "head_width_px": round(head_w, 2),
        "adaptive_threshold_px": round(threshold, 2),
        "player_confidence": round(player_conf, 4) if player_conf is not None else "",
        "head_confidence": round(float(head_conf), 4) if pd.notna(head_conf) else "",
        "ball_confidence": round(ball_conf, 4) if ball_conf is not None else "",
        "spatial_score": round(spatial_score, 4),
        "temporal_pattern_score": round(temporal_score, 4),
        "track_consistency_score": round(consistency_score, 4),
        "trajectory_change_deg": "" if angle is None else round(angle, 2),
        "trajectory_score": round(trajectory_score, 4),
        "header_confidence": round(float(confidence), 4),
        "is_header_candidate": bool(confidence >= args.confirm_threshold),
    }


# --------------------------------------------------------------------------- #
# Raw candidate detection, per track (4.1-4.3)
# --------------------------------------------------------------------------- #

def find_raw_candidates(series: dict, args) -> list:
    """Local minima of distance under the adaptive threshold, per track.
    Not yet deduplicated across tracks/frames - that's done in
    merge_into_events (4.8)."""
    raw = []
    for track_id, df in series.items():
        dist = df["distance"]
        threshold = args.k * df["head_w"]
        for t in dist.index:
            if not (dist[t] < threshold[t]):
                continue
            prev_ok = (t - 1 not in dist.index) or (dist[t] <= dist[t - 1])
            next_ok = (t + 1 not in dist.index) or (dist[t] <= dist[t + 1])
            if prev_ok and next_ok:
                raw.append({"track_id": track_id, "frame": t, "distance": float(dist[t])})
    return raw


# --------------------------------------------------------------------------- #
# Duplicate-event suppression (4.8)
# --------------------------------------------------------------------------- #

def merge_into_events(raw_candidates: list, merge_gap: int) -> list:
    """Group candidate frames that are temporally adjacent (within
    merge_gap frames of each other), regardless of which track triggered
    them, into single event windows. Returns windows sorted by start frame,
    each with its member raw candidates and the representative frame
    t* = argmin distance within the window."""
    if not raw_candidates:
        return []

    frames = sorted(set(c["frame"] for c in raw_candidates))
    windows = []
    cur_start = cur_end = frames[0]
    for f in frames[1:]:
        if f - cur_end <= merge_gap:
            cur_end = f
        else:
            windows.append((cur_start, cur_end))
            cur_start = cur_end = f
    windows.append((cur_start, cur_end))

    events = []
    for start, end in windows:
        members = [c for c in raw_candidates if start <= c["frame"] <= end]
        rep = min(members, key=lambda c: c["distance"])
        events.append({
            "window_start_frame": start,
            "window_end_frame": end,
            "event_frame": rep["frame"],
            "members": members,
        })
    return events


# --------------------------------------------------------------------------- #
# Performer attribution & ambiguity handling (4.2, 4.7)
# --------------------------------------------------------------------------- #

def attribute_performer(event_frame: int, series: dict, args):
    """At the representative contact frame, rank every track with a valid
    head-ball distance that frame. Returns the ranked list plus an
    ambiguity verdict based on the margin between the top two."""
    ranked = []
    for track_id, df in series.items():
        if event_frame in df.index:
            ranked.append((track_id, float(df["distance"][event_frame])))
    ranked.sort(key=lambda x: x[1])

    if not ranked:
        return None

    primary_id, primary_d = ranked[0]
    if len(ranked) > 1:
        second_id, second_d = ranked[1]
        margin = second_d - primary_d
    else:
        second_id, second_d, margin = None, None, None

    if margin is None:
        ambiguous, confidence_label, confidence_numeric = False, "high", 1.0
    else:
        # Saturating margin->[0,1] score: at margin==high_confidence_margin_px
        # the numeric score is 0.5, approaching 1.0 as the margin widens.
        confidence_numeric = float(margin / (margin + args.high_confidence_margin_px))
        if margin < args.ambiguity_margin_px:
            ambiguous, confidence_label = True, "low"
        elif margin < args.high_confidence_margin_px:
            ambiguous, confidence_label = False, "medium"
        else:
            ambiguous, confidence_label = False, "high"

    return {
        "primary_track_id": primary_id,
        "primary_distance_px": round(primary_d, 2),
        "second_track_id": second_id if second_id is not None else "",
        "second_distance_px": round(second_d, 2) if second_d is not None else "",
        "margin_px": round(margin, 2) if margin is not None else "",
        "ambiguous": ambiguous,
        "attribution_confidence": round(confidence_numeric, 4),
        "attribution_confidence_label": confidence_label,
    }


# --------------------------------------------------------------------------- #
# Per-video pipeline
# --------------------------------------------------------------------------- #

EVENT_COLUMNS = [
    "video_id", "event_id", "window_start_frame", "window_end_frame", "event_frame",
    "primary_track_id", "primary_distance_px", "second_track_id", "second_distance_px",
    "margin_px", "ambiguous", "attribution_confidence", "attribution_confidence_label",
    "head_width_px", "adaptive_threshold_px", "player_confidence", "head_confidence", "ball_confidence",
    "spatial_score", "temporal_pattern_score", "track_consistency_score",
    "trajectory_change_deg", "trajectory_score", "header_confidence", "is_header_candidate",
]


def detect_headers_for_video(video_id: str, video_dir: Path, args) -> pd.DataFrame:
    ball, heads, dets, tracks = load_video_data(video_dir)
    if ball.empty or heads.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    heads = attach_head_width(heads, dets)
    heads = heads.dropna(subset=["head_w"])
    series, ball_pos = build_distance_series(ball, heads)
    if not series:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    raw_candidates = find_raw_candidates(series, args)
    events = merge_into_events(raw_candidates, merge_gap=args.merge_gap)

    rows = []
    for event_id, ev in enumerate(events):
        t_star = ev["event_frame"]
        attribution = attribute_performer(t_star, series, args)
        if attribution is None:
            continue
        scores = score_candidate(attribution["primary_track_id"], t_star, series, ball_pos, ball, tracks, args)

        rows.append({
            "video_id": video_id,
            "event_id": event_id,
            "window_start_frame": ev["window_start_frame"],
            "window_end_frame": ev["window_end_frame"],
            "event_frame": t_star,
            **attribution,
            **scores,
        })

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    return pd.DataFrame(rows).sort_values("event_frame")


def main():
    parser = argparse.ArgumentParser(description="Detect header events (with performer attribution) from Phase 3 ball/head/player tracks.")
    parser.add_argument("--video_processing_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--video_id", type=str, default=None, help="Process a single video_id subfolder; default: all")
    parser.add_argument("--k", type=float, default=1.5, help="Adaptive threshold scale: T = k * head_box_width (tune empirically on validation data)")
    parser.add_argument("--confirm_threshold", type=float, default=0.55, help="Composite confidence needed to flag is_header_candidate")
    parser.add_argument("--w_spatial", type=float, default=0.30, help="Weight: ball-head proximity")
    parser.add_argument("--w_temporal", type=float, default=0.30, help="Weight: approach/departure distance pattern")
    parser.add_argument("--w_consistency", type=float, default=0.20, help="Weight: player-track continuity across the window")
    parser.add_argument("--w_trajectory", type=float, default=0.20, help="Weight: ball trajectory-change evidence")
    parser.add_argument("--merge_gap", type=int, default=5, help="Max frame gap between candidate frames to merge into one event window (4.8)")
    parser.add_argument("--ambiguity_margin_px", type=float, default=8.0, help="Below this second-vs-first distance margin, flag the event ambiguous (4.7) - tune on validation data")
    parser.add_argument("--high_confidence_margin_px", type=float, default=20.0, help="Above this margin, attribution confidence is 'high'; between the two thresholds it's 'medium'")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    root_dir = project_root / args.video_processing_dir

    if not root_dir.exists():
        print(f"Error: '{root_dir}' does not exist. Run scripts/inference/process_video.py first.")
        sys.exit(1)

    if args.video_id:
        video_dirs = [root_dir / args.video_id]
    else:
        video_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir() and (d / "ball_track.csv").exists())

    if not video_dirs:
        print(f"No processed videos found under '{root_dir}'.")
        sys.exit(0)

    total_events, total_confirmed, total_ambiguous = 0, 0, 0
    for video_dir in video_dirs:
        if not video_dir.exists():
            print(f"[Warning] '{video_dir}' does not exist, skipping.")
            continue
        video_id = video_dir.name
        result = detect_headers_for_video(video_id, video_dir, args)
        out_path = video_dir / "header_events.csv"
        result.to_csv(out_path, index=False)

        n_events = len(result)
        n_confirmed = int(result["is_header_candidate"].sum()) if not result.empty else 0
        n_ambiguous = int(result["ambiguous"].sum()) if not result.empty else 0
        total_events += n_events
        total_confirmed += n_confirmed
        total_ambiguous += n_ambiguous
        print(f"{video_id}: {n_events} event window(s), {n_confirmed} confirmed, {n_ambiguous} ambiguous -> {out_path}")

    print(f"\nTotal across {len(video_dirs)} video(s): {total_events} event(s), "
          f"{total_confirmed} confirmed header candidate(s), {total_ambiguous} flagged ambiguous for manual review.")


if __name__ == "__main__":
    main()
