#!/usr/bin/env python3
"""
Full Video Processing Pipeline (Phase 3)

For a single input video:
  Frame -> YOLO detector -> player + ball + head detections
  Player detections -> ByteTrack -> persistent track IDs
  Ball detections -> Kalman filter -> smoothed trajectory (detected/predicted)
  Head detections -> associated to the nearest compatible player track

Outputs (under outputs/video_processing/<video_id>/):
  frame_detections.csv   - every raw detection (3.2)
  player_tracks.csv      - ByteTrack player tracks (3.4)
  ball_track.csv         - Kalman-filtered ball trajectory (3.6)
  head_associations.csv  - head-to-player-track associations (3.7)
  video_metadata.json    - video_id, fps, resolution, frame count (3.1)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from device_utils import resolve_device

CLASS_NAMES = {0: "player", 1: "ball", 2: "head"}
PLAYER_CLASS, BALL_CLASS, HEAD_CLASS = 0, 1, 2


# --------------------------------------------------------------------------- #
# Ball tracking: constant-velocity Kalman filter (3.5-3.6)
# --------------------------------------------------------------------------- #

class BallKalmanTracker:
    """State St = [x, y, vx, vy] in image-plane coordinates.

    On a detected frame: predict() then correct() with the measurement,
    output the corrected (filtered) position, position_source='detected'.
    On a missed frame (within max_predict_gap frames of the last detection):
    predict() only, output the predicted position, position_source='predicted'.
    Beyond max_predict_gap consecutive misses the filter is reset (a ball
    ballistic/kicked trajectory is not reliably constant-velocity indefinitely),
    and position_source='missing' with no coordinates.
    """

    def __init__(self, process_noise=1.0, measurement_noise=10.0, max_predict_gap=15):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.max_predict_gap = max_predict_gap
        self.initialized = False
        self.misses = 0

    def _init_state(self, x, y):
        self.kf.statePost = np.array([x, y, 0, 0], dtype=np.float32).reshape(4, 1)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = True
        self.misses = 0

    def update(self, measurement):
        """measurement: (x, y) or None. Returns (x, y, source) or (None, None, 'missing')."""
        if measurement is not None:
            x, y = measurement
            if not self.initialized:
                self._init_state(x, y)
                return x, y, "detected"
            self.kf.predict()
            corrected = self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
            self.misses = 0
            return float(corrected[0, 0]), float(corrected[1, 0]), "detected"

        if not self.initialized:
            return None, None, "missing"

        self.misses += 1
        if self.misses > self.max_predict_gap:
            self.initialized = False
            return None, None, "missing"

        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0]), "predicted"


# --------------------------------------------------------------------------- #
# Head-to-player association (3.7)
# --------------------------------------------------------------------------- #

def associate_heads_to_players(heads, players, prev_track_had_head, upper_fraction=0.45):
    """heads: list of dicts with 'cx','cy'. players: list of dicts with
    'track_id','x1','y1','x2','y2'. Returns list of (head_idx, track_id, confidence)
    and the updated prev_track_had_head set for temporal-consistency scoring
    on the next frame."""
    assignments = []
    used_tracks = set()

    for h_idx, head in enumerate(heads):
        hx, hy = head["cx"], head["cy"]
        best_track, best_score = None, 0.0

        for p in players:
            if p["track_id"] in used_tracks:
                continue  # one head per player track per frame
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            box_h = max(1.0, y2 - y1)
            box_w = max(1.0, x2 - x1)

            if not (x1 <= hx <= x2 and y1 <= hy <= y2):
                continue  # head center must be inside the player box

            vert_ratio = (hy - y1) / box_h
            if vert_ratio > upper_fraction:
                continue  # only the upper portion of the player box is head-plausible

            containment_score = 1.0 - (vert_ratio / upper_fraction) * 0.3  # in [0.7, 1.0]
            horiz_offset_ratio = abs(hx - (x1 + x2) / 2.0) / (box_w / 2.0)
            proximity_score = max(0.0, 1.0 - horiz_offset_ratio)

            score = containment_score * proximity_score
            if p["track_id"] in prev_track_had_head:
                score = min(1.0, score + 0.05)  # temporal-consistency bonus

            if score > best_score:
                best_score, best_track = score, p["track_id"]

        if best_track is not None:
            assignments.append((h_idx, best_track, round(best_score, 4)))
            used_tracks.add(best_track)

    new_prev = {track_id for _, track_id, _ in assignments}
    return assignments, new_prev


# --------------------------------------------------------------------------- #
# Main processing loop
# --------------------------------------------------------------------------- #

def process_video(video_path: Path, model: YOLO, args, output_dir: Path, video_root: Path):
    # Prefix with the parent folder so e.g. videos/Header/h1.mp4 and
    # videos/Non Header/h1.mp4 don't collide in outputs/video_processing/.
    try:
        rel_parent = video_path.relative_to(video_root).parent
        group = rel_parent.as_posix().replace(" ", "_").replace("/", "_")
    except ValueError:
        group = video_path.parent.name.replace(" ", "_")
    video_id = f"{group}_{video_path.stem}" if group and group != "." else video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video '{video_path}'.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    metadata = {
        "video_id": video_id,
        "source_path": str(video_path),
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_sec": round(total_frames / fps, 3) if fps else None,
    }

    detection_rows = []
    player_track_rows = []
    ball_track_rows = []
    head_assoc_rows = []

    ball_tracker = BallKalmanTracker(
        process_noise=args.kf_process_noise,
        measurement_noise=args.kf_measurement_noise,
        max_predict_gap=args.ball_max_predict_gap,
    )
    prev_track_had_head = set()

    stream = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        classes=[PLAYER_CLASS, BALL_CLASS, HEAD_CLASS],
        conf=args.conf,
        device=args.device,
        stream=True,
        persist=True,
        verbose=False,
    )

    frame_number = 0
    for result in stream:
        timestamp_sec = frame_number / fps if fps else 0.0

        players, balls, heads = [], [], []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            ids = result.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(clss)

            for (x1, y1, x2, y2), conf, cid, track_id in zip(xyxy, confs, clss, ids):
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                detection_rows.append({
                    "video_id": video_id, "frame_number": frame_number,
                    "timestamp_sec": round(timestamp_sec, 4),
                    "class_id": int(cid), "class_name": CLASS_NAMES.get(int(cid), str(cid)),
                    "confidence": round(float(conf), 4),
                    "x1": round(float(x1), 2), "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2), "y2": round(float(y2), 2),
                    "center_x": round(float(cx), 2), "center_y": round(float(cy), 2),
                })

                if cid == PLAYER_CLASS:
                    p = {"track_id": int(track_id) if track_id is not None else -1,
                         "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                         "cx": cx, "cy": cy, "conf": float(conf)}
                    players.append(p)
                elif cid == BALL_CLASS:
                    balls.append({"cx": cx, "cy": cy, "conf": float(conf)})
                elif cid == HEAD_CLASS:
                    heads.append({"cx": cx, "cy": cy, "conf": float(conf)})

        # --- player tracks (3.4) ---
        for p in players:
            if p["track_id"] == -1:
                continue  # untracked (tracker hasn't confirmed an ID yet)
            player_track_rows.append({
                "video_id": video_id, "frame_number": frame_number,
                "timestamp": round(timestamp_sec, 4), "track_id": p["track_id"],
                "x1": round(p["x1"], 2), "y1": round(p["y1"], 2),
                "x2": round(p["x2"], 2), "y2": round(p["y2"], 2),
                "center_x": round(p["cx"], 2), "center_y": round(p["cy"], 2),
                "detection_confidence": round(p["conf"], 4),
            })

        # --- ball tracking via Kalman filter (3.5-3.6) ---
        best_ball = max(balls, key=lambda b: b["conf"]) if balls else None
        measurement = (best_ball["cx"], best_ball["cy"]) if best_ball else None
        bx, by, source = ball_tracker.update(measurement)
        if source != "missing":
            ball_track_rows.append({
                "video_id": video_id, "frame_number": frame_number,
                "timestamp": round(timestamp_sec, 4),
                "ball_x": round(bx, 2), "ball_y": round(by, 2),
                "confidence": round(best_ball["conf"], 4) if best_ball else "",
                "position_source": source,
            })

        # --- head-to-player association (3.7) ---
        tracked_players = [p for p in players if p["track_id"] != -1]
        assignments, prev_track_had_head = associate_heads_to_players(
            heads, tracked_players, prev_track_had_head, upper_fraction=args.head_upper_fraction
        )
        for h_idx, track_id, score in assignments:
            head = heads[h_idx]
            head_assoc_rows.append({
                "video_id": video_id, "frame": frame_number,
                "head_id": f"H{h_idx}", "head_x": round(head["cx"], 2), "head_y": round(head["cy"], 2),
                "player_track_id": track_id, "association_confidence": score,
            })

        frame_number += 1
        if args.limit and frame_number >= args.limit:
            break

    out_dir = output_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write an explicit column header, even with zero rows (e.g. a
    # clip with no ball ever detected) - a fully empty file breaks any
    # downstream pd.read_csv with EmptyDataError.
    detection_cols = ["video_id", "frame_number", "timestamp_sec", "class_id", "class_name",
                       "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
    player_track_cols = ["video_id", "frame_number", "timestamp", "track_id", "x1", "y1", "x2", "y2",
                          "center_x", "center_y", "detection_confidence"]
    ball_track_cols = ["video_id", "frame_number", "timestamp", "ball_x", "ball_y", "confidence", "position_source"]
    head_assoc_cols = ["video_id", "frame", "head_id", "head_x", "head_y", "player_track_id", "association_confidence"]

    pd.DataFrame(detection_rows, columns=detection_cols).to_csv(out_dir / "frame_detections.csv", index=False)
    pd.DataFrame(player_track_rows, columns=player_track_cols).to_csv(out_dir / "player_tracks.csv", index=False)
    pd.DataFrame(ball_track_rows, columns=ball_track_cols).to_csv(out_dir / "ball_track.csv", index=False)
    pd.DataFrame(head_assoc_rows, columns=head_assoc_cols).to_csv(out_dir / "head_associations.csv", index=False)
    with open(out_dir / "video_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    n_ball_detected = sum(1 for r in ball_track_rows if r["position_source"] == "detected")
    n_ball_predicted = sum(1 for r in ball_track_rows if r["position_source"] == "predicted")
    print(f"\n--- {video_id} ---")
    print(f"Frames processed       : {frame_number}")
    print(f"Detections             : {len(detection_rows)}")
    print(f"Player track points    : {len(player_track_rows)} ({len({r['track_id'] for r in player_track_rows})} unique tracks)")
    print(f"Ball points            : {len(ball_track_rows)} (detected={n_ball_detected}, predicted={n_ball_predicted})")
    print(f"Head-player associations: {len(head_assoc_rows)}")
    print(f"Saved to               : {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process a full soccer video: detection, tracking, ball Kalman filtering, head-player association.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (or a directory of videos)")
    parser.add_argument("--model", type=str, default="models/checkpoints/yolov8m/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--output_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. 'mps', 'cpu', '0'; auto-detects (CUDA > MPS > CPU) if unset")
    parser.add_argument("--kf_process_noise", type=float, default=1.0)
    parser.add_argument("--kf_measurement_noise", type=float, default=10.0)
    parser.add_argument("--ball_max_predict_gap", type=int, default=15, help="Max consecutive frames to Kalman-predict the ball without a detection")
    parser.add_argument("--head_upper_fraction", type=float, default=0.45, help="Fraction of the player box height (from top) considered head-plausible")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N frames (for testing)")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    project_root = Path(__file__).resolve().parent.parent.parent
    video_arg = project_root / args.video if not Path(args.video).is_absolute() else Path(args.video)
    model_path = project_root / args.model if not Path(args.model).is_absolute() else Path(args.model)
    output_dir = project_root / args.output_dir

    if not model_path.exists():
        print(f"Error: model weights not found at '{model_path}'.")
        sys.exit(1)

    if video_arg.is_dir():
        videos = sorted(video_arg.rglob("*.mp4"))
    elif video_arg.is_file():
        videos = [video_arg]
    else:
        print(f"Error: video path '{video_arg}' does not exist.")
        sys.exit(1)

    if not videos:
        print(f"No .mp4 videos found at '{video_arg}'.")
        sys.exit(0)

    print(f"Loading model '{model_path}'...")
    model = YOLO(str(model_path))

    # Root used to derive a collision-free video_id (e.g. Header_h1 vs
    # NonHeader_h1) - prefer the conventional videos/ folder, else whichever
    # directory was passed, else the single file's own parent.
    default_videos_root = project_root / "videos"
    if default_videos_root.exists() and str(video_arg).startswith(str(default_videos_root)):
        video_root = default_videos_root
    elif video_arg.is_dir():
        video_root = video_arg
    else:
        video_root = video_arg.parent

    for video_path in videos:
        process_video(video_path, model, args, output_dir, video_root)


if __name__ == "__main__":
    main()