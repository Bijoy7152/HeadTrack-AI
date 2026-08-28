#!/usr/bin/env python3
"""
Header Event Detection Evaluation (Phase 8, section 8.3)

Two evaluation modes:

1. PROXY (automatic, always runs): uses the dataset's own folder labels
   (videos/Header/ vs videos/Non Header/) as a weak clip-level ground
   truth - a Header_* clip is assumed to contain >=1 real header, a
   Non_Header_* clip is assumed to contain 0. This gives clip-level
   Precision/Recall/F1, NOT frame-accurate TP/FP/FN, since we don't know
   the true number or timestamp of headers within each Header_* clip.

2. TRUE (only if --ground_truth is supplied): frame-accurate TP/FP/FN
   against manually annotated true header timestamps, matching a confirmed
   event to a ground-truth timestamp within --tolerance_sec.

Use --write_template to generate a ground-truth CSV template (one row per
Header_* video) for you to fill in true_header_timestamps_sec by watching
each clip - this is required for true (non-proxy) evaluation and cannot be
generated automatically.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def run_proxy_evaluation(video_processing_dir: Path) -> dict:
    rows = []
    for video_dir in sorted(video_processing_dir.iterdir()):
        events_path = video_dir / "header_events.csv"
        if not video_dir.is_dir() or not events_path.exists():
            continue
        try:
            events = pd.read_csv(events_path)
        except pd.errors.EmptyDataError:
            events = pd.DataFrame()
        n_confirmed = int(events["is_header_candidate"].sum()) if not events.empty else 0
        is_header_clip = video_dir.name.startswith("Header_")
        rows.append({"video_id": video_dir.name, "is_header_clip": is_header_clip,
                     "n_confirmed_events": n_confirmed, "detected_any": n_confirmed > 0})

    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    tp = int(((df["is_header_clip"]) & (df["detected_any"])).sum())
    fn = int(((df["is_header_clip"]) & (~df["detected_any"])).sum())
    fp = int(((~df["is_header_clip"]) & (df["detected_any"])).sum())
    tn = int(((~df["is_header_clip"]) & (~df["detected_any"])).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"table": df, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def write_ground_truth_template(video_processing_dir: Path, out_path: Path):
    rows = []
    for video_dir in sorted(video_processing_dir.iterdir()):
        if video_dir.is_dir() and video_dir.name.startswith("Header_"):
            rows.append({"video_id": video_dir.name, "reviewed": "", "true_header_timestamps_sec": "", "notes": ""})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Ground-truth template written to '{out_path}' ({len(rows)} Header_* clips).")
    print("For each clip you watch: set 'reviewed' to TRUE, and fill "
          "'true_header_timestamps_sec' with semicolon-separated seconds (e.g. '2.9;4.1'), "
          "or leave it blank if you confirm the clip has zero real headers. Rows left with "
          "reviewed != TRUE are skipped entirely by the evaluator - only mark 'reviewed' once "
          "you've actually watched the clip, since a blank timestamp on an unreviewed row "
          "would otherwise be indistinguishable from a confirmed-zero clip.")


def run_true_evaluation(video_processing_dir: Path, ground_truth_path: Path, tolerance_sec: float) -> dict:
    gt = pd.read_csv(ground_truth_path)

    if "reviewed" not in gt.columns:
        print("Error: ground truth CSV has no 'reviewed' column - regenerate the template with "
              "--write_template (older templates predate this safety check).")
        sys.exit(1)

    def _is_reviewed(v):
        return str(v).strip().lower() in ("true", "yes", "1")

    reviewed_gt = gt[gt["reviewed"].apply(_is_reviewed)]
    skipped = len(gt) - len(reviewed_gt)
    if skipped:
        print(f"Skipping {skipped} row(s) not marked reviewed=TRUE (not yet watched).")
    if reviewed_gt.empty:
        print("No rows marked reviewed=TRUE - nothing to evaluate.")
        return {"matches": pd.DataFrame(), "tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp, fp, fn = 0, 0, 0
    match_rows = []

    for _, row in reviewed_gt.iterrows():
        video_id = row["video_id"]
        true_times = [float(t) for t in str(row["true_header_timestamps_sec"]).split(";") if t.strip()] \
            if pd.notna(row["true_header_timestamps_sec"]) and str(row["true_header_timestamps_sec"]).strip() else []

        events_path = video_processing_dir / video_id / "header_events.csv"
        meta_path = video_processing_dir / video_id / "video_metadata.json"
        detected_times = []
        if events_path.exists():
            try:
                events = pd.read_csv(events_path)
            except pd.errors.EmptyDataError:
                events = pd.DataFrame()
            if not events.empty:
                import json
                fps = 30.0
                if meta_path.exists():
                    fps = json.loads(meta_path.read_text()).get("fps", 30.0) or 30.0
                confirmed = events[events["is_header_candidate"] == True]  # noqa: E712
                detected_times = (confirmed["event_frame"] / fps).tolist()

        matched_true = set()
        matched_detected = set()
        for i, dt in enumerate(detected_times):
            for j, tt in enumerate(true_times):
                if j in matched_true:
                    continue
                if abs(dt - tt) <= tolerance_sec:
                    matched_true.add(j)
                    matched_detected.add(i)
                    tp += 1
                    match_rows.append({"video_id": video_id, "detected_sec": round(dt, 2),
                                        "true_sec": tt, "status": "TP"})
                    break

        for i, dt in enumerate(detected_times):
            if i not in matched_detected:
                fp += 1
                match_rows.append({"video_id": video_id, "detected_sec": round(dt, 2), "true_sec": "", "status": "FP"})
        for j, tt in enumerate(true_times):
            if j not in matched_true:
                fn += 1
                match_rows.append({"video_id": video_id, "detected_sec": "", "true_sec": tt, "status": "FN"})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"matches": pd.DataFrame(match_rows), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate header event detection (Phase 8, 8.3).")
    parser.add_argument("--video_processing_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--write_template", action="store_true", help="Write a ground-truth CSV template for manual annotation and exit")
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to a filled-in ground-truth CSV for TRUE frame-level evaluation")
    parser.add_argument("--tolerance_sec", type=float, default=1.0, help="Max seconds between a detected and true header timestamp to count as a match")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    video_processing_dir = project_root / args.video_processing_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_processing_dir.exists():
        print(f"Error: '{video_processing_dir}' does not exist.")
        sys.exit(1)

    if args.write_template:
        write_ground_truth_template(video_processing_dir, output_dir / "header_ground_truth_template.csv")
        return

    print("=== PROXY evaluation (clip-level, using Header_*/Non_Header_* folder labels) ===\n")
    proxy = run_proxy_evaluation(video_processing_dir)
    if proxy:
        print(f"TP={proxy['tp']}  FP={proxy['fp']}  FN={proxy['fn']}  TN={proxy['tn']}")
        print(f"Precision: {proxy['precision']:.3f}  Recall: {proxy['recall']:.3f}  F1: {proxy['f1']:.3f}")
        proxy["table"].to_csv(output_dir / "header_detection_proxy_evaluation.csv", index=False)
        print(f"Saved: {output_dir / 'header_detection_proxy_evaluation.csv'}")
        print("\nNOTE: this is a CLIP-LEVEL proxy (did we detect >=1 event where the folder "
              "label implies one exists), not frame-accurate TP/FP/FN. Use --write_template "
              "then --ground_truth for the true evaluation in section 8.3 of the spec.")

    if args.ground_truth:
        gt_path = project_root / args.ground_truth
        if not gt_path.exists():
            print(f"\nError: ground truth file '{gt_path}' does not exist.")
            sys.exit(1)
        print(f"\n=== TRUE evaluation (frame-level, against '{gt_path.name}') ===\n")
        true_eval = run_true_evaluation(video_processing_dir, gt_path, args.tolerance_sec)
        print(f"TP={true_eval['tp']}  FP={true_eval['fp']}  FN={true_eval['fn']}")
        print(f"Precision: {true_eval['precision']:.3f}  Recall: {true_eval['recall']:.3f}  F1: {true_eval['f1']:.3f}")
        true_eval["matches"].to_csv(output_dir / "header_detection_true_evaluation.csv", index=False)
        print(f"Saved: {output_dir / 'header_detection_true_evaluation.csv'}")


if __name__ == "__main__":
    main()
