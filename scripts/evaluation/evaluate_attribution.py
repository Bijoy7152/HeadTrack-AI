#!/usr/bin/env python3
"""
Player Attribution Evaluation (Phase 8, section 8.4)

For each CONFIRMED header event, was the correct player identified? This
cannot be evaluated automatically - it requires a human to watch the frame
(or a short clip around it) and judge whether the attributed track_id is
really the player who headed the ball.

Use --write_template to generate a fillable CSV: one row per confirmed
event, pre-populated with the pipeline's own attribution (video_id,
event_frame, predicted_track_id, ambiguous, second_track_id) plus a blank
'verdict' column for you to fill in as one of: correct / incorrect /
ambiguous. A preview image already exists for each event's frame at
outputs/video_processing/<video_id>/ - use those (or the source video) to
judge each row.

Once filled in, run without --write_template (passing --verdicts) to
compute:
  Player Attribution Accuracy = correctly_attributed / total_evaluable
  plus separate correct / incorrect / ambiguous counts.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def write_template(video_processing_dir: Path, out_path: Path):
    rows = []
    for video_dir in sorted(video_processing_dir.iterdir()):
        events_path = video_dir / "header_events.csv"
        if not video_dir.is_dir() or not events_path.exists():
            continue
        try:
            events = pd.read_csv(events_path)
        except pd.errors.EmptyDataError:
            continue
        confirmed = events[events["is_header_candidate"] == True]  # noqa: E712
        for _, row in confirmed.iterrows():
            rows.append({
                "video_id": video_dir.name,
                "event_frame": row["event_frame"],
                "predicted_track_id": row["primary_track_id"],
                "second_track_id": row["second_track_id"],
                "pipeline_flagged_ambiguous": row["ambiguous"],
                "verdict": "",  # fill in: correct / incorrect / ambiguous
                "notes": "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Template written to '{out_path}' ({len(df)} confirmed events).")
    print("For each row, review the event frame (in outputs/video_processing/<video_id>/ or "
          "the source video at the given frame_number/timestamp) and fill 'verdict' with "
          "exactly one of: correct, incorrect, ambiguous.")


def evaluate(verdicts_path: Path):
    df = pd.read_csv(verdicts_path)
    filled = df[df["verdict"].notna() & (df["verdict"].astype(str).str.strip() != "")]
    if filled.empty:
        print("No filled-in verdicts found - the 'verdict' column is empty for every row.")
        return

    valid_verdicts = {"correct", "incorrect", "ambiguous"}
    invalid = filled[~filled["verdict"].str.lower().isin(valid_verdicts)]
    if not invalid.empty:
        print(f"[Warning] {len(invalid)} row(s) have an unrecognized verdict value (expected "
              f"correct/incorrect/ambiguous), excluded from scoring:")
        print(invalid[["video_id", "event_frame", "verdict"]].to_string(index=False))
        filled = filled[filled["verdict"].str.lower().isin(valid_verdicts)]

    counts = filled["verdict"].str.lower().value_counts()
    n_correct = int(counts.get("correct", 0))
    n_incorrect = int(counts.get("incorrect", 0))
    n_ambiguous = int(counts.get("ambiguous", 0))
    total_evaluable = n_correct + n_incorrect  # ambiguous cases are excluded from the accuracy denominator

    accuracy = n_correct / total_evaluable if total_evaluable > 0 else float("nan")

    print(f"--- Player Attribution Evaluation (n={len(filled)} reviewed / {len(df)} total events) ---")
    print(f"Correct    : {n_correct}")
    print(f"Incorrect  : {n_incorrect}")
    print(f"Ambiguous  : {n_ambiguous}")
    print(f"\nPlayer Attribution Accuracy (correct / (correct + incorrect)): "
          f"{accuracy:.3f}" if total_evaluable > 0 else "\nNo evaluable (correct/incorrect) rows yet.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate header performer attribution (Phase 8, 8.4).")
    parser.add_argument("--video_processing_dir", type=str, default="outputs/video_processing")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--write_template", action="store_true", help="Write a fillable attribution-verdict CSV template and exit")
    parser.add_argument("--verdicts", type=str, default=None, help="Path to the filled-in verdicts CSV to score")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    video_processing_dir = project_root / args.video_processing_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_template:
        if not video_processing_dir.exists():
            print(f"Error: '{video_processing_dir}' does not exist.")
            sys.exit(1)
        write_template(video_processing_dir, output_dir / "attribution_verdict_template.csv")
        return

    if not args.verdicts:
        print("Error: pass --verdicts <path> (a filled-in template) to score, or --write_template to generate one.")
        sys.exit(1)

    verdicts_path = project_root / args.verdicts
    if not verdicts_path.exists():
        print(f"Error: '{verdicts_path}' does not exist.")
        sys.exit(1)

    evaluate(verdicts_path)


if __name__ == "__main__":
    main()
