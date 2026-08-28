#!/usr/bin/env python3
"""
Final Thesis Output Summary (Phase 8 consolidation)

Pulls together every evaluation artifact produced across Phases 1-8 into
one printed report plus outputs/evaluation/thesis_summary.json, and checks
off the 14-item "final research outputs" checklist from the Phase 8 spec.

Run this last, after scripts/evaluation/evaluate_tracking.py, evaluate_header_detection.py,
evaluate_attribution.py (optional - only if ground truth was filled in), and
evaluate_llm_reports.py.
"""

import json
import sys
from pathlib import Path

import pandas as pd


def safe_read_csv(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except pd.errors.EmptyDataError:
        return None


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs = project_root / "outputs"
    evaluation = outputs / "evaluation"

    summary = {}

    print("=" * 70)
    print("PHASE 8 — VERIFICATION AND EVALUATION SUMMARY")
    print("=" * 70)

    # 8.1 Detection
    det = safe_read_csv(evaluation / "model_comparison.csv")
    class_det = safe_read_csv(evaluation / "class_wise_metrics.csv")
    print("\n--- 8.1 Detection Evaluation ---")
    if det is not None:
        print(det.to_string(index=False))
        summary["detection_model_comparison"] = det.to_dict(orient="records")
    if class_det is not None:
        print()
        print(class_det.to_string(index=False))
        summary["detection_class_wise"] = class_det.to_dict(orient="records")
    if det is None and class_det is None:
        print("Not found - run scripts/evaluation/evaluate_models.py first.")

    # 8.2 Tracking
    trk = safe_read_csv(evaluation / "tracking_evaluation_summary.csv")
    print("\n--- 8.2 Tracking Evaluation (heuristic proxy, no MOT ground truth) ---")
    if trk is not None:
        total_switches = int(trk["candidate_id_switches"].sum())
        print(f"Videos analyzed: {len(trk)}, total tracks: {int(trk['n_tracks'].sum())}, "
              f"avg lifespan: {trk['avg_track_lifespan_frames'].mean():.1f} frames, "
              f"candidate ID switches: {total_switches}")
        summary["tracking_proxy"] = {
            "videos_analyzed": len(trk), "total_tracks": int(trk["n_tracks"].sum()),
            "avg_lifespan_frames": round(float(trk["avg_track_lifespan_frames"].mean()), 1),
            "total_candidate_id_switches": total_switches,
        }
    else:
        print("Not found - run scripts/evaluation/evaluate_tracking.py first.")

    # 8.3 Header detection
    hdr_proxy = safe_read_csv(evaluation / "header_detection_proxy_evaluation.csv")
    hdr_true = safe_read_csv(evaluation / "header_detection_true_evaluation.csv")
    print("\n--- 8.3 Header Event Evaluation ---")
    if hdr_proxy is not None:
        tp = int(((hdr_proxy["is_header_clip"]) & (hdr_proxy["detected_any"])).sum())
        fp = int(((~hdr_proxy["is_header_clip"]) & (hdr_proxy["detected_any"])).sum())
        fn = int(((hdr_proxy["is_header_clip"]) & (~hdr_proxy["detected_any"])).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"Proxy (clip-level): TP={tp} FP={fp} FN={fn}  P={p:.3f} R={r:.3f} F1={f1:.3f}")
        summary["header_detection_proxy"] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
    else:
        print("Proxy not found - run scripts/evaluation/evaluate_header_detection.py first.")
    if hdr_true is not None:
        tp = int((hdr_true["status"] == "TP").sum())
        fp = int((hdr_true["status"] == "FP").sum())
        fn = int((hdr_true["status"] == "FN").sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"True (manual ground truth): TP={tp} FP={fp} FN={fn}  P={p:.3f} R={r:.3f} F1={f1:.3f}")
        summary["header_detection_true"] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
    else:
        print("True evaluation not available (needs manually filled ground truth - see "
              "--write_template on scripts/evaluation/evaluate_header_detection.py).")

    # 8.4 Attribution
    attrib_template = evaluation / "attribution_verdict_template.csv"
    print("\n--- 8.4 Player Attribution Evaluation ---")
    if attrib_template.exists():
        df = pd.read_csv(attrib_template)
        reviewed = df[df["verdict"].notna() & (df["verdict"].astype(str).str.strip() != "")]
        if not reviewed.empty:
            counts = reviewed["verdict"].str.lower().value_counts()
            n_correct, n_incorrect = int(counts.get("correct", 0)), int(counts.get("incorrect", 0))
            n_ambiguous = int(counts.get("ambiguous", 0))
            denom = n_correct + n_incorrect
            acc = n_correct / denom if denom else float("nan")
            print(f"Reviewed: {len(reviewed)}/{len(df)}  Correct={n_correct} Incorrect={n_incorrect} "
                  f"Ambiguous={n_ambiguous}  Accuracy={acc:.3f}" if denom else "No evaluable rows yet.")
            summary["attribution_evaluation"] = {"reviewed": len(reviewed), "total_events": len(df),
                                                  "correct": n_correct, "incorrect": n_incorrect,
                                                  "ambiguous": n_ambiguous,
                                                  "accuracy": acc if denom else None}
        else:
            print(f"Template exists ({len(df)} events) but no verdicts filled in yet - see "
                  "scripts/evaluation/evaluate_attribution.py --write_template.")
    else:
        print("Not found - run scripts/evaluation/evaluate_attribution.py --write_template first.")

    # 8.5 LLM
    llm = safe_read_csv(evaluation / "llm_report_verification.csv")
    print("\n--- 8.5 LLM Report Verification ---")
    if llm is not None:
        overall = float(llm["all_checks_passed"].mean())
        halluc_free = float(llm["hallucination_free"].mean())
        print(f"Reports checked: {len(llm)}  All-checks-passed: {overall * 100:.1f}%  "
              f"Hallucination-free: {halluc_free * 100:.1f}%")
        summary["llm_verification"] = {"n_reports": len(llm), "all_checks_passed_rate": overall,
                                        "hallucination_free_rate": halluc_free}
    else:
        print("Not found - run scripts/evaluation/evaluate_llm_reports.py first.")

    # Final checklist
    checklist = [
        ("1. Trained player-ball-head detector", (project_root / "models/checkpoints/yolov8m/best.pt").exists()),
        ("2. Test-set detection results", det is not None),
        ("3. Full-video inference pipeline", (outputs / "video_processing").exists() and any((outputs / "video_processing").iterdir())),
        ("4. Player tracking system", (outputs / "player_tracks.csv").exists()),
        ("5. Ball trajectory tracking", (outputs / "ball_tracks.csv").exists()),
        ("6. Head-player association module", (outputs / "head_player_associations.csv").exists()),
        ("7. Header detection algorithm", (outputs / "header_events.csv").exists()),
        ("8. Header performer attribution", (outputs / "header_events.csv").exists()),
        ("9. Ambiguity detection", (outputs / "header_events.csv").exists()),
        ("10. Player-wise header exposure statistics", (outputs / "player_summary.csv").exists()),
        ("11. Structured CSV/JSON dataset", (outputs / "analysis_metadata.json").exists()),
        ("12. LLM-generated exposure reports", (outputs / "reports/all_reports.json").exists()),
        ("13. End-to-end evaluation", any((evaluation / n).exists() for n in
            ["model_comparison.csv", "tracking_evaluation_summary.csv",
             "header_detection_proxy_evaluation.csv", "llm_report_verification.csv"])),
        ("14. Error analysis", (evaluation / "missed_header_diagnostics.csv").exists()),
    ]
    print("\n--- Final Research Outputs Checklist ---")
    for name, done in checklist:
        print(f"  [{'x' if done else ' '}] {name}")
    summary["checklist"] = {name: done for name, done in checklist}

    evaluation.mkdir(parents=True, exist_ok=True)
    with open(evaluation / "thesis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {evaluation / 'thesis_summary.json'}")


if __name__ == "__main__":
    main()
