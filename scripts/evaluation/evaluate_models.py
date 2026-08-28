#!/usr/bin/env python3
"""
YOLO Detector Evaluation & Model Comparison (Phase 2, sections 2.8-2.10)

Evaluates each trained model's best.pt on the test split and reports:
- Per-class (player/ball/head) precision, recall, F1, AP50, mAP50-95
- Overall precision, recall, F1, mAP50, mAP50-95
- Inference speed (FPS) and parameter count, as model-selection tradeoffs
  (ball/head recall and inference cost matter as much as overall mAP here)

Outputs:
  outputs/evaluation/class_wise_metrics.csv
  outputs/evaluation/model_comparison.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from device_utils import resolve_device

CLASS_NAMES = {0: "player", 1: "ball", 2: "head"}


def benchmark_fps(model, imgsz, device, n_warmup=5, n_iters=30):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(n_warmup):
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)
    start = time.time()
    for _ in range(n_iters):
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)
    elapsed = time.time() - start
    return n_iters / elapsed if elapsed > 0 else 0.0


def evaluate_model(model_name, weights_path, data_yaml, imgsz, device, output_dir):
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=imgsz, device=device, plots=True, verbose=False,
        project=str(output_dir), name=f"{model_name}_val", exist_ok=True,
    )

    class_rows = []
    ap_class_index = [int(c) for c in metrics.box.ap_class_index]
    for i, cid in enumerate(ap_class_index):
        class_rows.append({
            "model": model_name,
            "class_id": cid,
            "class_name": CLASS_NAMES.get(cid, str(cid)),
            "precision": round(float(metrics.box.p[i]), 4),
            "recall": round(float(metrics.box.r[i]), 4),
            "f1": round(float(metrics.box.f1[i]), 4),
            "AP50": round(float(metrics.box.ap50[i]), 4),
            "mAP50-95": round(float(metrics.box.ap[i]), 4),
        })

    fps = benchmark_fps(model, imgsz, device)
    n_params = sum(p.numel() for p in model.model.parameters())

    mp, mr = float(metrics.box.mp), float(metrics.box.mr)
    overall_row = {
        "model": model_name,
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "f1": round(2 * mp * mr / (mp + mr), 4) if (mp + mr) > 0 else 0.0,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "FPS": round(fps, 2),
        "params_millions": round(n_params / 1e6, 2),
    }
    return class_rows, overall_row


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained YOLO detectors.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--models_dir", type=str, default="models/checkpoints",
                         help="Directory containing <model_name>/best.pt subfolders")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                         help="Specific model subfolder names to evaluate (default: all found)")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. 'mps', 'cpu', '0'; auto-detects (CUDA > MPS > CPU) if unset")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml = project_root / args.data
    models_dir = project_root / args.models_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        print(f"Error: data.yaml not found at '{data_yaml}'.")
        sys.exit(1)
    if not models_dir.exists():
        print(f"Error: models directory '{models_dir}' does not exist.")
        sys.exit(1)

    if args.models:
        model_names = args.models
    else:
        model_names = sorted(d.name for d in models_dir.iterdir() if d.is_dir() and (d / "best.pt").exists())

    if not model_names:
        print(f"No trained models (best.pt) found under '{models_dir}'.")
        sys.exit(0)

    all_class_rows, all_overall_rows = [], []
    for model_name in model_names:
        weights_path = models_dir / model_name / "best.pt"
        print(f"\nEvaluating {model_name} ({weights_path})...")
        class_rows, overall_row = evaluate_model(model_name, weights_path, data_yaml, args.imgsz, args.device, output_dir)
        all_class_rows.extend(class_rows)
        all_overall_rows.append(overall_row)

    class_df = pd.DataFrame(all_class_rows)
    overall_df = pd.DataFrame(all_overall_rows)

    class_csv = output_dir / "class_wise_metrics.csv"
    overall_csv = output_dir / "model_comparison.csv"
    class_df.to_csv(class_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)

    print("\n--- Class-Wise Metrics (Precision / Recall / F1 / AP50 / mAP50-95) ---")
    print(class_df.to_string(index=False))
    print("\n--- Model Comparison ---")
    print(overall_df.to_string(index=False))
    print(f"\nSaved: {class_csv}")
    print(f"Saved: {overall_csv}")


if __name__ == "__main__":
    main()
