#!/usr/bin/env python3
"""
YOLO Player/Ball/Head Detector Training (Phase 2, sections 2.4-2.7)

Trains one or more YOLO models (default: YOLOv8s baseline, then YOLOv8m for
comparison) on dataset/data.yaml. Augmentation is restricted to
soccer-appropriate transforms and only ever applies to the training split -
Ultralytics never augments validation/test data.

Excluded on purpose (would distort real match geometry):
  extreme rotation, vertical flip, heavy perspective, extreme distortion,
  mosaic/mixup/copy-paste (composite images).

Output layout per model:
  models/checkpoints/<model_name>/{best.pt, last.pt, results.csv, figures/}
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from device_utils import resolve_device


def reorganize_run(run_dir: Path, target_dir: Path):
    """Reshape Ultralytics' default run layout into the requested structure:
    target_dir/{best.pt, last.pt, results.csv, figures/*.png,*.jpg}"""
    target_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = run_dir / "weights"
    for name in ["best.pt", "last.pt"]:
        src = weights_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)

    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        shutil.copy2(results_csv, target_dir / "results.csv")

    figures_dir = target_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    for pattern in ("*.png", "*.jpg"):
        for img in run_dir.glob(pattern):
            shutil.copy2(img, figures_dir / img.name)


def train_model(model_name, weights, data_yaml, imgsz, epochs, patience, seed, device, project_dir, batch):
    print(f"\n{'=' * 70}\nTraining {model_name} ({weights})\n{'=' * 70}")
    model = YOLO(weights)
    run_name = f"{model_name}_run"
    model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        patience=patience,
        seed=seed,
        device=device,
        batch=batch,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
        # --- augmentation: train split only, soccer-appropriate (2.6) ---
        fliplr=0.5,          # horizontal flip
        flipud=0.0,          # avoid vertical flip
        degrees=0.0,         # avoid rotation
        translate=0.1,       # small translation
        scale=0.3,           # small scaling
        shear=0.0,           # avoid distortion
        perspective=0.0003,  # limited perspective
        hsv_h=0.015,         # limited hue variation
        hsv_s=0.5,           # moderate saturation/contrast variation
        hsv_v=0.3,           # moderate brightness variation
        mosaic=0.0,          # disabled: composites alter scene realism
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
    )
    run_dir = Path(model.trainer.save_dir)
    target_dir = project_dir / model_name
    reorganize_run(run_dir, target_dir)
    print(f"\n{model_name} training complete. Outputs organized at: {target_dir}")
    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Train YOLO player/ball/head detectors.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--models", type=str, nargs="+", default=["yolov8s", "yolov8m"],
                         help="Model variants to train, e.g. yolov8s yolov8m")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without mAP improvement)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto batch size")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. 'mps', 'cpu', '0'; auto-detects (CUDA > MPS > CPU) if unset")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    parser.add_argument("--pretrained_dir", type=str, default="models/pretrained", help="Directory containing the base <model_name>.pt weights to start training from")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml = project_root / args.data
    project_dir = project_root / args.output_dir
    pretrained_dir = project_root / args.pretrained_dir

    if not data_yaml.exists():
        print(f"Error: data.yaml not found at '{data_yaml}'.")
        sys.exit(1)

    project_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        weights = str(pretrained_dir / f"{model_name}.pt")
        train_model(model_name, weights, data_yaml, args.imgsz, args.epochs, args.patience,
                    args.seed, args.device, project_dir, args.batch)


if __name__ == "__main__":
    main()
