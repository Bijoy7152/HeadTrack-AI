# HeadTrack AI

Automated Soccer Header Exposure Tracking and LLM-Based Player Reporting.
Detects players/ball/heads in video (YOLO), tracks them (ByteTrack +
Kalman filter), infers header contact events, attributes each header to a
player, aggregates exposure per player, generates a natural-language report
per player via a local LLM (Ollama), and presents it all in a FastAPI
dashboard.

Built as a thesis project — see `Paper/` and `Thesis book/` for the
academic write-up (not code).

Classes (fixed, do not change): `0 = player`, `1 = ball`, `2 = head`.

```
Dataset Creation → Object Detection Training → Model Evaluation
  → Full Video Inference → Player Tracking → Ball Tracking
  → Head-to-Player Association → Ball-to-Head Distance
  → Temporal Contact Confirmation → Header Event Detection
  → Header Performer Attribution → Player-wise Exposure Analysis
  → Structured CSV/JSON → LLM Player Report → Website Dashboard
```

Every path in `scripts/` and `webapp/` is resolved relative to the project
root at runtime — nothing depends on a username, drive letter, or fixed
folder location. Clone this repo anywhere and it works.

---

## Table of contents

1. [Project structure](#1-project-structure)
2. [Setup](#2-setup)
3. [Dataset development](#3-dataset-development)
4. [Model training](#4-model-training)
5. [Model evaluation](#5-model-evaluation)
6. [Full video processing / inference](#6-full-video-processing--inference)
7. [Header detection & exposure analysis](#7-header-detection--exposure-analysis)
8. [LLM reporting](#8-llm-reporting)
9. [Website](#9-website)
10. [Quick start — clone and run (already-trained model)](#10-quick-start--clone-and-run-already-trained-model)
11. [Full command reference](#11-full-command-reference)

---

## 1. Project structure

Code is grouped by responsibility: dataset code, training code, evaluation
code, inference code, header-analysis code, LLM-reporting code, and website
code are each isolated in their own folder and don't import each other
except where the pipeline itself requires it (e.g. the website calls the
CV/LLM scripts as subprocesses — it never reimplements their logic).

```text
.
├── README.md                    You are here
├── requirements.txt              Installs scripts/ + webapp/ dependencies together
├── .env.example                   Every optional environment variable, documented
├── .gitignore
│
├── scripts/                       All CV/ML/LLM pipeline code, grouped by responsibility
│   ├── requirements.txt
│   ├── common/                    Shared utility with no phase-specific logic
│   │   └── device_utils.py        Inference device auto-detect (CUDA > MPS > CPU)
│   ├── dataset/                   1. DATASET DEVELOPMENT
│   │   ├── extract_frames.py
│   │   ├── clean_frames.py
│   │   ├── auto_annotate_players_heads.py
│   │   ├── auto_annotate_ball.py
│   │   ├── field_utils.py         (shared by the two auto_annotate_* scripts only)
│   │   ├── qc_auto_annotations.py
│   │   ├── validate_annotations.py
│   │   └── split_dataset.py
│   ├── training/                   2. MODEL TRAINING
│   │   └── train_yolo.py
│   ├── evaluation/                 3. MODEL EVALUATION
│   │   ├── evaluate_models.py
│   │   ├── evaluate_tracking.py
│   │   ├── evaluate_header_detection.py
│   │   ├── evaluate_attribution.py
│   │   ├── evaluate_llm_reports.py
│   │   ├── diagnose_missed_headers.py
│   │   └── generate_thesis_summary.py
│   ├── inference/                  4. FULL VIDEO PROCESSING / INFERENCE
│   │   ├── process_video.py
│   │   ├── kalman_filter.py        not imported anywhere; process_video.py
│   │   │                            has its own inline Kalman implementation
│   │   └── head_association.py     not imported anywhere; process_video.py
│   │                                has its own inline association logic
│   ├── header_analysis/            5. HEADER DETECTION & EXPOSURE ANALYSIS
│   │   ├── detect_headers.py
│   │   ├── analyze_exposure.py
│   │   └── export_structured_data.py
│   └── llm_reporting/              6. LLM REPORTING
│       └── generate_reports.py
│
├── webapp/                         7. WEBSITE APPLICATION
│   ├── app.py                      FastAPI routes/pages
│   ├── pipeline_runner.py          Calls scripts/*/*.py as subprocesses — no CV logic here
│   ├── database.py                  SQLite persistence
│   ├── requirements.txt
│   ├── static/, templates/           Bootstrap/Chart.js UI (server-rendered Jinja2)
│   ├── jobs/                          Per-upload pipeline outputs (gitignored)
│   ├── uploads/                        Uploaded videos (gitignored)
│   └── headtrack.db                     SQLite database (gitignored)
│
├── models/                          8. MODELS / CHECKPOINTS
│   ├── pretrained/                   Base weights (COCO-pretrained; used as
│   │   │                              training starting points and by the
│   │   │                              auto-annotation scripts)
│   │   └── yolo11x.pt, yolo11x-pose.pt, yolov8m.pt, yolov8s.pt, yolov8n.pt, yolov8n-pose.pt
│   └── checkpoints/                   Trained model outputs (from train_yolo.py)
│       ├── yolov8m/{best.pt, last.pt, results.csv, figures/}
│       ├── yolov8s/{best.pt, last.pt, results.csv, figures/}
│       └── yolov8m_run/, yolov8s_run/   Full Ultralytics run artifacts
│
├── outputs/                          9. OUTPUTS / RESULTS
│   ├── video_processing/<video_id>/   Per-video detections/tracks (Phase 3)
│   ├── exposure_analysis/              Per-video exposure aggregates (Phase 5)
│   ├── evaluation/                      All evaluation CSVs/JSON (Phase 8)
│   ├── reports/                          LLM-generated player reports (Phase 7)
│   ├── preview_ball/, preview_player_head/  Auto-annotation preview images
│   ├── detections.csv, player_tracks.csv, ball_tracks.csv,
│   │   head_player_associations.csv, header_events.csv,
│   │   player_summary.csv/json, analysis_metadata.json  (Phase 6 consolidated exports)
│   ├── dataset_statistics.csv, dataset_validation_errors.csv
│   └── ball_auto_annotation.csv, player_head_auto_annotation.csv, manual_review_required.csv
│
├── videos/                            Raw source clips: videos/Header/, videos/"Non Header"/
├── extracted_frames/                   Frames extracted from videos/ (Phase 1)
├── auto_labels/                        Auto-generated YOLO pseudo-labels (Phase 1)
├── dataset/                             Final train/valid/test split + data.yaml (Phase 2)
├── frames_metadata.csv                   Per-frame extraction metadata
│
├── Paper/                                Conference paper (academic deliverable, not code)
└── Thesis book/                          Thesis manuscript + methodology notes (not code)
```

`videos/`, `extracted_frames/`, `auto_labels/`, `dataset/`, `outputs/`,
`models/pretrained/*.pt`, `models/checkpoints/*/best.pt` and
`webapp/{jobs,uploads,headtrack.db}` are all listed in `.gitignore` — they
are regenerable or too large for git and are not part of a fresh clone. See
§2.6 for what that means when you clone this repo.

---

## 2. Setup

### 2.1 Prerequisites

- **Python**: developed and tested with Python 3.12.11. No minimum version
  is pinned in the repo; 3.10+ is likely fine given the dependencies
  (`ultralytics`, `fastapi`) but this hasn't been verified on anything but
  3.12.
- **Ollama** (only needed for LLM report generation, §8): install from
  [ollama.com](https://ollama.com), then:
  ```bash
  ollama pull llama3.1:8b
  ```
- **GPU**: optional. Inference/training auto-detects CUDA > Apple Silicon
  (MPS) > CPU at runtime. CPU works, just slower.

### 2.2 Clone and create the environment

```bash
git clone <repo-url>
cd "HeadTrack AI"                  # or whatever you named the folder — the
                                    # project does not depend on this name
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
```

### 2.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt     # installs scripts/requirements.txt + webapp/requirements.txt
```

`requirements.txt` (root) is just an include file:
```text
-r scripts/requirements.txt
-r webapp/requirements.txt
```
so you can install either half standalone if you only need one side of the
project (e.g. `pip install -r scripts/requirements.txt` for CV/ML work with
no website).

### 2.4 Configure `.env` (optional)

```bash
cp .env.example .env
```

Every variable is optional and commented out — the project runs with zero
`.env` file present, using paths resolved relative to the project root.
Only uncomment something if a default doesn't fit your machine:

| Variable | Default if unset |
|---|---|
| `DEVICE` | auto-detect (CUDA > MPS > CPU) |
| `MODEL_PATH` | `models/checkpoints/yolov8m/best.pt` |
| `JOBS_DIR` | `webapp/jobs` |
| `UPLOADS_DIR` | `webapp/uploads` |
| `DB_PATH` | `webapp/headtrack.db` |
| `WEBAPP_HOST` / `WEBAPP_PORT` / `WEBAPP_RELOAD` | `127.0.0.1` / `8000` / `false` (only used by `python app.py`, not the `uvicorn` CLI form) |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | Ollama's own default / `llama3.1:8b` |

Full reference with comments: [.env.example](.env.example).

### 2.5 Verify the install

```bash
python -c "import cv2, pandas, numpy, yaml, tqdm, ultralytics, ollama; print('scripts deps OK')"
python -c "import fastapi, uvicorn, jinja2, reportlab; print('webapp deps OK')"
python -c "import sys; sys.path.insert(0, 'scripts/common'); from device_utils import resolve_device; print('resolved device:', resolve_device())"
```

### 2.6 What a fresh `git clone` gives you vs. what's gitignored

If you received this project as a **full copy** (not a fresh empty clone),
everything below already exists and you do not need to regenerate it. If
you did a **fresh `git clone`**, the following are gitignored (not
committed — too large or regenerable) and start empty:

| What | Where | Regenerate with (section) |
|---|---|---|
| Raw video clips | `videos/Header/`, `videos/"Non Header"/` | — (source data, not regenerable by script) |
| Extracted frames | `extracted_frames/` | §3.2 |
| Auto-generated pseudo-labels | `auto_labels/` | §3.4–3.5 |
| Final train/valid/test split | `dataset/` + `dataset/data.yaml` | §3.8 |
| Base pretrained weights | `models/pretrained/*.pt` | auto-downloaded by Ultralytics on first use if missing |
| Trained detector checkpoints | `models/checkpoints/yolov8m/best.pt` (**required for the website**) | §4 |
| Per-video inference outputs | `outputs/video_processing/` | §6 |
| Evaluation results | `outputs/evaluation/` | §5 |
| LLM reports | `outputs/reports/` | §8 |
| Website's own job/upload data | `webapp/jobs/`, `webapp/uploads/`, `webapp/headtrack.db` | created automatically the first time you run the website |

**The website will not work on a fresh clone until a trained model exists**
at `models/checkpoints/yolov8m/best.pt` (or `MODEL_PATH` in `.env` points at
one) — either train one yourself (§3 → §4) or copy an existing
`models/checkpoints/` folder in from elsewhere.

---

## 3. Dataset development

Code: `scripts/dataset/` — separate from training, evaluation, inference,
and website code. This is the actual pipeline used to build the dataset in
this repo (auto-annotation, not manual Roboflow annotation — see the note
at the end of this section).

```
videos/ → extract_frames.py → extracted_frames/
        → clean_frames.py (report/move blurry+duplicate frames)
        → auto_annotate_players_heads.py → auto_labels/ (player + head boxes)
        → auto_annotate_ball.py → auto_labels/ (+ ball boxes, appended)
        → qc_auto_annotations.py (sanity-check the pseudo-labels)
        → validate_annotations.py (integrity + statistics)
        → split_dataset.py → dataset/{train,valid,test} + dataset/data.yaml
```

### 3.1 Organize raw videos

Place clips under `videos/Header/` and `videos/"Non Header"/` (the folder
name has a space — quote it in shell commands). This already exists in the
current repo; only needed if starting from zero.

### 3.2 Extract frames

```bash
python scripts/dataset/extract_frames.py \
  --video_dir videos \
  --output_dir extracted_frames \
  --header_fps 10 \
  --non_header_fps 2
```
Header clips are sampled denser (10 FPS) around the contact moment; Non
Header clips at 2 FPS. Writes `extracted_frames/frames_metadata.csv`.

### 3.3 Clean frames

```bash
python scripts/dataset/clean_frames.py --action report   # writes extracted_frames/frame_cleaning_report.csv
python scripts/dataset/clean_frames.py --action move      # moves rejects to rejected_frames/
```
Flags blurred frames (Laplacian variance) and near-duplicate consecutive
frames (mean pixel diff). Manually spot-check the kept set for
scoreboard-only/crowd-only/pure-transition frames.

### 3.4 Auto-annotate players + heads

```bash
python scripts/dataset/auto_annotate_players_heads.py \
  --input_dir extracted_frames \
  --output_dir auto_labels \
  --player_model models/pretrained/yolo11x.pt \
  --pose_model models/pretrained/yolo11x-pose.pt
```
Runs a pretrained Ultralytics detector (COCO person class) for players and a
pretrained pose model (facial keypoints) to derive head boxes. Writes YOLO
label files (`0 = player`, `2 = head`) to `auto_labels/`.

### 3.5 Auto-annotate ball

```bash
python scripts/dataset/auto_annotate_ball.py \
  --input_dir extracted_frames \
  --labels_dir auto_labels \
  --model models/pretrained/yolo11x.pt
```
Detects the COCO 'sports ball' class and appends class `1` labels to the
existing files in `auto_labels/` — never overwrites the player/head lines
already there.

### 3.6 QC the pseudo-labels

```bash
python scripts/dataset/qc_auto_annotations.py --labels_dir auto_labels
```
Sanity-checks class IDs, coordinate ranges, and player/head/ball
relationships. Never modifies or deletes label files — only reports issues.

### 3.7 Validate annotation integrity

```bash
python scripts/dataset/validate_annotations.py --source auto_labels
```
Or, once split, validate the final dataset:
```bash
python scripts/dataset/validate_annotations.py --source dataset
```
Checks image-label correspondence, valid class IDs (0/1/2), valid YOLO
coordinates, and reports per-class box counts. Writes
`outputs/dataset_statistics.csv` and `outputs/dataset_validation_errors.csv`.

### 3.8 Split into train/valid/test

```bash
python scripts/dataset/split_dataset.py \
  --images_dir extracted_frames \
  --labels_dir auto_labels \
  --output_dir dataset \
  --train 0.70 --valid 0.15 --test 0.15
```
Frames from the same source video stay together in one split (prevents
leakage). Populates `dataset/{train,valid,test}/{images,labels}` and
generates `dataset/data.yaml`. The generated `data.yaml`'s `path:` field is
written **relative to the project root** (e.g. `path: dataset`), not an
absolute path, so it keeps working if the project folder is renamed or
moved — as long as scripts are run from the project root (the convention
used throughout this repo).

### Alternative: Roboflow-annotated source

`validate_annotations.py` and `split_dataset.py` both accept
`--source <folder>` for a single flat folder containing both images and
labels side by side (e.g. a Roboflow YOLO-format export), as an alternative
to the mirrored `extracted_frames/` + `auto_labels/` layout. `roboflow` is
listed in `scripts/requirements.txt` for this path — but the dataset
currently in this repo was built via the auto-annotation route above
(`auto_labels/` is populated; no `roboflow_export/` folder exists), so
whether a Roboflow account/project was ever actually used for this thesis
is unconfirmed from the code alone.

### Current dataset stats (from the last `validate_annotations.py` run in this repo)

- 7,495 images / 7,495 labels, integrity: PASSED
- 82,663 player boxes, 2,035 ball boxes, 14,602 head boxes
- Split: 5,422 train / 928 valid / 1,145 test images

---

## 4. Model training

Code: `scripts/training/train_yolo.py` — separate from dataset code,
evaluation code, and website code. Its only inputs are `dataset/data.yaml`
(§3) and base weights in `models/pretrained/`.

Trains one or more YOLO detector variants (default: `yolov8s` then
`yolov8m`) on `dataset/data.yaml`, detecting classes
`0=player, 1=ball, 2=head`. Augmentation is restricted to
soccer-appropriate transforms and only ever applied to the training split —
extreme rotation, vertical flip, heavy perspective/distortion, and
mosaic/mixup/copy-paste are excluded on purpose (they'd distort real match
geometry).

```bash
python scripts/training/train_yolo.py \
  --data dataset/data.yaml \
  --models yolov8s yolov8m \
  --imgsz 960 \
  --epochs 100 \
  --patience 20 \
  --batch -1 \
  --output_dir models/checkpoints \
  --pretrained_dir models/pretrained
```

All flags shown are the defaults except `--models` (spelled out for
clarity) — running `python scripts/training/train_yolo.py` with no flags
does the same thing.

- `--device`: leave unset to auto-detect (CUDA > Apple MPS > CPU); pass
  `cpu`, `mps`, or a CUDA index (`0`) to force one.
- `--batch -1`: Ultralytics auto-selects batch size for the device.
- `--pretrained_dir`: where `<model_name>.pt` base weights are read from
  before training starts (already present in this repo at
  `models/pretrained/yolov8s.pt`, `models/pretrained/yolov8m.pt`, etc.). If
  a requested variant isn't there, Ultralytics downloads it from its
  official releases into that folder.

Output:
```
models/checkpoints/<model_name>/
  best.pt        highest-validation-mAP checkpoint
  last.pt        final-epoch checkpoint
  results.csv    per-epoch metrics
  figures/       training curves
models/checkpoints/<model_name>_run/   full raw Ultralytics run directory
```

The website (§9) and the inference/evaluation scripts default to
`models/checkpoints/yolov8m/best.pt` as *the* trained detector. If you train
a different variant and want it used instead, either train `yolov8m`, or
override `MODEL_PATH` in `.env` and pass `--model_path`/`--models_dir`
explicitly where scripts accept it.

---

## 5. Model evaluation

Code: `scripts/evaluation/` — separate from training and dataset code. Every
script here is read-only with respect to `dataset/`, `videos/`, and
`models/`; they only read existing outputs and write CSV/JSON reports into
`outputs/evaluation/`.

Run these **after** you have: a trained model (§4), and ideally some
processed videos (`outputs/video_processing/`, §6) and generated LLM
reports (`outputs/reports/`, §8).

### 5.1 Detector evaluation & model comparison

```bash
python scripts/evaluation/evaluate_models.py \
  --data dataset/data.yaml \
  --models_dir models/checkpoints \
  --output_dir outputs/evaluation
```
Runs each `models/checkpoints/<name>/best.pt` against the test split.
Writes `outputs/evaluation/class_wise_metrics.csv` (per-class
precision/recall/F1/AP50/mAP50-95) and `model_comparison.csv` (overall
metrics + inference FPS + parameter count).

### 5.2 Tracking evaluation (heuristic proxy)

```bash
python scripts/evaluation/evaluate_tracking.py --video_processing_dir outputs/video_processing
```
No dense MOT ground truth exists for this dataset, so this reports an
automated heuristic proxy: track count/lifespan stats and candidate
ID-switch events (a track disappearing and a new one appearing nearby
shortly after — a classic ByteTrack occlusion failure mode). Writes
`outputs/evaluation/tracking_evaluation_summary.csv` and
`tracking_id_switch_candidates.csv`.

### 5.3 Header event detection evaluation

```bash
python scripts/evaluation/evaluate_header_detection.py --video_processing_dir outputs/video_processing
```
Always runs a PROXY evaluation automatically: uses `videos/Header/` vs
`videos/"Non Header"/` folder membership as weak clip-level ground truth
(clip-level Precision/Recall/F1, not frame-accurate).

For frame-accurate TRUE evaluation, first generate a fillable template:
```bash
python scripts/evaluation/evaluate_header_detection.py --write_template
```
fill in the true header timestamps by hand, then:
```bash
python scripts/evaluation/evaluate_header_detection.py --ground_truth outputs/evaluation/header_ground_truth_template.csv
```

### 5.4 Player attribution evaluation (manual)

Requires a human to judge each confirmed header event.
```bash
python scripts/evaluation/evaluate_attribution.py --write_template
```
This writes `outputs/evaluation/attribution_verdict_template.csv`,
pre-populated with the pipeline's own attribution — fill in the `verdict`
column (`correct`/`incorrect`/`ambiguous`) using the preview images in
`outputs/video_processing/<video_id>/` or the source video, then:
```bash
python scripts/evaluation/evaluate_attribution.py --verdicts outputs/evaluation/attribution_verdict_template.csv
```

### 5.5 LLM report verification (fully automated)

```bash
python scripts/evaluation/evaluate_llm_reports.py \
  --reports_json outputs/reports/all_reports.json \
  --output_csv outputs/evaluation/llm_report_verification.csv
```
Cross-checks every generated report's text against the exact structured
JSON it came from: header counts, match/player IDs, verbatim
`review_priority`, ambiguous-event mentions, a forbidden-content
hallucination check (g-force, m/s, acceleration, concussion/diagnosis
claims), and presence of the required disclaimer.

### 5.6 Diagnostics — why did a clip get zero confirmed headers?

```bash
python scripts/evaluation/diagnose_missed_headers.py --video_glob "Header_*"
```
For clips with zero confirmed candidates, reports whether it's because no
ball/head was ever detected, or a genuine near-miss on the adaptive
distance threshold (candidate for tuning `--k` in §7.1).

### 5.7 Final consolidated summary (run last)

```bash
python scripts/evaluation/generate_thesis_summary.py
```
Pulls together every evaluation artifact above into one printed report plus
`outputs/evaluation/thesis_summary.json`, and checks off the 14-item final
research-outputs checklist. Run this after 5.1–5.6 above (5.4 is optional —
only if you filled in attribution ground truth).

---

## 6. Full video processing / inference

Code: `scripts/inference/process_video.py` — separate from dataset creation,
training, and evaluation code. This is the one stage that runs the trained
detector end-to-end over a full video.

```
Frame → YOLO detector → player + ball + head detections
Player detections → ByteTrack → persistent track IDs
Ball detections → Kalman filter (inline, cv2.KalmanFilter-based) → smoothed trajectory
Head detections → associated to the nearest compatible player track (inline)
```

```bash
python scripts/inference/process_video.py \
  --video videos/Header/h1.mp4 \
  --model models/checkpoints/yolov8m/best.pt \
  --output_dir outputs/video_processing
```

- `--video` accepts a single file or a directory of videos.
- `--model` defaults to `models/checkpoints/yolov8m/best.pt` if omitted.
- `--device` leave unset to auto-detect (CUDA > MPS > CPU).

This is the CPU/GPU-heavy step — it processes every frame of a full video
through the detector.

Output, per video, under `outputs/video_processing/<video_id>/`:

| File | Contents |
|---|---|
| `frame_detections.csv` | every raw detection |
| `player_tracks.csv` | ByteTrack player tracks |
| `ball_track.csv` | Kalman-filtered ball trajectory (detected/predicted per frame) |
| `head_associations.csv` | head-to-player-track associations |
| `video_metadata.json` | video_id, fps, resolution, frame count |

`<video_id>` is normally the input filename's stem; when given a directory
of videos, `process_video.py` may prefix it with the parent folder name.

The website (§9) calls this exact script as a subprocess, scoped to a
per-job output directory (`webapp/jobs/<job_id>/video_processing/`) instead
of the shared `outputs/video_processing/`.

---

## 7. Header detection & exposure analysis

Code: `scripts/header_analysis/` — separate from general video inference
(§6) and from website code (§9). Takes `process_video.py`'s per-video CSVs
as input; no video/model access here.

```
outputs/video_processing/<video_id>/{ball_track.csv, head_associations.csv, frame_detections.csv, player_tracks.csv}
  → detect_headers.py       → header_events.csv (per video, confirmed events + attribution)
  → analyze_exposure.py     → outputs/exposure_analysis/ (player-wise exposure stats)
  → export_structured_data.py → outputs/ (consolidated, LLM-ready CSV/JSON)
```

### 7.1 Header event detection + performer attribution

```bash
python scripts/header_analysis/detect_headers.py --video_processing_dir outputs/video_processing
```

No separate action classifier — a header is inferred from combined signals,
none individually decisive:
- ball-to-head distance vs. an adaptive threshold `T = k * head_box_width`
  (`--k`; scales with apparent player/head size instead of a fixed pixel
  value)
- temporal verification: distance should approach then recede over a
  ±2-frame window
- ball trajectory-change angle as supporting (not required) evidence
- a weighted composite confidence (`--w_spatial`, `--w_temporal`,
  `--w_consistency`, `--w_trajectory`) against `--confirm_threshold`

Ambiguity: when the gap between the best and second-best candidate player's
distance margin is below `--ambiguity_margin_px`, the event is flagged
`ambiguous` rather than confidently attributed. Process a single video with
`--video_id <id>` instead of the whole directory.

### 7.2 Player-wise exposure aggregation

```bash
python scripts/header_analysis/analyze_exposure.py \
  --video_processing_dir outputs/video_processing \
  --output_dir outputs/exposure_analysis
```

Aggregates confirmed events per `(video_id, track_id)` — ByteTrack IDs are
only unique within one video, there's no cross-video player
re-identification. Computes: total exposure, first/second-half split
(`--half_split_sec`, default 45 min — mostly moot for short highlight
clips), inter-header interval stats, rolling 5/10/15-minute exposure
windows, and a review-priority tier (`--pr_*` thresholds).

### 7.3 Structured data export

```bash
python scripts/header_analysis/export_structured_data.py \
  --video_processing_dir outputs/video_processing \
  --exposure_dir outputs/exposure_analysis \
  --output_dir outputs \
  --model_path models/checkpoints/yolov8m/best.pt
```

Consolidates every video's per-video CSVs into the flat, LLM-ready
structure:
```
outputs/
├── detections.csv                every raw detection, all videos
├── player_tracks.csv             ByteTrack player tracks, all videos
├── ball_tracks.csv               Kalman-filtered ball trajectories, all videos
├── head_player_associations.csv  head-to-player associations, all videos
├── header_events.csv             CONFIRMED header events only
├── player_summary.csv / .json    per-player exposure summary — this is the
│                                  input to LLM reporting (§8)
└── analysis_metadata.json        pipeline configuration & run statistics
```

`--model_path` is recorded in `analysis_metadata.json` purely for
provenance (which detector produced this data) — it does not load the
model.

The website (§9) calls these same three scripts as subprocesses, scoped to
`webapp/jobs/<job_id>/`. The website contains **no duplicate
implementation** of detection/attribution/exposure logic — it only calls
these scripts and reads their CSV/JSON output.

---

## 8. LLM reporting

Code: `scripts/llm_reporting/generate_reports.py` — separate from all
computer-vision code. It never touches video or raw CV data, only the
already-validated structured JSON from §7.3.

```
CV Pipeline (§3–§7) → Validated Structured JSON → LLM → Report
```

The CV pipeline is responsible for detection, tracking, contact inference,
attribution, and exposure counting. The LLM is responsible only for
summarization, explanation, and report formatting — never for deciding
whether a header occurred.

### Prerequisite: Ollama running locally

```bash
ollama pull llama3.1:8b     # default model; only needed once
```
Runs entirely locally via [Ollama](https://ollama.com) — no external API
calls, no per-request cost.

### Run

```bash
python scripts/llm_reporting/generate_reports.py \
  --player_summary outputs/player_summary.json \
  --output_dir outputs/reports
```

- `--model`: Ollama model name, must already be pulled. Defaults to the
  `OLLAMA_MODEL` env var if set, else `llama3.1:8b`.
- `--temperature 0.2` (default): low, for factual/consistent reports.
- `--limit N`: only generate the first N players' reports (useful while
  testing).
- Ollama host defaults to `http://127.0.0.1:11434`; override with
  `OLLAMA_HOST` in `.env` if Ollama runs elsewhere.

Output:
```
outputs/reports/
├── <video_id>__track<track_id>.txt   one plain-text report per player
└── all_reports.json                  all reports, structured (used by evaluate_llm_reports.py)
```

The system prompt enforces: use only the supplied structured JSON (never
invent an event/count/timestamp); never invent symptoms/injuries/behavior;
never diagnose concussion/TBI; never convert pixel displacement into
force/acceleration/g-force; flag `ambiguous_events > 0` clearly; report
`review_priority` verbatim; always end with a research-not-diagnosis
disclaimer. `scripts/evaluation/evaluate_llm_reports.py` (§5.5)
automatically verifies every generated report against these rules.

The website (§9) calls this script as a subprocess after a video's
structured export completes, only if at least one player was found. No LLM
prompt logic is duplicated in the website code.

---

## 9. Website

Code: `webapp/` — separate from dataset, training, and evaluation code. It
*uses* the existing CV/LLM pipeline scripts (as subprocesses via
`pipeline_runner.py`); it never reimplements their logic.

### Architecture — there is no separate frontend build step

This is a server-rendered app: FastAPI (`webapp/app.py`) + Jinja2 templates
(`webapp/templates/`) + Bootstrap 5 and Chart.js loaded from a CDN
(`webapp/templates/base.html`) — no React/Vue/webpack, no `npm install`, no
separate frontend dev server. Starting the backend *is* starting the
website; there's nothing else to start.

Bootstrap/Chart.js are loaded from `cdn.jsdelivr.net` at page-load time in
the browser — the machine running the app needs no internet access, but a
browser viewing the dashboard does (for those two CDN assets).

### Prerequisites

- Dependencies installed (§2.3).
- A trained model at `models/checkpoints/yolov8m/best.pt` (or `MODEL_PATH`
  in `.env` pointing elsewhere) — §4.
- Ollama running locally with `llama3.1:8b` pulled, for the LLM report step
  — §8. The dashboard still works without it; only the per-player LLM
  report generation step of a job will fail.

### Start the website

```bash
cd webapp
uvicorn app:app --reload --port 8000
```
Open http://127.0.0.1:8000

Alternative (reads `WEBAPP_HOST`/`WEBAPP_PORT`/`WEBAPP_RELOAD` from `.env`,
defaults `127.0.0.1:8000`, no reload):
```bash
cd webapp
python app.py
```

### What happens on upload

1. Upload a `.mp4` on the Upload page → saved to `webapp/uploads/<job_id>.mp4`.
2. A background task runs, scoped to `webapp/jobs/<job_id>/`:
   `inference/process_video.py` → `header_analysis/detect_headers.py` →
   `header_analysis/analyze_exposure.py` →
   `header_analysis/export_structured_data.py` →
   `llm_reporting/generate_reports.py` (all under `scripts/` — see
   `webapp/pipeline_runner.py` for the exact subprocess calls).
3. Progress is tracked in `webapp/headtrack.db` (SQLite) and polled from the
   Processing Status page.
4. Every dashboard page reads directly from that job's `structured/` and
   `reports/` output files under `webapp/jobs/<job_id>/` — nothing is
   cached or mocked.
5. Ambiguous-event accept/reject decisions go into a separate SQLite table
   (`ambiguous_reviews`), keeping the pipeline's own CSVs immutable/reproducible.

### Pages

Dashboard · Upload · Processing Status · Detection Results · Player
Tracking · Ball Tracking · Header Events (video viewer with clickable event
markers) · Player Exposure Summary · Player Detail (timeline + LLM report) ·
Ambiguous Event Review · LLM Reports · Export (CSV/JSON/PDF).

### Data this repo already ships with

`webapp/jobs/`, `webapp/uploads/`, and `webapp/headtrack.db` are gitignored
— a fresh clone starts with an empty dashboard (no prior jobs) until you
upload a video.

---

## 10. Quick start — clone and run (already-trained model)

If `models/checkpoints/yolov8m/best.pt` already exists (either you copied
the full project, or you already ran §3–§4 yourself):

```bash
git clone <repo-url> && cd "HeadTrack AI"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # optional
cd webapp
uvicorn app:app --reload --port 8000
```
Open http://127.0.0.1:8000

If you cloned an **empty** repo with no trained model, run the full
pipeline first: §3 (dataset) → §4 (train) → then this section.

---

## 11. Full command reference

Every command below is exact and copy-pasteable, run from the project root
unless stated otherwise.

```bash
# --- 0. Setup (once) ---
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env                 # optional

# --- 1. Dataset development ---
python scripts/dataset/extract_frames.py --video_dir videos --output_dir extracted_frames --header_fps 10 --non_header_fps 2
python scripts/dataset/clean_frames.py --action report
python scripts/dataset/auto_annotate_players_heads.py --input_dir extracted_frames --output_dir auto_labels --player_model models/pretrained/yolo11x.pt --pose_model models/pretrained/yolo11x-pose.pt
python scripts/dataset/auto_annotate_ball.py --input_dir extracted_frames --labels_dir auto_labels --model models/pretrained/yolo11x.pt
python scripts/dataset/qc_auto_annotations.py --labels_dir auto_labels
python scripts/dataset/validate_annotations.py --source auto_labels
python scripts/dataset/split_dataset.py --images_dir extracted_frames --labels_dir auto_labels --output_dir dataset --train 0.70 --valid 0.15 --test 0.15

# --- 2. Model training ---
python scripts/training/train_yolo.py --data dataset/data.yaml --models yolov8s yolov8m --imgsz 960 --epochs 100 --output_dir models/checkpoints --pretrained_dir models/pretrained

# --- 3. Model evaluation ---
python scripts/evaluation/evaluate_models.py --data dataset/data.yaml --models_dir models/checkpoints --output_dir outputs/evaluation
python scripts/evaluation/evaluate_tracking.py --video_processing_dir outputs/video_processing
python scripts/evaluation/evaluate_header_detection.py --video_processing_dir outputs/video_processing
python scripts/evaluation/evaluate_attribution.py --write_template
python scripts/evaluation/evaluate_llm_reports.py --reports_json outputs/reports/all_reports.json --output_csv outputs/evaluation/llm_report_verification.csv
python scripts/evaluation/diagnose_missed_headers.py --video_glob "Header_*"
python scripts/evaluation/generate_thesis_summary.py

# --- 4. Full video processing / inference ---
python scripts/inference/process_video.py --video videos/Header/h1.mp4 --model models/checkpoints/yolov8m/best.pt --output_dir outputs/video_processing

# --- 5. Header detection & exposure analysis ---
python scripts/header_analysis/detect_headers.py --video_processing_dir outputs/video_processing
python scripts/header_analysis/analyze_exposure.py --video_processing_dir outputs/video_processing --output_dir outputs/exposure_analysis
python scripts/header_analysis/export_structured_data.py --video_processing_dir outputs/video_processing --exposure_dir outputs/exposure_analysis --output_dir outputs --model_path models/checkpoints/yolov8m/best.pt

# --- 6. LLM reporting ---
ollama pull llama3.1:8b     # once
python scripts/llm_reporting/generate_reports.py --player_summary outputs/player_summary.json --output_dir outputs/reports

# --- 7. Website ---
cd webapp && uvicorn app:app --reload --port 8000
# Open http://127.0.0.1:8000
```

Every script's exact flags and defaults for the code as it stands can also
be seen directly:
```bash
python <path-to-script>.py --help
```