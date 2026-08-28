#!/usr/bin/env python3
"""
LLM-Based Player Exposure Report Generation (Phase 7)

The LLM never inspects video or raw CV data - it only receives the already
validated, structured per-player summary produced by Phase 6
(outputs/player_summary.json) and turns it into a natural-language report.

  CV Pipeline (Phases 1-6) -> Validated Structured JSON -> LLM -> Report

The CV pipeline is responsible for detection, tracking, contact inference,
attribution, and exposure counting. The LLM is responsible only for
summarization, explanation, and report formatting (7.1) - never for
deciding whether a header occurred.

Runs entirely locally via Ollama (no external API calls, no per-request
cost) - see the --model flag to swap the local model.

Output:
  outputs/reports/<video_id>__track<track_id>.txt   (one report per player)
  outputs/reports/all_reports.json                  (all reports, structured)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import ollama
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

SYSTEM_PROMPT = """You are a research reporting assistant that writes player heading-exposure \
reports from video-derived soccer analytics data, for research and coaching-review purposes.

You will be given a single JSON object describing one player's confirmed header events in one \
video, already computed by a computer-vision pipeline (detection, tracking, contact inference, \
player attribution, exposure counting). You did not watch the video and have no other information.

Follow these rules exactly:
- Use only the supplied structured data. Do not add any header event, count, or timestamp that \
is not present in the JSON.
- Do not invent player symptoms, injuries, or behavior.
- Do not diagnose concussion or traumatic brain injury, and do not suggest one may have occurred.
- Do not convert pixel displacement into physical acceleration, force, or g-force unless those \
exact values are explicitly provided in the JSON (they will not be).
- If ambiguous_events is greater than zero, clearly mention that some events had ambiguous \
player attribution and should be manually reviewed.
- Report the review_priority value exactly as supplied - do not upgrade, downgrade, or rephrase \
it into a different severity word.
- End every report with a short disclaimer that this is a research/review summary, not a medical \
diagnosis, and does not determine whether a concussion or brain injury occurred.

Write the report in this structure: a title line, a Match/Player identification block, the \
header counts (total, first half, second half), a sentence about the shortest interval and any \
notable clustering (headers_last_5_min/10_min/15_min), a note on ambiguous events, a "Review \
Priority:" line, and the closing disclaimer paragraph. Keep it concise and factual."""


def build_llm_input(record: dict) -> dict:
    """Maps our internal player_summary.json field names to the spec's
    7.2 schema for what the LLM actually receives."""
    return {
        "player_id": record["track_id"],
        "match_id": record["video_id"],
        "total_headers": record["total_headers"],
        "first_half": record["first_half_headers"],
        "second_half": record["second_half_headers"],
        "shortest_interval_sec": record["shortest_interval_sec"],
        "headers_last_5_min": record.get("headers_last_5_min", ""),
        "headers_last_10_min": record.get("headers_last_10_min", ""),
        "headers_last_15_min": record["headers_last_15_min"],
        "ambiguous_events": record["ambiguous_events"],
        "review_priority": record["review_priority"],
    }


def generate_report(client_model: str, llm_input: dict, temperature: float) -> str:
    response = ollama.chat(
        model=client_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Here is the structured player exposure data:\n\n"
                f"{json.dumps(llm_input, indent=2)}\n\n"
                "Generate the player header exposure report now."
            )},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description="Generate natural-language player exposure reports via a local LLM (Phase 7).")
    parser.add_argument("--player_summary", type=str, default="outputs/player_summary.json")
    parser.add_argument("--output_dir", type=str, default="outputs/reports")
    parser.add_argument("--model", type=str, default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"), help="Ollama model name (must already be pulled: `ollama pull <model>`); defaults to $OLLAMA_MODEL")
    parser.add_argument("--temperature", type=float, default=0.2, help="Low temperature for factual, consistent reports")
    parser.add_argument("--limit", type=int, default=None, help="Only generate reports for the first N players (for testing)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    summary_path = project_root / args.player_summary
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        print(f"Error: '{summary_path}' does not exist. Run scripts/header_analysis/export_structured_data.py first.")
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.limit:
        records = records[:args.limit]

    if not records:
        print("No player records found.")
        sys.exit(0)

    print(f"Generating {len(records)} report(s) with local model '{args.model}'...")

    all_reports = []
    for i, record in enumerate(records, 1):
        llm_input = build_llm_input(record)
        report_text = generate_report(args.model, llm_input, args.temperature)

        video_id, track_id = record["video_id"], record["track_id"]
        out_name = f"{video_id}__track{track_id}.txt"
        (output_dir / out_name).write_text(report_text, encoding="utf-8")

        all_reports.append({"video_id": video_id, "track_id": track_id, "llm_input": llm_input, "report": report_text})
        print(f"  [{i}/{len(records)}] {out_name}")

    with open(output_dir / "all_reports.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\nDone. {len(all_reports)} report(s) written to '{output_dir}'.")


if __name__ == "__main__":
    main()
