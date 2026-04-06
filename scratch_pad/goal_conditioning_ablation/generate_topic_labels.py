#!/usr/bin/env python3
"""
Generate short topic labels for each episode in a JSONL dataset.

Examples of output:
  "React scrollIntoView conditional triggering"
  "LLM model comparison and differentiation"
  "election campaign propaganda writing"
  "Ion Stoica biography"

Usage:
    # Test on 3 samples first
    python generate_topic_labels.py --dataset chatbot_arena --dry-run --limit 3

    # Generate for all datasets
    python generate_topic_labels.py --dataset chatbot_arena
    python generate_topic_labels.py --dataset clariq
    python generate_topic_labels.py --dataset oasst1
    python generate_topic_labels.py --dataset qulac
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

TOPIC_SYSTEM_PROMPT = (
    "You are a topic labeler. Given a conversation, output ONLY a short noun phrase "
    "(3-8 words) describing the subject matter. "
    "Do NOT describe the user's intent, goal, persona, or tone. "
    "Do NOT use words like 'user', 'asking', 'seeking', 'help', 'question'. "
    "Output the phrase only — no punctuation, no explanation."
)

TOPIC_USER_TEMPLATE = """\
Conversation:
{conversation}

Short topic label (3-8 words, subject matter only):"""

DATASETS = {
    "chatbot_arena": REPO_ROOT / "scratch_pad/data/chatbot_arena/chatbot_arena_mirror.jsonl",
    "clariq": REPO_ROOT / "scratch_pad/data/clariq/clariq_mirror.jsonl",
    "oasst1": REPO_ROOT / "scratch_pad/data/oasst1/oasst1_mirror.jsonl",
    "qulac": REPO_ROOT / "scratch_pad/data/qulac/qulac_mirror.jsonl",
}

OUT_DIR = Path(__file__).parent / "data"


def format_conversation(turns: list[dict], max_turns: int = 4) -> str:
    lines = []
    for t in turns[:max_turns]:
        role = t.get("role", "?").capitalize()
        content = t.get("content", "").strip()[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def make_llm_client():
    from model_configs import MODEL_CONFIGS
    from mirrorbench.core.registry import registry

    cfg = MODEL_CONFIGS["gpt-4o"]
    factory = registry.factory("model_clients", cfg["client"])
    return factory(**cfg["client_params"])


def generate_topic(client, turns: list[dict]) -> str:
    from mirrorbench.core.models.messages import Message, Role

    conversation_text = format_conversation(turns)
    messages = [
        Message(role=Role.SYSTEM, content=TOPIC_SYSTEM_PROMPT),
        Message(role=Role.USER, content=TOPIC_USER_TEMPLATE.format(conversation=conversation_text)),
    ]
    response = client.invoke(messages=messages, temperature=0)
    return response.message.content.strip()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(records)} records -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print generated topics without writing output file")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N records (for testing)")
    args = parser.parse_args()

    src_path = DATASETS[args.dataset]
    if not src_path.exists():
        print(f"ERROR: {src_path} not found")
        sys.exit(1)

    records = load_jsonl(src_path)
    if args.limit:
        records = records[:args.limit]

    out_path = OUT_DIR / f"{args.dataset}_mirror_goal_topic.jsonl"

    # Resume: load already-processed records from existing output file
    already_done: dict[str, dict] = {}
    if not args.dry_run and out_path.exists():
        for r in load_jsonl(out_path):
            already_done[r["conversation_id"]] = r
        print(f"Resuming: {len(already_done)} records already processed, skipping.")

    print(f"Loaded {len(records)} records from {src_path.name}")
    print(f"Generating topic labels using GPT-4o...\n")

    client = make_llm_client()
    out_records = []
    skipped = 0
    failed = 0

    for i, rec in enumerate(records):
        conv_id = rec.get("conversation_id", "?")

        # Resume: reuse already-processed record
        if conv_id in already_done:
            out_records.append(already_done[conv_id])
            skipped += 1
            continue

        turns = rec.get("turns", [])
        full_task_desc = rec.get("task_description", "")

        try:
            topic = generate_topic(client, turns)
            status = "OK"
        except Exception as e:
            # Content filter or other API error — skip with empty topic
            topic = ""
            status = f"FAILED ({type(e).__name__})"
            failed += 1

        print(f"[{i+1}/{len(records)}] {conv_id}  [{status}]")
        print(f"  full task_description : {full_task_desc[:120]}")
        print(f"  generated topic       : {topic if topic else '[skipped]'}")
        print()

        r = dict(rec)
        r["task_description"] = topic
        out_records.append(r)

        # Write incrementally every 10 records so progress is saved on crash
        if not args.dry_run and (i + 1) % 10 == 0:
            write_jsonl(out_records, out_path)

    print(f"\nSummary: {len(out_records)} total, {skipped} resumed, {failed} failed/skipped.")

    if args.dry_run:
        print("[dry-run] Not writing output file.")
        return

    # Filter out records with empty topic (content-filtered episodes)
    valid = [r for r in out_records if r.get("task_description", "").strip()]
    dropped = len(out_records) - len(valid)
    if dropped:
        print(f"Dropping {dropped} records with empty topic (content filter failures).")

    write_jsonl(valid, out_path)
    print(f"Done. Topic-only JSONL written to: {out_path}")


if __name__ == "__main__":
    main()
