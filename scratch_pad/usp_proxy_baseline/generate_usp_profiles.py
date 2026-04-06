#!/usr/bin/env python3
"""
Generate USP profiles for all episodes in our mirror datasets.

For each episode, extract all reference user turns and feed them to
Profile_Generator to produce a persona description paragraph.
The output JSONL is identical to the input except each episode gains
a `usp_profile` field in its metadata.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=4 /home/ccloud/miniconda3/bin/python \
        rebuttal_experiments/exp7_usp_baseline/generate_usp_profiles.py \
        --model rebuttal_experiments/exp7_usp_baseline/models/Profile_Generator \
        --datasets chatbot_arena_mirror clariq_mirror oasst1_mirror qulac_mirror \
        [--limit 5] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOTS = {
    "chatbot_arena_mirror": "scratch_pad/data/chatbot_arena/chatbot_arena_mirror.jsonl",
    "clariq_mirror": "scratch_pad/data/clariq/clariq_mirror.jsonl",
    "oasst1_mirror": "scratch_pad/data/oasst1/oasst1_mirror.jsonl",
    "qulac_mirror": "scratch_pad/data/qulac/qulac_mirror.jsonl",
}
OUTPUT_DIR = Path(__file__).parent / "data"


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def generate_profile(utterances: list[str], model, tokenizer) -> str:
    import torch

    system_prompt = "You are an expert in creating user profile descriptions based on dialogue analysis."
    instruction = "Analyze the user utterances marked by [User] to generate a comprehensive and descriptive user profile"
    user_prompt = "".join(f"[User]: {u}\n---\n" for u in utterances)

    formatted_msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n{user_prompt}"},
    ]
    input_text = tokenizer.apply_chat_template(
        formatted_msg, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(
        model.device
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def process_dataset(
    dataset_key: str,
    model,
    tokenizer,
    limit: int | None,
    dry_run: bool,
) -> None:
    input_path = REPO_ROOT / DATA_ROOTS[dataset_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{dataset_key}_usp.jsonl"

    # Resume support: load already-processed conversation_ids
    existing: dict[str, dict] = {}
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                ep = json.loads(line)
                cid = ep.get("conversation_id", ep.get("episode_id", ""))
                existing[cid] = ep
        print(f"  Resuming: {len(existing)} episodes already processed")

    episodes = []
    with input_path.open() as f:
        for line in f:
            ep = json.loads(line)
            episodes.append(ep)
    if limit:
        episodes = episodes[:limit]

    print(f"  Processing {len(episodes)} episodes for {dataset_key} ...")

    out_f = output_path.open("a")
    written = 0

    for i, ep in enumerate(episodes):
        cid = ep.get("conversation_id", ep.get("episode_id", str(i)))
        if cid in existing:
            continue  # already done

        # Extract reference user turns
        user_turns = [
            t["content"]
            for t in ep.get("turns", [])
            if t.get("role") == "user" and t.get("content", "").strip()
        ]
        if not user_turns:
            ep.setdefault("metadata", {})["usp_profile"] = ""
            out_f.write(json.dumps(ep) + "\n")
            written += 1
            continue

        if dry_run:
            print(f"    [{i}] {cid}: {len(user_turns)} user turns → (dry-run)")
            ep["task_description"] = "[DRY-RUN USP PROFILE]"
            ep.setdefault("metadata", {})["task_description"] = "[DRY-RUN USP PROFILE]"
        else:
            try:
                profile = generate_profile(user_turns, model, tokenizer)
                # Replace task_description with USP profile so MirrorBench injects it
                # as the proxy's task description via build_user_proxy_system_prompt()
                ep["task_description"] = profile
                ep.setdefault("metadata", {})["task_description"] = profile
                if i < 2:
                    print(f"    [{i}] {cid}: profile[:120] = {profile[:120]!r}")
            except Exception as exc:
                print(f"    [{i}] {cid}: ERROR — {exc}")
                ep["task_description"] = ep.get("task_description", "")
                ep.setdefault("metadata", {})["task_description"] = ep["task_description"]

        out_f.write(json.dumps(ep) + "\n")
        written += 1

        if written % 20 == 0:
            out_f.flush()
            print(f"    ... {written + len(existing)} / {len(episodes)} done")

    out_f.flush()
    out_f.close()
    print(f"  Done. Written {written} new episodes → {output_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATA_ROOTS.keys()),
        choices=list(DATA_ROOTS.keys()),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        print(f"Loading Profile_Generator from {args.model} ...")
        model, tokenizer = load_model(args.model)
        print("Model loaded.\n")
    else:
        model = tokenizer = None

    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        process_dataset(ds, model, tokenizer, args.limit, args.dry_run)

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
