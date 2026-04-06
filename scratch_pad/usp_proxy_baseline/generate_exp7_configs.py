#!/usr/bin/env python3
"""
Generate YAML configs for Exp7: USP Baseline.

Runs the 3 main proxies (gpt-4o, claude-4-sonnet, gemini-2.5-pro) over
the USP-profiled JSONL files (where task_description = USP-generated profile).
This directly tests whether USP-style profile conditioning changes proxy quality
vs. the standard LLM-synthesized goal descriptions.

Run from repo root:
    python rebuttal_experiments/exp7_usp_baseline/generate_exp7_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from model_configs import MODEL_CONFIGS  # noqa: E402

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("PyYAML is required. Install with: uv add pyyaml")

DATA_DIR = Path("rebuttal_experiments/exp7_usp_baseline/data")

USP_MODEL_PATH = "rebuttal_experiments/exp7_usp_baseline/models/USP"


def _usp_proxy_block() -> dict:
    return {
        "name": "proxy:usp/llama",
        "adapter": "adapter:usp/local",
        "params": {
            "model_path": USP_MODEL_PATH,
            "max_new_tokens": 256,
            "device": "cuda:0",
        },
    }
EXP7_DATASETS = {
    "chatbot_arena_mirror": str(DATA_DIR / "chatbot_arena_mirror_usp.jsonl"),
    "clariq_mirror": str(DATA_DIR / "clariq_mirror_usp.jsonl"),
    "oasst1_mirror": str(DATA_DIR / "oasst1_mirror_usp.jsonl"),
    "qulac_mirror": str(DATA_DIR / "qulac_mirror_usp.jsonl"),
}


def _user_proxy_block(model_key: str) -> dict:
    cfg = MODEL_CONFIGS[model_key]
    params: dict = {
        "model_client": cfg["client"],
        "client_params": cfg["client_params"],
        "request_params": {},
    }
    if cfg.get("combine_system_and_history"):
        params["combine_system_and_history"] = True
    return {
        "name": f"proxy:usp/{model_key}",
        "adapter": "adapter:generic/llm",
        "params": params,
    }


def _judge_metrics_block(judge_key: str = "claude-4-sonnet") -> list[dict]:
    judge_cfg = MODEL_CONFIGS[judge_key]
    return [
        {
            "name": "metric:judge/gteval",
            "label": "GTEval Realism",
            "params": {
                "judge_client_name": judge_cfg["client"],
                "judge_params": judge_cfg["client_params"],
                "num_judge_samples": 1,
                "compute_controls": True,
            },
        },
        {
            "name": "metric:judge/pi_pairwise",
            "label": "Pairwise Indistinguishability",
            "params": {
                "judge_client_name": judge_cfg["client"],
                "judge_params": judge_cfg["client_params"],
                "num_judge_samples": 3,
                "compute_controls": True,
            },
        },
        {
            "name": "metric:judge/rubric_and_reason",
            "label": "Rubric & Reason",
            "params": {
                "judge_client_name": judge_cfg["client"],
                "judge_params": judge_cfg["client_params"],
                "num_judge_samples": 2,
                "compute_controls": True,
            },
        },
    ]


def _dataset_block(dataset_key: str, jsonl_path: str, max_examples: int = 200) -> dict:
    return {
        "name": f"dataset:jsonl/{dataset_key}",
        "split": "default",
        "label": dataset_key.replace("_", " ").title(),
        "params": {"path": jsonl_path, "max_examples": max_examples},
    }


def _assistant_task_driver(dataset_key: str, assistant_key: str = "gpt-4o") -> dict:
    asst_cfg = MODEL_CONFIGS[assistant_key]
    return {
        f"dataset:jsonl/{dataset_key}": {
            "driver": "task:mirror/conversation",
            "params": {
                "assistant_model_client": asst_cfg["client"],
                "client_params": asst_cfg["client_params"],
                "request_params": {},
            },
        }
    }


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  Written: {path.relative_to(REPO_ROOT)}")


def main() -> None:
    out_root = REPO_ROOT / "rebuttal_experiments/exp7_usp_baseline/configs"
    print("\n=== Exp7: USP Baseline ===")

    for ds_key, jsonl_path in EXP7_DATASETS.items():
        run_name = f"exp7_usp_{ds_key}"
        job = {
            "run": {
                "name": run_name,
                "seeds": [0],
                "engine": "sync",       # local model — no async concurrency
                "max_concurrency": 1,
                "timeout_seconds": 86400,
                "cache": {"enabled": True},
                "observability": {"log_json": False, "log_level": "INFO"},
            },
            "user_proxies": [_usp_proxy_block()],
            "datasets": [_dataset_block(ds_key, jsonl_path)],
            "metrics": _judge_metrics_block(),
            "scorecards": [
                {
                    "name": "human_likeness",
                    "weights": {
                        "metric:judge/gteval": 1.0,
                        "metric:judge/pi_pairwise": 1.0,
                        "metric:judge/rubric_and_reason": 1.0,
                    },
                }
            ],
            "task_drivers": _assistant_task_driver(ds_key),
        }
        out_path = out_root / f"{ds_key}.yaml"
        write_yaml(out_path, job)

    print("\nAll configs written.")


if __name__ == "__main__":
    main()
