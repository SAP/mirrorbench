#!/usr/bin/env python3
"""Generate MirrorBench job configs and run them for selected combinations."""

from __future__ import annotations

import itertools
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to run this script. Install it with `uv add pyyaml`."
    ) from exc

from model_configs import (
    ASSISTANTS,
    DATASETS,
    JUDGE_DEFAULT_MAP,
    JUDGES,
    LEXICAL_METRIC_DEFAULTS,
    MODEL_CONFIGS,
    USER_PROXIES,
)

from mirrorbench.io.paths import Paths

CONFIG_ROOT = Path("configs")
DEFAULT_CONFIG_FILENAME = "job.yaml"

LEXICAL_DEFAULT_KEYS = list(LEXICAL_METRIC_DEFAULTS.keys())
JUDGE_DEFAULT_KEYS = list(JUDGE_DEFAULT_MAP.keys())


@dataclass(frozen=True)
class UserProxy:
    label: str
    definition: dict[str, Any]


@dataclass(frozen=True)
class Assistant:
    label: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Judge:
    label: str
    metric_params: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Dataset:
    key: str
    label: str
    params: dict[str, Any]
    task_driver_params: dict[str, Any] = field(default_factory=dict)
    lexical_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    judge_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_model(ref: Any) -> dict[str, Any]:
    if isinstance(ref, str):
        try:
            return deepcopy(MODEL_CONFIGS[ref])
        except KeyError as exc:
            raise KeyError(f"Unknown model label '{ref}'") from exc
    if isinstance(ref, dict):
        base_ref = ref.get("base")
        if base_ref:
            base = resolve_model(base_ref)
            overrides = ref.get("overrides", {})
            return deep_merge(base, overrides)
        return deepcopy(ref)
    if ref is None:
        return {}
    raise TypeError(f"Unsupported model reference type: {type(ref)!r}")


def build_user_proxy(spec: dict[str, Any]) -> UserProxy:
    label = spec["label"]
    model_details = resolve_model(spec.get("model"))
    request_params = deep_merge(
        model_details.get("request_params", {}), spec.get("request_params", {})
    )
    params = {
        "model_client": spec.get("model_client", model_details.get("client")),
        "client_params": deep_merge(
            model_details.get("client_params", {}), spec.get("client_params", {})
        ),
        "request_params": request_params,
    }
    # Add combine_system_and_history flag if present in model config
    if "combine_system_and_history" in model_details:
        params["combine_system_and_history"] = model_details["combine_system_and_history"]
    definition = {
        "name": spec.get("name", f"proxy:{label}"),
        "adapter": spec.get("adapter", "adapter:generic/llm"),
        "params": params,
    }
    return UserProxy(label=label, definition=definition)


def build_assistant(spec: dict[str, Any]) -> Assistant:
    label = spec["label"]
    model_details = resolve_model(spec.get("model"))
    params = {
        "assistant_model_client": spec.get("assistant_model_client", model_details.get("client")),
        "client_params": deep_merge(
            model_details.get("client_params", {}), spec.get("client_params", {})
        ),
        "request_params": deep_merge(
            model_details.get("request_params", {}), spec.get("request_params", {})
        ),
    }
    if "system_prompt" in spec:
        params["request_params"]["system_prompt"] = spec["system_prompt"]
    return Assistant(label=label, params=params)


def _build_metric_params(
    *,
    metric_key: str,
    base_model_ref: Any,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    base_model = resolve_model(base_model_ref)
    params: dict[str, Any] = {
        "judge_client_name": base_model.get("client"),
        "judge_params": deepcopy(base_model.get("client_params", {})),
    }
    if "request_params" in base_model:
        params["request_params"] = deepcopy(base_model["request_params"])

    defaults = JUDGE_DEFAULT_MAP[metric_key]["params"]
    for key, value in defaults.items():
        params[key] = deepcopy(value)

    if overrides:
        params = deep_merge(params, overrides)

    return params


def build_judge(spec: dict[str, Any]) -> Judge:
    label = spec["label"]
    base_model = spec.get("model")
    metric_models = spec.get("metric_models", {})
    metric_overrides = spec.get("metric_overrides", {})

    metric_params: dict[str, dict[str, Any]] = {}
    for metric_key in JUDGE_DEFAULT_KEYS:
        model_ref = metric_models.get(metric_key, base_model)
        overrides = metric_overrides.get(metric_key, {})
        metric_params[metric_key] = _build_metric_params(
            metric_key=metric_key,
            base_model_ref=model_ref,
            overrides=overrides,
        )

    return Judge(label=label, metric_params=metric_params)


def build_dataset(spec: dict[str, Any]) -> Dataset:
    lexical_overrides: dict[str, dict[str, Any]] = {}
    for metric in spec.get("lexical_metrics", []) or []:
        key = metric.get("name") or metric.get("key")
        if not key:
            raise KeyError("Lexical metric override requires a 'name' or 'key'")
        lexical_overrides[key] = deepcopy(metric)

    judge_overrides: dict[str, dict[str, Any]] = {}
    for metric in spec.get("judge_metrics", []) or []:
        key = metric.get("key") or metric.get("name")
        if not key:
            raise KeyError("Judge metric override requires a 'key' or 'name'")
        judge_overrides[key] = deepcopy(metric)

    return Dataset(
        key=spec["key"],
        label=spec["label"],
        params=deepcopy(spec.get("params", {})),
        task_driver_params=deepcopy(spec.get("task_driver_params") or {}),
        lexical_overrides=lexical_overrides,
        judge_overrides=judge_overrides,
    )


def resolve_configs() -> tuple[list[Dataset], list[UserProxy], list[Assistant], list[Judge]]:
    datasets = [build_dataset(item) for item in DATASETS]
    user_proxies = [build_user_proxy(item) for item in USER_PROXIES]
    assistants = [build_assistant(item) for item in ASSISTANTS]
    judges = [build_judge(item) for item in JUDGES]
    return datasets, user_proxies, assistants, judges


def ensure_yaml_dump(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def build_lexical_metrics(dataset: Dataset) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    remaining_overrides = dict(dataset.lexical_overrides)

    for key in LEXICAL_DEFAULT_KEYS:
        default = LEXICAL_METRIC_DEFAULTS[key]
        metric = deepcopy(default)
        if key in remaining_overrides:
            metric = deep_merge(metric, remaining_overrides.pop(key))
        metrics.append(metric)

    # Append any extra lexical metrics specified by the dataset
    metrics.extend(remaining_overrides.values())
    return metrics


def build_judge_metrics(dataset: Dataset, judge: Judge) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    remaining_overrides = dict(dataset.judge_overrides)

    for key in JUDGE_DEFAULT_KEYS:
        default = JUDGE_DEFAULT_MAP[key]
        metric = {
            "name": default["name"],
            "label": default["label"],
            "params": deepcopy(judge.metric_params[key]),
        }
        override = remaining_overrides.pop(key, None)
        if override:
            if "name" in override:
                metric["name"] = override["name"]
            if "label" in override:
                metric["label"] = override["label"]
            if "params" in override:
                metric["params"] = deep_merge(metric["params"], override["params"])
        metrics.append(metric)

    metrics.extend(remaining_overrides.values())
    return metrics


def build_job_config(
    *,
    dataset: Dataset,
    user_proxy: UserProxy,
    assistant: Assistant,
    judge: Judge,
) -> dict[str, Any]:
    run_name = f"{dataset.key}-{user_proxy.label}-{assistant.label}-{judge.label}"

    metrics = build_lexical_metrics(dataset)
    metrics.extend(build_judge_metrics(dataset, judge))

    dataset_entry = {
        "name": f"dataset:jsonl/{dataset.key}",
        "split": "default",
        "label": dataset.label,
        "params": dataset.params,
    }

    config = {
        "run": {
            "name": run_name,
            "seeds": [0],
            "engine": "sync",
            "max_concurrency": 1,
            "timeout_seconds": 600,
            "cache": {"enabled": True},
            "observability": {"log_json": False, "log_level": "INFO"},
        },
        "user_proxies": [user_proxy.definition],
        "datasets": [dataset_entry],
        "metrics": metrics,
        "scorecards": [
            {
                "name": "human_likeness",
                "weights": {key: 1.0 for key in JUDGE_DEFAULT_KEYS},
            },
            {
                "name": "lexical_quality",
                "weights": {key: 1.0 for key in LEXICAL_DEFAULT_KEYS},
            },
        ],
        "task_drivers": {
            dataset_entry["name"]: {
                "driver": "task:mirror/conversation",
                "params": deep_merge(assistant.params, dataset.task_driver_params),
            }
        },
    }

    return config, run_name


def write_config(
    *,
    dataset: Dataset,
    user_proxy: UserProxy,
    assistant: Assistant,
    judge: Judge,
    config: dict[str, Any],
) -> Path:
    target_dir = CONFIG_ROOT / dataset.key / user_proxy.label / assistant.label / judge.label
    target_dir.mkdir(parents=True, exist_ok=True)
    config_path = target_dir / DEFAULT_CONFIG_FILENAME
    config_path.write_text(ensure_yaml_dump(config))
    return config_path


def run_job(config_path: Path, run_name: str) -> None:
    run_dir = Paths.default().runs_dir() / run_name
    cmd = [
        sys.executable,
        "-m",
        "mirrorbench.cli",
        "run",
        "-c",
        str(config_path),
        "--run-id",
        str(run_name),
    ]
    if run_dir.exists():
        cmd.append("--resume")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    datasets, user_proxies, assistants, judges = resolve_configs()
    CONFIG_ROOT.mkdir(exist_ok=True)

    combinations = list(itertools.product(datasets, user_proxies, assistants, judges))

    for dataset, user_proxy, assistant, judge in combinations:
        config, run_name = build_job_config(
            dataset=dataset,
            user_proxy=user_proxy,
            assistant=assistant,
            judge=judge,
        )
        config_path = write_config(
            dataset=dataset,
            user_proxy=user_proxy,
            assistant=assistant,
            judge=judge,
            config=config,
        )

        print(f"[mirrorbench-script] running job: {config_path}")
        run_job(config_path, run_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        sys.exit(130)
