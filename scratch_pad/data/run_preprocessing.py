from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Type

from chatbot_arena import ChatbotArenaPreprocessor
from clariq import ClariQPreprocessor
from oasst1 import OASST1Preprocessor
from qulac import QulacPreprocessor

from common.base import BaseDatasetPreprocessor
from common.task_description import TaskDescriptionBuilder

DATASET_REGISTRY: Dict[str, Type[BaseDatasetPreprocessor]] = {
    "oasst1": OASST1Preprocessor,
    "chatbot_arena": ChatbotArenaPreprocessor,
    "qulac": QulacPreprocessor,
    "clariq": ClariQPreprocessor,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MirrorBench dataset preprocessing runner")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=DATASET_REGISTRY.keys(),
        default=list(DATASET_REGISTRY.keys()),
        help="Datasets to preprocess (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of stratified samples per dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for sampling",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where dataset outputs will be written",
    )
    parser.add_argument(
        "--task-desc-client",
        default="client:langchain/chat",
        help="Registry name of the model client used for task description generation",
    )
    parser.add_argument(
        "--task-desc-model-import",
        default="langchain_openai.AzureChatOpenAI",
        help="Dotted import path for the underlying chat model (passed to the client)",
    )
    parser.add_argument(
        "--task-desc-model-kwargs",
        default='{"azure_deployment": "gpt-4o","api_version": "2025-01-01-preview"}',
        help="JSON-encoded keyword arguments forwarded to the chat model constructor",
    )
    parser.add_argument(
        "--task-desc-invoke-kwargs",
        default=None,
        help="JSON-encoded keyword arguments supplied per LLM invocation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing outputs instead of starting from scratch",
    )
    return parser


def run_preprocessors(
    datasets: List[str],
    max_samples: int,
    seed: int,
    output_root: Path,
    task_builder: TaskDescriptionBuilder,
    *,
    resume: bool,
) -> List[Path]:
    outputs: List[Path] = []
    for dataset_name in datasets:
        preprocessor_cls = DATASET_REGISTRY[dataset_name]
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        preprocessor = preprocessor_cls(
            output_dir=str(dataset_dir),
            max_samples=max_samples,
            seed=seed,
            task_builder=task_builder,
        )
        output_path = preprocessor.run(resume=resume)
        outputs.append(output_path)
    return outputs


def _parse_json_mapping(payload: str | None, *, label: str) -> Mapping[str, object] | None:
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - user input validation
        raise ValueError(f"Invalid JSON supplied for {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object for {label}")
    return parsed


def _create_task_builder(args: argparse.Namespace) -> TaskDescriptionBuilder:
    model_kwargs = _parse_json_mapping(args.task_desc_model_kwargs, label="--task-desc-model-kwargs")
    invoke_kwargs = _parse_json_mapping(args.task_desc_invoke_kwargs, label="--task-desc-invoke-kwargs")

    return TaskDescriptionBuilder.create_default(
        client_name=args.task_desc_client,
        model_import=args.task_desc_model_import,
        model_kwargs=model_kwargs,
        invoke_kwargs=invoke_kwargs,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        task_builder = _create_task_builder(args)
    except ValueError as exc:
        parser.error(str(exc))

    outputs = run_preprocessors(
        datasets=list(args.datasets),
        max_samples=args.max_samples,
        seed=args.seed,
        output_root=args.output_root,
        task_builder=task_builder,
        resume=args.resume,
    )

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
