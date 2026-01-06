"""Loader backend that proxies the Hugging Face `datasets` library."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable, Iterator, Mapping
from types import ModuleType
from typing import Any

from mirrorbench.core.constants import HF_TOKEN_ENV
from mirrorbench.datasets.loaders import register_loader
from mirrorbench.datasets.loaders.base import DatasetLoaderBackend, DatasetLoaderError


def _load_datasets_module() -> ModuleType:
    try:
        return importlib.import_module("datasets")
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "Install the 'datasets' package to use the Hugging Face loader"
        raise DatasetLoaderError(msg) from exc


@register_loader("huggingface")
class HuggingFaceLoader(DatasetLoaderBackend):
    """Load records from the Hugging Face Hub via the `datasets` library."""

    def load_split(
        self,
        *,
        split: str,
        limit: int | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        datasets_module = _load_datasets_module()
        dataset_name = self.params.get("dataset_name") or self.params.get("name")
        if not dataset_name:
            msg = "Hugging Face loader requires 'dataset_name' parameter"
            raise DatasetLoaderError(msg)
        subset = self.params.get("subset")
        revision = self.params.get("revision")
        auth_token = self.params.get("token") or os.getenv(HF_TOKEN_ENV)
        streaming = bool(self.params.get("streaming", False))
        trust_remote_code = bool(self.params.get("trust_remote_code", False))

        load_kwargs: dict[str, Any] = {
            "path": dataset_name,
            "split": split,
            "streaming": streaming,
            "trust_remote_code": trust_remote_code,
        }
        if subset:
            load_kwargs["name"] = subset
        if revision:
            load_kwargs["revision"] = revision
        if auth_token:
            load_kwargs["token"] = auth_token

        try:
            dataset = datasets_module.load_dataset(**load_kwargs)
        except Exception as exc:  # pragma: no cover - pass-through for hf errors
            msg = f"Failed to load Hugging Face dataset '{dataset_name}' with error: {exc}"
            raise DatasetLoaderError(msg) from exc

        def iterate() -> Iterator[Mapping[str, Any]]:
            for index, item in enumerate(dataset):
                if limit is not None and index >= limit:
                    break
                yield dict(item)

        return iterate()


__all__ = ["HuggingFaceLoader"]
