"""Datasets package exposing built-in dataset implementations and loaders."""

from __future__ import annotations

# Import dataset registry module to trigger dataset registrations.
from mirrorbench.datasets import registry as _registry  # noqa: F401
from mirrorbench.datasets.artifacts import (
    ARTIFACT_MANIFEST_FILENAME,
    EPISODES_FILENAME,
    DatasetArtifact,
    compute_pipeline_hash,
)
from mirrorbench.datasets.base_dataset import BaseDataset, DatasetError

# Import built-in loaders so they register with the loader registry.
from mirrorbench.datasets.loaders import (
    DatasetLoaderBackend,
    DatasetLoaderError,
    available_loaders,
    create_loader,
    get_loader,
    huggingface as _huggingface_loader,  # noqa: F401
    jsonl as _jsonl_loader,  # noqa: F401
    register_loader,
)

# ruff: noqa: RUF022
__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "BaseDataset",
    "DatasetArtifact",
    "DatasetError",
    "DatasetLoaderBackend",
    "DatasetLoaderError",
    "EPISODES_FILENAME",
    "available_loaders",
    "compute_pipeline_hash",
    "create_loader",
    "get_loader",
    "register_loader",
]
