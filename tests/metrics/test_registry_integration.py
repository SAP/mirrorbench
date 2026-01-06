"""Integration tests verifying metrics register correctly with the global registry."""

from __future__ import annotations

import pytest

import mirrorbench.metrics  # noqa: F401 - Triggers auto-registration
from mirrorbench.core.constants import REGISTRY_GROUP_METRICS
from mirrorbench.core.models.messages import Role
from mirrorbench.core.registry import registry

LEXICAL_METRIC = "metric:lexical/ttr"
JUDGE_METRIC = "metric:judge/gteval"


def test_ttr_metric_registered() -> None:
    """Verify TypeTokenRatioMetric is registered in the global registry."""
    entry = registry.get(REGISTRY_GROUP_METRICS, LEXICAL_METRIC)

    assert entry is not None
    assert entry.name == LEXICAL_METRIC
    assert entry.metadata is not None
    assert entry.metadata.name == LEXICAL_METRIC
    assert entry.metadata.needs_references is False
    assert entry.metadata.needs_judge is False
    assert entry.metadata.category == "lexical_diversity"


def test_gteval_metric_registered() -> None:
    """Verify GTEvalMetric is registered in the global registry."""
    entry = registry.get(REGISTRY_GROUP_METRICS, JUDGE_METRIC)

    assert entry is not None
    assert entry.name == JUDGE_METRIC
    assert entry.metadata is not None
    assert entry.metadata.name == JUDGE_METRIC
    assert entry.metadata.needs_references is True
    assert entry.metadata.needs_judge is True
    assert entry.metadata.category == "human_likeness"


def test_ttr_metric_factory_instantiation() -> None:
    """Verify TypeTokenRatioMetric can be instantiated via registry factory."""
    factory = registry.factory(REGISTRY_GROUP_METRICS, LEXICAL_METRIC)
    metric = factory(min_tokens=10, tokenizer_model="gpt-4", target_role=Role.ASSISTANT)

    assert metric is not None
    assert hasattr(metric, "evaluate")
    assert hasattr(metric, "aggregate")
    assert metric.min_tokens == 10  # noqa: PLR2004
    assert metric.tokenizer_model == "gpt-4"
    assert metric.target_role == Role.ASSISTANT


def test_gteval_metric_factory_instantiation() -> None:
    """Verify GTEvalMetric can be instantiated via registry factory."""
    factory = registry.factory(REGISTRY_GROUP_METRICS, JUDGE_METRIC)
    metric = factory(
        judge_client_name="test:fake-client",
        rubric_version="test-v1",
    )

    assert metric is not None
    assert hasattr(metric, "evaluate")
    assert hasattr(metric, "aggregate")
    assert metric.rubric_version == "test-v1"
    assert metric.judge_client_name == "test:fake-client"


def test_list_all_metrics() -> None:
    """Verify we can list all registered metrics."""
    entries = registry.list_entries(REGISTRY_GROUP_METRICS)

    metric_names = {entry.name for entry in entries}

    # At minimum, we should have our two built-in metrics
    assert LEXICAL_METRIC in metric_names
    assert JUDGE_METRIC in metric_names


@pytest.mark.parametrize(
    "metric_name,expected_needs_judge",
    [
        (LEXICAL_METRIC, False),
        (JUDGE_METRIC, True),
    ],
)
def test_metric_metadata_judge_requirement(
    metric_name: str,
    expected_needs_judge: bool,
) -> None:
    """Verify metrics correctly declare judge requirements in metadata."""
    entry = registry.get(REGISTRY_GROUP_METRICS, metric_name)
    assert entry.metadata.needs_judge == expected_needs_judge


@pytest.mark.parametrize(
    "metric_name,expected_needs_refs",
    [
        (LEXICAL_METRIC, False),
        (JUDGE_METRIC, True),
    ],
)
def test_metric_metadata_reference_requirement(
    metric_name: str,
    expected_needs_refs: bool,
) -> None:
    """Verify metrics correctly declare reference requirements in metadata."""
    entry = registry.get(REGISTRY_GROUP_METRICS, metric_name)
    assert entry.metadata.needs_references == expected_needs_refs


def test_metrics_have_info_attribute() -> None:
    """Verify all registered metrics expose a ClassVar info attribute."""
    entries = registry.list_entries(REGISTRY_GROUP_METRICS)

    for entry in entries:
        factory = registry.factory(REGISTRY_GROUP_METRICS, entry.name)
        metric_cls = factory

        # Verify the class has an 'info' attribute (ClassVar metadata)
        assert hasattr(metric_cls, "info"), f"Metric {entry.name} missing 'info' ClassVar"
        info = metric_cls.info
        assert info.name == entry.name
