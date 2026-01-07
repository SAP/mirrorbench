from __future__ import annotations

from mirrorbench.core.models.run import MetricValue
from mirrorbench.metrics.base import default_metric_aggregate
from mirrorbench.metrics.judge.gteval import METRIC_NAME, GTEvalMetric
from mirrorbench.metrics.util.bootstrap import BootstrapConfig, bootstrap_mean
from mirrorbench.metrics.util.stats import mean_stdev_ci

EXPECTED_MEAN = 3.0
T_CRIT_DF4 = 2.776
CI_TOLERANCE = 1e-6
BOOTSTRAP_MIN_VALUE = 0.1
BOOTSTRAP_MAX_VALUE = 0.9


def test_mean_stdev_ci_uses_t_distribution():
    values = [1, 2, 3, 4, 5]
    mean_value, stdev_value, ci_half = mean_stdev_ci(values)
    assert round(mean_value, 5) == EXPECTED_MEAN
    assert stdev_value > 0.0
    # Known 95% CI half-width for n=5 (df=4, t=2.776)
    expected_half_width = T_CRIT_DF4 * stdev_value / (len(values) ** 0.5)
    assert abs(ci_half - expected_half_width) < CI_TOLERANCE


def test_bootstrap_mean_returns_deterministic_results():
    config = BootstrapConfig(iterations=100, confidence=0.90, random_seed=42)
    result = bootstrap_mean([0.1, 0.5, 0.9], config)
    assert BOOTSTRAP_MIN_VALUE <= result.mean <= BOOTSTRAP_MAX_VALUE
    assert result.standard_deviation >= 0.0
    assert result.ci_lower <= result.mean <= result.ci_upper


def test_default_metric_aggregate_populates_confidence_interval():
    metric_values = [
        MetricValue(metric_name="metric:test", values=[0.1]),
        MetricValue(metric_name="metric:test", values=[0.5]),
        MetricValue(metric_name="metric:test", values=[0.9]),
    ]
    aggregate = default_metric_aggregate(metric_values, metric_name="metric:test")
    assert aggregate.confidence_interval is not None
    assert aggregate.confidence_interval >= 0.0


def test_gteval_metric_aggregate_uses_bootstrap():
    metric = GTEvalMetric(
        judge_client_name="client:test/chat",
        judge_params={},
        bootstrap={"iterations": 50, "confidence": 0.90, "random_seed": 123},
    )
    metric_values = [
        MetricValue(metric_name=METRIC_NAME, values=[0.2]),
        MetricValue(metric_name=METRIC_NAME, values=[0.5]),
        MetricValue(metric_name=METRIC_NAME, values=[0.7]),
    ]
    aggregate = metric.aggregate(metric_values)
    assert aggregate.confidence_interval is not None
    assert aggregate.extras["bootstrap"] is not None
