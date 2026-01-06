from __future__ import annotations

import textwrap

import pytest

from mirrorbench.core.config import (
    DEFAULT_CACHE_TTL_SECONDS,
    JobConfig,
    ScorecardConfig,
    load_job_config,
)
from mirrorbench.core.constants import DEFAULT_RETRY_BACKOFF_SECONDS

ADAPTER_NAME = "adapter:test/echo"
DATASET_NAME = "dataset:test/static"
METRIC_A = "metric:test/distinct_n"
METRIC_B = "metric:test/length_diff"


def test_job_config_valid(tmp_path):
    config_text = textwrap.dedent(
        """
        run:
          name: sample
          seeds: [0, 1]
        user_proxies:
          - name: echo-proxy
            adapter: adapter:test/echo
        datasets:
          - name: dataset:test/static
            split: test
        metrics:
          - name: metric:test/distinct_n
        """
    )
    path = tmp_path / "job.yaml"
    path.write_text(config_text, encoding="utf-8")

    cfg = load_job_config(path)
    assert isinstance(cfg, JobConfig)
    assert cfg.run.seeds == [0, 1]
    assert cfg.user_proxies[0].adapter == ADAPTER_NAME
    assert cfg.scorecards is not None
    assert len(cfg.scorecards) == 1
    default_scorecard = cfg.scorecards[0]
    assert isinstance(default_scorecard, ScorecardConfig)
    assert default_scorecard.name == "mirror_scorecard"
    # single metric -> weight 1.0
    assert default_scorecard.weights == {METRIC_A: 1.0}
    assert cfg.run.timeout_seconds is None
    assert cfg.run.max_retries == 0
    assert cfg.run.retry_backoff_seconds == DEFAULT_RETRY_BACKOFF_SECONDS
    assert cfg.run.observability.log_json is True
    assert cfg.run.observability.log_level == "INFO"


def test_job_config_requires_lists(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(
        "run:\n  seeds: []\n  name: bad\nuser_proxies: []\ndatasets: []\nmetrics: []\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        load_job_config(path)
    assert "Invalid job configuration" in str(exc.value)


def test_scorecard_weights_normalized(tmp_path):
    config_text = textwrap.dedent(
        """
        run:
          name: sample
          seeds: [0]
        user_proxies:
          - name: echo-proxy
            adapter: adapter:test/echo
        datasets:
          - name: dataset:test/static
            split: test
        metrics:
          - name: metric:test/distinct_n
          - name: metric:test/length_diff
        scorecards:
          - name: custom
            weights:
              metric:test/distinct_n: 2
              metric:test/length_diff: 1
        """
    )
    path = tmp_path / "job.yaml"
    path.write_text(config_text, encoding="utf-8")

    cfg = load_job_config(path)
    assert cfg.scorecards is not None
    custom = cfg.scorecards[0]
    assert custom.name == "custom"
    assert pytest.approx(custom.weights[METRIC_A]) == 2 / 3
    assert pytest.approx(custom.weights[METRIC_B]) == 1 / 3


def test_run_config_coerces_boolean_cache(tmp_path):
    config_text = textwrap.dedent(
        """
        run:
          name: sample
          seeds: [0]
          cache: false
        user_proxies:
          - name: echo-proxy
            adapter: adapter:test/echo
        datasets:
          - name: dataset:test/static
            split: test
        metrics:
          - name: metric:test/distinct_n
        """
    )
    path = tmp_path / "job_cache.yaml"
    path.write_text(config_text, encoding="utf-8")

    cfg = load_job_config(path)
    assert cfg.run.cache.enabled is False
    assert cfg.run.cache.ttl_seconds == DEFAULT_CACHE_TTL_SECONDS


def test_run_config_rejects_invalid_retry_values(tmp_path):
    config_text = textwrap.dedent(
        """
        run:
          name: faulty
          seeds: [0]
          max_retries: -1
        user_proxies:
          - name: echo-proxy
            adapter: adapter:test/echo
        datasets:
          - name: dataset:test/static
            split: test
        metrics:
          - name: metric:test/distinct_n
        """
    )
    path = tmp_path / "invalid.yaml"
    path.write_text(config_text, encoding="utf-8")

    with pytest.raises(SystemExit):
        load_job_config(path)
