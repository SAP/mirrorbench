"""Utilities for calibrating pairwise judge metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean

from mirrorbench.core.models.run import MetricValue

HH_LABEL = "HH"
PP_LABEL = "PP"
CONTROL_METADATA_KEY = "control"


@dataclass(slots=True)
class CalibrationAnchors:
    """HH/PP anchor means used to calibrate pairwise metrics."""

    hh_mean: float
    pp_mean: float

    @property
    def gap(self) -> float:
        """Return HH minus PP gap; zero indicates no separation."""

        return self.hh_mean - self.pp_mean


def derive_anchors(values: Sequence[MetricValue]) -> CalibrationAnchors | None:
    """Derive HH/PP anchor means from metric values metadata."""

    hh_scores: list[float] = []
    pp_scores: list[float] = []

    for value in values:
        control_label = value.metadata.get(CONTROL_METADATA_KEY)
        if control_label == HH_LABEL:
            hh_scores.extend(value.values)
        elif control_label == PP_LABEL:
            pp_scores.extend(value.values)

        controls = value.metadata.get("controls")
        if isinstance(controls, dict):
            hh_meta = controls.get("hh")
            if isinstance(hh_meta, dict) and "score" in hh_meta:
                hh_scores.append(float(hh_meta["score"]))
            pp_meta = controls.get("pp")
            if isinstance(pp_meta, dict) and "score" in pp_meta:
                pp_scores.append(float(pp_meta["score"]))

    if not hh_scores or not pp_scores:
        return None

    return CalibrationAnchors(hh_mean=mean(hh_scores), pp_mean=mean(pp_scores))


def apply_linear_calibration(
    scores: Iterable[float],
    anchors: CalibrationAnchors,
) -> list[float]:
    """Apply linear calibration using HH/PP anchors."""

    denominator = max(1e-6, anchors.gap)
    calibrated: list[float] = []
    for score in scores:
        normalised = (score - anchors.pp_mean) / denominator
        calibrated.append(min(1.0, max(0.0, normalised)))
    return calibrated


def calibration_summary(
    *,
    raw_mean: float,
    anchors: CalibrationAnchors | None,
    calibrated_scores: Sequence[float] | None,
) -> dict[str, float | None]:
    """Build a metadata payload describing calibration results."""

    if anchors is None:
        return {
            "enabled": False,
            "raw_mean": raw_mean,
            "calibrated_mean": None,
            "hh_mean": None,
            "pp_mean": None,
            "gap": None,
        }

    calibrated_mean = mean(calibrated_scores) if calibrated_scores else None
    return {
        "enabled": True,
        "raw_mean": raw_mean,
        "calibrated_mean": calibrated_mean,
        "hh_mean": anchors.hh_mean,
        "pp_mean": anchors.pp_mean,
        "gap": anchors.gap,
    }


__all__ = [
    "CalibrationAnchors",
    "CONTROL_METADATA_KEY",
    "HH_LABEL",
    "PP_LABEL",
    "apply_linear_calibration",
    "calibration_summary",
    "derive_anchors",
]
