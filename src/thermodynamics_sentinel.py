"""Rolling information-drift monitoring for AI data ingestion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time
from typing import Callable, Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class IngestionBatch:
    """A single batch observed by the sentinel."""

    synthetic_vol: float
    human_vol: float
    ingested_at: float

    @property
    def total_volume(self) -> float:
        return self.synthetic_vol + self.human_vol

    @property
    def grounding_ratio(self) -> float:
        if self.total_volume == 0:
            return 1.0
        return self.human_vol / self.total_volume


InterventionCallback = Callable[[IngestionBatch, float], None]


class ModelCollapseSentinel:
    """Monitor a rolling grounding ratio and flag information drift.

    The rolling sums make ``ingest_batch`` and ``calculate_drift`` O(1) with
    respect to the configured window size. The callback fires only when the
    drift margin crosses from at-or-above the threshold to below it. If the
    stream recovers and crosses again later, a new intervention is emitted.
    """

    def __init__(
        self,
        baseline_grounding_ratio: float,
        window_size: int = 100,
        drift_threshold: float = -0.20,
        intervention_triggered: Optional[InterventionCallback] = None,
    ) -> None:
        if not 0.0 <= baseline_grounding_ratio <= 1.0:
            raise ValueError("baseline_grounding_ratio must be between 0 and 1")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if drift_threshold >= 0.0:
            raise ValueError("drift_threshold must be negative")

        self.baseline_grounding_ratio = baseline_grounding_ratio
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.intervention_triggered = intervention_triggered
        self._batches: Deque[IngestionBatch] = deque(maxlen=window_size)
        self._synthetic_total = 0.0
        self._human_total = 0.0
        self._previous_drift: Optional[float] = None
        self.quarantined_batches: list[IngestionBatch] = []

    def ingest_batch(self, synthetic_vol: float, human_vol: float) -> Dict[str, object]:
        """Record a batch and return its current monitoring snapshot.

        Volumes are non-negative counts or weights. A batch that causes a
        threshold crossing is recorded in ``quarantined_batches`` and sent to
        the optional intervention callback so a caller can halt downstream
        training or move the batch to isolated storage.
        """
        if synthetic_vol < 0 or human_vol < 0:
            raise ValueError("synthetic_vol and human_vol must be non-negative")
        if synthetic_vol + human_vol == 0:
            raise ValueError("a batch must contain a positive total volume")

        batch = IngestionBatch(synthetic_vol, human_vol, time())
        if len(self._batches) == self.window_size:
            expired = self._batches[0]
            self._synthetic_total -= expired.synthetic_vol
            self._human_total -= expired.human_vol
        self._batches.append(batch)
        self._synthetic_total += synthetic_vol
        self._human_total += human_vol

        drift = self.calculate_drift()
        crossed_threshold = (
            drift < self.drift_threshold
            and (self._previous_drift is None or self._previous_drift >= self.drift_threshold)
        )
        if crossed_threshold:
            self.quarantined_batches.append(batch)
            if self.intervention_triggered is not None:
                self.intervention_triggered(batch, drift)
        self._previous_drift = drift
        return self.get_status()

    def calculate_drift(self) -> float:
        """Return current rolling grounding ratio minus its baseline."""
        total = self._synthetic_total + self._human_total
        current_ratio = self._human_total / total if total else self.baseline_grounding_ratio
        return current_ratio - self.baseline_grounding_ratio

    def get_status(self) -> Dict[str, object]:
        """Return a serializable status snapshot for monitoring systems."""
        total = self._synthetic_total + self._human_total
        ratio = self._human_total / total if total else self.baseline_grounding_ratio
        drift = ratio - self.baseline_grounding_ratio
        return {
            "status": "CRITICAL" if drift < self.drift_threshold else "HEALTHY",
            "grounding_ratio": ratio,
            "baseline": self.baseline_grounding_ratio,
            "drift_margin": drift,
            "window_batches": len(self._batches),
            "quarantined_batches": len(self.quarantined_batches),
        }


__all__ = ["IngestionBatch", "ModelCollapseSentinel"]
