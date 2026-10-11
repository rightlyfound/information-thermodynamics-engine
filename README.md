# Information Thermodynamics Engine: Mathematical Defense Against AI Model Collapse

The **Information Thermodynamics Engine** is a lightweight, dependency-free Python sentinel for detecting information drift while data is entering an AI training pipeline. It turns a silent dataset-quality problem into an observable, testable control signal.

> This project is part of the sovereign, drift-native computing initiative, cryptographically anchored by Gameover.nft.

## The Problem: Model Collapse

When synthetic or low-quality material overwhelms human-grounded material, future models can lose diversity, factual grounding, and useful signal. Conventional training workflows often discover this only after expensive compute has already been spent.

## The Solution: Information Drift

`ModelCollapseSentinel` maintains a fixed-size rolling window of ingestion batches. Every batch contributes to two running totals:

- **Expansion:** synthetic, low-perplexity, or unverified volume.
- **Contraction:** human-verified, high-perplexity, or cryptographically signed volume.

Because expired batches are subtracted as new batches arrive, monitoring is **O(1)** per ingestion operation with respect to the window size. When the drift margin crosses the configured threshold, the sentinel calls an intervention callback and records the triggering batch for quarantine.

## Installation

The engine uses only the Python standard library. Copy `src/thermodynamics_sentinel.py` into your project or install the repository in editable mode with your preferred packaging configuration.

Run the tests:

```bash
PYTHONPATH=src python -m pytest -q
```

## Usage Example

```python
from thermodynamics_sentinel import ModelCollapseSentinel


def halt_and_quarantine(batch, drift):
    print(f"Intervention: quarantine {batch.total_volume} items; drift={drift:.3f}")
    # Stop the downstream training consumer here.

sentinel = ModelCollapseSentinel(
    baseline_grounding_ratio=0.80,
    window_size=100,
    drift_threshold=-0.20,
    intervention_triggered=halt_and_quarantine,
)

status = sentinel.ingest_batch(synthetic_vol=100, human_vol=900)
print(status["status"])  # HEALTHY

status = sentinel.ingest_batch(synthetic_vol=2000, human_vol=0)
print(status["status"])  # CRITICAL when the rolling margin crosses -0.20
```

The callback is edge-triggered: it fires once when the margin moves from at-or-above the threshold to below it. If the stream recovers and later crosses again, a new intervention is emitted.

## The Math

For a rolling window, let `E` be the total expansion volume and `C` be the total contraction volume:

```text
Grounding Ratio = C / (E + C)
Information Drift Margin = Grounding Ratio - Baseline
```

The default intervention rule is:

```text
Information Drift Margin < -0.20  =>  CRITICAL / quarantine
```

For example, with a baseline grounding ratio of `0.80`, a rolling ratio below `0.60` crosses the threshold. This is a transparent operational guardrail, not a claim that one scalar can fully characterize model quality: production systems should combine it with provenance, deduplication, evaluation, and cryptographic verification signals.

## Project Layout

```text
src/thermodynamics_sentinel.py  # Core rolling-window sentinel
 tests/test_model_collapse.py   # Healthy/flood/recovery proof and validation tests
```

## License

Released as an open-source reference implementation. Add the license and contribution policy that best match your intended governance model.
