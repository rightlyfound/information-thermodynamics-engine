from thermodynamics_sentinel import ModelCollapseSentinel


def test_healthy_ingestion_then_synthetic_flood_triggers_once_at_crossing():
    interventions = []

    def quarantine(batch, drift):
        interventions.append((batch, drift))

    sentinel = ModelCollapseSentinel(
        baseline_grounding_ratio=0.80,
        window_size=4,
        intervention_triggered=quarantine,
    )

    # Phase 1: high human verification keeps the rolling ratio healthy.
    for _ in range(3):
        status = sentinel.ingest_batch(synthetic_vol=10, human_vol=90)
        assert status["status"] == "HEALTHY"
    assert interventions == []

    # Phase 2: a massive flood pushes the four-batch rolling ratio below 0.60.
    status = sentinel.ingest_batch(synthetic_vol=300, human_vol=0)
    assert status["status"] == "CRITICAL"
    assert status["drift_margin"] < -0.20
    assert len(interventions) == 1
    assert len(sentinel.quarantined_batches) == 1

    # Remaining critical batches do not repeatedly fire the same intervention.
    status = sentinel.ingest_batch(synthetic_vol=100, human_vol=0)
    assert status["status"] == "CRITICAL"
    assert len(interventions) == 1


def test_recovery_allows_a_later_threshold_crossing():
    interventions = []
    sentinel = ModelCollapseSentinel(
        baseline_grounding_ratio=0.80,
        window_size=2,
        intervention_triggered=lambda batch, drift: interventions.append(drift),
    )

    sentinel.ingest_batch(0, 100)
    sentinel.ingest_batch(100, 0)
    assert len(interventions) == 1

    sentinel.ingest_batch(0, 100)  # rolling window recovers to 50% grounding
    sentinel.ingest_batch(0, 100)  # healthy at 100%, so the latch resets
    sentinel.ingest_batch(100, 0)
    sentinel.ingest_batch(100, 0)
    assert len(interventions) == 2


def test_invalid_batches_are_rejected():
    sentinel = ModelCollapseSentinel(baseline_grounding_ratio=0.8)
    for values in [(-1, 10), (10, -1), (0, 0)]:
        try:
            sentinel.ingest_batch(*values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid batch should raise ValueError")
