"""
Unit tests for the S&OP consensus layer (adjustments + sign-off).

No API keys or forecast model needed — consensus.py is pure bookkeeping over
CSVs, deliberately isolated from forecast_engine.py the same way the rest of
the deterministic core is tested independently of the agent layer.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from consensus import ConsensusStore


@pytest.fixture
def store(tmp_path):
    return ConsensusStore(data_dir=str(tmp_path / "sop_data"))


def test_no_adjustments_consensus_equals_baseline(store):
    result = store.get_consensus("2026-07", "SKU-1003", baseline_total=1000.0)
    assert result.consensus_total == 1000.0
    assert result.adjustments == []
    assert result.is_signed_off is False


def test_adjustment_requires_rationale(store):
    with pytest.raises(ValueError):
        store.add_adjustment("2026-07", "SKU-1003", "Sales", 100, "")
    with pytest.raises(ValueError):
        store.add_adjustment("2026-07", "SKU-1003", "Sales", 100, "   ")


def test_adjustment_rejects_unknown_function(store):
    with pytest.raises(ValueError):
        store.add_adjustment("2026-07", "SKU-1003", "Finance", 100, "some rationale")


def test_consensus_sums_multiple_adjustments(store):
    store.add_adjustment("2026-07", "SKU-1003", "Sales", 120, "Confirmed bulk reorder.")
    store.add_adjustment("2026-07", "SKU-1003", "Marketing", -40, "Promo pulled forward.")

    result = store.get_consensus("2026-07", "SKU-1003", baseline_total=1000.0)
    assert result.consensus_total == 1080.0
    assert len(result.adjustments) == 2


def test_adjustments_are_scoped_to_cycle_and_sku(store):
    store.add_adjustment("2026-07", "SKU-1003", "Sales", 120, "July bump.")
    store.add_adjustment("2026-08", "SKU-1003", "Sales", 999, "August bump — different cycle.")
    store.add_adjustment("2026-07", "SKU-1004", "Sales", 999, "Different SKU.")

    result = store.get_consensus("2026-07", "SKU-1003", baseline_total=1000.0)
    assert result.consensus_total == 1120.0
    assert len(result.adjustments) == 1


def test_sign_off_marks_consensus_final(store):
    store.add_adjustment("2026-07", "SKU-1003", "Sales", 50, "Minor bump.")
    store.sign_off("2026-07", "SKU-1003", signed_off_by="Jane (S&OP Lead)")

    result = store.get_consensus("2026-07", "SKU-1003", baseline_total=1000.0)
    assert result.is_signed_off is True
    assert result.signed_off_by == "Jane (S&OP Lead)"
    assert result.signed_off_at is not None


def test_signed_off_cycle_rejects_new_adjustments(store):
    store.sign_off("2026-07", "SKU-1003", signed_off_by="Jane (S&OP Lead)")
    with pytest.raises(ValueError):
        store.add_adjustment("2026-07", "SKU-1003", "Sales", 50, "Too late.")


def test_persists_across_store_instances(tmp_path):
    data_dir = str(tmp_path / "sop_data")
    store_a = ConsensusStore(data_dir=data_dir)
    store_a.add_adjustment("2026-07", "SKU-1003", "Product", 30, "New pack size launching.")

    store_b = ConsensusStore(data_dir=data_dir)  # simulates a fresh app process
    result = store_b.get_consensus("2026-07", "SKU-1003", baseline_total=500.0)
    assert result.consensus_total == 530.0
