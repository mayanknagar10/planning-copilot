"""
Unit tests for the deterministic forecast engine.

Run: pytest tests/ -v

These tests deliberately do NOT require any API keys — they only test
forecast_engine.py, which never calls an LLM. Testing the deterministic
core in isolation from the agent layer is itself part of the "LLM never
computes the number" design: the numbers need to be correct on their own,
independent of anything the agent says about them.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecast_engine import DemandForecastEngine, ForecastResult

DATA_PATH = str(Path(__file__).parent.parent / "data" / "demand_history.csv")


@pytest.fixture(scope="module")
def engine():
    return DemandForecastEngine(DATA_PATH)


def test_list_skus_returns_expected_count(engine):
    skus = engine.list_skus()
    assert len(skus) == 10
    assert "SKU-1001" in skus


def test_forecast_returns_valid_result(engine):
    result = engine.forecast("SKU-1003", horizon_days=30)
    assert isinstance(result, ForecastResult)
    assert len(result.forecast_df) == 30
    assert result.safety_stock >= 0
    assert result.reorder_point >= result.safety_stock  # reorder point must include safety stock


def test_forecast_accuracy_is_reasonable(engine):
    result = engine.forecast("SKU-1003", horizon_days=30)
    # On synthetic data with ~12% noise, WAPE should land well under 30%
    assert result.wape < 30, f"WAPE too high: {result.wape}% — model may be misconfigured"


def test_unknown_sku_raises(engine):
    with pytest.raises(ValueError):
        engine.forecast("SKU-9999")


def test_promo_scenario_increases_demand(engine):
    """A promo scenario should never predict LESS demand than baseline."""
    baseline = engine.forecast("SKU-1003", horizon_days=14, promo_scenario=[0] * 14)
    promo = engine.forecast("SKU-1003", horizon_days=14, promo_scenario=[1] * 14)
    assert promo.forecast_df["yhat"].sum() > baseline.forecast_df["yhat"].sum()


def test_longer_lead_time_increases_reorder_point(engine):
    """Reorder point must monotonically increase with lead time — a basic
    sanity check that would catch a broken formula immediately."""
    short = engine.forecast("SKU-1005", horizon_days=30, lead_time_days=7)
    long = engine.forecast("SKU-1005", horizon_days=30, lead_time_days=28)
    assert long.reorder_point > short.reorder_point


def test_exception_detection_returns_native_bool(engine):
    """Regression test: is_exception must be a native Python bool, not
    numpy.bool_, or it silently breaks JSON serialization downstream."""
    result = engine.detect_exceptions("SKU-1003")
    assert isinstance(result["is_exception"], bool)


def test_detect_exceptions_unknown_sku_raises(engine):
    """detect_exceptions must validate sku_id the same way forecast() and
    get_sku_history() do — otherwise an unknown SKU falls through to Prophet
    fitting on an empty/near-empty frame and fails with an obscure error
    instead of a clean ValueError the caller can handle."""
    with pytest.raises(ValueError):
        engine.detect_exceptions("SKU-9999")


def test_run_scenario_matches_forecast_with_same_params(engine):
    """run_scenario is a convenience wrapper — verify it produces the same
    numbers as calling forecast() directly with equivalent parameters."""
    direct = engine.forecast("SKU-1002", horizon_days=10,
                              promo_scenario=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              lead_time_days=14)
    via_scenario = engine.run_scenario("SKU-1002", horizon_days=10,
                                        extra_promo_days=3, lead_time_override=14)
    assert direct.forecast_df["yhat"].sum() == pytest.approx(
        via_scenario.forecast_df["yhat"].sum(), rel=1e-6
    )


# ── PRD Phase 1 — SMA/ETS baseline comparison ────────────────────────────────

def test_get_sku_history_returns_expected_columns(engine):
    history = engine.get_sku_history("SKU-1003")
    assert list(history.columns) == ["ds", "y", "on_promotion"]
    assert len(history) > 0


def test_get_sku_history_unknown_sku_raises(engine):
    with pytest.raises(ValueError):
        engine.get_sku_history("SKU-9999")


def test_compare_baselines_includes_all_three_methods(engine):
    result = engine.compare_baselines("SKU-1003", horizon_days=28)
    assert set(result["methods"].keys()) == {"sma", "ets", "prophet"}
    for method, scores in result["methods"].items():
        assert "error" in scores or ("mape_pct" in scores and "wape_pct" in scores), (
            f"{method} result missing expected score keys: {scores}"
        )


def test_compare_baselines_scores_are_non_negative(engine):
    result = engine.compare_baselines("SKU-1005", horizon_days=28)
    for method, scores in result["methods"].items():
        if "error" in scores:
            continue
        assert scores["mape_pct"] >= 0
        assert scores["wape_pct"] >= 0
