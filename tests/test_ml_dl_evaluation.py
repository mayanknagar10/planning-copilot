"""
Unit tests for the Phase 3 ML/DL evaluation module.

Like test_forecast_engine.py, these tests require no API keys — ml_dl_evaluation.py
never calls an LLM. Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forecast_engine import DemandForecastEngine
from ml_dl_evaluation import evaluate_ml, evaluate_dl, compare_to_baseline

DATA_PATH = str(Path(__file__).parent.parent / "data" / "demand_history.csv")
HORIZON = 21  # kept short relative to test_forecast_engine.py to keep the suite fast


@pytest.fixture(scope="module")
def engine():
    return DemandForecastEngine(DATA_PATH)


def test_evaluate_ml_returns_valid_scores(engine):
    result = evaluate_ml("SKU-1003", engine, horizon_days=HORIZON)
    assert result.method == "ml_gradient_boosting"
    assert result.mape_pct >= 0
    assert result.wape_pct >= 0


def test_evaluate_dl_returns_valid_scores(engine):
    result = evaluate_dl("SKU-1003", engine, horizon_days=HORIZON)
    assert result.method == "dl_mlp"
    assert result.mape_pct >= 0
    assert result.wape_pct >= 0


def test_compare_to_baseline_includes_all_three_methods(engine):
    result = compare_to_baseline("SKU-1003", engine, horizon_days=HORIZON)
    assert set(result.keys()) >= {"prophet_baseline", "ml", "dl"}
    for key in ("prophet_baseline", "ml", "dl"):
        assert "mape_pct" in result[key]
        assert "wape_pct" in result[key]


def test_compare_to_baseline_unknown_sku_raises(engine):
    with pytest.raises(ValueError):
        compare_to_baseline("SKU-9999", engine, horizon_days=HORIZON)
