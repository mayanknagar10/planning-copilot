"""
ml_dl_evaluation.py — PRD Phase 3: ML/DL evaluation (evaluation-only).

NOT wired into agent.py, mcp_server.py, or app.py, and NOT an agent tool.
Per the PRD (Section 8.3 / Section 4.2 non-goals), Phase 3 is an evaluation
exercise, not a production deployment: the question this module answers is
"does a heavier model beat the Phase 1 Prophet baseline enough to justify the
added complexity — before committing to an in-house build or a commercial
platform (Kinaxis / Blue Yonder / o9 Solutions / Anaplan)?" It intentionally
does not touch the deterministic-core-only guarantee forecast_engine.py makes
for the production path.

DESIGN CHOICES — read before swapping the models out:
- Classical ML → scikit-learn's HistGradientBoostingRegressor, standing in for
  XGBoost/LightGBM (same lag-feature gradient-boosted-tree family; swap the
  estimator for a real XGBoost/LightGBM model with no other code changes here
  if you want a closer match to what a commercial engine uses internally).
- "DL" → scikit-learn's MLPRegressor (a small feed-forward net), standing in
  for a proper sequence model (LSTM / Temporal Fusion Transformer / DeepAR).
  This project deliberately avoids a PyTorch/TensorFlow dependency for a
  Phase 3 evaluation stub — same free-tier, no-heavy-install philosophy as
  the rest of the project (see README's LLM provider section). If Phase 3
  clears its decision gate, swap evaluate_dl()'s model for a real sequence
  model; the feature-engineering/scoring harness below doesn't need to change.

Both methods are lag/calendar-feature regressors, not sequence models — this
is a lightweight comparison, not a production DL pipeline.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from forecast_engine import DemandForecastEngine

N_LAGS = 14  # trailing days of demand used as model features


@dataclass
class EvalResult:
    method: str
    mape_pct: float
    wape_pct: float
    notes: str


def _make_lag_features(sku_df: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """
    Reshapes forecast_engine's (ds, y, on_promotion) history into a supervised
    learning frame: lag_1..lag_n of demand + day-of-week + on_promotion →
    next-day demand. Same source rows Prophet trains on, just tabular instead
    of a time-series API.
    """
    df = sku_df.copy().reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["dow"] = df["ds"].dt.weekday
    return df.dropna().reset_index(drop=True)


def _fit_predict(model, sku_df: pd.DataFrame, horizon_days: int, n_lags: int = N_LAGS) -> np.ndarray:
    """
    Recursive (iterative) multi-step forecast: fit on everything except the
    holdout window, then predict one day at a time, feeding each day's
    prediction back in as a lag for the next — the standard way to turn a
    tabular regressor into a multi-day forecaster.
    """
    feat_df = _make_lag_features(sku_df, n_lags)
    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)] + ["dow", "on_promotion"]
    train = feat_df.iloc[:-horizon_days]

    X_train, y_train = train[feature_cols], train["y"]
    scaler = StandardScaler().fit(X_train)
    model.fit(scaler.transform(X_train), y_train)

    history = list(sku_df["y"].iloc[-(horizon_days + n_lags):-horizon_days])
    last_dow = sku_df["ds"].iloc[-horizon_days - 1].weekday()

    preds = []
    for step in range(horizon_days):
        lags = history[-n_lags:][::-1]  # lag_1 = most recent day
        dow = (last_dow + step + 1) % 7
        promo = 0  # no future promo assumed — matches forecast_engine.forecast()'s default scenario
        row = pd.DataFrame([lags + [dow, promo]], columns=feature_cols)
        pred = max(float(model.predict(scaler.transform(row))[0]), 0)
        preds.append(pred)
        history.append(pred)

    return np.array(preds)


def evaluate_ml(sku_id: str, engine: DemandForecastEngine, horizon_days: int = 28) -> EvalResult:
    """Classical ML candidate — gradient-boosted trees on lag/calendar features."""
    sku_df = engine.get_sku_history(sku_id)
    actual = sku_df["y"].iloc[-horizon_days:].to_numpy()
    pred = _fit_predict(HistGradientBoostingRegressor(max_depth=4, random_state=42), sku_df, horizon_days)
    mape, wape = DemandForecastEngine._score(actual, pred).values()
    return EvalResult(
        method="ml_gradient_boosting",
        mape_pct=mape, wape_pct=wape,
        notes="scikit-learn HistGradientBoostingRegressor — stands in for XGBoost/LightGBM",
    )


def evaluate_dl(sku_id: str, engine: DemandForecastEngine, horizon_days: int = 28) -> EvalResult:
    """'DL-lite' candidate — small feed-forward neural net on lag/calendar features."""
    sku_df = engine.get_sku_history(sku_id)
    actual = sku_df["y"].iloc[-horizon_days:].to_numpy()
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16), max_iter=2000, early_stopping=True, random_state=42,
    )
    pred = _fit_predict(model, sku_df, horizon_days)
    mape, wape = DemandForecastEngine._score(actual, pred).values()
    return EvalResult(
        method="dl_mlp",
        mape_pct=mape, wape_pct=wape,
        notes="scikit-learn MLPRegressor — stands in for a sequence model (LSTM/TFT/DeepAR) "
              "if Phase 3 clears its decision gate",
    )


def compare_to_baseline(sku_id: str, engine: DemandForecastEngine, horizon_days: int = 28) -> dict:
    """
    Runs the ML and DL candidates against the same holdout window
    forecast_engine.py already backtests Prophet on, so the Phase 3
    decision-gate question has a direct, apples-to-apples answer instead of
    an eyeballed one.
    """
    baseline = engine.forecast(sku_id, horizon_days=horizon_days)
    ml_result = evaluate_ml(sku_id, engine, horizon_days)
    dl_result = evaluate_dl(sku_id, engine, horizon_days)

    return {
        "sku_id": sku_id,
        "horizon_days": horizon_days,
        "prophet_baseline": {"mape_pct": baseline.mape, "wape_pct": baseline.wape},
        "ml": {"mape_pct": ml_result.mape_pct, "wape_pct": ml_result.wape_pct, "notes": ml_result.notes},
        "dl": {"mape_pct": dl_result.mape_pct, "wape_pct": dl_result.wape_pct, "notes": dl_result.notes},
    }


if __name__ == "__main__":
    # Quick smoke test — no API keys needed, same as forecast_engine.py's own smoke test.
    engine = DemandForecastEngine("../data/demand_history.csv")
    for sku in ["SKU-1003", "SKU-1007"]:
        result = compare_to_baseline(sku, engine)
        print(f"\n{sku} — {result['horizon_days']}-day holdout")
        print(f"  Prophet (Phase 1)  MAPE {result['prophet_baseline']['mape_pct']}%  "
              f"WAPE {result['prophet_baseline']['wape_pct']}%")
        print(f"  ML  (GBT)          MAPE {result['ml']['mape_pct']}%  WAPE {result['ml']['wape_pct']}%")
        print(f"  DL  (MLP)          MAPE {result['dl']['mape_pct']}%  WAPE {result['dl']['wape_pct']}%")
