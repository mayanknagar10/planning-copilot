"""
forecast_engine.py — the deterministic forecasting core of PlanningCopilot.

IMPORTANT DESIGN PRINCIPLE:
This module NEVER calls an LLM. All numbers come from Prophet, a well-established
statistical forecasting library. The LLM layer (agent.py) sits ON TOP of this
module and is only allowed to call these functions and narrate their output —
it never estimates a demand number itself. This separation is the core
engineering decision behind this whole project.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from prophet import Prophet
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


@dataclass
class ForecastResult:
    sku_id: str
    forecast_df: pd.DataFrame          # columns: ds, yhat, yhat_lower, yhat_upper
    history_df: pd.DataFrame           # actual historical demand used for training
    mape: float                        # backtested accuracy on a holdout window
    wape: float                        # weighted absolute percentage error (more robust for demand)
    model_params: dict                 # what parameters were used, for auditability
    safety_stock: float                # recommended safety stock units
    reorder_point: float               # recommended reorder point units


class DemandForecastEngine:
    """
    Wraps Prophet to produce SKU-level demand forecasts with safety stock
    and reorder point recommendations. Every method returns numbers only —
    no natural language. That happens one layer up, in agent.py.
    """

    def __init__(self, data_path: str = "data/demand_history.csv"):
        self.df = pd.read_csv(data_path, parse_dates=["date"])
        self._model_cache: dict[str, Prophet] = {}

    def list_skus(self) -> list[str]:
        return sorted(self.df["sku_id"].unique().tolist())

    def _prepare_sku_data(self, sku_id: str) -> pd.DataFrame:
        sku_df = self.df[self.df["sku_id"] == sku_id].copy()
        sku_df = sku_df.rename(columns={"date": "ds", "demand": "y"})
        return sku_df[["ds", "y", "on_promotion"]].sort_values("ds")

    def _backtest(self, sku_df: pd.DataFrame, horizon_days: int = 28) -> tuple[float, float]:
        """
        Simple holdout backtest: train on all but the last `horizon_days`,
        predict that window, compare to actuals. Returns (MAPE, WAPE).
        """
        train = sku_df.iloc[:-horizon_days]
        test = sku_df.iloc[-horizon_days:]

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        m.add_regressor("on_promotion")
        m.fit(train)

        future = test[["ds", "on_promotion"]].copy()
        forecast = m.predict(future)

        actual = test["y"].values
        pred = forecast["yhat"].values
        pred = np.maximum(pred, 0)

        # MAPE (can be unstable near zero demand days)
        nonzero = actual > 0
        mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")

        # WAPE — more robust for intermittent/low-volume demand, preferred metric in supply chain planning
        wape = float(np.sum(np.abs(actual - pred)) / np.sum(actual) * 100) if actual.sum() > 0 else float("nan")

        return round(mape, 2), round(wape, 2)

    def forecast(
        self,
        sku_id: str,
        horizon_days: int = 30,
        promo_scenario: Optional[list[int]] = None,
        lead_time_days: int = 14,
        service_level_z: float = 1.65,  # ~95% service level
    ) -> ForecastResult:
        """
        Produce a forecast for a SKU.

        promo_scenario: optional list of 0/1 flags of length horizon_days,
                         overriding whether each future day is a promo day.
                         This is what powers "what-if" scenario questions —
                         the agent calls this again with different promo flags,
                         it never invents the answer itself.
        lead_time_days: supplier lead time, used for reorder point calculation.
        service_level_z: z-score for desired service level (1.65 ≈ 95%, 2.33 ≈ 99%).
        """
        if sku_id not in self.df["sku_id"].unique():
            raise ValueError(f"Unknown SKU: {sku_id}")

        sku_df = self._prepare_sku_data(sku_id)
        mape, wape = self._backtest(sku_df)

        model_params = {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "regressors": ["on_promotion"],
            "backtest_horizon_days": 28,
        }

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        m.add_regressor("on_promotion")
        m.fit(sku_df)

        future = m.make_future_dataframe(periods=horizon_days)
        # default: no promos scheduled in the future unless a scenario is given
        future["on_promotion"] = 0
        future.loc[future.index[-horizon_days:], "on_promotion"] = (
            promo_scenario if promo_scenario is not None else [0] * horizon_days
        )

        forecast = m.predict(future)
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

        future_forecast = forecast.tail(horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)

        # Safety stock & reorder point — standard supply chain planning formulas
        # demand_std estimated from the forecast interval width (proxy for uncertainty)
        avg_daily_demand = future_forecast["yhat"].mean()
        demand_std = (future_forecast["yhat_upper"] - future_forecast["yhat_lower"]).mean() / (2 * 1.96)  # from 95% CI

        safety_stock = service_level_z * demand_std * np.sqrt(lead_time_days)
        reorder_point = avg_daily_demand * lead_time_days + safety_stock

        return ForecastResult(
            sku_id=sku_id,
            forecast_df=future_forecast,
            history_df=sku_df.rename(columns={"ds": "date", "y": "demand"}),
            mape=mape,
            wape=wape,
            model_params=model_params,
            safety_stock=round(float(safety_stock), 1),
            reorder_point=round(float(reorder_point), 1),
        )

    def run_scenario(
        self,
        sku_id: str,
        horizon_days: int = 30,
        extra_promo_days: int = 0,
        lead_time_override: Optional[int] = None,
    ) -> ForecastResult:
        """
        Convenience wrapper the agent calls for "what-if" questions, e.g.
        "what if we run a promo for the first week" or "what if lead time
        grows to 21 days". This is the ONLY way the agent can answer such
        questions — by re-running the real model, never by guessing.
        """
        promo_scenario = [1] * extra_promo_days + [0] * (horizon_days - extra_promo_days)
        lead_time = lead_time_override if lead_time_override is not None else 14

        return self.forecast(
            sku_id=sku_id,
            horizon_days=horizon_days,
            promo_scenario=promo_scenario,
            lead_time_days=lead_time,
        )

    def detect_exceptions(self, sku_id: str, threshold_pct: float = 15.0) -> dict:
        """
        Flags whether recent actual demand deviated significantly from what
        a prior forecast would have predicted — the trigger for an "exception
        narrative" in the S&OP report. Pure statistics, no LLM.
        """
        sku_df = self._prepare_sku_data(sku_id)
        recent = sku_df.tail(14)

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode="multiplicative")
        m.add_regressor("on_promotion")
        m.fit(sku_df.iloc[:-14])

        future = recent[["ds", "on_promotion"]].copy()
        forecast = m.predict(future)

        actual_total = recent["y"].sum()
        predicted_total = forecast["yhat"].clip(lower=0).sum()
        deviation_pct = ((actual_total - predicted_total) / predicted_total * 100) if predicted_total > 0 else 0

        return {
            "sku_id": sku_id,
            "actual_last_14d": int(actual_total),
            "predicted_last_14d": round(float(predicted_total), 1),
            "deviation_pct": round(float(deviation_pct), 1),
            "is_exception": bool(abs(deviation_pct) > threshold_pct),
        }


if __name__ == "__main__":
    engine = DemandForecastEngine("../data/demand_history.csv")
    print("Available SKUs:", engine.list_skus())

    result = engine.forecast("SKU-1003", horizon_days=30)
    print(f"\nForecast for {result.sku_id}")
    print(f"MAPE: {result.mape}%  |  WAPE: {result.wape}%")
    print(f"Safety stock: {result.safety_stock} units")
    print(f"Reorder point: {result.reorder_point} units")
    print(result.forecast_df.head())
