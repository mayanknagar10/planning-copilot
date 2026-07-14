"""
Synthetic demand data generator for PlanningCopilot.

Generates realistic SKU-level daily demand history with:
- Long-term trend
- Weekly seasonality (weekend spikes for some categories)
- Yearly seasonality (holiday season lift)
- Promotional spikes
- Random noise + occasional stockout dips

Schema matches what you'd get from a real POS/ERP extract, so this can be
swapped for a real dataset (e.g. Kaggle M5, Online Retail II) later by
producing a CSV with the same columns: date, sku_id, category, demand,
on_promotion, price.

Run: python generate_data.py
Output: demand_history.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 12, 31)
N_DAYS = (END_DATE - START_DATE).days + 1

SKUS = [
    {"sku_id": "SKU-1001", "category": "Beverages",   "base_demand": 120, "weekend_lift": 1.4, "yearly_amp": 0.25, "price": 4.99},
    {"sku_id": "SKU-1002", "category": "Beverages",   "base_demand": 80,  "weekend_lift": 1.3, "yearly_amp": 0.20, "price": 3.49},
    {"sku_id": "SKU-1003", "category": "Snacks",      "base_demand": 200, "weekend_lift": 1.6, "yearly_amp": 0.15, "price": 2.99},
    {"sku_id": "SKU-1004", "category": "Snacks",      "base_demand": 150, "weekend_lift": 1.5, "yearly_amp": 0.15, "price": 3.99},
    {"sku_id": "SKU-1005", "category": "Household",   "base_demand": 60,  "weekend_lift": 1.1, "yearly_amp": 0.10, "price": 12.99},
    {"sku_id": "SKU-1006", "category": "Household",   "base_demand": 45,  "weekend_lift": 1.05,"yearly_amp": 0.10, "price": 8.99},
    {"sku_id": "SKU-1007", "category": "Electronics", "base_demand": 25,  "weekend_lift": 1.3, "yearly_amp": 0.60, "price": 49.99},
    {"sku_id": "SKU-1008", "category": "Electronics", "base_demand": 18,  "weekend_lift": 1.2, "yearly_amp": 0.70, "price": 89.99},
    {"sku_id": "SKU-1009", "category": "Seasonal",    "base_demand": 30,  "weekend_lift": 1.2, "yearly_amp": 1.20, "price": 15.99},
    {"sku_id": "SKU-1010", "category": "Seasonal",    "base_demand": 22,  "weekend_lift": 1.15,"yearly_amp": 1.10, "price": 19.99},
]


def generate_sku_series(sku_cfg, dates):
    n = len(dates)
    t = np.arange(n)

    # long-term trend: slow linear growth with a slight slowdown in year 3 (realistic business dynamic)
    trend = sku_cfg["base_demand"] * (1 + 0.00015 * t - 0.00000015 * t**2)

    # weekly seasonality: weekend lift
    dow = np.array([d.weekday() for d in dates])  # 0=Mon ... 6=Sun
    weekly = np.where(dow >= 5, sku_cfg["weekend_lift"], 1.0)

    # yearly seasonality: holiday season lift (Nov-Dec), smoothed sine wave otherwise
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    yearly = 1 + sku_cfg["yearly_amp"] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # extra November/December holiday bump
    holiday_bump = np.where(
        np.isin([d.month for d in dates], [11, 12]),
        1 + sku_cfg["yearly_amp"] * 0.8,
        1.0,
    )

    # promotions: random 3-7 day promo windows, ~8 times a year, +40-90% lift
    on_promotion = np.zeros(n, dtype=int)
    promo_lift = np.ones(n)
    n_promos = int(4 * (N_DAYS / 365))
    for _ in range(n_promos):
        start = np.random.randint(0, n - 10)
        length = np.random.randint(3, 8)
        lift = np.random.uniform(1.4, 1.9)
        on_promotion[start:start + length] = 1
        promo_lift[start:start + length] = lift

    # occasional stockout dips (2-4 day near-zero demand, ~3 times/year, realistic supply issue)
    stockout_mask = np.ones(n)
    n_stockouts = int(3 * (N_DAYS / 365))
    for _ in range(n_stockouts):
        start = np.random.randint(0, n - 5)
        length = np.random.randint(2, 5)
        stockout_mask[start:start + length] = np.random.uniform(0.05, 0.2)

    # noise
    noise = np.random.normal(1.0, 0.12, n)

    demand = trend * weekly * yearly * holiday_bump * promo_lift * stockout_mask * noise
    demand = np.maximum(demand, 0).round().astype(int)

    return demand, on_promotion


def main():
    dates = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]
    rows = []

    for sku_cfg in SKUS:
        demand, on_promotion = generate_sku_series(sku_cfg, dates)
        for i, d in enumerate(dates):
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "sku_id": sku_cfg["sku_id"],
                "category": sku_cfg["category"],
                "demand": demand[i],
                "on_promotion": on_promotion[i],
                "price": sku_cfg["price"],
            })

    df = pd.DataFrame(rows)
    out_path = "demand_history.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows across {len(SKUS)} SKUs")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to {out_path}")
    print("\nSample:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
