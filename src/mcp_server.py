"""
mcp_server.py — exposes PlanningCopilot's tools via MCP (Model Context Protocol).

This is deliberately a THIN wrapper. It does not duplicate any logic — every
function here calls straight into forecast_engine.py or knowledge_base.py,
the exact same deterministic core used by the LangGraph agent in agent.py.
There is exactly one source of truth for "how a forecast is computed" in this
whole project, regardless of which interface (Streamlit chat, MCP client) is
asking for it.

Run standalone:
    python3 mcp_server.py

Or configure in Claude Desktop's claude_desktop_config.json (see README):
    {
      "mcpServers": {
        "planning-copilot": {
          "command": "python3",
          "args": ["/absolute/path/to/planning-copilot/src/mcp_server.py"]
        }
      }
    }
"""

import json
from typing import Optional
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()  # Claude Desktop launches this as a subprocess with its own
                # environment — this ensures .env is read regardless of how
                # the process was started.

from forecast_engine import DemandForecastEngine
from knowledge_base import PlanningKnowledgeBase

mcp = FastMCP("planning-copilot")

_engine: Optional[DemandForecastEngine] = None
_kb: Optional[PlanningKnowledgeBase] = None


def get_engine() -> DemandForecastEngine:
    global _engine
    if _engine is None:
        _engine = DemandForecastEngine("../data/demand_history.csv")
    return _engine


def get_kb() -> PlanningKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = PlanningKnowledgeBase(persist_dir="./chroma_db", embedding_mode="default")
        _kb.build()
    return _kb


@mcp.tool()
def list_available_skus() -> str:
    """Returns the list of SKU IDs available for demand forecasting."""
    return json.dumps(get_engine().list_skus())


@mcp.tool()
def get_forecast(sku_id: str, horizon_days: int = 30) -> str:
    """
    Get the baseline demand forecast for a SKU. Returns forecast totals,
    backtested accuracy (MAPE/WAPE), recommended safety stock, and reorder
    point — all computed by Prophet, never estimated.
    """
    result = get_engine().forecast(sku_id, horizon_days=horizon_days)
    return json.dumps({
        "sku_id": result.sku_id,
        "horizon_days": horizon_days,
        "total_forecast_demand": round(float(result.forecast_df["yhat"].sum()), 1),
        "avg_daily_demand": round(float(result.forecast_df["yhat"].mean()), 1),
        "forecast_accuracy_mape_pct": result.mape,
        "forecast_accuracy_wape_pct": result.wape,
        "recommended_safety_stock_units": result.safety_stock,
        "recommended_reorder_point_units": result.reorder_point,
    })


@mcp.tool()
def run_what_if_scenario(sku_id: str, horizon_days: int = 30, extra_promo_days: int = 0,
                          lead_time_days: int = 14) -> str:
    """
    Re-runs the real Prophet forecast under a hypothetical scenario — e.g.
    a promo of a given length, or a different supplier lead time. Always
    re-computes rather than estimating the answer.
    """
    engine = get_engine()
    baseline = engine.forecast(sku_id, horizon_days=horizon_days)
    scenario = engine.run_scenario(
        sku_id, horizon_days=horizon_days,
        extra_promo_days=extra_promo_days, lead_time_override=lead_time_days,
    )
    baseline_total = float(baseline.forecast_df["yhat"].sum())
    scenario_total = float(scenario.forecast_df["yhat"].sum())
    return json.dumps({
        "sku_id": sku_id,
        "scenario_params": {"extra_promo_days": extra_promo_days, "lead_time_days": lead_time_days},
        "baseline_total_demand": round(baseline_total, 1),
        "scenario_total_demand": round(scenario_total, 1),
        "demand_delta_pct": round((scenario_total / baseline_total - 1) * 100, 1) if baseline_total > 0 else None,
        "baseline_reorder_point": baseline.reorder_point,
        "scenario_reorder_point": scenario.reorder_point,
    })


@mcp.tool()
def check_demand_exception(sku_id: str, threshold_pct: float = 15.0) -> str:
    """
    Checks whether a SKU's recent actual demand deviated significantly from
    the statistical forecast — the trigger for an S&OP exception review.
    """
    return json.dumps(get_engine().detect_exceptions(sku_id, threshold_pct=threshold_pct))


@mcp.tool()
def search_planning_notes(query: str, category_filter: Optional[str] = None) -> str:
    """
    Searches planning policy documents and historical S&OP meeting notes.
    category_filter can be "policy" or "meeting_notes". Any number found in
    a retrieved document should be attributed to that document, not treated
    as a live forecast figure.
    """
    docs = get_kb().search(query, k=3, category_filter=category_filter)
    return json.dumps([
        {"title": d.metadata["title"], "category": d.metadata["category"], "text": d.text}
        for d in docs
    ])


if __name__ == "__main__":
    mcp.run(transport="stdio")
