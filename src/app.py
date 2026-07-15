"""
app.py — Streamlit dashboard for PlanningCopilot.

Four tabs:
  1. Portfolio Overview — fixed-horizon snapshot across all SKUs: KPI tiles,
     a category-level demand rollup, and a per-SKU status table. Pure numbers
     from forecast_engine, no LLM.
  2. Forecast Explorer — Single SKU deep dive, or Compare SKUs side by side.
     Pure numbers from forecast_engine, no LLM.
  3. S&OP Consensus — Sales/Marketing/Product submit baseline adjustments with
     rationale, reconciled into a signed-off consensus number (PRD Section 9,
     FR-5/FR-6). Backed by consensus.py; pure bookkeeping, no LLM.
  4. Planning Assistant — chat with the agent for explanations, what-if
     scenarios, and exception narratives. Every number the agent states here
     traces back to a tool call, shown in an expandable "trace" for transparency.
"""

import os
import sys
import json
import time
from datetime import date
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ — must run before checking API keys below

sys.path.insert(0, str(Path(__file__).parent))

from forecast_engine import DemandForecastEngine
from consensus import ConsensusStore, FUNCTIONS

st.set_page_config(page_title="PlanningCopilot", page_icon="📦", layout="wide")

DATA_PATH = str(Path(__file__).parent.parent / "data" / "demand_history.csv")
SOP_DATA_DIR = str(Path(__file__).parent.parent / "data")

# ── Validated color palette (fixed categorical order — color follows the
# entity's identity, never the current selection/filter) ───────────────────
CATEGORICAL_PALETTE = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
ACTUAL_LINE_COLOR = "#52514e"  # neutral ink — "actual" is a role, not an entity


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


@st.cache_resource
def get_engine():
    return DemandForecastEngine(DATA_PATH)


@st.cache_resource
def get_agent_module():
    # Imported lazily so the Forecast Explorer tab works even without API keys set
    import agent as agent_module
    return agent_module


@st.cache_resource
def get_consensus_store():
    return ConsensusStore(data_dir=SOP_DATA_DIR)


@st.cache_resource
def get_sku_category_map(_engine: DemandForecastEngine) -> dict:
    return _engine.df[["sku_id", "category"]].drop_duplicates().set_index("sku_id")["category"].to_dict()


@st.cache_resource
def get_category_color_map(_engine: DemandForecastEngine) -> dict:
    categories = sorted(_engine.df["category"].unique())
    return {cat: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, cat in enumerate(categories)}


@st.cache_resource
def get_sku_dash_map(_engine: DemandForecastEngine) -> dict:
    """Two SKUs share a category's color; dash style tells them apart, fixed
    by sorted SKU id — never by current selection order."""
    sku_cat = get_sku_category_map(_engine)
    dash_map = {}
    for cat in set(sku_cat.values()):
        skus_in_cat = sorted(s for s, c in sku_cat.items() if c == cat)
        for i, sku in enumerate(skus_in_cat):
            dash_map[sku] = "solid" if i == 0 else "dash"
    return dash_map


OVERVIEW_HORIZON_DAYS = 30
OVERVIEW_LEAD_TIME_DAYS = 14


@st.cache_data(show_spinner="Computing portfolio snapshot (10 SKUs, ~20 Prophet fits)...")
def compute_portfolio_snapshot(_engine: DemandForecastEngine) -> pd.DataFrame:
    sku_cat = get_sku_category_map(_engine)
    rows = []
    for sku in _engine.list_skus():
        fc = _engine.forecast(sku, horizon_days=OVERVIEW_HORIZON_DAYS, lead_time_days=OVERVIEW_LEAD_TIME_DAYS)
        exc = _engine.detect_exceptions(sku)
        rows.append({
            "sku_id": sku,
            "category": sku_cat[sku],
            "total_forecast_demand": round(float(fc.forecast_df["yhat"].sum()), 1),
            "wape": fc.wape,
            "safety_stock": fc.safety_stock,
            "reorder_point": fc.reorder_point,
            "is_exception": exc["is_exception"],
            "deviation_pct": exc["deviation_pct"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=32)
def get_forecast_cached(_engine: DemandForecastEngine, sku_id: str, horizon_days: int, lead_time_days: int):
    return _engine.forecast(sku_id, horizon_days=horizon_days, lead_time_days=lead_time_days)


engine = get_engine()
sku_list = engine.list_skus()
category_color_map = get_category_color_map(engine)
sku_category_map = get_sku_category_map(engine)
sku_dash_map = get_sku_dash_map(engine)

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📦 PlanningCopilot")
    st.caption("AI-augmented demand forecasting & inventory planning")

    with st.container(border=True):
        st.markdown(
            "**Design principle:** the forecast number always comes from Prophet. "
            "The AI layer only explains, reasons about scenarios, and drafts narratives — "
            "it never estimates a number itself."
        )

    st.markdown("#### System status")
    with st.container(border=True):
        groq_set = bool(os.environ.get("GROQ_API_KEY"))
        google_set = bool(os.environ.get("GOOGLE_API_KEY"))
        st.markdown(f"{'🟢' if groq_set else '⚪'} Groq · Llama 3.3 70B (primary)")
        st.markdown(f"{'🟢' if google_set else '⚪'} Google Gemini (fallback)")
        last_provider = st.session_state.get("last_provider")
        if last_provider:
            st.caption(f"Last answered by: {last_provider}")

tab_overview, tab_explorer, tab_consensus, tab_chat = st.tabs(
    ["🧭 Portfolio Overview", "📊 Forecast Explorer", "🤝 S&OP Consensus", "💬 Planning Assistant"]
)

# ── Tab 1: Portfolio Overview ───────────────────────────────────────────────

with tab_overview:
    st.caption(
        f"Fixed snapshot at a {OVERVIEW_HORIZON_DAYS}-day horizon and a "
        f"{OVERVIEW_LEAD_TIME_DAYS}-day lead time, computed once per session — "
        "adjust horizon/lead time per SKU in Forecast Explorer."
    )

    snapshot = compute_portfolio_snapshot(engine)

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        with st.container(border=True):
            st.metric("SKUs tracked", len(snapshot))
    with kpi_cols[1]:
        with st.container(border=True):
            st.metric("Open exceptions", int(snapshot["is_exception"].sum()))
    with kpi_cols[2]:
        with st.container(border=True):
            st.metric("Portfolio avg WAPE", f"{snapshot['wape'].mean():.1f}%")
    with kpi_cols[3]:
        with st.container(border=True):
            st.metric("Total reorder units", f"{snapshot['reorder_point'].sum():,.0f}")

    col_chart, col_table = st.columns([2, 3])

    with col_chart:
        by_category = snapshot.groupby("category")["total_forecast_demand"].sum().reset_index()
        by_category = by_category.sort_values("category")
        fig = go.Figure(go.Bar(
            x=by_category["category"],
            y=by_category["total_forecast_demand"],
            marker_color=[category_color_map[c] for c in by_category["category"]],
        ))
        fig.update_layout(
            title=f"{OVERVIEW_HORIZON_DAYS}-day forecasted demand by category",
            xaxis_title="Category", yaxis_title="Units", height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        display_df = snapshot.copy()
        display_df["status"] = display_df["is_exception"].map({True: "⚠️ Exception", False: "✅ On track"})
        st.dataframe(
            display_df[["sku_id", "category", "total_forecast_demand", "wape",
                        "safety_stock", "reorder_point", "status"]],
            column_config={
                "sku_id": st.column_config.TextColumn("SKU"),
                "category": st.column_config.TextColumn("Category"),
                "total_forecast_demand": st.column_config.NumberColumn(f"{OVERVIEW_HORIZON_DAYS}d Forecast", format="%.0f"),
                "wape": st.column_config.ProgressColumn("WAPE", min_value=0, max_value=100, format="%.1f%%"),
                "safety_stock": st.column_config.NumberColumn("Safety Stock", format="%.0f"),
                "reorder_point": st.column_config.NumberColumn("Reorder Point", format="%.0f"),
                "status": st.column_config.TextColumn("Status"),
            },
            hide_index=True,
            use_container_width=True,
            height=420,
        )

# ── Tab 2: Forecast Explorer ────────────────────────────────────────────────

with tab_explorer:
    mode = st.radio("View", ["Single SKU", "Compare SKUs"], horizontal=True, key="explorer_mode")

    if mode == "Single SKU":
        col_a, col_b = st.columns([1, 3])

        with col_a:
            selected_sku = st.selectbox("Select SKU", sku_list, key="single_sku")
            horizon = st.slider("Forecast horizon (days)", 7, 90, 30, key="single_horizon")
            lead_time = st.slider("Supplier lead time (days)", 3, 30, 14, key="single_leadtime")

            result = get_forecast_cached(engine, selected_sku, horizon, lead_time)

            st.metric("Backtested WAPE", f"{result.wape}%",
                       help="Weighted Absolute Percentage Error on a 28-day holdout window — lower is better.")
            st.metric("Backtested MAPE", f"{result.mape}%")
            st.metric("Recommended safety stock", f"{result.safety_stock:,.0f} units")
            st.metric("Recommended reorder point", f"{result.reorder_point:,.0f} units")

        with col_b:
            hist = result.history_df.tail(90)
            fc = result.forecast_df
            sku_color = category_color_map[sku_category_map[selected_sku]]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["date"], y=hist["demand"], mode="lines",
                name="Actual (last 90 days)", line=dict(color=ACTUAL_LINE_COLOR, width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat"], mode="lines",
                name="Forecast", line=dict(color=sku_color, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=list(fc["ds"]) + list(fc["ds"][::-1]),
                y=list(fc["yhat_upper"]) + list(fc["yhat_lower"][::-1]),
                fill="toself", fillcolor=hex_to_rgba(sku_color, 0.15),
                line=dict(color="rgba(255,255,255,0)"),
                name="95% confidence interval", showlegend=True,
            ))
            fig.update_layout(
                title=f"Demand forecast — {selected_sku} ({sku_category_map[selected_sku]})",
                xaxis_title="Date", yaxis_title="Units",
                height=480, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Exception check — last 14 days vs. forecast"):
                exc = engine.detect_exceptions(selected_sku)
                if exc["is_exception"]:
                    st.warning(
                        f"⚠️ Actual demand deviated {exc['deviation_pct']}% from forecast "
                        f"({exc['actual_last_14d']} actual vs {exc['predicted_last_14d']} predicted)."
                    )
                else:
                    st.success(
                        f"✅ Demand tracking within {exc['deviation_pct']}% of forecast — no exception flagged."
                    )

            with st.expander("📚 Related planning notes & policy for this SKU"):
                try:
                    from knowledge_base import PlanningKnowledgeBase
                    kb = st.session_state.get("_kb")
                    if kb is None:
                        kb = PlanningKnowledgeBase(
                            persist_dir=str(Path(__file__).parent / "chroma_db"),
                            embedding_mode="default",
                        )
                        kb.build()
                        st.session_state["_kb"] = kb
                    docs = kb.search(f"{selected_sku} demand history decisions policy", k=3)
                    if docs:
                        for d in docs:
                            st.markdown(f"**{d.metadata['title']}** · *{d.metadata['category']}*")
                            st.caption(d.text)
                    else:
                        st.caption("No related notes found.")
                except Exception as e:
                    st.caption(
                        f"Notes search needs the embedding model downloaded on first use "
                        f"(requires internet once). Error: {e}"
                    )

    else:  # Compare SKUs
        col_ctl, col_chart = st.columns([1, 3])

        with col_ctl:
            compare_skus = st.multiselect(
                "Select SKUs to compare", sku_list,
                default=sku_list[:2], max_selections=4, key="compare_skus",
            )
            horizon = st.slider("Forecast horizon (days)", 7, 90, 30, key="compare_horizon")
            lead_time = st.slider("Supplier lead time (days)", 3, 30, 14, key="compare_leadtime")

        with col_chart:
            if not compare_skus:
                st.info("Select at least one SKU to compare.")
            else:
                results = {
                    sku: get_forecast_cached(engine, sku, horizon, lead_time)
                    for sku in compare_skus
                }

                fig = go.Figure()
                for sku, result in results.items():
                    cat = sku_category_map[sku]
                    fc = result.forecast_df
                    fig.add_trace(go.Scatter(
                        x=fc["ds"], y=fc["yhat"], mode="lines",
                        name=f"{sku} ({cat})",
                        line=dict(color=category_color_map[cat], width=2, dash=sku_dash_map[sku]),
                    ))
                fig.update_layout(
                    title=f"{horizon}-day forecast comparison",
                    xaxis_title="Date", yaxis_title="Units",
                    height=420, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                compare_df = pd.DataFrame([{
                    "sku_id": sku,
                    "category": sku_category_map[sku],
                    "total_forecast_demand": round(float(r.forecast_df["yhat"].sum()), 1),
                    "wape": r.wape,
                    "safety_stock": r.safety_stock,
                    "reorder_point": r.reorder_point,
                } for sku, r in results.items()])
                st.dataframe(
                    compare_df,
                    column_config={
                        "sku_id": st.column_config.TextColumn("SKU"),
                        "category": st.column_config.TextColumn("Category"),
                        "total_forecast_demand": st.column_config.NumberColumn(f"{horizon}d Forecast", format="%.0f"),
                        "wape": st.column_config.NumberColumn("WAPE %", format="%.1f"),
                        "safety_stock": st.column_config.NumberColumn("Safety Stock", format="%.0f"),
                        "reorder_point": st.column_config.NumberColumn("Reorder Point", format="%.0f"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

# ── Tab 3: S&OP Consensus ────────────────────────────────────────────────

with tab_consensus:
    st.caption(
        "Sales, Marketing, and Product submit adjustments to the baseline with "
        "documented rationale; the S&OP lead reconciles them into a signed-off "
        "consensus number. Every adjustment and sign-off is persisted for audit."
    )

    store = get_consensus_store()

    col_cycle, col_sku = st.columns([1, 1])
    with col_cycle:
        default_cycle = date.today().strftime("%Y-%m")
        cycle = st.text_input("Forecast cycle", value=default_cycle, key="sop_cycle",
                                help="Free-form label for the monthly forecast cycle being reconciled, e.g. 2026-07.")
    with col_sku:
        consensus_sku = st.selectbox("SKU", sku_list, key="sop_sku")

    baseline_result = get_forecast_cached(engine, consensus_sku, OVERVIEW_HORIZON_DAYS, OVERVIEW_LEAD_TIME_DAYS)
    baseline_total = float(baseline_result.forecast_df["yhat"].sum())

    cr = store.get_consensus(cycle, consensus_sku, baseline_total)

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        with st.container(border=True):
            st.metric(f"Baseline ({OVERVIEW_HORIZON_DAYS}d)", f"{cr.baseline_total:,.0f} units")
    with kpi_cols[1]:
        with st.container(border=True):
            st.metric("Net adjustment", f"{cr.consensus_total - cr.baseline_total:+,.0f} units")
    with kpi_cols[2]:
        with st.container(border=True):
            st.metric("Consensus forecast", f"{cr.consensus_total:,.0f} units")

    if cr.is_signed_off:
        st.success(f"✅ Signed off by **{cr.signed_off_by}** at {cr.signed_off_at} — adjustments are locked for this cycle.")
    else:
        st.info("Not yet signed off — Sales, Marketing, and Product can still submit adjustments.")

    st.markdown("#### Submit an adjustment")
    with st.form("adjustment_form", clear_on_submit=True):
        form_cols = st.columns([1, 1, 3])
        with form_cols[0]:
            function = st.selectbox("Function", FUNCTIONS, key="sop_function")
        with form_cols[1]:
            delta_units = st.number_input("Delta (units)", value=0.0, step=10.0, key="sop_delta")
        with form_cols[2]:
            rationale = st.text_input("Rationale (required)", key="sop_rationale",
                                        placeholder="e.g. Regional account confirmed a bulk reorder")
        submitted = st.form_submit_button("Submit adjustment", disabled=cr.is_signed_off)

        if submitted:
            try:
                store.add_adjustment(cycle, consensus_sku, function, delta_units, rationale)
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    st.markdown("#### Audit trail — adjustments this cycle")
    if cr.adjustments:
        adj_df = pd.DataFrame(cr.adjustments)
        st.dataframe(
            adj_df[["function", "delta_units", "rationale", "submitted_at"]],
            column_config={
                "function": st.column_config.TextColumn("Function"),
                "delta_units": st.column_config.NumberColumn("Delta", format="%+.0f"),
                "rationale": st.column_config.TextColumn("Rationale", width="large"),
                "submitted_at": st.column_config.TextColumn("Submitted at"),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("No adjustments submitted yet for this cycle/SKU.")

    if not cr.is_signed_off:
        st.markdown("#### S&OP sign-off")
        signoff_cols = st.columns([2, 1])
        with signoff_cols[0]:
            signed_off_by = st.text_input("Signing off as", value="S&OP Lead", key="sop_signoff_name")
        with signoff_cols[1]:
            st.write("")
            if st.button("Sign off consensus forecast", type="primary", use_container_width=True):
                store.sign_off(cycle, consensus_sku, signed_off_by=signed_off_by or "S&OP Lead")
                st.rerun()

# ── Tab 4: Planning Assistant (chat) ────────────────────────────────────────

with tab_chat:
    header_col, clear_col = st.columns([5, 1])
    with header_col:
        st.markdown(
            "Ask about forecasts, run what-if scenarios, or request an exception summary. "
            "Every number in the answer traces back to a tool call — expand **Trace** to verify."
        )
    with clear_col:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    api_key_missing = not os.environ.get("GROQ_API_KEY")
    if api_key_missing:
        st.info(
            "Set `GROQ_API_KEY` (and optionally `GOOGLE_API_KEY` for fallback) "
            "as environment variables to enable the chat assistant. See README for setup."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    example_qs = [
        "What's the forecast for SKU-1003 over the next 30 days?",
        "What if we ran a promo on SKU-1003 for 2 weeks — how does the reorder point change?",
        "Any demand exceptions I should flag for SKU-1007?",
        "Have we seen a stockout on SKU-1005 before — what happened?",
    ]
    st.caption("Try: " + " · ".join(f"*{q}*" for q in example_qs))

    if question := st.chat_input("Ask the planning assistant...", disabled=api_key_missing):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            state = {"provider": None, "tool_calls": [], "fell_back": False, "usage": None, "error": None}

            def token_stream():
                agent_module = get_agent_module()
                for event_type, payload in agent_module.stream_agent_with_fallback(question):
                    if event_type == "token":
                        yield payload
                    elif event_type == "done":
                        state["provider"] = payload["provider"]
                        state["tool_calls"] = payload["tool_calls"]
                        state["usage"] = payload.get("usage")
                    elif event_type == "fell_back":
                        state["fell_back"] = True
                    elif event_type == "error":
                        state["error"] = payload

            start_time = time.perf_counter()
            try:
                answer = st.write_stream(token_stream())
            except Exception as e:
                answer = f"Error calling the LLM provider: {e}\n\nCheck your API keys in .env."
                st.markdown(answer)
            latency_s = time.perf_counter() - start_time

            if state["error"]:
                answer = state["error"]
                st.error(answer)

            provider_used = state["provider"]
            if provider_used:
                st.session_state["last_provider"] = provider_used

            if state["fell_back"] and provider_used:
                st.toast(f"Primary provider rate-limited — answered via {provider_used}", icon="⚡")
                st.warning(f"⚡ Primary provider was rate-limited — answered by the fallback provider: **{provider_used}**.")

            meta_bits = []
            if provider_used:
                meta_bits.append(f"⚙️ {provider_used}")
            meta_bits.append(f"⏱️ {latency_s:.1f}s")
            usage = state["usage"] or {}
            if usage.get("total_tokens"):
                meta_bits.append(f"🔢 {usage['total_tokens']} tokens")
            st.caption(" · ".join(meta_bits))

            tool_calls_made = state["tool_calls"]
            if tool_calls_made:
                with st.expander("🔍 Trace — tool calls behind this answer"):
                    for tc in tool_calls_made:
                        with st.container(border=True):
                            st.markdown(f"**`{tc['tool']}`**")
                            try:
                                st.json(json.loads(tc["output"]), expanded=False)
                            except (json.JSONDecodeError, TypeError):
                                st.code(tc["output"])

        st.session_state.messages.append({"role": "assistant", "content": answer})
