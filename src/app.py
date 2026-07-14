"""
app.py — Streamlit dashboard for PlanningCopilot.

Two panels:
  1. Forecast Explorer — pick a SKU, see the chart, safety stock, reorder point,
     and backtested accuracy. Pure numbers from forecast_engine, no LLM.
  2. Planning Assistant — chat with the agent for explanations, what-if
     scenarios, and exception narratives. Every number the agent states here
     traces back to a tool call, shown in an expandable "trace" for transparency.
"""

import os
import sys
import json
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ — must run before checking API keys below

sys.path.insert(0, str(Path(__file__).parent))

from forecast_engine import DemandForecastEngine

st.set_page_config(page_title="PlanningCopilot", page_icon="📦", layout="wide")

DATA_PATH = str(Path(__file__).parent.parent / "data" / "demand_history.csv")


@st.cache_resource
def get_engine():
    return DemandForecastEngine(DATA_PATH)


@st.cache_resource
def get_agent_module():
    # Imported lazily so the Forecast Explorer tab works even without API keys set
    import agent as agent_module
    return agent_module


# ── Sidebar ───────────────────────────────────────────────────────────────

st.sidebar.title("📦 PlanningCopilot")
st.sidebar.caption("AI-augmented demand forecasting & inventory planning")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Design principle:** the forecast number always comes from Prophet. "
    "The AI layer only explains, reasons about scenarios, and drafts narratives — "
    "it never estimates a number itself."
)
st.sidebar.markdown("---")

engine = get_engine()
sku_list = engine.list_skus()

tab1, tab2 = st.tabs(["📊 Forecast Explorer", "💬 Planning Assistant"])

# ── Tab 1: Forecast Explorer ────────────────────────────────────────────────

with tab1:
    col_a, col_b = st.columns([1, 3])

    with col_a:
        selected_sku = st.selectbox("Select SKU", sku_list)
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
        lead_time = st.slider("Supplier lead time (days)", 3, 30, 14)

        with st.spinner("Running Prophet forecast..."):
            result = engine.forecast(selected_sku, horizon_days=horizon, lead_time_days=lead_time)

        st.metric("Backtested WAPE", f"{result.wape}%",
                   help="Weighted Absolute Percentage Error on a 28-day holdout window — lower is better.")
        st.metric("Backtested MAPE", f"{result.mape}%")
        st.metric("Recommended safety stock", f"{result.safety_stock:,.0f} units")
        st.metric("Recommended reorder point", f"{result.reorder_point:,.0f} units")

    with col_b:
        hist = result.history_df.tail(90)
        fc = result.forecast_df

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["demand"], mode="lines",
            name="Actual (last 90 days)", line=dict(color="#4C6EF5", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat"], mode="lines",
            name="Forecast", line=dict(color="#F76707", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=list(fc["ds"]) + list(fc["ds"][::-1]),
            y=list(fc["yhat_upper"]) + list(fc["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(247,112,7,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% confidence interval", showlegend=True,
        ))
        fig.update_layout(
            title=f"Demand forecast — {selected_sku}",
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

# ── Tab 2: Planning Assistant (chat) ────────────────────────────────────────

with tab2:
    st.markdown(
        "Ask about forecasts, run what-if scenarios, or request an exception summary. "
        "Every number in the answer traces back to a tool call — expand **Trace** to verify."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    api_key_missing = not os.environ.get("OPENROUTER_API_KEY")
    if api_key_missing:
        st.info(
            "Set `OPENROUTER_API_KEY` (and optionally `GOOGLE_API_KEY` for fallback) "
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
            with st.spinner("Thinking..."):
                try:
                    agent_module = get_agent_module()
                    agent = agent_module.build_agent()
                    from langchain_core.messages import HumanMessage
                    result = agent.invoke({"messages": [HumanMessage(content=question)]})
                    answer = result["messages"][-1].content

                    # Extract tool calls for the trace panel
                    tool_calls_made = [
                        {"tool": m.name, "output": m.content}
                        for m in result["messages"]
                        if hasattr(m, "type") and m.type == "tool"
                    ]
                except Exception as e:
                    answer = f"Error calling the LLM provider: {e}\n\nCheck your API keys in .env."
                    tool_calls_made = []

                st.markdown(answer)
                if tool_calls_made:
                    with st.expander("🔍 Trace — tool calls behind this answer"):
                        for tc in tool_calls_made:
                            st.code(f"{tc['tool']}() → {tc['output']}", language="json")

        st.session_state.messages.append({"role": "assistant", "content": answer})
