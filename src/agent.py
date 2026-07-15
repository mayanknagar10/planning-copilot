"""
agent.py — the LLM reasoning layer of PlanningCopilot.

CORE DESIGN PRINCIPLE (read this before touching this file):
The agent NEVER computes a forecast number itself. Every numeric answer comes
from calling a tool that wraps forecast_engine.DemandForecastEngine, which
runs the real Prophet model. The LLM's job is strictly to:
  1. Decide which tool to call based on the planner's question
  2. Narrate the tool's output in plain language
  3. Flag its own uncertainty when it isn't sure what the planner is asking

This separation is what makes the system trustworthy for planning decisions —
an LLM "estimating" a demand number would be actively dangerous in this
context, since a wrong number silently propagates into a real inventory order.

Provider setup: primary model via Groq (free, open-weight Llama models served
on Groq's LPU hardware — fast, generous rate limits, no credit card), automatic
fallback to Google Gemini if Groq is rate-limited or unavailable. Both are
LangChain-native chat models, so swapping either is a one-line change.
"""

import os
import json
from typing import Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ — needed when agent.py is run/imported standalone

from forecast_engine import DemandForecastEngine
from knowledge_base import PlanningKnowledgeBase

# ── LLM setup: primary + fallback ────────────────────────────────────────────

def build_primary_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY") or "not-set",
        temperature=0.1,  # low temperature: this is a planning tool, not creative writing
    )


def build_fallback_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        # "gemini-flash-latest" is Google's auto-updating alias — it always
        # points to their current-generation Flash model, hot-swapped by
        # Google on every release. We use it deliberately instead of a
        # pinned version string (e.g. "gemini-2.5-flash") after hitting a
        # real 404 in testing: Google retired that pinned model for new
        # users within months of this project being built. A fallback
        # provider that itself needs manual updates defeats the point of
        # having a fallback — the alias avoids that failure mode going
        # forward. Trade-off: Google could change behavior/pricing under
        # you between releases; pin to a specific version instead if you
        # need that guarantee for a production system.
        model="gemini-flash-latest",
        google_api_key=os.environ.get("GOOGLE_API_KEY") or "not-set",
        temperature=0.1,
    )


# NOTE on why fallback is handled manually, not via LangChain's
# `.with_fallbacks()`: that mechanism wraps a chat model at the raw-LLM
# level, but `create_react_agent` needs to call `.bind_tools()` on the model
# to enable tool calling — and `RunnableWithFallbacks` doesn't reliably
# propagate `.bind_tools()` through to the wrapped fallback model. In
# practice this means the fallback silently doesn't get tools bound, so a
# rate-limited primary provider surfaces its original error instead of
# failing over. Building two complete agents and catching the failure at
# the *agent invocation* level (see invoke_agent_with_fallback below)
# sidesteps this entirely and is the reliable pattern for agent+fallback.


# ── Tools: the ONLY way the agent can produce a number ───────────────────────
# Each tool is a thin wrapper around DemandForecastEngine. The agent calls
# these; it does not see or touch Prophet directly, and it cannot bypass them.

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
        # embedding_mode="default" uses Chroma's built-in local embedding model
        # (free, one-time ~90MB download, then fully offline). See
        # knowledge_base.py docstring for the "tfidf" testing fallback.
        _kb = PlanningKnowledgeBase(persist_dir="./chroma_db", embedding_mode="default")
        _kb.build()  # no-ops if already ingested
    return _kb


@tool
def list_available_skus() -> str:
    """Returns the list of SKU IDs available for forecasting."""
    return json.dumps(get_engine().list_skus())


@tool
def get_forecast(sku_id: str, horizon_days: int = 30) -> str:
    """
    Get the baseline demand forecast for a SKU over the next `horizon_days`.
    Returns forecast totals, backtested accuracy (MAPE/WAPE), safety stock,
    and reorder point. Use this for any question about expected future demand,
    inventory recommendations, or forecast accuracy.
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
        "model_params_used": result.model_params,
    })


@tool
def run_what_if_scenario(sku_id: str, horizon_days: int = 30, extra_promo_days: int = 0,
                          lead_time_days: int = 14) -> str:
    """
    Re-runs the REAL forecasting model under a hypothetical scenario. Use this
    whenever a planner asks a "what if" question — e.g. "what if we run a promo
    for the first two weeks" (extra_promo_days=14) or "what if supplier lead
    time grows to 21 days" (lead_time_days=21). NEVER estimate a what-if answer
    yourself — always call this tool, since it reruns the actual model.
    """
    baseline = get_engine().forecast(sku_id, horizon_days=horizon_days)
    scenario = get_engine().run_scenario(
        sku_id, horizon_days=horizon_days,
        extra_promo_days=extra_promo_days, lead_time_override=lead_time_days,
    )
    return json.dumps({
        "sku_id": sku_id,
        "scenario_params": {"extra_promo_days": extra_promo_days, "lead_time_days": lead_time_days},
        "baseline_total_demand": round(float(baseline.forecast_df["yhat"].sum()), 1),
        "scenario_total_demand": round(float(scenario.forecast_df["yhat"].sum()), 1),
        "demand_delta_pct": round(
            (scenario.forecast_df["yhat"].sum() / baseline.forecast_df["yhat"].sum() - 1) * 100, 1
        ) if baseline.forecast_df["yhat"].sum() > 0 else None,
        "baseline_safety_stock": baseline.safety_stock,
        "scenario_safety_stock": scenario.safety_stock,
        "baseline_reorder_point": baseline.reorder_point,
        "scenario_reorder_point": scenario.reorder_point,
    })


@tool
def check_demand_exception(sku_id: str, threshold_pct: float = 15.0) -> str:
    """
    Checks whether a SKU's recent actual demand (last 14 days) deviated
    significantly from what the model would have forecasted. Use this to
    answer questions like "any exceptions I should know about for SKU X"
    or when drafting an S&OP exception summary.
    """
    result = get_engine().detect_exceptions(sku_id, threshold_pct=threshold_pct)
    return json.dumps(result)


@tool
def search_planning_notes(query: str, category_filter: Optional[str] = None) -> str:
    """
    Searches planning policy documents and historical S&OP meeting notes for
    institutional context. Use this when a planner asks about PAST decisions,
    precedent, or POLICY — e.g. "have we seen this deviation before," "what's
    our safety stock policy for seasonal items," or "any known issues with
    this supplier." Optionally filter category_filter to "policy" or
    "meeting_notes". Always attribute any number found in a retrieved
    document to that document explicitly (e.g. "per company policy...") —
    never present it as a live Prophet-computed forecast figure.
    """
    docs = get_kb().search(query, k=3, category_filter=category_filter)
    return json.dumps([
        {"title": d.metadata["title"], "category": d.metadata["category"], "text": d.text}
        for d in docs
    ])


TOOLS = [list_available_skus, get_forecast, run_what_if_scenario, check_demand_exception, search_planning_notes]


def extract_message_text(message) -> str:
    """
    Normalizes a LangChain message's `.content` into plain text. Most
    providers (Groq/Llama) return a plain string, but Gemini returns a
    list of content blocks (e.g. `[{"type": "text", "text": "...", "extras":
    {"signature": "..."}}]`) — without this, the raw block list (signature
    and all) gets displayed to the planner verbatim.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


SYSTEM_PROMPT = """You are PlanningCopilot, an AI assistant for supply chain planners.

Your job is to help planners understand demand forecasts, run what-if scenarios,
and prepare for S&OP (sales & operations planning) meetings.

CRITICAL RULES YOU MUST FOLLOW:
1. You NEVER compute or estimate a demand number, forecast, safety stock level,
   or reorder point yourself. Every number in your response MUST come from a
   tool call. If you don't have a tool result for a number, say you don't know
   rather than guessing.
2. For any "what if" question, you MUST call run_what_if_scenario — never
   reason about hypothetical demand changes without re-running the model.
3. When you present numbers, always mention the forecast accuracy (MAPE/WAPE)
   from the tool output so the planner knows how much to trust the figure.
4. If a planner's question is ambiguous (e.g. they say "the beverage SKU" when
   there are two), ask which SKU they mean rather than guessing — call
   list_available_skus first if you need to check.
5. Be concise. Planners are busy — lead with the number and the recommendation,
   then briefly explain the driver.
6. When drafting an S&OP exception narrative, be specific about the deviation
   percentage and whether it's above or below forecast, using check_demand_exception.
7. When a planner asks about past decisions, precedent, or company policy,
   use search_planning_notes. Never blend a retrieved policy/notes number
   with a current Prophet-computed forecast number — they come from
   different tools and must stay clearly attributed in your answer.

Your tone is that of a sharp, no-nonsense planning analyst — clear, numbers-first,
never salesy or overly enthusiastic."""


def build_agent(use_fallback: bool = False):
    """
    Builds a single agent on one provider. Kept for callers that only need
    one provider (e.g. simple scripts). For the reliable primary→fallback
    behavior, use invoke_agent_with_fallback() instead — see note above.
    """
    llm = build_fallback_llm() if use_fallback else build_primary_llm()
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)


def invoke_agent_with_fallback(question: str) -> dict:
    """
    Tries the primary provider (Groq) first; if it raises for any reason
    (rate limit, timeout, auth error), transparently retries on the fallback
    provider (Gemini) with a fresh agent. Returns a dict with the full
    LangGraph result (so app.py can still extract the tool-call trace) plus
    which provider actually answered, for UI transparency.
    """
    messages = {"messages": [HumanMessage(content=question)]}

    try:
        primary_agent = create_react_agent(build_primary_llm(), TOOLS, prompt=SYSTEM_PROMPT)
        result = primary_agent.invoke(messages)
        return {"result": result, "provider": "Groq (Llama 3.3 70B)", "fell_back": False}
    except Exception as primary_error:
        try:
            fallback_agent = create_react_agent(build_fallback_llm(), TOOLS, prompt=SYSTEM_PROMPT)
            result = fallback_agent.invoke(messages)
            return {"result": result, "provider": "Google Gemini 2.5 Flash (fallback)", "fell_back": True}
        except Exception as fallback_error:
            raise RuntimeError(
                f"Both LLM providers failed.\n"
                f"Primary (Groq) error: {primary_error}\n"
                f"Fallback (Gemini) error: {fallback_error}\n"
                f"Check both GROQ_API_KEY and GOOGLE_API_KEY in .env."
            )


def stream_agent_with_fallback(question: str):
    """
    Generator version of invoke_agent_with_fallback(), for token-by-token
    rendering (e.g. st.write_stream in app.py). Yields:
      ("token", str)   — an answer text delta, as it's generated
      ("fell_back", str)  — emitted once, if the primary provider raised
      ("done", {"provider": str, "tool_calls": [...], "usage": dict|None})
      ("error", str)   — emitted only if BOTH providers fail

    Streaming failures are handled the same way as invoke_agent_with_fallback:
    if the primary provider raises (almost always immediately — an auth or
    rate-limit error surfaces before any token is streamed, not mid-answer),
    we fall back to a fresh agent on Gemini and stream from that instead.
    """
    messages = {"messages": [HumanMessage(content=question)]}

    def run(llm, label):
        agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
        tool_calls = []
        usage = None
        for chunk, _metadata in agent.stream(messages, stream_mode="messages"):
            if isinstance(chunk, ToolMessage):
                tool_calls.append({"tool": chunk.name, "output": chunk.content})
                continue
            text = extract_message_text(chunk)
            if text:
                yield ("token", text)
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
        yield ("done", {"provider": label, "tool_calls": tool_calls, "usage": usage})

    try:
        for event in run(build_primary_llm(), "Groq (Llama 3.3 70B)"):
            yield event
    except Exception as primary_error:
        yield ("fell_back", str(primary_error))
        try:
            for event in run(build_fallback_llm(), "Google Gemini 2.5 Flash (fallback)"):
                yield event
        except Exception as fallback_error:
            yield ("error", (
                f"Both LLM providers failed.\n"
                f"Primary (Groq) error: {primary_error}\n"
                f"Fallback (Gemini) error: {fallback_error}\n"
                f"Check both GROQ_API_KEY and GOOGLE_API_KEY in .env."
            ))


def ask(question: str) -> str:
    """Convenience function for one-off questions — uses the primary→fallback chain."""
    response = invoke_agent_with_fallback(question)
    return extract_message_text(response["result"]["messages"][-1])


if __name__ == "__main__":
    # Quick smoke test — requires GROQ_API_KEY (and optionally GOOGLE_API_KEY) in env
    test_questions = [
        "What SKUs are available?",
        "What's the demand forecast for SKU-1003 over the next 30 days, and how confident should I be in it?",
        "What if we ran a promo on SKU-1003 for the first 2 weeks — how would that change the reorder point?",
        "Any demand exceptions I should flag for SKU-1007 in this week's S&OP meeting?",
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("-" * 80)
