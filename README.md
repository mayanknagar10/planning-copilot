# PlanningCopilot

**AI-augmented demand forecasting & inventory planning — where the AI explains, and the statistics decide.**

PlanningCopilot is a supply chain planning assistant that sits on top of a classical
statistical forecasting engine. A planner can ask it things like *"what's the forecast
for SKU-1003"* or *"what if we ran a promo for two weeks — how does the reorder point
change"* and get back a grounded, numerically accurate answer with a full trace of
how it was computed.

## The core design decision

**The LLM never computes a forecast number. It only explains one.**

Every number that reaches the planner — a forecast total, a safety stock level, a
reorder point, a percentage deviation — is produced by [Prophet](https://facebook.github.io/prophet/),
a well-established statistical forecasting library, running inside `forecast_engine.py`.
The LLM agent (`agent.py`) can only:

1. Decide *which* deterministic tool to call based on the planner's question
2. Narrate that tool's output in plain language
3. Re-run the model with different parameters to answer "what if" questions
4. Say "I don't know" rather than estimate a number it wasn't given by a tool

This matters because a wrong number in a planning tool isn't a cosmetic bug — it
silently propagates into a real inventory order. An LLM that "estimates" a demand
figure when it's unsure is actively dangerous in this context. Separating the
compute layer from the reasoning layer is the single most important engineering
decision in this project, and it's enforced structurally: the agent has no code
path that lets it return a number without a tool call behind it.

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   Streamlit UI       │────────▶│   Forecast Explorer   │
│   (app.py)           │         │   tab — direct calls  │
│                       │         │   to forecast_engine  │
│                       │         │   + notes search       │
│                       │         └──────────────────────┘
│                       │
│                       │         ┌──────────────────────┐
│                       │────────▶│   Planning Assistant  │
└─────────────────────┘         │   tab — chat via      │
                                   │   LangGraph agent      │
                                   └──────────┬───────────┘
┌─────────────────────┐                     │ tool calls only
│  MCP client            │                     │
│  (Claude Desktop, etc) │────────────────────┤
└─────────────────────┘                     │
                                              ▼
                          ┌───────────────────────────────────┐
                          │  5 tools — same source of truth    │
                          │  used by BOTH agent.py (LangChain) │
                          │  and mcp_server.py (MCP protocol)  │
                          ├───────────────────┬─────────────────┤
                          │ forecast_engine.py │ knowledge_base.py│
                          │ (Prophet, no LLM)  │ (ChromaDB vector │
                          │                     │  store, no LLM) │
                          │ • forecast()        │                  │
                          │ • run_scenario()    │ • build()        │
                          │ • detect_exceptions()│ • search()      │
                          └──────────┬─────────┴────────┬────────┘
                                     ▼                    ▼
                          demand_history.csv      chroma_db/
                                              (planning policy docs +
                                               S&OP meeting notes)
```

The agent (`agent.py`) is a [LangGraph](https://langchain-ai.github.io/langgraph/) ReAct
agent, and `mcp_server.py` exposes the identical capability set via the
[Model Context Protocol](https://modelcontextprotocol.io) — both call the exact same
underlying functions, so there is exactly one source of truth for how a number or a
retrieved document is produced, regardless of which client is asking.

| Tool | What it does | Backed by |
|---|---|---|
| `list_available_skus` | Lists SKUs the planner can ask about | `forecast_engine.py` |
| `get_forecast` | Baseline forecast + accuracy + safety stock + reorder point | `forecast_engine.py` |
| `run_what_if_scenario` | Re-runs the model under a hypothetical (promo, lead time) | `forecast_engine.py` |
| `check_demand_exception` | Flags whether recent actuals deviated from forecast | `forecast_engine.py` |
| `search_planning_notes` | Retrieves relevant planning policy docs & S&OP meeting notes | `knowledge_base.py` |

## The vector store — why it exists and what it holds

`search_planning_notes` retrieves from a small corpus of planning policy documents
and past S&OP meeting notes (`knowledge_base.py::PLANNING_DOCUMENTS`) — the kind
of institutional memory a forecast number alone can't answer. "What's our safety
stock policy for seasonal SKUs" or "any known issues with the Electronics
supplier" are genuinely common planner questions that live in text, not in Prophet.

This is deliberately **not** a general-purpose document store — it only holds two
kinds of documents (policy docs and meeting notes), matching a real planning
team's actual knowledge base rather than being RAG-for-its-own-sake.

**Retrieval is backed by [ChromaDB](https://www.trychroma.com)**, using its default
local embedding model (a small ONNX-exported MiniLM model, downloaded once on
first use — roughly 90MB — then fully offline and free from then on). ChromaDB
runs embedded, with no separate server process to manage, so the only setup cost
is `pip install chromadb` plus that one-time model download.

```python
from knowledge_base import PlanningKnowledgeBase

kb = PlanningKnowledgeBase(persist_dir="./chroma_db", embedding_mode="default")
kb.build()
kb.search("what's our safety stock policy for seasonal items", k=3)
```

`knowledge_base.py` also ships a `embedding_mode="tfidf"` fallback (scikit-learn
TF-IDF, no download required) — this exists purely so the retrieval logic can be
tested in network-restricted environments (CI, sandboxes). It is **not** what the
shipped app uses by default; semantic quality is lower than the real embedding
model, so don't demo on it.

## Baseline method comparison — SMA vs. ETS vs. Prophet

Prophet remains the one production forecasting path — it's what `forecast()`,
`run_scenario()`, and every agent/MCP tool actually use. `forecast_engine.py`
also implements two lighter statistical methods, purely for comparison:

- **SMA** (Simple Moving Average) — repeats the trailing 28-day average forward.
  Deliberately naive; it's the floor every other method needs to beat.
- **ETS** (Holt-Winters Exponential Smoothing, via `statsmodels`) — reacts to
  trend and weekly seasonality, but without Prophet's holiday/regressor handling.

`DemandForecastEngine.compare_baselines(sku_id, horizon_days=28)` backtests all
three methods on the same holdout window and returns MAPE/WAPE for each, so
"why Prophet and not something simpler" has evidence behind it rather than
being asserted:

```python
from forecast_engine import DemandForecastEngine

engine = DemandForecastEngine("data/demand_history.csv")
engine.compare_baselines("SKU-1003", horizon_days=28)
# {"sku_id": "SKU-1003", "horizon_days": 28,
#  "methods": {"sma": {...}, "ets": {...}, "prophet": {...}}}
```

This isn't wired into the agent or MCP tools — it's a Phase 1 evaluation utility.

## ML/DL evaluation — `ml_dl_evaluation.py` (Phase 3, evaluation-only)

A separate module answers a different question than the comparison above: does
a *heavier* model (classical ML, or a deep-learning approach) beat the Phase 1
Prophet baseline enough to justify the added complexity — before committing
engineering time to an in-house build, or evaluating a commercial platform?

- **ML candidate** — scikit-learn's `HistGradientBoostingRegressor` on
  lag/calendar features, standing in for XGBoost/LightGBM.
- **"DL" candidate** — scikit-learn's `MLPRegressor` (a small feed-forward net),
  standing in for a proper sequence model (LSTM / Temporal Fusion Transformer /
  DeepAR). This project deliberately avoids adding a PyTorch/TensorFlow
  dependency for an evaluation stub — same free-tier, no-heavy-install
  philosophy as the LLM provider choices above. If this phase clears its
  decision gate, swap `evaluate_dl()`'s model for a real sequence model; the
  feature-engineering/scoring harness doesn't need to change.

Both are lag-feature regressors, not sequence models — this is a lightweight
comparison, not a production DL pipeline.

```python
from forecast_engine import DemandForecastEngine
from ml_dl_evaluation import compare_to_baseline

engine = DemandForecastEngine("data/demand_history.csv")
compare_to_baseline("SKU-1003", engine, horizon_days=28)
# {"sku_id": ..., "prophet_baseline": {...}, "ml": {...}, "dl": {...}}
```

Like the SMA/ETS comparison, this module is **not** called by `agent.py`,
`mcp_server.py`, or `app.py` — it exists purely to generate the accuracy-lift
evidence a real Phase 3 decision needs.

### Reference — existing commercial forecasting platforms

Before spending engineering time on a heavier in-house model, it's worth
knowing what commercial demand-planning platforms already offer at enterprise
scale — these are reference points for the Phase 3 build-vs-buy decision, not
things this project tries to replicate:

- **[Kinaxis](https://www.kinaxis.com)** (RapidResponse) — concurrent planning
  with scenario simulation.
- **[Blue Yonder](https://blueyonder.com)** — end-to-end supply chain platform
  with ML-based demand forecasting.
- **[o9 Solutions](https://o9solutions.com)** — AI-driven integrated business
  planning on a graph-based data model.
- **[Anaplan](https://www.anaplan.com)** — connected planning platform, often
  used for the S&OP consensus/collaboration layer itself.

## S&OP consensus workflow — `consensus.py`

The PRD's Section 9 process (baseline → function adjustments → S&OP sign-off →
consensus forecast) is implemented as its own deterministic, no-LLM module —
same separation-of-concerns rule as everything else here. `ConsensusStore`
persists two things to CSV under `data/`, so the audit trail survives app
restarts:

- **Adjustments** (`sop_adjustments.csv`) — one row per Sales/Marketing/Product
  adjustment to the baseline, each with a **required** rationale. An adjustment
  with no rationale, or from an unrecognized function, is rejected outright.
- **Sign-offs** (`sop_signoffs.csv`) — marks a forecast cycle + SKU as final.
  Once signed off, further adjustments to that cycle/SKU are rejected — the
  number is locked until the next monthly cycle.

`ConsensusStore.get_consensus(cycle, sku_id, baseline_total)` sums the recorded
adjustments against the baseline to produce the consensus total:

```python
from consensus import ConsensusStore

store = ConsensusStore(data_dir="data")
store.add_adjustment("2026-07", "SKU-1003", "Sales", 120, "Confirmed bulk reorder.")
store.get_consensus("2026-07", "SKU-1003", baseline_total=1000.0)
# ConsensusResult(baseline_total=1000.0, consensus_total=1120.0, is_signed_off=False, ...)
```

The Streamlit app's **🤝 S&OP Consensus** tab is a thin UI over this same store —
pick a cycle and SKU, submit an adjustment with rationale, and sign off once
the number is agreed. Nothing here calls the LLM or the agent; it's pure
bookkeeping, matching FR-5/FR-6/FR-8.

## MCP server — query PlanningCopilot from Claude Desktop directly

`mcp_server.py` exposes all 5 tools over the Model Context Protocol, so you can
talk to PlanningCopilot from Claude Desktop with no Streamlit app running at all.

Add this to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "planning-copilot": {
      "command": "python3",
      "args": ["/absolute/path/to/planning-copilot/src/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop, and you can ask it things like *"using planning-copilot,
what's the reorder point for SKU-1003"* directly in a normal Claude conversation.

Run it standalone to verify it works before wiring it into Claude Desktop:
```bash
cd src && python3 mcp_server.py
```

## LLM provider setup — free tier, with automatic fallback

This project runs entirely on free-tier LLM APIs — no cost to build or demo.

**Primary: [Groq](https://console.groq.com/keys)** — one API key, routes to
`llama-3.3-70b-versatile`. Groq serves open-weight models on custom LPU
hardware rather than GPUs, so responses come back dramatically faster than
typical free-tier LLM APIs; its free tier is a persistent rate-limited
allowance (not a trial that expires), and it supports tool calling.

**Fallback: [Google AI Studio (Gemini)](https://ai.google.dev)** — kicks in
automatically if Groq is rate-limited or unavailable. This isn't just a
workaround for free-tier limits — it's the same primary/fallback LLM-gateway
pattern used in production systems, and it's demoed live in this project
rather than just described.

**Implementation note — worth reading if you're extending this file.**
Fallback is handled by trying a complete primary agent, catching any
exception, and retrying on a complete fallback agent (see
`invoke_agent_with_fallback()` in `agent.py`) — not LangChain's built-in
`.with_fallbacks()`. That method wraps a raw chat model, but `create_react_agent`
needs to call `.bind_tools()` on the model to enable tool calling, and
`RunnableWithFallbacks` doesn't reliably propagate `.bind_tools()` through to
the wrapped fallback model. In practice that means the fallback silently
doesn't get tools bound, so a rate-limited primary surfaces its original
error instead of failing over — exactly the bug this project hit during
testing. Building two complete agents and catching the failure at the agent
invocation level sidesteps the issue entirely. The same primary→fallback
pattern is used again for streaming — see `stream_agent_with_fallback()`,
which yields answer text token-by-token instead of blocking for the full
response, falling back to Gemini mid-generator if Groq raises before
streaming any tokens (the common case: auth/rate-limit errors surface
immediately, before the first chunk).

```python
def invoke_agent_with_fallback(question: str) -> dict:
    try:
        primary_agent = create_react_agent(build_primary_llm(), TOOLS, prompt=SYSTEM_PROMPT)
        return {"result": primary_agent.invoke(...), "provider": "Groq", "fell_back": False}
    except Exception:
        fallback_agent = create_react_agent(build_fallback_llm(), TOOLS, prompt=SYSTEM_PROMPT)
        return {"result": fallback_agent.invoke(...), "provider": "Gemini", "fell_back": True}
```

The Streamlit chat tab streams the answer token-by-token and surfaces which
provider actually answered (plus response latency and token usage) whenever a
fallback occurs, so the switch is visible rather than silent — useful both
for debugging and for demonstrating the resilience pattern live.

**Second implementation note — model string pinning.** `build_fallback_llm()`
uses `"gemini-flash-latest"` rather than a specific pinned version like
`"gemini-2.5-flash"`. This is a deliberate choice made after hitting a real
404 during testing: Google retired that pinned model for new users within
months of this project being built. Google publishes auto-updating aliases
(`-latest`) specifically for this reason — they get hot-swapped to Google's
current-generation model on every release, so a *fallback* provider doesn't
itself need manual maintenance to keep working. The trade-off is that
behavior and pricing can shift under you between Google's releases; pin to
an exact version string instead if you need that guarantee for a production
system. For a portfolio/demo project prioritizing "it keeps working without
me touching it," the alias is the better choice.

Neither provider requires a credit card. Get your keys at the links above, then
create a `.env` file in the project root with:

```bash
GROQ_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here   # optional — only needed for the fallback path
```

**Important:** `.env` is already listed in `.gitignore` (along with
`.env.example`, should you choose to add one), so your real keys never get
pushed to GitHub. Also note that `.env` needs `load_dotenv()` to actually be
read into the environment — this project calls it at the top of `app.py`,
`agent.py`, and `mcp_server.py`, so you don't need to do anything extra beyond
creating the file.

## Getting started

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd planning-copilot
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows (cmd/PowerShell)
pip install -r requirements.txt

# 2. Generate the synthetic demand dataset
cd data && python3 generate_data.py && cd ..

# 3. Set up API keys (see above) — create a .env file in the project root
#    with GROQ_API_KEY and (optionally) GOOGLE_API_KEY

# 4. Run the test suite (no API keys needed — tests only the deterministic core)
pytest tests/ -v

# 5. Launch the app
cd src && streamlit run app.py

# Optional: run the MCP server standalone to verify it, or wire it into Claude Desktop
cd src && python3 mcp_server.py
```

On Windows, use `python` instead of `python3` if your Python install doesn't
expose a `python3` alias.

The **Forecast Explorer** tab works immediately with no API keys — it only calls
`forecast_engine.py` directly. The related-notes panel and the **Planning Assistant**
chat tab both use `knowledge_base.py`, which needs one internet connection on first
use to download the local embedding model (~90MB, one-time only) — after that it
runs fully offline. The chat tab additionally needs `GROQ_API_KEY` set for the
LLM narration layer. The **MCP server** needs no API keys to run itself — it's the
LLM client (Claude Desktop, etc.) connecting to it that provides the reasoning layer.

## About the data

This project ships with a **synthetic-but-realistic** multi-SKU demand generator
(`data/generate_data.py`) rather than a downloaded dataset, so the whole pipeline
runs standalone with no external data dependency. The synthetic data includes:

- 10 SKUs across 4 categories (Beverages, Snacks, Household, Electronics, Seasonal)
- 4 years of daily history with realistic trend, weekly seasonality (weekend lifts),
  yearly seasonality (holiday season demand), promotional spikes, and occasional
  stockout dips
- The same schema (`date, sku_id, category, demand, on_promotion, price`) as a
  typical POS/ERP extract, so it's a one-line swap to point `forecast_engine.py`
  at a real dataset instead — for example
  [Kaggle's M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
  (Walmart) or [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI).

If you swap in a real dataset, the forecasting accuracy will be genuinely
meaningful rather than a property of how the synthetic generator was tuned —
worth doing before quoting a MAPE/WAPE number in an interview.

## What each file does

```
planning-copilot/
├── data/
│   ├── generate_data.py      # synthetic demand data generator
│   ├── demand_history.csv    # generated output (run generate_data.py first)
│   ├── sop_adjustments.csv   # S&OP audit trail — created on first adjustment submitted
│   └── sop_signoffs.csv      # S&OP audit trail — created on first sign-off
├── src/
│   ├── forecast_engine.py    # deterministic core — Prophet (+ SMA/ETS comparison), no LLM, fully unit-tested
│   ├── ml_dl_evaluation.py   # Phase 3 ML/DL evaluation (evaluation-only, not agent-wired)
│   ├── knowledge_base.py     # ChromaDB vector store over planning docs — no LLM
│   ├── consensus.py           # S&OP adjustments + sign-off bookkeeping — no LLM
│   ├── agent.py                # LangGraph agent — tool-calling only, narrates results
│   ├── mcp_server.py          # exposes the same 5 tools via MCP for Claude Desktop etc.
│   ├── app.py                  # Streamlit dashboard — Overview, Explorer, S&OP Consensus, chat
│   └── chroma_db/              # persisted vector index (auto-created on first run)
├── tests/
│   ├── test_forecast_engine.py  # unit tests for the deterministic core (incl. SMA/ETS comparison)
│   ├── test_ml_dl_evaluation.py # unit tests for the Phase 3 ML/DL evaluation module
│   ├── test_consensus.py        # unit tests for the S&OP adjustments/sign-off workflow
│   └── test_knowledge_base.py   # unit tests for retrieval (uses tfidf mode, no download)
├── requirements.txt
├── .env                         # you create this — see "LLM provider setup" above
└── README.md
```

## Design notes worth knowing for a walkthrough

- **WAPE over MAPE as the primary accuracy metric.** MAPE is unstable near
  low/zero-demand days, which is common in real SKU-level data. WAPE (weighted
  absolute percentage error) is more robust and is the metric supply chain
  planners actually use day to day.
- **Safety stock and reorder point use standard formulas** (`z * σ * √(lead_time)`
  for safety stock; `avg_daily_demand * lead_time + safety_stock` for reorder
  point), not anything LLM-derived — see `forecast_engine.py::forecast()`.
- **The agent's system prompt explicitly forbids number estimation** and requires
  it to state forecast accuracy (MAPE/WAPE) alongside any number it reports, so
  a planner always knows how much to trust the figure.
- **Retrieval and forecasting stay cleanly separated.** `search_planning_notes`
  returns policy/meeting-note text; the agent's system prompt explicitly requires
  attributing any number found there to its source document rather than blending
  it with a live Prophet-computed forecast figure.
- **The vector store uses ChromaDB's local embedding model by default**, with a
  documented TF-IDF fallback used only for testing in network-restricted
  environments — see "The vector store" section above.
- **Baseline method choice is backtested, not asserted.** `compare_baselines()`
  in `forecast_engine.py` scores SMA, ETS, and Prophet on the same holdout
  window; `ml_dl_evaluation.py` does the same for a classical-ML and a
  "DL-lite" candidate. Neither is wired into the agent's tools — Prophet stays
  the one production path until a Phase 3 decision explicitly changes that.
- **The MCP server and the LangChain agent share one source of truth.** Both
  call the exact same `DemandForecastEngine` and `PlanningKnowledgeBase` instances —
  there's no risk of the two interfaces disagreeing about how a number or a
  retrieved document is produced.
- **Every chat answer includes an expandable trace** in the Streamlit UI showing
  the exact tool call and raw output behind it — full auditability, not a black box.

## Extending this project

- Swap in a real dataset (see "About the data" above)
- Point `PLANNING_DOCUMENTS` in `knowledge_base.py` at your own markdown/text
  files instead of the synthetic corpus — the `build()`/`search()` interface
  stays the same
- Add a multi-SKU aggregate view for category-level S&OP planning
- Add `Langfuse` tracking to log every agent decision for later review
- Deploy the Streamlit app to Streamlit Community Cloud or Azure Container Apps
  for a live demo link

---

Built as part of an AI Engineer portfolio. See `tests/` for proof the deterministic
core is correct independent of anything the LLM says about it.
