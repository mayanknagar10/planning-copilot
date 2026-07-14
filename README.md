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

**Primary: [OpenRouter](https://openrouter.ai)** — one API key, routes to
`meta-llama/llama-3.3-70b-instruct:free`, which supports tool calling.

**Fallback: [Google AI Studio (Gemini)](https://ai.google.dev)** — kicks in
automatically if OpenRouter is rate-limited or unavailable. This isn't just a
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
invocation level sidesteps the issue entirely.

```python
def invoke_agent_with_fallback(question: str) -> dict:
    try:
        primary_agent = create_react_agent(build_primary_llm(), TOOLS, prompt=SYSTEM_PROMPT)
        return {"result": primary_agent.invoke(...), "provider": "OpenRouter", "fell_back": False}
    except Exception:
        fallback_agent = create_react_agent(build_fallback_llm(), TOOLS, prompt=SYSTEM_PROMPT)
        return {"result": fallback_agent.invoke(...), "provider": "Gemini", "fell_back": True}
```

The Streamlit chat tab surfaces which provider actually answered whenever a
fallback occurs, so the switch is visible rather than silent — useful both
for debugging and for demonstrating the resilience pattern live.

Neither provider requires a credit card. Get your keys at the links above, then:

```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY and (optionally) GOOGLE_API_KEY
```

**Important:** `.env` is where your real keys go — `.env.example` is only a
committed template with empty placeholders. `.env` is already listed in
`.gitignore`, so your real keys never get pushed to GitHub. Also note that
`.env` needs `load_dotenv()` to actually be read into the environment — this
project calls it at the top of `app.py`, `agent.py`, and `mcp_server.py`, so
you don't need to do anything extra beyond filling in the file.

## Getting started

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd planning-copilot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate the synthetic demand dataset
cd data && python3 generate_data.py && cd ..

# 3. Set up API keys (see above)
cp .env.example .env   # then fill in your keys

# 4. Run the test suite (no API keys needed — tests only the deterministic core)
pytest tests/ -v

# 5. Launch the app
cd src && streamlit run app.py

# Optional: run the MCP server standalone to verify it, or wire it into Claude Desktop
cd src && python3 mcp_server.py
```

The **Forecast Explorer** tab works immediately with no API keys — it only calls
`forecast_engine.py` directly. The related-notes panel and the **Planning Assistant**
chat tab both use `knowledge_base.py`, which needs one internet connection on first
use to download the local embedding model (~90MB, one-time only) — after that it
runs fully offline. The chat tab additionally needs `OPENROUTER_API_KEY` set for the
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
│   └── demand_history.csv    # generated output (run generate_data.py first)
├── src/
│   ├── forecast_engine.py    # deterministic core — Prophet, no LLM, fully unit-tested
│   ├── knowledge_base.py     # ChromaDB vector store over planning docs — no LLM
│   ├── agent.py                # LangGraph agent — tool-calling only, narrates results
│   ├── mcp_server.py          # exposes the same 5 tools via MCP for Claude Desktop etc.
│   ├── app.py                  # Streamlit dashboard — Forecast Explorer + chat
│   └── chroma_db/              # persisted vector index (auto-created on first run)
├── tests/
│   └── test_forecast_engine.py  # unit tests for the deterministic core
├── requirements.txt
├── .env.example
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
