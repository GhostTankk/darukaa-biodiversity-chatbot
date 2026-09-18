# Darukaa.Earth — AI Biodiversity Intelligence Chatbot

An AI environmental scientist, not a chatbot: retrieves grounded evidence from a
biodiversity/soil/climate knowledge base, reasons across multiple environmental
variables at once, and returns structured, evidence-backed recommendations —
never "use sustainable practices."

Built for the Darukaa.Earth Hackathon Challenge.

## Architecture

```
User (text or JSON) ──▶ ConversationManager ──▶ KnowledgeBase (FAISS + embeddings)
                              │  (slot-filling,          │
                              │   clarifying Qs,          │  top-k evidence chunks
                              │   multi-turn memory)      ▼
                              └──────────────────▶ ReasoningEngine (Gemini API)
                                                          │
                                                          ▼
                                            Structured ChatbotResponse
                                       (recommendations + metrics + sources)
```

**Why this split matters:** retrieval is deterministic, free, and runs locally
(sentence-transformers + FAISS) — it decides *what evidence is relevant*. The
LLM call is the one place that costs money and can hallucinate, so it's
constrained to reason *only* over the evidence it's handed, and forced to
return strict JSON so the app never has to regex-parse prose. This is what
"knowledge-driven," not "LLM-only," means in practice.

### Components (`src/`)

| File | Responsibility |
|---|---|
| `schemas.py` | Pydantic contracts for site data, evidence chunks, and recommendations. Defines what "complete" input looks like. |
| `knowledge_base.py` | Loads `data/knowledge_base.json`, embeds every entry with `all-MiniLM-L6-v2` (local, free), builds a FAISS cosine-similarity index. |
| `pdf_ingest.py` | Optional: chunks PDFs in `data/pdfs/` and folds them into the same FAISS index, so real research papers can be indexed alongside the curated JSON. |
| `conversation.py` | Multi-turn state. Extracts structured slots from free text (regex/keyword based — swap for an LLM extractor for more robustness), decides when a clarifying question is needed vs. when to proceed with partial data. |
| `reasoning_engine.py` | The only component that calls Gemini. System prompt hard-constrains it to reason only over retrieved evidence, connect ≥2 variables where possible, and return strict JSON. |
| `app.py` | Streamlit UI — chat panel + a structured-input sidebar (form or raw JSON) for the "structured input" requirement, including optional lat/long. |
| `cli.py` | Terminal chat loop for fast local testing without spinning up Streamlit. |

## Knowledge base / schema

`data/knowledge_base.json` is a list of entries shaped like:

```json
{
  "id": "kb001",
  "topic": "soil_carbon",
  "trigger_conditions": { "soil_organic_carbon_pct": {"max": 0.8}, "land_use": ["monoculture"] },
  "recommendation": "...",
  "reasoning": "...",
  "impacted_metrics": ["soil_organic_carbon_pct", "microbial_diversity", "..."],
  "expected_effect": "quantified where possible",
  "time_horizon": "short|medium|long",
  "confidence": "low|medium|high",
  "sources": [{"name": "...", "org": "..."}]
}
```

Six seed entries cover the five required domains (soil, land use, biodiversity,
climate, human impact/pollution/deforestation), each citing FAO / IPCC / IPBES
/ peer-reviewed sources. Every entry is written to connect at least two
environmental variables (e.g. `kb003` ties rainfall scarcity to both soil
moisture *and* species richness), satisfying the multi-metric reasoning
requirement at the data layer, not just the prompt layer.

Add more entries by hand, or drop PDFs into `data/pdfs/` and run
`python -m src.pdf_ingest` to chunk and embed them into the same index.

## Local setup

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd darukaa-biodiversity-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=...
```

Get a free key (no credit card needed) at [aistudio.google.com](https://aistudio.google.com):
"Get API key" → create a key → paste it into `.env`. It's read automatically via
`python-dotenv` in both `cli.py` and `app.py`.

The first run downloads the `all-MiniLM-L6-v2` embedding model from
Hugging Face (~90MB) and caches it locally — after that, retrieval works
fully offline. It runs comfortably on CPU; no GPU required, though it'll use
one automatically if available.

Run the terminal demo:
```bash
python -m src.cli
```

Run the web UI:
```bash
streamlit run src/app.py
```

Run tests (offline — no API key needed; these cover retrieval and
slot-filling logic, not live LLM calls):
```bash
pytest tests/ -v
```

## Deploying the live demo

Easiest free option: [Streamlit Community Cloud](https://streamlit.io/cloud).
1. Push this repo to GitHub.
2. On Streamlit Cloud, "New app" → point at this repo, main file `src/app.py`.
3. In the app's **Secrets**, add:
   ```toml
   GEMINI_API_KEY = "..."
   ```
4. Deploy. You'll get a `*.streamlit.app` URL — that's your live demo link.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:
- installs dependencies
- lints with `ruff`
- runs the offline test suite (`pytest tests/`)

Live-API reasoning calls are deliberately excluded from CI (no key, and no
reason to spend API credits on every push) — they're covered by manual /
demo testing instead.

## Example interaction

```
User: Biodiversity is declining on my land.
Assistant: Could you share soil organic carbon %, pH, or moisture; the
           current land use; and rainfall/region type?

User: Soil organic carbon is 0.3%, rainfall is low, monoculture wheat,
      semi-arid region.
Assistant:
  1. Transition part of the monoculture wheat plot to agroforestry/
     intercropping with a deep-rooted, drought-tolerant species.
     Why: connects two failure modes at once — low SOC (nothing anchoring
     moisture or microbial activity) and low erratic rainfall (shallow-
     rooted wheat has no buffer through dry spells).
     Impacts: soil_organic_carbon_pct, soil_moisture, species_richness
     Expected effect: improved dry-season moisture retention; SOC gains
     compound with cover cropping over 2-3 years
     Time horizon: medium | Confidence: medium
     Sources: FAO – Agroforestry for Landscape Restoration; IPCC AR6 WG2
```

## Known limitations / next steps

- Free-text slot extraction is regex/keyword-based, not LLM-based — good
  enough for the demo, but a dedicated extraction call (or function-calling)
  would handle more phrasings.
- The knowledge base is seeded with 6 hand-curated entries; the PDF
  ingestion pipeline is there to scale it up with real papers/reports.
- No persistent database — conversation state lives in memory per session
  (Streamlit `session_state`). Fine for a hackathon demo; swap in Redis/
  Postgres for multi-user production use.
