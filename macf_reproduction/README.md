# MACF Reproduction (Reconstructed)

This repository provides a **faithful, runnable reconstruction** of the core method described in:

> **"Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations"**

## Important transparency note
- This implementation does **not** claim access to hidden/original author prompts.
- All prompt text is explicitly marked **"reconstructed / inferred from paper"**.

## Paper-grounded vs reconstructed

### Grounded in paper description
- Three roles: **Orchestrator**, **User Agents**, **Item Agents**.
- Four tools:
  - `GetSimilarUsers(user_id, n)`
  - `GetRelevantItems(user_id, query, n)`
  - `RetrieveByQuery(query, k)`
  - `RetrieveByItem(item_id, k)`
- Dynamic Agent Recruitment (DAR)
- Personalized Collaboration Instruction (PCI)
- Adaptive Tool Use (ATU)
- Multi-round discussion (`Tmax=5`)
- Final top-K list (`K=10`)
- Shared backend config: `model=gpt-4o`, `temperature=0.3`

### Reconstructed / inferred choices
- Exact prompt wording and templates
- Heuristic ranking aggregation and sufficiency checks
- Mock retrieval backend and deterministic agent behaviors
- JSON schemas used for offline reproducible runs

## Repo layout

```text
macf_reproduction/
  README.md
  pyproject.toml
  requirements.txt
  .env.example
  config/default.yaml
  src/macf/...
  tests/...
```

## Setup

```bash
cd macf_reproduction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run demo

```bash
python -m macf.main --config config/default.yaml --query "thoughtful emotional sci-fi set in space"
```

The script prints structured JSON with per-session output and final top-K list.

## Test sanity checks

```bash
pytest -q
```

## Swapping mock tools with real retrieval
1. Implement `RetrievalTools` protocol in `src/macf/tools/base.py`.
2. Replace `MockRetrievalTools` calls in `runner.py`.
3. Keep signatures identical for minimal changes.

## Plugging in a real LLM provider
1. Implement `LLMBackend.generate_json` in `src/macf/llm.py`.
2. Use provider/model from `config/default.yaml`.
3. Keep agent output schema as strict JSON.
