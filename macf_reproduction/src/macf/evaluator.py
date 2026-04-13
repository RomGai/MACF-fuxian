from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .config import AppConfig
from .data import build_target_user, load_metadata_csv, load_query_csv
from .llm import build_llm_backend
from .pipelines.discussion import run_discussion
from .tools import CSVRetrievalTools
from .types import MACFSessionState, QueryContext


@dataclass
class EvalRow:
    user_id: str
    query: str
    target_item_id: str
    rank: int | None


def _calc_metrics(ranks: list[int | None], k: int) -> dict[str, float]:
    hits = [1.0 if r is not None and r <= k else 0.0 for r in ranks]
    ndcgs = [1.0 / math.log2(r + 1) if r is not None and r <= k else 0.0 for r in ranks]
    n = len(ranks) or 1
    return {f"hit@{k}": sum(hits) / n, f"ndcg@{k}": sum(ndcgs) / n}


def evaluate_from_csv(config: AppConfig, query_csv: str, metadata_csv: str) -> dict:
    items = load_metadata_csv(metadata_csv)
    records = load_query_csv(query_csv)
    llm_backend = build_llm_backend(config.llm)

    rows: list[EvalRow] = []
    for rec in records:
        target_user = build_target_user(rec)
        tools = CSVRetrievalTools(items=items, query_records=records, current_target=target_user)
        state = MACFSessionState(
            target_user=target_user,
            query=QueryContext(text=rec.query),
            top_k=config.macf.top_k,
            max_rounds=config.macf.max_rounds,
        )
        state = run_discussion(state, tools, default_n=config.macf.default_n, llm_backend=llm_backend)

        ranked_ids = [x.item_id for x in state.final_ranked_items]
        rank = ranked_ids.index(rec.target_item_id) + 1 if rec.target_item_id in ranked_ids else None
        rows.append(EvalRow(user_id=rec.user_id, query=rec.query, target_item_id=rec.target_item_id, rank=rank))

    ranks = [r.rank for r in rows]
    metrics = {}
    metrics.update(_calc_metrics(ranks, 10))
    metrics.update(_calc_metrics(ranks, 20))
    metrics.update(_calc_metrics(ranks, 40))
    return {
        "num_cases": len(rows),
        "metrics": metrics,
        "details": [r.__dict__ for r in rows],
    }


def format_eval_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)
