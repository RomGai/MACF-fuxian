from __future__ import annotations

from collections import defaultdict

from .types import CandidateItem, RankedItem, RankedListDraft


def build_ranked_draft(candidates: list[CandidateItem], limit: int = 50) -> RankedListDraft:
    score_map: dict[str, float] = defaultdict(float)
    reason_map: dict[str, list[str]] = defaultdict(list)
    for c in candidates:
        score_map[c.item_id] += max(0.0, min(1.0, c.confidence))
        reason_map[c.item_id].append(f"{c.source_agent_id}: {c.rationale}")

    ordered = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    items = [
        RankedItem(rank=i + 1, item_id=item_id, reason=" | ".join(reason_map[item_id][:2]), score=score)
        for i, (item_id, score) in enumerate(ordered)
    ]
    return RankedListDraft(items=items)
