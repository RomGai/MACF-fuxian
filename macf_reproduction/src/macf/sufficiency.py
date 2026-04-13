from __future__ import annotations

from .types import RankedListDraft, SufficiencyResult


def evaluate_sufficiency(draft: RankedListDraft, top_k: int) -> SufficiencyResult:
    unique_ids = {item.item_id for item in draft.items}
    enough_items = len(unique_ids) >= top_k
    clear_relevance = all(len(item.reason.strip()) > 10 for item in draft.items[:top_k]) if draft.items else False
    scores = [item.score for item in draft.items]
    clear_ordering = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)) if scores else False
    overall_pass = enough_items and clear_relevance and clear_ordering
    return SufficiencyResult(
        enough_items=enough_items,
        clear_relevance=clear_relevance,
        clear_ordering=clear_ordering,
        overall_pass=overall_pass,
        notes="Pass" if overall_pass else "Needs another round",
    )
