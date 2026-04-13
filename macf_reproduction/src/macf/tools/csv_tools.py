from __future__ import annotations

import logging
import math
from collections import defaultdict

from ..data import QueryRecord
from ..types import ItemProfile, SimilarUserProfile, TargetUser

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return {t.strip(".,!?()[]{}\"'`).:-").lower() for t in text.split() if t.strip()}


class CSVRetrievalTools:
    """Retrieval backend over metadata.csv + query_data1.csv.

    Paper-grounded: the 4 tool interfaces.
    Reconstructed: token-overlap scoring heuristics.
    """

    def __init__(self, items: dict[str, ItemProfile], query_records: list[QueryRecord], current_target: TargetUser) -> None:
        self.items = items
        self.query_records = query_records
        self.target_user = current_target
        self._user_histories: dict[str, set[str]] = defaultdict(set)
        for r in query_records:
            self._user_histories[r.user_id].update(r.history_item_ids)

    def _item_text(self, item: ItemProfile) -> str:
        return f"{item.title} {item.description} {' '.join(item.tags)}"

    def _query_score(self, query_tokens: set[str], item: ItemProfile) -> float:
        item_tokens = _tokenize(self._item_text(item))
        if not item_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(item_tokens))
        norm = math.sqrt(len(item_tokens))
        return overlap / (norm or 1.0)

    def get_similar_users(self, user_id: str, n: int) -> list[SimilarUserProfile]:
        logger.info("Tool call GetSimilarUsers(user_id=%s, n=%s)", user_id, n)
        target_hist = set(self.target_user.history_item_ids)
        sims: list[SimilarUserProfile] = []
        for uid, hist in self._user_histories.items():
            if uid == user_id or not hist:
                continue
            inter = len(target_hist.intersection(hist))
            union = len(target_hist.union(hist)) or 1
            score = inter / union
            sims.append(
                SimilarUserProfile(
                    user_id=uid,
                    profile_text=f"History overlap={inter}/{union}",
                    history_item_ids=sorted(hist),
                    similarity_score=score,
                )
            )
        sims.sort(key=lambda x: x.similarity_score, reverse=True)
        return sims[:n]

    def get_relevant_items(self, user_id: str, query: str, n: int) -> list[ItemProfile]:
        logger.info("Tool call GetRelevantItems(user_id=%s, query=%s, n=%s)", user_id, query, n)
        q_tokens = _tokenize(query)
        cand = [self.items[i] for i in self.target_user.history_item_ids if i in self.items]
        cand.sort(key=lambda it: self._query_score(q_tokens, it), reverse=True)
        return cand[:n]

    def retrieve_by_query(self, query: str, k: int) -> list[ItemProfile]:
        logger.info("Tool call RetrieveByQuery(query=%s, k=%s)", query, k)
        q_tokens = _tokenize(query)
        ranked = sorted(self.items.values(), key=lambda it: self._query_score(q_tokens, it), reverse=True)
        return ranked[:k]

    def retrieve_by_item(self, item_id: str, k: int) -> list[ItemProfile]:
        logger.info("Tool call RetrieveByItem(item_id=%s, k=%s)", item_id, k)
        anchor = self.items.get(item_id)
        if anchor is None:
            return list(self.items.values())[:k]
        anchor_tokens = _tokenize(self._item_text(anchor))

        def score(it: ItemProfile) -> float:
            if it.item_id == item_id:
                return -1
            tokens = _tokenize(self._item_text(it))
            return len(anchor_tokens.intersection(tokens)) / (len(anchor_tokens.union(tokens)) or 1)

        ranked = sorted(self.items.values(), key=score, reverse=True)
        return ranked[:k]
