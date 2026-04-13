from __future__ import annotations

import logging

from ..demo.sample_data import load_sample_data
from ..types import ItemProfile, SimilarUserProfile

logger = logging.getLogger(__name__)


class MockRetrievalTools:
    def __init__(self) -> None:
        self.target_user, self.similar_users, self.items = load_sample_data()

    def get_similar_users(self, user_id: str, n: int) -> list[SimilarUserProfile]:
        logger.info("Tool call GetSimilarUsers(user_id=%s, n=%s)", user_id, n)
        return sorted(self.similar_users.values(), key=lambda u: u.similarity_score, reverse=True)[:n]

    def get_relevant_items(self, user_id: str, query: str, n: int) -> list[ItemProfile]:
        logger.info("Tool call GetRelevantItems(user_id=%s, query=%s, n=%s)", user_id, query, n)
        query_terms = set(query.lower().split())
        scored: list[tuple[float, ItemProfile]] = []
        for item_id in self.target_user.history_item_ids:
            item = self.items[item_id]
            text = (item.title + " " + item.description + " " + " ".join(item.tags)).lower()
            score = sum(1 for t in query_terms if t in text)
            scored.append((score, item))
        return [x[1] for x in sorted(scored, key=lambda kv: kv[0], reverse=True)[:n]]

    def retrieve_by_query(self, query: str, k: int) -> list[ItemProfile]:
        logger.info("Tool call RetrieveByQuery(query=%s, k=%s)", query, k)
        terms = set(query.lower().split())
        scored: list[tuple[float, ItemProfile]] = []
        for item in self.items.values():
            text = (item.title + " " + item.description + " " + " ".join(item.tags)).lower()
            score = sum(1 for t in terms if t in text)
            scored.append((score, item))
        return [x[1] for x in sorted(scored, key=lambda kv: kv[0], reverse=True)[:k]]

    def retrieve_by_item(self, item_id: str, k: int) -> list[ItemProfile]:
        logger.info("Tool call RetrieveByItem(item_id=%s, k=%s)", item_id, k)
        anchor = self.items[item_id]
        anchor_tags = set(anchor.tags)
        scored: list[tuple[int, ItemProfile]] = []
        for item in self.items.values():
            if item.item_id == item_id:
                continue
            overlap = len(anchor_tags.intersection(item.tags))
            scored.append((overlap, item))
        return [x[1] for x in sorted(scored, key=lambda kv: kv[0], reverse=True)[:k]]
