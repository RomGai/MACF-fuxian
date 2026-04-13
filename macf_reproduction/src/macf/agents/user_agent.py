from __future__ import annotations

from dataclasses import dataclass

from .base import AgentBase
from ..types import CandidateItem, SimilarUserProfile, TargetUser
from ..tools.base import RetrievalTools


@dataclass
class UserAgent(AgentBase):
    similar_user: SimilarUserProfile

    def act(
        self,
        target_user: TargetUser,
        query: str,
        instruction: str,
        tools: RetrievalTools,
        round_index: int,
    ) -> dict:
        use_query = round_index == 0 or "replace" in instruction.lower() or "expand" in instruction.lower()
        retrieved = tools.retrieve_by_query(query, 12) if use_query else tools.retrieve_by_item(self.similar_user.history_item_ids[0], 12)
        offset = sum(ord(ch) for ch in self.agent_id) % 4
        pool = retrieved[offset:] + retrieved[:offset]
        cands = [
            {
                "item_id": item.item_id,
                "rationale": f"Similar-user({self.similar_user.user_id}) alignment: {self.similar_user.profile_text}",
                "confidence": round(min(0.95, 0.55 + 0.05 * i + self.similar_user.similarity_score * 0.2), 3),
            }
            for i, item in enumerate(pool[:6])
        ]
        return {
            "action": "propose" if round_index == 0 else "revise",
            "inferred_preference": self.similar_user.profile_text,
            "candidates": cands,
            "supported_items": [],
            "rejected_items": [],
            "ranking_feedback": f"Focus on preferences adjacent to {self.similar_user.user_id}.",
            "tool_calls_used": ["RetrieveByQuery" if use_query else "RetrieveByItem"],
        }

    def to_candidates(self, payload: dict) -> list[CandidateItem]:
        return [
            CandidateItem(
                item_id=c["item_id"],
                rationale=c["rationale"],
                confidence=float(c["confidence"]),
                source_agent_id=self.agent_id,
            )
            for c in payload.get("candidates", [])
        ]
