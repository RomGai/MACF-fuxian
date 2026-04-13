from __future__ import annotations

from dataclasses import dataclass

from .base import AgentBase
from ..types import CandidateItem, ItemProfile, TargetUser
from ..tools.base import RetrievalTools


@dataclass
class ItemAgent(AgentBase):
    anchor_item: ItemProfile

    def act(
        self,
        target_user: TargetUser,
        query: str,
        instruction: str,
        tools: RetrievalTools,
        round_index: int,
    ) -> dict:
        weak_anchor = "too broad" in instruction.lower() or "off-topic" in instruction.lower()
        use_item = not weak_anchor
        retrieved = tools.retrieve_by_item(self.anchor_item.item_id, 12) if use_item else tools.retrieve_by_query(query, 12)
        offset = sum(ord(ch) for ch in self.agent_id) % 5
        pool = retrieved[offset:] + retrieved[:offset]
        cands = [
            {
                "item_id": item.item_id,
                "rationale": f"Anchor-item({self.anchor_item.item_id}) path via tags {','.join(self.anchor_item.tags)}",
                "confidence": round(0.6 + 0.04 * i, 3),
            }
            for i, item in enumerate(pool[:6])
        ]
        return {
            "action": "propose" if round_index == 0 else "revise",
            "anchor_analysis": f"{self.anchor_item.title} signals target preference for {'/'.join(self.anchor_item.tags)}.",
            "candidates": cands,
            "supported_items": [],
            "rejected_items": [],
            "ranking_feedback": "Strengthen anchor-consistent items.",
            "tool_calls_used": ["RetrieveByItem" if use_item else "RetrieveByQuery"],
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
