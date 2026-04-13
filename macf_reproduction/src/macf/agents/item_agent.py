from __future__ import annotations

from dataclasses import dataclass, field

from .base import AgentBase
from ..llm import LLMBackend, MockLLMBackend
from ..prompts import (
    ITEM_AGENT_INITIAL_PROMPT_TEMPLATE,
    ITEM_AGENT_REFINE_PROMPT_TEMPLATE,
    ITEM_AGENT_SYSTEM_PROMPT,
)
from ..types import CandidateItem, ItemProfile, TargetUser
from ..tools.base import RetrievalTools


@dataclass
class ItemAgent(AgentBase):
    anchor_item: ItemProfile
    llm_backend: LLMBackend = field(default_factory=MockLLMBackend)

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
        fallback = {
            "action": "propose" if round_index == 0 else "revise",
            "anchor_analysis": f"{self.anchor_item.title} signals target preference for {'/'.join(self.anchor_item.tags)}.",
            "candidates": cands,
            "supported_items": [],
            "rejected_items": [],
            "ranking_feedback": "Strengthen anchor-consistent items.",
            "tool_calls_used": ["RetrieveByItem" if use_item else "RetrieveByQuery"],
        }
        if round_index == 0:
            prompt = ITEM_AGENT_INITIAL_PROMPT_TEMPLATE.format(item_json=str(self.anchor_item), target_user_json=str(target_user), query=query)
        else:
            prompt = ITEM_AGENT_REFINE_PROMPT_TEMPLATE.format(item_json=str(self.anchor_item), discussion_history_json="[]", draft_json="[]", instruction=instruction)
        return self.llm_backend.generate_json(ITEM_AGENT_SYSTEM_PROMPT, prompt, fallback)

    def to_candidates(self, payload: dict) -> list[CandidateItem]:
        rows = list(payload.get("candidates", []))
        rows.extend(payload.get("new_candidates", []))
        out: list[CandidateItem] = []
        for c in rows:
            item_id = c.get("item_id")
            if not item_id:
                continue
            rationale = c.get("rationale") or c.get("reason") or "Item-agent recommendation."
            confidence = float(c.get("confidence", 0.5))
            out.append(
                CandidateItem(
                    item_id=item_id,
                    rationale=rationale,
                    confidence=confidence,
                    source_agent_id=self.agent_id,
                )
            )
        return out
