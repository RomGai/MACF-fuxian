from __future__ import annotations

from dataclasses import dataclass, field

from .base import AgentBase
from ..llm import LLMBackend, MockLLMBackend
from ..prompts import (
    USER_AGENT_INITIAL_PROMPT_TEMPLATE,
    USER_AGENT_REFINE_PROMPT_TEMPLATE,
    USER_AGENT_SYSTEM_PROMPT,
)
from ..types import CandidateItem, SimilarUserProfile, TargetUser
from ..tools.base import RetrievalTools


@dataclass
class UserAgent(AgentBase):
    similar_user: SimilarUserProfile
    llm_backend: LLMBackend = field(default_factory=MockLLMBackend)

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
        fallback = {
            "action": "propose" if round_index == 0 else "revise",
            "inferred_preference": self.similar_user.profile_text,
            "candidates": cands,
            "supported_items": [],
            "rejected_items": [],
            "ranking_feedback": f"Focus on preferences adjacent to {self.similar_user.user_id}.",
            "tool_calls_used": ["RetrieveByQuery" if use_query else "RetrieveByItem"],
        }
        if round_index == 0:
            prompt = USER_AGENT_INITIAL_PROMPT_TEMPLATE.format(similar_user_json=str(self.similar_user), target_user_json=str(target_user), query=query)
        else:
            prompt = USER_AGENT_REFINE_PROMPT_TEMPLATE.format(similar_user_json=str(self.similar_user), discussion_history_json="[]", draft_json="[]", instruction=instruction)
        return self.llm_backend.generate_json(USER_AGENT_SYSTEM_PROMPT, prompt, fallback)

    def to_candidates(self, payload: dict) -> list[CandidateItem]:
        return [
            CandidateItem(
                item_id=c["item_id"],
                rationale=c["rationale"],
                confidence=float(c["confidence"]),
                source_agent_id=self.agent_id,
            )
            for c in payload.get("candidates", [])
            if all(k in c for k in ["item_id", "rationale", "confidence"])
        ]
