from __future__ import annotations

from dataclasses import dataclass

from ..ranking import build_ranked_draft
from ..sufficiency import evaluate_sufficiency
from ..types import AgentInstruction, CandidateItem, RankedListDraft, TargetUser
from ..tools.base import RetrievalTools


@dataclass
class OrchestratorAgent:
    agent_id: str = "orchestrator"

    def recruit_and_instruct(self, target: TargetUser, query: str, tools: RetrievalTools, n: int) -> tuple[list[dict], list[dict], list[AgentInstruction]]:
        similar_users = tools.get_similar_users(target.user_id, n)
        history_items = tools.get_relevant_items(target.user_id, query, n)
        recruited_users = []
        recruited_items = []
        instructions: list[AgentInstruction] = []

        for su in similar_users:
            aid = f"user_agent_{su.user_id}"
            recruited_users.append({"agent_id": aid, "similar_user_id": su.user_id, "why_recruited": f"High similarity {su.similarity_score}"})
            instructions.append(AgentInstruction(aid, "user", f"Based on your preference for {su.profile_text}, suggest query-aligned alternatives."))

        for it in history_items:
            aid = f"item_agent_{it.item_id}"
            recruited_items.append({"agent_id": aid, "history_item_id": it.item_id, "why_recruited": "High query relevance from history"})
            instructions.append(AgentInstruction(aid, "item", f"Examine whether anchor item {it.title} is too broad for the query and refine."))

        return recruited_users, recruited_items, instructions

    def decide_next(
        self,
        round_index: int,
        accumulated_candidates: list[CandidateItem],
        top_k: int,
        max_rounds: int,
        prior_active_agents: list[str],
    ) -> dict:
        draft = build_ranked_draft(accumulated_candidates)
        suff = evaluate_sufficiency(draft, top_k)
        if suff.overall_pass or round_index >= max_rounds - 1:
            return {
                "decision": "end",
                "draft": draft,
                "sufficiency": suff,
                "selected_agents": [],
                "instructions": [],
                "uncertainty_summary": "None" if suff.overall_pass else "Reached max rounds",
            }

        reduced = prior_active_agents[: max(2, len(prior_active_agents) // 2)]
        instructions = [
            {"agent_id": aid, "instruction": "Replace low-relevance options; resolve ties near ranks 8-12 with stronger query alignment."}
            for aid in reduced
        ]
        return {
            "decision": "continue",
            "draft": draft,
            "sufficiency": suff,
            "selected_agents": reduced,
            "instructions": instructions,
            "uncertainty_summary": "Tie/conflict near cutoff.",
        }
