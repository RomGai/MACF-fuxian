from __future__ import annotations

import logging

from ..agents import ItemAgent, OrchestratorAgent, UserAgent
from ..types import AgentMessage, DiscussionRound, MACFSessionState, RankedItem

logger = logging.getLogger(__name__)


def _finalize_top_k(state: MACFSessionState, tools) -> None:
    final_items = list(state.draft.items)
    seen = {x.item_id for x in final_items}
    if len(seen) < state.top_k:
        for item in tools.items.values():
            if item.item_id in seen:
                continue
            final_items.append(RankedItem(rank=len(final_items) + 1, item_id=item.item_id, reason="Backfill for top-k completeness.", score=0.01))
            seen.add(item.item_id)
            if len(seen) >= state.top_k:
                break
    state.final_ranked_items = final_items[: state.top_k]


def run_discussion(state: MACFSessionState, tools, default_n: int) -> MACFSessionState:
    orch = OrchestratorAgent()
    ru, ri, instructions = orch.recruit_and_instruct(state.target_user, state.query.text, tools, default_n)

    user_agents = {r["agent_id"]: UserAgent(agent_id=r["agent_id"], agent_type="user", similar_user=tools.similar_users[r["similar_user_id"]]) for r in ru}
    item_agents = {r["agent_id"]: ItemAgent(agent_id=r["agent_id"], agent_type="item", anchor_item=tools.items[r["history_item_id"]]) for r in ri}
    all_agents = {**user_agents, **item_agents}

    active_ids = list(all_agents.keys())
    current_instructions = {inst.agent_id: inst.instruction for inst in instructions}

    for round_idx in range(state.max_rounds):
        round_obj = DiscussionRound(round_index=round_idx, active_agent_ids=active_ids, instructions=[i for i in instructions if i.agent_id in active_ids])
        logger.info("Round %s active agents: %s", round_idx, active_ids)

        for aid in active_ids:
            agent = all_agents[aid]
            payload = agent.act(state.target_user, state.query.text, current_instructions.get(aid, ""), tools, round_idx)
            msg = AgentMessage(agent_id=aid, agent_type=agent.agent_type, round_index=round_idx, payload=payload)
            round_obj.agent_messages.append(msg)
            state.accumulated_candidates.extend(agent.to_candidates(payload))

        state.rounds.append(round_obj)
        decision = orch.decide_next(round_idx, state.accumulated_candidates, state.top_k, state.max_rounds, active_ids)
        state.draft = decision["draft"]
        logger.info("Sufficiency at round %s: %s", round_idx, decision["sufficiency"])

        if decision["decision"] == "end":
            _finalize_top_k(state, tools)
            return state

        active_ids = decision["selected_agents"]
        current_instructions = {x["agent_id"]: x["instruction"] for x in decision["instructions"]}

    _finalize_top_k(state, tools)
    return state
