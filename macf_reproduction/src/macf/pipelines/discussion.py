from __future__ import annotations

import json
import logging

from ..agents import ItemAgent, OrchestratorAgent, UserAgent
from ..llm import LLMBackend, MockLLMBackend
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


def run_discussion(
    state: MACFSessionState,
    tools,
    default_n: int,
    llm_backend: LLMBackend | None = None,
    verbose_agent_trace: bool = False,
) -> MACFSessionState:
    llm = llm_backend or MockLLMBackend()
    orch = OrchestratorAgent(llm_backend=llm)
    ru, ri, instructions = orch.recruit_and_instruct(state.target_user, state.query.text, tools, default_n)

    if verbose_agent_trace:
        print("[AgentTrace] Recruited user agents:", json.dumps(ru, ensure_ascii=False))
        print("[AgentTrace] Recruited item agents:", json.dumps(ri, ensure_ascii=False))
        print(
            "[AgentTrace] Initial instructions:",
            json.dumps([{"agent_id": x.agent_id, "agent_type": x.agent_type, "instruction": x.instruction} for x in instructions], ensure_ascii=False),
        )

    similar_profiles = {u.user_id: u for u in tools.get_similar_users(state.target_user.user_id, max(default_n * 3, 20))}
    relevant_items = {i.item_id: i for i in tools.get_relevant_items(state.target_user.user_id, state.query.text, max(default_n * 3, 20))}

    user_agents = {
        r["agent_id"]: UserAgent(agent_id=r["agent_id"], agent_type="user", similar_user=similar_profiles[r["similar_user_id"]], llm_backend=llm)
        for r in ru
        if r["similar_user_id"] in similar_profiles
    }
    item_agents = {
        r["agent_id"]: ItemAgent(agent_id=r["agent_id"], agent_type="item", anchor_item=relevant_items[r["history_item_id"]], llm_backend=llm)
        for r in ri
        if r["history_item_id"] in relevant_items
    }
    all_agents = {**user_agents, **item_agents}

    active_ids = list(all_agents.keys())
    current_instructions = {inst.agent_id: inst.instruction for inst in instructions if inst.agent_id in all_agents}

    for round_idx in range(state.max_rounds):
        round_obj = DiscussionRound(round_index=round_idx, active_agent_ids=active_ids, instructions=[i for i in instructions if i.agent_id in active_ids])
        logger.info("Round %s active agents: %s", round_idx, active_ids)
        if verbose_agent_trace:
            print(f"[AgentTrace] ===== Round {round_idx} | active_agents={active_ids}")

        for aid in active_ids:
            agent = all_agents[aid]
            payload = agent.act(state.target_user, state.query.text, current_instructions.get(aid, ""), tools, round_idx)
            msg = AgentMessage(agent_id=aid, agent_type=agent.agent_type, round_index=round_idx, payload=payload)
            round_obj.agent_messages.append(msg)
            state.accumulated_candidates.extend(agent.to_candidates(payload))
            if verbose_agent_trace:
                print(f"[AgentTrace] {aid} output: {json.dumps(payload, ensure_ascii=False)}")

        state.rounds.append(round_obj)
        decision = orch.decide_next(round_idx, state.accumulated_candidates, state.top_k, state.max_rounds, active_ids)
        state.draft = decision["draft"]
        logger.info("Sufficiency at round %s: %s", round_idx, decision["sufficiency"])

        if verbose_agent_trace:
            draft_preview = [{"rank": x.rank, "item_id": x.item_id, "score": x.score} for x in state.draft.items[:10]]
            print(f"[AgentTrace] Round {round_idx} draft_top10: {json.dumps(draft_preview, ensure_ascii=False)}")
            print(f"[AgentTrace] Round {round_idx} decision: {decision['decision']} | suff={decision['sufficiency']}")

        if decision["decision"] == "end":
            _finalize_top_k(state, tools)
            if verbose_agent_trace:
                final_preview = [{"rank": x.rank, "item_id": x.item_id, "score": x.score} for x in state.final_ranked_items]
                print(f"[AgentTrace] Final topK: {json.dumps(final_preview, ensure_ascii=False)}")
            return state

        active_ids = [x for x in decision["selected_agents"] if x in all_agents]
        current_instructions = {x["agent_id"]: x["instruction"] for x in decision["instructions"] if x["agent_id"] in all_agents}

    _finalize_top_k(state, tools)
    if verbose_agent_trace:
        final_preview = [{"rank": x.rank, "item_id": x.item_id, "score": x.score} for x in state.final_ranked_items]
        print(f"[AgentTrace] Final topK: {json.dumps(final_preview, ensure_ascii=False)}")
    return state
