from macf.agents.item_agent import ItemAgent
from macf.agents.user_agent import UserAgent
from macf.types import ItemProfile, SimilarUserProfile


def test_user_agent_to_candidates_accepts_new_candidates() -> None:
    agent = UserAgent(
        agent_id="user_agent_uX",
        agent_type="user",
        similar_user=SimilarUserProfile(user_id="uX", profile_text="p", history_item_ids=["i1"], similarity_score=0.2),
    )
    payload = {
        "new_candidates": [
            {"item_id": "i100", "reason": "better fit"},
            {"item_id": "i101", "rationale": "good", "confidence": 0.9},
        ]
    }
    out = agent.to_candidates(payload)
    assert [x.item_id for x in out] == ["i100", "i101"]


def test_item_agent_to_candidates_accepts_reason_without_confidence() -> None:
    agent = ItemAgent(
        agent_id="item_agent_iX",
        agent_type="item",
        anchor_item=ItemProfile(item_id="iX", title="t", tags=[], description="d"),
    )
    payload = {"candidates": [{"item_id": "i200", "reason": "anchor neighbor"}]}
    out = agent.to_candidates(payload)
    assert len(out) == 1
    assert out[0].item_id == "i200"
