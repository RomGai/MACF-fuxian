from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TargetUser:
    user_id: str
    profile_text: str
    history_item_ids: list[str]


@dataclass
class SimilarUserProfile:
    user_id: str
    profile_text: str
    history_item_ids: list[str]
    similarity_score: float


@dataclass
class ItemProfile:
    item_id: str
    title: str
    tags: list[str]
    description: str


@dataclass
class QueryContext:
    text: str


@dataclass
class CandidateItem:
    item_id: str
    rationale: str
    confidence: float
    source_agent_id: str


@dataclass
class AgentInstruction:
    agent_id: str
    agent_type: Literal["user", "item"]
    instruction: str


@dataclass
class AgentMessage:
    agent_id: str
    agent_type: Literal["user", "item", "orchestrator"]
    round_index: int
    payload: dict


@dataclass
class DiscussionRound:
    round_index: int
    active_agent_ids: list[str]
    instructions: list[AgentInstruction]
    agent_messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class RankedItem:
    rank: int
    item_id: str
    reason: str
    score: float


@dataclass
class RankedListDraft:
    items: list[RankedItem]


@dataclass
class SufficiencyResult:
    enough_items: bool
    clear_relevance: bool
    clear_ordering: bool
    overall_pass: bool
    notes: str


@dataclass
class MACFSessionState:
    target_user: TargetUser
    query: QueryContext
    top_k: int
    max_rounds: int
    rounds: list[DiscussionRound] = field(default_factory=list)
    accumulated_candidates: list[CandidateItem] = field(default_factory=list)
    draft: RankedListDraft = field(default_factory=lambda: RankedListDraft(items=[]))
    final_ranked_items: list[RankedItem] = field(default_factory=list)
