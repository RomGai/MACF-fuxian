from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentBase:
    agent_id: str
    agent_type: str
