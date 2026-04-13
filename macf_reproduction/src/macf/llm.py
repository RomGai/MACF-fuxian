from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str


class LLMBackend:
    def generate_json(self, request: LLMRequest) -> dict:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    """Deterministic stub backend; useful for offline reproducibility."""

    def generate_json(self, request: LLMRequest) -> dict:
        return {
            "note": "MockLLMBackend is active",
            "system_prompt_hint": request.system_prompt[:80],
            "user_prompt_hint": request.user_prompt[:120],
        }
