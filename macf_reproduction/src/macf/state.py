from __future__ import annotations

from .types import MACFSessionState


def append_round(state: MACFSessionState, round_obj) -> None:
    state.rounds.append(round_obj)
