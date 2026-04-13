from macf.config import load_config
from macf.runner import run_session


def test_discussion_loop_produces_top_k() -> None:
    cfg = load_config("config/default.yaml")
    state = run_session(cfg, "emotional space sci-fi")
    assert len(state.final_ranked_items) == cfg.macf.top_k
    assert 1 <= len(state.rounds) <= cfg.macf.max_rounds
