from __future__ import annotations

from .config import AppConfig
from .demo.sample_data import load_sample_data
from .llm import build_llm_backend
from .pipelines.discussion import run_discussion
from .tools.mock_tools import MockRetrievalTools
from .types import MACFSessionState, QueryContext


def run_session(config: AppConfig, query: str) -> MACFSessionState:
    target_user, _, _ = load_sample_data()
    tools = MockRetrievalTools()
    llm_backend = build_llm_backend(config.llm)
    state = MACFSessionState(
        target_user=target_user,
        query=QueryContext(text=query),
        top_k=config.macf.top_k,
        max_rounds=config.macf.max_rounds,
    )
    return run_discussion(state=state, tools=tools, default_n=config.macf.default_n, llm_backend=llm_backend)
