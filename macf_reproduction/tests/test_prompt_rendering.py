from macf import prompts


def test_prompts_marked_reconstructed() -> None:
    assert "RECONSTRUCTED" in prompts.ORCHESTRATOR_SYSTEM_PROMPT
    rendered = prompts.USER_AGENT_INITIAL_PROMPT_TEMPLATE.format(similar_user_json="{}", target_user_json="{}", query="q")
    assert "query" in rendered.lower()
