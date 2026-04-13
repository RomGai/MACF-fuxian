from macf.config import LLMConfig, load_config
from macf.llm import build_llm_backend


def test_build_qwen_backend_with_fallback_json() -> None:
    backend = build_llm_backend(LLMConfig(provider="qwen_local", model="Qwen/Qwen3-8B"))
    output = backend.generate_json("sys", "usr", {"ok": True})
    assert output["ok"] is True


def test_default_config_thinking_disabled() -> None:
    cfg = load_config("config/default.yaml")
    assert cfg.llm.enable_thinking is False
