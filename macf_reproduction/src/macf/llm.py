from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMBackend:
    def generate_json(self, system_prompt: str, user_prompt: str, fallback: dict) -> dict:
        raise NotImplementedError


def _merge_with_fallback(parsed: dict, fallback: dict) -> dict:
    """Keep parsed content, but preserve critical fallback keys when missing/empty."""
    out = dict(parsed)
    for key, fallback_val in fallback.items():
        if key not in out:
            out[key] = fallback_val
            continue
        current = out[key]
        if isinstance(current, dict) and isinstance(fallback_val, dict):
            out[key] = _merge_with_fallback(current, fallback_val)
            continue
        if current in (None, "", []) and fallback_val not in (None, "", []):
            out[key] = fallback_val
    return out


@dataclass
class MockLLMBackend(LLMBackend):
    """Deterministic offline backend: returns fallback payload unchanged."""

    def generate_json(self, system_prompt: str, user_prompt: str, fallback: dict) -> dict:
        return fallback


@dataclass
class QwenLocalBackend(LLMBackend):
    model_name: str
    max_new_tokens: int = 1024
    enable_thinking: bool = True

    def __post_init__(self) -> None:
        self._available = False
        self._tokenizer = None
        self._model = None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self._available = True
            logger.info("Loaded Qwen backend: %s", self.model_name)
        except Exception as exc:
            logger.warning("Qwen backend unavailable, fallback to heuristic outputs. reason=%s", exc)

    def _extract_json(self, text: str) -> dict | None:
        candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
        for c in candidates[::-1]:
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                continue
        return None

    def generate_json(self, system_prompt: str, user_prompt: str, fallback: dict) -> dict:
        if not self._available:
            return fallback

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Qwen thinking token split compatible with official example.
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self._tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        parsed = self._extract_json(content)
        if not isinstance(parsed, dict):
            return fallback
        return _merge_with_fallback(parsed, fallback)


def build_llm_backend(cfg: LLMConfig) -> LLMBackend:
    if cfg.provider == "qwen_local":
        return QwenLocalBackend(
            model_name=cfg.model,
            max_new_tokens=cfg.max_new_tokens,
            enable_thinking=cfg.enable_thinking,
        )
    return MockLLMBackend()
