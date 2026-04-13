from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str = "qwen_local"
    model: str = "Qwen/Qwen3-8B"
    temperature: float = 0.3
    max_new_tokens: int = 1024
    enable_thinking: bool = False


@dataclass
class MACFConfig:
    top_k: int = 10
    max_rounds: int = 5
    default_n: int = 5
    default_k: int = 15


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AppConfig:
    llm: LLMConfig
    macf: MACFConfig
    logging: LoggingConfig


def _parse_simple_yaml(text: str) -> dict:
    current = None
    out: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith("-"):
            current = line[:-1]
            out[current] = {}
            continue
        if ":" in line and current:
            k, v = [x.strip() for x in line.split(":", 1)]
            if v.lower() in {"true", "false"}:
                val = v.lower() == "true"
            else:
                try:
                    val = int(v)
                except ValueError:
                    try:
                        val = float(v)
                    except ValueError:
                        val = v.strip('"').strip("'")
            out[current][k] = val
    return out


def load_config(path: str | Path) -> AppConfig:
    data = _parse_simple_yaml(Path(path).read_text())
    return AppConfig(
        llm=LLMConfig(**data.get("llm", {})),
        macf=MACFConfig(**data.get("macf", {})),
        logging=LoggingConfig(**data.get("logging", {})),
    )
