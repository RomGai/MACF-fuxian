from __future__ import annotations

import argparse
import json

from .config import load_config
from .logging_utils import setup_logging
from .runner import run_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MACF reproduction demo")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--query", default="thoughtful emotional sci-fi set in space")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)
    result = run_session(cfg, args.query)

    output = {
        "target_user_id": result.target_user.user_id,
        "query": result.query.text,
        "rounds": len(result.rounds),
        "final_top_k": [
            {"rank": x.rank, "item_id": x.item_id, "reason": x.reason, "score": x.score}
            for x in result.final_ranked_items
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
