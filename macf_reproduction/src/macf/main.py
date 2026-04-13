from __future__ import annotations

import argparse
import json

from .config import load_config
from .evaluator import evaluate_from_csv, format_eval_json
from .logging_utils import setup_logging
from .runner import run_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MACF reproduction")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--mode", choices=["demo", "evaluate"], default="demo")
    parser.add_argument("--query", default="thoughtful emotional sci-fi set in space")
    parser.add_argument("--query-csv", default="amazon_beauty/query_data1.csv")
    parser.add_argument("--metadata-csv", default="amazon_beauty/metadata.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.logging.level)

    if args.mode == "demo":
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
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    eval_result = evaluate_from_csv(cfg, query_csv=args.query_csv, metadata_csv=args.metadata_csv)
    print(format_eval_json(eval_result))


if __name__ == "__main__":
    main()
