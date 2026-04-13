#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure direct script execution works without setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parent
MACF_SRC = REPO_ROOT / "macf_reproduction" / "src"
if str(MACF_SRC) not in sys.path:
    sys.path.insert(0, str(MACF_SRC))

from macf.config import load_config
from macf.evaluator import evaluate_from_csv, format_eval_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MACF retrieval baseline evaluation on amazon_beauty CSV data.")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--query-data", required=True, help="Path to query_data1.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to store evaluation artifacts")
    parser.add_argument("--config", default=str(REPO_ROOT / "macf_reproduction" / "config" / "default.yaml"), help="Path to MACF config file")
    parser.add_argument("--top-k", type=int, default=40, help="Final recommendation cutoff K (default: 40).")
    parser.add_argument(
        "--query-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use query + metadata for retrieval (default: true).",
    )
    parser.add_argument(
        "--preference-only",
        action="store_true",
        help="Deprecated compatibility flag. It is ignored; retrieval remains query-based.",
    )
    parser.add_argument(
        "--verbose-agent-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-round agent actions/outputs. Use --no-verbose-agent-trace to disable.",
    )
    args = parser.parse_args()

    if args.preference_only:
        print("[Warning] --preference-only is deprecated and ignored. Retrieval still uses query + metadata.")

    if not args.query_only:
        print("[Warning] --no-query-only is not supported in current baseline; forcing query-only retrieval.")

    cfg = load_config(args.config)
    cfg.macf.top_k = args.top_k
    result = evaluate_from_csv(
        cfg,
        query_csv=args.query_data,
        metadata_csv=args.metadata,
        verbose_agent_trace=args.verbose_agent_trace,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "macf_eval_result.json"
    out_json.write_text(format_eval_json(result), encoding="utf-8")

    summary = {
        "output_file": str(out_json),
        "retrieval_mode": "query_only",
        "verbose_agent_trace": args.verbose_agent_trace,
        "top_k": cfg.macf.top_k,
        "metrics": result.get("metrics", {}),
        "num_cases": result.get("num_cases", 0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
