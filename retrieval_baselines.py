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
    parser.add_argument("--preference-only", action="store_true", help="Compatibility flag; currently reserved for future filtering strategy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = evaluate_from_csv(cfg, query_csv=args.query_data, metadata_csv=args.metadata)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "macf_eval_result.json"
    out_json.write_text(format_eval_json(result), encoding="utf-8")

    summary = {
        "output_file": str(out_json),
        "preference_only": args.preference_only,
        "metrics": result.get("metrics", {}),
        "num_cases": result.get("num_cases", 0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
