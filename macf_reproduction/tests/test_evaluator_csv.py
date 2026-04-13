from pathlib import Path

from macf.config import load_config
from macf.evaluator import evaluate_from_csv


def test_evaluate_from_csv_outputs_metrics(tmp_path: Path, capsys) -> None:
    metadata = tmp_path / "metadata.csv"
    query = tmp_path / "query_data1.csv"

    metadata.write_text(
        "item_id,title,description,categories\n"
        "a1,Moisturizer,hydrating cream,sensitive skin\n"
        "a2,Cleanser,gentle daily wash,sensitive skin\n"
        "a3,Sunscreen,spf uv protection,sun care\n"
        "a4,Serum,vitamin c brightening,brightening\n",
        encoding="utf-8",
    )

    query.write_text(
        "user_id,query,history_item_ids,target_item_id\n"
        "u1,hydrating sensitive skin,a2|a3,a1\n"
        "u2,uv protection,a3|a2,a3\n",
        encoding="utf-8",
    )

    cfg = load_config("config/default.yaml")
    result = evaluate_from_csv(cfg, query_csv=str(query), metadata_csv=str(metadata))
    printed = capsys.readouterr().out

    assert result["num_cases"] == 2
    assert "hit@10" in result["metrics"]
    assert "ndcg@40" in result["metrics"]
    assert len(result["details"]) == 2
    assert "[Progress] Processing user 1/2" in printed
    assert "AvgMetrics after 2 users" in printed
