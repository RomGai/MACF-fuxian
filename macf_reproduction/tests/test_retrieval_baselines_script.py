from pathlib import Path
import subprocess
import sys


def test_retrieval_baselines_script_runs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "retrieval_baselines.py"

    metadata = tmp_path / "metadata.csv"
    query = tmp_path / "query_data1.csv"
    out_dir = tmp_path / "out"

    metadata.write_text(
        "item_id,title,description,categories\n"
        "a1,Moisturizer,hydrating cream,sensitive skin\n"
        "a2,Cleanser,gentle daily wash,sensitive skin\n",
        encoding="utf-8",
    )
    query.write_text(
        "user_id,query,history_item_ids,target_item_id\n"
        "u1,hydrating sensitive skin,a2,a1\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(script),
        "--metadata",
        str(metadata),
        "--query-data",
        str(query),
        "--output-dir",
        str(out_dir),
        "--preference-only",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert "output_file" in proc.stdout
    assert (out_dir / "macf_eval_result.json").exists()
