from pathlib import Path

from macf.data import load_metadata_csv, load_query_csv


def test_data_loader_supports_new_schema(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    query = tmp_path / "query_data1.csv"

    metadata.write_text(
        "id,title,description,category,price\n"
        "p1,ItemA,good lotion,skin care,19.9\n"
        "p2,ItemB,gentle cleanser,skin care,9.9\n",
        encoding="utf-8",
    )
    query.write_text(
        "user_id,query,remaining_interaction_string,targets\n"
        "u1,lotion for dry skin,p2|p3,p1\n",
        encoding="utf-8",
    )

    items = load_metadata_csv(metadata)
    records = load_query_csv(query)

    assert "p1" in items
    assert "price:" in items["p1"].description
    assert len(records) == 1
    assert records[0].history_item_ids == ["p2", "p3"]
    assert records[0].target_item_id == "p1"
