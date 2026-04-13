from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .types import ItemProfile, TargetUser


@dataclass
class QueryRecord:
    user_id: str
    query: str
    history_item_ids: list[str]
    target_item_id: str


def _pick_column(fieldnames: list[str], candidates: list[str]) -> str:
    lowered = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    raise ValueError(f"Missing required columns. candidates={candidates}, got={fieldnames}")


def _split_history(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    for sep in ["|", ";", ",", " "]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]
    return [text]


def load_metadata_csv(path: str | Path) -> dict[str, ItemProfile]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("metadata.csv has no header")
        item_col = _pick_column(reader.fieldnames, ["item_id", "asin", "parent_asin", "id"])
        title_col = _pick_column(reader.fieldnames, ["title", "name"])
        desc_col = _pick_column(reader.fieldnames, ["description", "desc", "features"])
        tags_col = _pick_column(reader.fieldnames, ["categories", "category", "tags"])

        items: dict[str, ItemProfile] = {}
        for row in reader:
            item_id = (row.get(item_col) or "").strip()
            if not item_id:
                continue
            tags = [x.strip() for x in (row.get(tags_col) or "").replace("[", "").replace("]", "").replace("'", "").split(",") if x.strip()]
            items[item_id] = ItemProfile(
                item_id=item_id,
                title=(row.get(title_col) or "").strip() or item_id,
                tags=tags,
                description=(row.get(desc_col) or "").strip(),
            )
    return items


def load_query_csv(path: str | Path) -> list[QueryRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("query_data csv has no header")
        user_col = _pick_column(reader.fieldnames, ["user_id", "reviewerID", "uid"])
        query_col = _pick_column(reader.fieldnames, ["query", "instruction", "search_query"])
        history_col = _pick_column(reader.fieldnames, ["history_item_ids", "history", "history_items", "clicked_items"])
        target_col = _pick_column(reader.fieldnames, ["target_item_id", "target", "label_item", "ground_truth_item"])

        records: list[QueryRecord] = []
        for row in reader:
            q = (row.get(query_col) or "").strip()
            t = (row.get(target_col) or "").strip()
            if not q or not t:
                continue
            records.append(
                QueryRecord(
                    user_id=(row.get(user_col) or "unknown_user").strip(),
                    query=q,
                    history_item_ids=_split_history(row.get(history_col) or ""),
                    target_item_id=t,
                )
            )
    return records


def build_target_user(record: QueryRecord) -> TargetUser:
    return TargetUser(
        user_id=record.user_id,
        profile_text=f"Derived from history size={len(record.history_item_ids)}",
        history_item_ids=record.history_item_ids,
    )
