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


def _pick_optional_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    lowered = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _split_history(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    cleaned = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    for sep in ["|", ";", ","]:
        if sep in cleaned:
            return [x.strip() for x in cleaned.split(sep) if x.strip()]
    return [x.strip() for x in cleaned.split() if x.strip()]


def _parse_target(raw: str) -> str:
    vals = _split_history(raw)
    return vals[0] if vals else (raw or "").strip()


def load_metadata_csv(path: str | Path) -> dict[str, ItemProfile]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("metadata.csv has no header")
        item_col = _pick_column(reader.fieldnames, ["item_id", "asin", "parent_asin", "id"])
        title_col = _pick_optional_column(reader.fieldnames, ["title", "name"])
        desc_col = _pick_optional_column(reader.fieldnames, ["description", "desc", "features"])
        tags_col = _pick_optional_column(reader.fieldnames, ["categories", "category", "tags"])
        price_col = _pick_optional_column(reader.fieldnames, ["price"])

        items: dict[str, ItemProfile] = {}
        for row in reader:
            item_id = (row.get(item_col) or "").strip()
            if not item_id:
                continue
            tags_raw = (row.get(tags_col) or "") if tags_col else ""
            tags = [x.strip() for x in tags_raw.replace("[", "").replace("]", "").replace("'", "").split(",") if x.strip()]
            title = ((row.get(title_col) or "").strip() if title_col else "") or item_id
            desc = (row.get(desc_col) or "").strip() if desc_col else ""
            price = (row.get(price_col) or "").strip() if price_col else ""
            merged_desc = f"{desc} price:{price}".strip()
            items[item_id] = ItemProfile(item_id=item_id, title=title, tags=tags, description=merged_desc)
    return items


def load_query_csv(path: str | Path) -> list[QueryRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("query_data csv has no header")
        user_col = _pick_column(reader.fieldnames, ["user_id", "reviewerID", "uid"])
        query_col = _pick_column(reader.fieldnames, ["query", "instruction", "search_query", "new_query"])
        history_col = _pick_optional_column(
            reader.fieldnames,
            ["history_item_ids", "history", "history_items", "clicked_items", "remaining_interaction_string"],
        )
        target_col = _pick_column(reader.fieldnames, ["target_item_id", "target", "label_item", "ground_truth_item", "targets"])

        records: list[QueryRecord] = []
        for row in reader:
            q = (row.get(query_col) or "").strip()
            t = _parse_target(row.get(target_col) or "")
            if not q or not t:
                continue
            history_raw = (row.get(history_col) or "") if history_col else ""
            records.append(
                QueryRecord(
                    user_id=(row.get(user_col) or "unknown_user").strip(),
                    query=q,
                    history_item_ids=_split_history(history_raw),
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
