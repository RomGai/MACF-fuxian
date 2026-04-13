from macf.sufficiency import evaluate_sufficiency
from macf.types import RankedItem, RankedListDraft


def test_sufficiency_passes_with_enough_ordered_items() -> None:
    items = [RankedItem(rank=i + 1, item_id=f"i{i}", reason="relevant reason text", score=1.0 - i * 0.01) for i in range(10)]
    draft = RankedListDraft(items=items)
    result = evaluate_sufficiency(draft, top_k=10)
    assert result.overall_pass
