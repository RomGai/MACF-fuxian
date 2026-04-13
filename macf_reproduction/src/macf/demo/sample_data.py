from __future__ import annotations

from ..types import ItemProfile, SimilarUserProfile, TargetUser


def load_sample_data() -> tuple[TargetUser, dict[str, SimilarUserProfile], dict[str, ItemProfile]]:
    items = {
        "i1": ItemProfile("i1", "Interstellar", ["sci-fi", "space"], "Epic emotional space odyssey."),
        "i2": ItemProfile("i2", "Arrival", ["sci-fi", "linguistic"], "Thoughtful first-contact drama."),
        "i3": ItemProfile("i3", "The Martian", ["sci-fi", "survival"], "Science-driven survival on Mars."),
        "i4": ItemProfile("i4", "Blade Runner 2049", ["sci-fi", "neo-noir"], "Moody future mystery."),
        "i5": ItemProfile("i5", "Dune", ["sci-fi", "epic"], "Political desert-space epic."),
        "i6": ItemProfile("i6", "Ex Machina", ["sci-fi", "ai"], "Intimate AI ethics thriller."),
        "i7": ItemProfile("i7", "Gravity", ["space", "thriller"], "Intense orbital survival film."),
        "i8": ItemProfile("i8", "Annihilation", ["sci-fi", "mind-bending"], "Psychological speculative journey."),
        "i9": ItemProfile("i9", "Moon", ["sci-fi", "character"], "Lonely lunar identity story."),
        "i10": ItemProfile("i10", "Ad Astra", ["space", "drama"], "Personal journey through deep space."),
        "i11": ItemProfile("i11", "Oblivion", ["sci-fi", "action"], "Post-apocalyptic memory mystery."),
        "i12": ItemProfile("i12", "Sunshine", ["space", "thriller"], "Crew mission to revive the sun."),
    }
    similar_users = {
        "u2": SimilarUserProfile("u2", "Likes cerebral sci-fi and emotional arcs.", ["i2", "i9", "i6"], 0.92),
        "u3": SimilarUserProfile("u3", "Prefers grounded science and space survival.", ["i3", "i7", "i12"], 0.89),
        "u4": SimilarUserProfile("u4", "Enjoys epic worldbuilding and philosophical tone.", ["i1", "i5", "i4"], 0.87),
        "u5": SimilarUserProfile("u5", "Likes tense AI or identity mysteries.", ["i6", "i11", "i4"], 0.84),
        "u6": SimilarUserProfile("u6", "Space drama with introspective characters.", ["i10", "i1", "i9"], 0.81),
    }
    target = TargetUser("u1", "Enjoys thoughtful sci-fi, space settings, and emotional depth.", ["i1", "i2", "i7"])
    return target, similar_users, items
