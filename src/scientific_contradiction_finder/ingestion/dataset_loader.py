"""Dataset loaders and adapters for benchmarks (CAS-Con150, SciFact, Evidence Inference)."""

import json
from pathlib import Path

from scientific_contradiction_finder.schemas import (
    ConditionSlots,
    DirectionLabel,
    EvidenceSpan,
    Finding,
    FindingPair,
    RelationLabel,
)


class BenchmarkLoader:
    """Loads and standardizes scientific contradiction and claim verification benchmarks."""

    def __init__(self, data_dir: Path | None = None):
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

    def load_cas_con150(self) -> list[FindingPair]:
        """Loads the CAS-Con150 gold standard benchmark."""
        file_path = self.data_dir / "annotations" / "cas_con150.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Benchmark file not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs: list[FindingPair] = []
        for item in data:
            fa_raw = item["finding_a"]
            fb_raw = item["finding_b"]

            fa = Finding(
                finding_id=fa_raw["finding_id"],
                paper_id=fa_raw["paper_id"],
                paper_title=fa_raw["paper_title"],
                paper_doi=fa_raw.get("paper_doi"),
                paper_url=fa_raw.get("paper_url"),
                evidence=EvidenceSpan(**fa_raw["evidence"]),
                finding_claim=fa_raw["finding_claim"],
                direction=DirectionLabel(fa_raw["direction"]),
                conditions=ConditionSlots(**fa_raw["conditions"]),
            )

            fb = Finding(
                finding_id=fb_raw["finding_id"],
                paper_id=fb_raw["paper_id"],
                paper_title=fb_raw["paper_title"],
                paper_doi=fb_raw.get("paper_doi"),
                paper_url=fb_raw.get("paper_url"),
                evidence=EvidenceSpan(**fb_raw["evidence"]),
                finding_claim=fb_raw["finding_claim"],
                direction=DirectionLabel(fb_raw["direction"]),
                conditions=ConditionSlots(**fb_raw["conditions"]),
            )

            pair = FindingPair(
                pair_id=item["pair_id"],
                finding_a=fa,
                finding_b=fb,
                ground_truth_relation=RelationLabel(item["ground_truth_relation"]),
                domain=item.get("domain"),
                notes=item.get("notes"),
            )
            pairs.append(pair)

        return pairs

    def get_train_test_split(
        self, pairs: list[FindingPair], test_ratio: float = 0.3, random_seed: int = 42
    ) -> tuple[list[FindingPair], list[FindingPair]]:
        """Splits pairs deterministically while preserving stratified label distributions."""
        import random
        from collections import defaultdict

        rng = random.Random(random_seed)
        by_label: dict[str, list[FindingPair]] = defaultdict(list)
        for p in pairs:
            by_label[p.ground_truth_relation.value].append(p)

        train_pairs: list[FindingPair] = []
        test_pairs: list[FindingPair] = []

        for label, group in by_label.items():
            shuffled = list(group)
            rng.shuffle(shuffled)
            n_test = max(1, int(len(shuffled) * test_ratio))
            test_pairs.extend(shuffled[:n_test])
            train_pairs.extend(shuffled[n_test:])

        # Shuffle again deterministically
        rng.shuffle(train_pairs)
        rng.shuffle(test_pairs)
        return train_pairs, test_pairs
