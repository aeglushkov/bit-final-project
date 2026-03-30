"""Create small dataset subsets for initial SpatialScore experiments.

Run from literature/spatialscore/code/:
    python ../../../experiments/spatialscore/create_subsets.py

Creates two filtered JSONs in dataset/:
  - SpatialScore_test50.json  -- 50 MMVP samples (simplest case: single-image, multi-choice)
  - SpatialScore_diverse.json -- ~10 samples per source, single-image only (tests all eval branches)
"""

import json
import os
from collections import defaultdict

DATASET_PATH = "./dataset/SpatialScore.json"


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run this script from literature/spatialscore/code/"
        )
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} total samples")
    return data


def save_subset(subset, filename):
    path = os.path.join("./dataset", filename)
    with open(path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved {len(subset)} samples to {path}")


def print_stats(subset, label):
    """Print breakdown of a subset by source, category, question_type."""
    by_source = defaultdict(int)
    by_category = defaultdict(int)
    by_qtype = defaultdict(int)
    for s in subset:
        by_source[s.get("source", "unknown")] += 1
        by_category[s.get("category", "unknown")] += 1
        by_qtype[s.get("question_type", "unknown")] += 1

    print(f"\n--- {label} ---")
    print(f"Total: {len(subset)} samples")
    print(f"\nBy source:")
    for k, v in sorted(by_source.items()):
        print(f"  {k}: {v}")
    print(f"\nBy category:")
    for k, v in sorted(by_category.items()):
        print(f"  {k}: {v}")
    print(f"\nBy question type:")
    for k, v in sorted(by_qtype.items()):
        print(f"  {k}: {v}")


def create_test50(data):
    """50 MMVP samples -- simplest evaluation path (single-image, multi-choice)."""
    mmvp = [d for d in data if d.get("source") == "MMVP"][:50]
    if len(mmvp) < 50:
        print(f"WARNING: Only found {len(mmvp)} MMVP samples (expected 50)")
    return mmvp


def create_diverse(data, per_source=10):
    """~10 samples per source, single-image only.

    This tests all prompt templates and evaluation branches
    without the complexity of video/multi-frame inputs.
    """
    by_source = defaultdict(list)
    for d in data:
        if d.get("input_modality") == "single-image":
            by_source[d.get("source", "unknown")].append(d)

    subset = []
    for source, items in sorted(by_source.items()):
        n = min(per_source, len(items))
        subset.extend(items[:n])
        print(f"  {source}: {n} samples (of {len(items)} available)")

    return subset


def main():
    data = load_dataset()

    # Subset 1: 50 MMVP samples
    test50 = create_test50(data)
    print_stats(test50, "test50 (MMVP)")
    save_subset(test50, "SpatialScore_test50.json")

    # Subset 2: diverse single-image samples
    print("\nCreating diverse subset (10 per source, single-image):")
    diverse = create_diverse(data)
    print_stats(diverse, "diverse (all single-image sources)")
    save_subset(diverse, "SpatialScore_diverse.json")


if __name__ == "__main__":
    main()
