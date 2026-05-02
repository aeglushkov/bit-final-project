"""Build YOLO-World detection vocabulary by extracting candidate object terms
from VSI-Bench questions/options/ground-truth and merging with the EVA paper's
`customized_classes`.

Heuristic — output is intended for manual review (the user can edit the
generated file). Tunables: --min-count, --max-classes, --include-options."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "at", "to", "from", "is", "are", "was",
        "were", "be", "been", "being", "do", "does", "did", "doing", "have", "has",
        "had", "having", "will", "would", "should", "could", "can", "may", "might",
        "must", "shall",
        "i", "you", "we", "they", "he", "she", "it", "this", "that", "these", "those",
        "my", "your", "our", "their", "his", "her", "its",
        "what", "which", "where", "when", "who", "why", "how", "whose", "whom",
        "and", "or", "but", "with", "by", "for", "as", "if", "than", "then", "so",
        "not", "no", "yes", "true", "false", "any", "all", "some", "each", "every",
        "many", "much", "more", "most", "few", "less", "least", "very", "just", "only",
        "about", "above", "below", "behind", "between", "among", "across", "around",
        "near", "next", "into", "onto", "out", "outside", "inside", "through", "over",
        "under",
        "front", "back", "side", "left", "right", "top", "bottom", "center", "middle",
        "facing", "walking", "standing", "sitting", "looking", "moving", "go", "going",
        "came", "come", "came", "approach", "approaching",
        "room", "scene", "video", "image", "frame", "frames", "place", "places",
        "object", "objects", "thing", "things", "item", "items",
        "color", "shape", "size", "distance", "direction", "order", "path", "route",
        "appearance", "estimation", "planning", "counting", "absolute", "relative",
        "easy", "medium", "hard",
        "answer", "option", "options", "choose", "select", "based", "given", "asked",
        "ask", "show", "shows", "showing", "describe", "describes", "according",
        "you're", "i'm", "im", "youre",
        # measurement / metric scaffolding (not objects)
        "length", "width", "height", "depth", "dimension", "dimensions", "volume",
        "area", "diameter", "radius", "perimeter",
        "centimeters", "centimeter", "meters", "meter", "feet", "foot", "inches", "inch",
        "measured", "measure", "measuring", "measurement", "measurements",
        "longest", "shortest", "largest", "smallest", "tallest", "biggest",
        "closest", "farthest", "furthest", "nearest", "next", "last", "first",
        "single", "double", "triple", "multiple", "several",
        "edge", "end", "ends", "corner", "corners", "point", "points",
        "turn", "turning", "moving", "stand", "stands", "stood", "stops", "stopped",
        "facing", "faced", "face", "look", "looks", "looked", "see", "sees", "seen", "view", "viewed",
        "following", "follow", "follows", "follower", "followed",
        "approximately", "exactly", "roughly", "about", "around",
        "yes", "no", "none", "either", "neither", "both",
        # generic colors and material adjectives that aren't objects
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "grey", "beige", "dark", "light",
        "wooden", "metal", "metallic", "plastic", "glass", "fabric", "leather",
        # part-of-question vocabulary
        "side", "sides", "above", "below", "behind", "behind", "front", "back", "right", "left",
        "horizontal", "vertical", "diagonal",
        # quantifiers + counting words
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "second", "third", "fourth", "fifth", "lot",
    }
)


def extract_candidates(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    return [t for t in tokens if t not in STOPWORDS]


def load_paper_classes(paper_code_dir: Path) -> list[str]:
    paper_code_dir = paper_code_dir.resolve()
    sys.path.insert(0, str(paper_code_dir))
    try:
        from detection_classes import customized_classes  # type: ignore[import-not-found]

        return list(customized_classes)
    finally:
        if str(paper_code_dir) in sys.path:
            sys.path.remove(str(paper_code_dir))


def collect_vsibench_terms(min_count: int, include_options: bool) -> Counter:
    from datasets import load_dataset

    ds = load_dataset("nyu-visionx/VSI-Bench", split="test")
    counter: Counter = Counter()
    for row in ds:
        for field in ("question", "ground_truth"):
            v = row.get(field)
            if v is None:
                continue
            if isinstance(v, list):
                v = " ".join(map(str, v))
            for cand in extract_candidates(str(v)):
                counter[cand] += 1
        if include_options:
            opts = row.get("options")
            if opts:
                for opt in opts:
                    for cand in extract_candidates(str(opt)):
                        counter[cand] += 1
    return Counter({k: v for k, v in counter.items() if v >= min_count})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-code-dir", type=Path, required=True, help="Path to literature/EmbodiedVideoAgent/code/")
    ap.add_argument("--output", type=Path, required=True, help="Where to write the merged class list.")
    ap.add_argument("--min-count", type=int, default=3, help="Drop candidates appearing in fewer than N rows.")
    ap.add_argument("--max-classes", type=int, default=200, help="Cap on total classes (YOLO-World slows with many).")
    ap.add_argument("--include-options", action="store_true", help="Also harvest terms from MCA options.")
    args = ap.parse_args()

    paper = load_paper_classes(args.paper_code_dir)
    print(f"Paper vocab: {len(paper)} classes")

    vsi = collect_vsibench_terms(min_count=args.min_count, include_options=args.include_options)
    print(f"VSI-Bench candidates with count >= {args.min_count}: {len(vsi)}")

    paper_set = {c.lower() for c in paper}
    new_classes = [c for c, _ in vsi.most_common() if c not in paper_set]

    merged = list(paper) + new_classes
    if len(merged) > args.max_classes:
        dropped = len(merged) - args.max_classes
        merged = merged[: args.max_classes]
        print(f"Capped at --max-classes={args.max_classes}; dropped {dropped} low-frequency entries", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(merged) + "\n")
    print(f"Wrote {len(merged)} classes to {args.output}")


if __name__ == "__main__":
    main()
