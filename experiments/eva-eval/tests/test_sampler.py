from collections import Counter

from eva_eval.eval.sampler import stratified_indices


def _qtypes(per_type: dict[str, int]) -> list[str]:
    out = []
    for qt, n in per_type.items():
        out.extend([qt] * n)
    return out


def test_stratified_balances_across_types():
    qtypes = _qtypes({"a": 50, "b": 50, "c": 50})
    idxs = stratified_indices(qtypes, total=30, seed=0)
    counts = Counter(qtypes[i] for i in idxs)
    assert sum(counts.values()) == 30
    assert all(c == 10 for c in counts.values())


def test_stratified_handles_uneven_remainder():
    qtypes = _qtypes({"a": 10, "b": 10, "c": 10})
    idxs = stratified_indices(qtypes, total=10, seed=0)
    counts = Counter(qtypes[i] for i in idxs)
    assert sum(counts.values()) == 10
    assert max(counts.values()) - min(counts.values()) <= 1


def test_stratified_caps_at_available():
    qtypes = _qtypes({"a": 3, "b": 100})
    idxs = stratified_indices(qtypes, total=50, seed=0)
    counts = Counter(qtypes[i] for i in idxs)
    assert counts["a"] <= 3


def test_stratified_deterministic_with_same_seed():
    qtypes = _qtypes({"a": 50, "b": 50})
    a = stratified_indices(qtypes, total=20, seed=42)
    b = stratified_indices(qtypes, total=20, seed=42)
    assert a == b


def test_stratified_changes_with_seed():
    qtypes = _qtypes({"a": 50, "b": 50})
    a = stratified_indices(qtypes, total=20, seed=1)
    b = stratified_indices(qtypes, total=20, seed=2)
    assert a != b


def test_stratified_zero_total():
    assert stratified_indices(["a", "b", "c"], total=0, seed=0) == []


def test_stratified_returns_sorted_indices():
    qtypes = _qtypes({"a": 50, "b": 50})
    idxs = stratified_indices(qtypes, total=20, seed=0)
    assert idxs == sorted(idxs)
