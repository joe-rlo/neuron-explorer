import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import experiment as ex  # noqa: E402

FOOD = {"name": "food words", "type": "words", "words": ["pizza", "pasta", "cheese", "dinner", "recipe"]}
PIZZA = {"name": "mentions pizza", "type": "contains", "word": "pizza"}


# ---------- measures ----------
def test_words_per_100_whole_word_case_insensitive():
    assert ex.score(FOOD, "Pizza and PASTA for dinner tonight") == pytest.approx(100 * 3 / 6)
    assert ex.score(FOOD, "pizzas, cheesecake and a pizzeria") == 0  # whole words only
    assert ex.score(FOOD, "") == 0


def test_words_multiword_phrase():
    m = {"type": "words", "words": ["ice cream"]}
    assert ex.score(m, "I like ice cream and ice") == pytest.approx(100 * 1 / 6)


def test_words_stems_match_word_starts():
    m = {"type": "words", "words": ["alien", "extr"], "stems": True}
    assert ex.score(m, "Aliens from extraterrestrial space, not mine") == pytest.approx(100 * 2 / 6)
    assert ex.score(m, "salient") == 0  # still anchored at the start of a word


def test_contains():
    assert ex.score(PIZZA, "I love Pizza!") == 1
    assert ex.score(PIZZA, "pizzeria") == 0


def test_regex_counts_matches():
    m = {"type": "regex", "pattern": r"\bpepperoni\b"}
    assert ex.score(m, "pepperoni, pepperoni and pepperonis") == 2


def test_unknown_measure():
    with pytest.raises(ValueError):
        ex.score({"type": "vibes"}, "x")


def test_script_mix():
    assert ex.dominant_script("Who are you?") == "LATIN"
    assert ex.script_mix("hello", "LATIN") == 0
    assert ex.script_mix("ab你好", "LATIN") == pytest.approx(0.5)
    assert ex.script_mix("123 !!", "LATIN") == 0  # no letters
    assert ex.dominant_script("你好吗 ok") == "CJK"


# ---------- statistics ----------
def test_bootstrap_fixed_seed_fixed_interval():
    x = np.random.default_rng(1).normal(3, 1, 20)
    a, b = ex.bootstrap_ci(x, seed=7), ex.bootstrap_ci(x, seed=7)
    assert a == b
    assert a[0] < x.mean() < a[1]
    assert ex.bootstrap_ci([1.0]) is None


def test_bootstrap_covers_true_mean_mostly():
    rng, hits = np.random.default_rng(0), 0
    for s in range(200):
        lo, hi = ex.bootstrap_ci(rng.normal(0, 1, 30), B=500, seed=s)
        hits += lo <= 0 <= hi
    assert 0.88 <= hits / 200 <= 0.99


def test_cohens_d_paired():
    x, y = np.array([3.0, 5, 4, 6]), np.array([1.0, 2, 2, 3])
    d = x - y
    assert ex.cohens_d_paired(x, y) == pytest.approx(d.mean() / d.std(ddof=1))
    assert ex.cohens_d_paired([1, 2], [1, 2]) == 0.0
    assert ex.cohens_d_paired([2, 3], [1, 2]) is None  # constant non-zero difference: infinite
    assert ex.cohens_d_paired([1], [0]) is None


def test_paired_skips_missing():
    p = ex.paired([1.0, None, 3.0], [0.0, 1.0, 1.0])
    assert p["n"] == 2 and p["diff"] == pytest.approx(1.5)


def test_sentence():
    p = {"diff": 2.1, "ci": [0.9, 3.4], "d": 1.2}
    assert ex.sentence("+0.5", FOOD, p) == \
        "+0.5 raised food words by 2.10 per 100 words (95% CI 0.90 to 3.40) versus no steering."
    s = ex.sentence("−0.5", PIZZA, {"diff": -0.25, "ci": [-0.5, 0.1], "d": -0.4})
    assert s.startswith("−0.5 lowered mentions pizza by 25 percentage points (95% CI -50 to 10)")
    assert "not distinguishable" in s


# ---------- dose-response ----------
def _sweep(effect, coherence):
    noise = np.random.default_rng(0).normal(0, 0.1, 8)  # shared across doses: samples are paired by seed
    pts = {}
    for d in (-1.0, -0.5, 0, 0.5, 1.0):
        pts[d] = {"measure": list(effect(d) + noise), "coherence": list(coherence(d) + noise * 0),
                  "script": [0.0] * 8}
    return pts


def test_window_effect_and_toxicity():
    pts = _sweep(lambda d: 5 * d if d > 0 else 0, lambda d: -1.0 - 3 * abs(d) ** 2)
    w = ex.window(pts, {"coherence_drop": 1.0, "script_mix": 0.2})
    pos = w["positive"]
    assert pos["ed"] == 0.5 and pos["td"] == 1.0
    assert pos["window"] == pytest.approx(0.5) and pos["index"] == pytest.approx(2.0)
    assert w["negative"]["note"] == "no effect in range"
    assert ex.best_index(w) == pytest.approx(2.0)


def test_window_no_toxic_dose():
    pts = _sweep(lambda d: 5 * d, lambda d: -1.0)
    w = ex.window(pts, {"coherence_drop": 1.0, "script_mix": 0.2})
    assert w["positive"]["ed"] == 0.5 and w["positive"]["td"] is None
    assert w["positive"]["note"] == "TD > 1"
    assert ex.best_index(w) == pytest.approx(2.0)  # max dose over ED


def test_window_script_mix_is_toxic():
    pts = _sweep(lambda d: 5 * d, lambda d: -1.0)
    pts[1.0]["script"] = [0.5] * 8
    w = ex.window(pts, {"coherence_drop": 1.0, "script_mix": 0.2})
    assert w["positive"]["td"] == 1.0


# ---------- tuning curves ----------
def test_item_selectivity():
    assert ex.item_selectivity([1.0, 1.0, 1.0]) == pytest.approx(0)  # flat
    assert ex.item_selectivity([2.0, 0, 0]) == pytest.approx(1)  # one item only
    assert ex.item_selectivity([3.0, 1.0, 1.0]) == pytest.approx((3 - 1) / (3 + 1))
    assert ex.item_selectivity([0.0, 0.0]) == 0
    r = np.array([[3.0, 1.0], [1.0, 1.0], [1.0, 1.0]])  # vectorized over neurons
    assert ex.item_selectivity(r) == pytest.approx([0.5, 0.0])


def test_category_d_known_value():
    cats = ["food"] * 3 + ["other"] * 3
    r = np.array([5.0, 6, 7, 1, 2, 3])
    sd = np.sqrt((2 * 1.0 + 2 * 1.0) / 4)
    assert ex.category_d(r, cats, "food") == pytest.approx(4 / sd)
    assert ex.category_d(r, cats, "other") == pytest.approx(-4 / sd)


def test_food_vs_pizza_neuron():
    """A food neuron: high category d', low item selectivity. A pizza neuron: both high."""
    items = ["pizza", "burger", "sushi", "salad", "car", "dog", "chair", "river"]
    cats = ["food"] * 4 + ["other"] * 4
    food = np.array([5.0, 5.2, 4.8, 5.1, 1.0, 1.1, 0.9, 1.0])
    pizza = np.array([8.0, 1.6, 1.4, 1.5, 0.5, 0.6, 0.4, 0.5])
    assert ex.category_d(food, cats, "food") > 5 and ex.item_selectivity(food) < 0.4
    assert ex.category_d(pizza, cats, "food") > 0.8 and ex.item_selectivity(pizza) > 0.7
    assert items[int(pizza.argmax())] == "pizza"
