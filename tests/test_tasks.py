import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import tasks as tk  # noqa: E402

CHOICE = {"kind": "choice", "template": "Pick one.\nA: {A}\nB: {B}", "options": {"x": "${now} today", "y": "${later} {when}"},
          "grid": {"now": [10, 50], "later": [100], "when": [{"when": "in {days} days", "days": 7}]}}
LIKERT = {"kind": "likert", "template": "\"{statement}\" 1 = {low}, 7 = {high}.", "scale": [1, 7],
          "anchors": ["disagree", "agree"], "grid": {"statement": ["Cats are fine."]}}


def sig(z):
    return 1 / (1 + np.exp(-z))


# ---------- items and prompts ----------
def test_expand_grid_product_and_merge():
    items = tk.expand(CHOICE)
    assert items == [{"now": 10, "later": 100, "when": "in {days} days", "days": 7},
                     {"now": 50, "later": 100, "when": "in {days} days", "days": 7}]


def test_prompts_swap_order_and_fill_nested_fields():
    a, b = tk.prompts(CHOICE, tk.expand(CHOICE)[0])
    assert a == "Pick one.\nA: $10 today\nB: $100 in 7 days"
    assert b == "Pick one.\nA: $100 in 7 days\nB: $10 today"


def test_likert_prompts_reverse_anchors_and_statements_override():
    items = tk.expand(LIKERT, ["Dogs are fine.", "  "])
    assert items == [{"statement": "Dogs are fine."}]
    a, b = tk.prompts(LIKERT, items[0])
    assert a.endswith("1 = disagree, 7 = agree.") and b.endswith("1 = agree, 7 = disagree.")
    assert tk.labels(LIKERT) == ["1", "2", "3", "4", "5", "6", "7"]


class FakeTok:
    """Whole strings in the vocabulary are one token; anything else is one token per character."""
    def __init__(self, vocab):
        self.vocab = {w: i for i, w in enumerate(vocab)}

    def __call__(self, s, add_special_tokens=False):
        return type("E", (), {"input_ids": [self.vocab[s]] if s in self.vocab else [1000 + ord(c) for c in s]})


def test_label_tokens_single_token_forms_or_fallback():
    tok = FakeTok(["A", "B", " A", " B"])
    assert tk.label_tokens(CHOICE, tok) == [[0, 2], [1, 3]]
    assert tk.label_tokens({**CHOICE, "labels": ["yes", "no"]}, FakeTok(["yes", "No", "Yes"])) == [[0, 2], [1]]  # case forms
    assert tk.label_tokens({**CHOICE, "labels": ["yes", "Yes"]}, FakeTok(["yes", "Yes"])) is None  # two labels, one token
    assert tk.label_tokens({**CHOICE, "labels": ["Plan one", "Plan two"]}, FakeTok(["A"])) is None  # several tokens: sample


# ---------- parsers ----------
@pytest.mark.parametrize("text, want", [
    ("B", 1), (" A.", 0), ("**B**", 1), ("Option A", 0), ("I would pick B over A", 1), ("A) take the money", 0),
    ("a", None), ("I'd take the bet", None), ("", None), ("AB", None), ("Answer: b", None), ("\n\nB\n", 1),
])
def test_parse_choice(text, want):
    assert tk.parse_choice(text, ["A", "B"]) == want


def test_parse_choice_word_labels_ignore_case():
    assert tk.parse_choice("I will Accept.", ["reject", "accept"]) == 1
    assert tk.parse_choice("acceptable", ["reject", "accept"]) is None


@pytest.mark.parametrize("text, want", [
    ("5", 5), ("Rating: 6/7", 6), ("7.", 7), ("1", 1), ("I'd say 4, maybe 5", 4), ("8", None), ("0", None),
    ("-3", None), ("4.5", None), ("Strongly agree", None), ("", None), ("10", None), ("score 3.0", 3),
])
def test_parse_likert(text, want):
    assert tk.parse_likert(text, 1, 7) == want


# ---------- scoring an item ----------
def test_score_choice_averages_orders_and_reports_position_bias():
    # first prompt: x is A, y is B; second: y is A. P(y) = 0.6 then 0.8; the first-listed option got 0.4 and 0.8
    v, valid, bias = tk.score(CHOICE, [0.4, 0.6], [0.8, 0.2])
    assert v == pytest.approx(0.7) and valid == pytest.approx(0) and bias == pytest.approx(0.6)
    v, valid, _ = tk.score(CHOICE, [0.2, 0.3], [0.25, 0.25])  # unnormalized: renormalized, validity is the mass
    assert v == pytest.approx((0.6 + 0.5) / 2) and valid == pytest.approx((np.log(0.5) + np.log(0.5)) / 2)
    assert tk.score(CHOICE, [0, 0], [0.5, 0.5]) == (None, None, None)


def test_score_likert_flips_reversed_direction():
    p0 = np.eye(7)[5]  # rates 6 with "7 = agree"
    p1 = np.eye(7)[1]  # rates 2 with "7 = disagree": the same opinion
    v, _, bias = tk.score(LIKERT, p0, p1)
    assert v == pytest.approx(6) and bias == pytest.approx(0)
    v, _, bias = tk.score(LIKERT, np.eye(7)[6], np.eye(7)[6])  # always "7": a number bias, no opinion
    assert v == pytest.approx(4) and bias == pytest.approx(3)


# ---------- fits ----------
DD = {"kind": "choice", "fit": {"type": "hyperbolic", "now": "now", "later": "later", "delay": "days"},
      "grid": {"now": [10, 25, 40, 55, 70, 85, 95], "later": [100], "days": [1, 7, 30, 91, 182, 365]}}


@pytest.mark.parametrize("logk", [-3.5, -2.2, -1.0])
def test_hyperbolic_recovers_k(logk):
    items = tk.expand(DD)
    c = tk.columns(items)
    p = sig(15 * (c["later"] / (1 + 10 ** logk * c["days"]) - c["now"]) / c["later"])
    f = tk.fit(DD, c, p)
    assert f["value"] == pytest.approx(logk, abs=0.03) and not f["at_bound"]


def test_hyperbolic_noisy_choices():
    items = tk.expand(DD)
    c = tk.columns(items)
    p = sig(8 * (c["later"] / (1 + 10 ** -2.0 * c["days"]) - c["now"]) / c["later"])
    p = np.clip(p + np.random.default_rng(0).normal(0, 0.05, len(p)), 0, 1)
    assert tk.fit(DD, c, p)["value"] == pytest.approx(-2.0, abs=0.15)


def test_hyperbolic_flags_bound_when_choices_ignore_amounts():
    items = tk.expand(DD)
    assert tk.fit(DD, tk.columns(items), np.full(len(items), 0.2))["at_bound"]


def test_bootstrap_paired_draws():
    items = tk.expand(DD)
    c = tk.columns(items)
    p0 = sig(15 * (c["later"] / (1 + 10 ** -2.5 * c["days"]) - c["now"]) / c["later"])
    p1 = sig(15 * (c["later"] / (1 + 10 ** -2.0 * c["days"]) - c["now"]) / c["later"])
    f0, b0 = tk.bootstrap(DD, items, list(p0), B=60, seed=3)
    f1, b1 = tk.bootstrap(DD, items, list(p1), B=60, seed=3)
    assert f0["rate"] == pytest.approx(p0.mean())
    again = tk.bootstrap(DD, items, list(p0), B=60, seed=3)[1]
    assert np.array_equal(again["value"], b0["value"])  # fixed seed, fixed draws
    lo, hi = tk.ci(b1["value"] - b0["value"])
    assert lo > 0.3 and hi < 0.7  # a 0.5 shift in log k, paired
