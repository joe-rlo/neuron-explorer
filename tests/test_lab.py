import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import lab  # noqa: E402

M = {"name": "food words", "type": "words", "words": ["pizza"]}


def pt(dose, sentence=None):
    return {"dose": dose, **({"vs0": {"sentence": sentence}} if sentence else {})}


# ---------- notebook ----------
def test_describe_and_headline_per_kind():
    chat = {"id": "x", "kind": "chat", "doses": [0.5, 0, -0.5], "mode": "add", "prompt": "Who are you?", "measures": [M],
            "summary": [{"vs0": {"sentence": "+0.5 raised it."}}, {}, {"vs0": {"sentence": "−0.5 lowered it."}}]}
    assert lab.describe(chat) == "steer ±0.5 · “Who are you?” · food words"
    assert lab.headline(chat) == "+0.5 raised it. −0.5 lowered it."
    sweep = {"kind": "sweep", "prompt": "Hi", "measures": [M],
             "summary": [pt(-1, "−1 a."), pt(-0.5, "−0.5 b."), pt(0), pt(0.5, "+0.5 c."), pt(1, "+1 d.")]}
    assert lab.headline(sweep) == "+1 d. −1 a."  # the extreme doses on each side
    assert lab.headline({**sweep, "cancelled": True}).startswith("cancelled")
    ablate = {"kind": "ablate", "method": "mean", "neurons": [{}, {}], "prompt": "Hi", "measures": [M],
              "summary": [{}, {"vs0": {"sentence": "Ablating lowered it."}}]}
    assert lab.describe(ablate) == "mean ablation of 2 neurons · “Hi” · food words"
    assert lab.headline(ablate) == "Ablating lowered it."
    task = {"kind": "task", "task": {"name": "risk"}, "measures": [{"name": "CE", "type": "task"}], "summary": [pt(0)]}
    assert lab.describe(task) == "task · risk · CE" and lab.headline(task) == ""


def test_long_prompts_clip_and_entry_fields():
    rec = {"id": "20261028-101010-L1N2-sweep-abcd", "kind": "sweep", "prompt": "word " * 40, "measures": [M], "summary": [],
           "layer": 1, "neurons": [{"layer": 1, "n": 2}], "prediction": "  ", "verdict": None}
    e = lab.entry(rec)
    assert len(e["what"]) < 100 and "…" in e["what"]
    assert e["prediction"] is None  # blank predictions count as none
    assert lab.EXPERIMENT_FILE.match(rec["id"]) and not lab.EXPERIMENT_FILE.match("../windows")


# ---------- blind rating ----------
def test_binomial_tail_matches_direct_sum():
    assert lab.binom_sf(0, 10, 1 / 3) == 1.0
    assert abs(lab.binom_sf(10, 10, 1 / 3) - (1 / 3) ** 10) < 1e-15
    assert abs(lab.binom_sf(7, 10, 1 / 3) - 0.01966) < 1e-4  # 7 of 10 at chance 1/3
    assert abs(lab.binom_sf(2, 3, 0.5) - 0.5) < 1e-12


def trial(prop, target, picked):
    return {"property": prop, "target": target, "picked": picked}


def test_blind_stats_counts_hits_against_target():
    ts = [trial("aliens", "up", "up")] * 7 + [trial("aliens", "up", "zero")] * 2 + [trial("Aliens ", "down", "up")]
    s = lab.blind_summary(ts)
    assert s["n"] == 10 and s["hits"] == 7 and abs(s["accuracy"] - 0.7) < 1e-12
    assert s["p"] < 0.05 and "unlikely to be luck" in s["sentence"]
    assert list(s["by_property"]) == ["aliens"]  # properties group without case or spaces
    few = lab.blind_stats([trial("x", "up", "up")] * 2)
    assert "too few" in few["sentence"] and lab.blind_stats([])["accuracy"] is None
