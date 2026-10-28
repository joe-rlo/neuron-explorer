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


# ---------- challenges ----------
CH = {"id": "c", "measure": {"name": "mentions pizza", "type": "contains", "word": "pizza"}, "op": ">=", "target": 0.5, "minimize": "dose"}
TOX = {"coherence_drop": 1.0, "script_mix": 0.2}


def test_challenge_passes_only_when_goal_met_and_coherent():
    ok = lab.challenge_result(CH, [1, 1, 0, 1], [-0.5] * 4, [-1.0] * 4, [0] * 4, TOX)
    assert ok["passed"] and ok["value"] == 0.75 and abs(ok["coherence_drop"] - 0.5) < 1e-12
    assert ok["sentence"].startswith("Passed. mentions pizza: 75% of replies (goal ≥ 50% of replies)")
    broke = lab.challenge_result(CH, [1, 1, 1, 1], [-0.5] * 4, [-2.0] * 4, [0] * 4, TOX)
    assert broke["met"] and not broke["passed"] and "text broke" in broke["sentence"]
    script = lab.challenge_result(CH, [1] * 4, [-0.5] * 4, [-0.6] * 4, [0.5] * 4, TOX)
    assert not script["coherent"] and "other script" in script["sentence"]
    low = lab.challenge_result({**CH, "op": "<=", "target": 0.2}, [0, 0, 1, 0, 0, 0], [-0.5] * 6, [-0.5] * 6, [0] * 6, TOX)
    assert low["passed"]  # 1 in 6 is under 0.2


def test_best_orders_by_the_challenges_constraint():
    a = [{"challenge": "c", "passed": True, "dose": 0.6, "neurons": [1]},
         {"challenge": "c", "passed": True, "dose": -0.3, "neurons": [1, 2, 3]},
         {"challenge": "c", "passed": False, "dose": 0.1, "neurons": [1]},
         {"challenge": "other", "passed": True, "dose": 0.05, "neurons": [1]}]
    assert [x["dose"] for x in lab.best(CH, a)] == [-0.3, 0.6]
    assert [x["dose"] for x in lab.best({**CH, "minimize": "neurons"}, a)] == [0.6, -0.3]
