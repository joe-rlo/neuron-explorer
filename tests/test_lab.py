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
