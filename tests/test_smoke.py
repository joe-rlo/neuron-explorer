"""Smoke test against a running explorer with the live model (EXPLORER_URL, default http://127.0.0.1:8791): a short
dose sweep on a real neuron finishes and writes a valid experiment file. Skipped when no server answers."""
import json
import os
import urllib.request

import pytest

URL = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


@pytest.fixture(scope="module")
def live():
    try:
        with urllib.request.urlopen(URL + "/api/experiment/config", timeout=3) as r:
            json.load(r)
    except Exception:
        pytest.skip(f"no explorer with experiment mode at {URL}")


def test_sweep_writes_experiment(live):
    measure = {"name": "mentions alien", "type": "contains", "word": "alien"}
    r = post("/api/experiment/sweep", {"layer": 22, "n": 2742, "prompt": "Who are you?", "measure": measure,
                                       "doses": [-0.5, 0, 0.5], "samples": 2, "max_new": 20, "temperature": 0.7, "seed": 0})
    assert "id" in r, r
    events = []
    with urllib.request.urlopen(f"{URL}/api/experiment/events?id={r['id']}", timeout=900) as s:
        for line in s:
            if line.startswith(b"data: "):
                events.append(json.loads(line[6:]))
    errors = [e["error"] for e in events if "error" in e]
    assert not errors, errors
    end = events[-1]
    assert end.get("end") and not end["cancelled"] and end["id"]
    assert sorted(e["point"]["dose"] for e in events if "point" in e) == [-0.5, 0, 0.5]
    with open(os.path.join(ROOT, "experiments", end["id"] + ".json")) as f:
        rec = json.load(f)
    for k in ("id", "timestamp", "model", "git_commit", "layer", "neurons", "action", "doses", "seeds", "n",
              "temperature", "max_tokens", "prompt", "measures", "raw", "summary", "window"):
        assert k in rec, k
    assert rec["seeds"] == [0, 1] and rec["n"] == 2 and rec["neurons"] == [{"layer": 22, "n": 2742}]
    assert all(len(r["texts"]) == 2 for r in rec["raw"].values())
    assert set(rec["window"]) == {"positive", "negative"}


def test_chat_samples_are_paired_and_seeded(live):
    """n samples per column with a measure: the stream carries every (column, sample) and a summary per column; the
    same seed gives the same unsteered text twice."""
    body = {"layer": 22, "n": 2742, "prompt": "Who are you?", "dose": 0.5, "mode": "add", "samples": 2, "temperature": 0.7,
            "seed": 3, "max_new": 16, "measure": {"name": "mentions alien", "type": "contains", "word": "alien"}}

    def run():
        req = urllib.request.Request(URL + "/api/steer", json.dumps(body).encode(), {"Content-Type": "application/json"})
        texts, summary = {}, None
        with urllib.request.urlopen(req, timeout=900) as s:
            for line in s:
                if not line.startswith(b"data: "):
                    continue
                ev = json.loads(line[6:])
                assert "error" not in ev, ev
                if ev.get("row") is not None:
                    texts[(ev["row"], ev["s"])] = texts.get((ev["row"], ev["s"]), "") + ev["t"]
                summary = ev.get("summary", summary)
        return texts, summary
    a, sa = run()
    b, _ = run()
    assert set(a) == {(c, i) for c in range(3) for i in range(2)}
    assert [c["label"] for c in sa["columns"]] == ["+0.5", "0", "−0.5"]
    assert "vs0" in sa["columns"][0] and "vs0" not in sa["columns"][1]
    assert a[(1, 0)] == b[(1, 0)] and a[(1, 1)] == b[(1, 1)]


def test_ablate_writes_experiment(live):
    """Zero ablation of a set, n = 2: intact and ablated columns paired by seed, saved as an ablate experiment."""
    measure = {"name": "mentions pizza", "type": "contains", "word": "pizza"}
    r = post("/api/experiment/ablate", {"neurons": [{"layer": 7, "n": 8659}, {"layer": 6, "n": 6756}], "method": "zero",
                                        "prompt": "Name a popular Italian dish.", "measure": measure, "samples": 2,
                                        "max_new": 12, "temperature": 0.7, "seed": 0})
    assert "id" in r, r
    events = []
    with urllib.request.urlopen(f"{URL}/api/experiment/events?id={r['id']}", timeout=900) as s:
        for line in s:
            if line.startswith(b"data: "):
                events.append(json.loads(line[6:]))
    assert not [e for e in events if "error" in e]
    summary = next(e["summary"] for e in events if "summary" in e)
    assert [c["label"] for c in summary["columns"]] == ["intact", "ablated"]
    assert "the intact model" in summary["columns"][1]["vs0"]["sentence"]
    with open(os.path.join(ROOT, "experiments", events[-1]["id"] + ".json")) as f:
        rec = json.load(f)
    assert rec["action"] == "ablate" and rec["method"] == "zero" and rec["values"] == [0.0, 0.0] and len(rec["neurons"]) == 2
