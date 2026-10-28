"""Lab: the records and games built on experiment mode. The notebook's one-line view of each saved experiment (what
was run, its result and the prediction made before it). Pure python; storage and the model are in server.py.
"""
import re

VERDICTS = ("confirmed", "disconfirmed", "inconclusive")
EXPERIMENT_FILE = re.compile(r"^\d{8}-\d{6}-L\d+N\d+-[a-z]+-[0-9a-f]{4}$")


def _clip(s, k=70):
    s = " ".join(str(s or "").split())
    return s if len(s) <= k else s[:k - 1] + "…"


def _measure_name(rec):
    m = (rec.get("measures") or [None])[0]
    return m.get("name", "") if isinstance(m, dict) else str(m or "")


def describe(rec):
    """What an experiment ran, in a few words."""
    kind, prompt, m = rec.get("kind"), _clip(rec.get("prompt"), 60), _measure_name(rec)
    if kind == "chat":
        d = rec["doses"][0]
        return f"steer {'×' if rec.get('mode') == 'mul' else '±'}{d:g} · “{prompt}” · {m}"
    if kind == "sweep":
        return f"dose sweep · “{prompt}” · {m}"
    if kind == "ablate":
        k = len(rec.get("neurons") or [])
        return f"{rec.get('method', '')} ablation{f' of {k} neurons' if k > 1 else ''} · “{prompt}” · {m}"
    if kind == "task":
        return f"task · {rec['task']['name']} · {m}"
    if kind == "challenge":
        return f"challenge · {rec['challenge']['name']} · {rec['dose']:+g} on {len(rec['neurons'])} neuron{'s' * (len(rec['neurons']) > 1)}"
    return kind or "experiment"


def _extremes(points):
    """The sentences at the largest positive and negative doses that have one."""
    pts = [p for p in points or [] if p.get("vs0")]
    out = [max((p for p in pts if p["dose"] > 0), key=lambda p: p["dose"], default=None),
           min((p for p in pts if p["dose"] < 0), key=lambda p: p["dose"], default=None)]
    return " ".join(p["vs0"]["sentence"] for p in out if p)


def headline(rec):
    """The experiment's result in a line or two, from its saved summary."""
    if rec.get("cancelled"):
        return "cancelled before it finished"
    kind, s = rec.get("kind"), rec.get("summary")
    if kind == "chat":
        return " ".join(c["vs0"]["sentence"] for c in s or [] if c.get("vs0"))
    if kind in ("sweep", "task"):
        return _extremes(s)
    if kind == "ablate":
        return (s or [{}, {}])[1].get("vs0", {}).get("sentence", "")
    if kind == "challenge":
        return (s or {}).get("sentence", "")
    return ""


def entry(rec):
    """A notebook row."""
    return {"id": rec["id"], "timestamp": rec.get("timestamp"), "kind": rec.get("kind"), "layer": rec.get("layer"),
            "neurons": rec.get("neurons") or [], "what": describe(rec), "result": headline(rec),
            "prediction": (rec.get("prediction") or "").strip() or None, "verdict": rec.get("verdict")}
