"""Experiment Mode: outcome measures that score one reply, and the statistics over repeated samples (bootstrap
intervals, paired differences against dose 0, dose-response windows). Pure numpy; the model side is in runtime.py.

Samples are paired: sample i in every condition used the same seed, so the difference between conditions is taken
sample by sample before averaging.
"""
import json
import math
import os
import re
import unicodedata

import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_config.json")


def config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ---------- measures ----------
WORD = re.compile(r"\w+(?:'\w+)?")


def _words_re(words, stems=False):
    alts = sorted({w.strip().lower() for w in words if w.strip()}, key=len, reverse=True)
    end = "" if stems else r"(?!\w)"
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(w) for w in alts) + r")" + end, re.I) if alts else None


def score(measure, text):
    """One reply's score under a measure: {type: words, words: [...]} hits per 100 words (case-insensitive,
    whole-word, or word-start with stems: true); {type: contains, word} 1 or 0; {type: regex, pattern} match count."""
    kind = measure["type"]
    if kind == "words":
        rx = _words_re(measure.get("words") or [], measure.get("stems", False))
        n = len(WORD.findall(text))
        return 0.0 if not rx or not n else 100.0 * len(rx.findall(text)) / n
    if kind == "contains":
        rx = _words_re([measure.get("word", "")])
        return float(bool(rx and rx.search(text)))
    if kind == "regex":
        return float(sum(1 for _ in re.finditer(measure["pattern"], text)))
    raise ValueError(f"unknown measure type {kind!r}")


def unit(measure):
    """How the measure's numbers read in a sentence."""
    return {"words": "per 100 words", "contains": "", "regex": "matches"}.get(measure["type"], "")


def script_of(ch):
    """The writing system of one letter (LATIN, CJK, CYRILLIC, ...), from its Unicode name; None for non-letters."""
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    first = name.split(" ")[0]
    return "CJK" if first in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL") else first or None


def dominant_script(text):
    counts = {}
    for ch in text:
        s = script_of(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else "LATIN"


def script_mix(text, script):
    """Share of the reply's letters outside `script` (the prompt's dominant one); 0 when it has no letters."""
    ss = [s for s in map(script_of, text) if s]
    return 0.0 if not ss else sum(s != script for s in ss) / len(ss)


# ---------- statistics ----------
def bootstrap_ci(x, B=2000, seed=0, level=0.95):
    """Percentile bootstrap interval of the mean. A fixed seed gives a fixed interval."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def summary(x, B=2000, seed=0):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return {"n": 0, "mean": None, "sd": None, "ci": None}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "ci": bootstrap_ci(x, B, seed)}


def cohens_d_paired(x, y):
    """Paired Cohen's d (d_z): mean of the per-sample differences over their SD. None when it is undefined."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) < 2:
        return None
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() == 0 else None
    return float(d.mean() / sd)


def paired(x, y, B=2000, seed=0):
    """x against y sample by sample (same seeds): mean difference, its bootstrap interval and paired d."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not pairs:
        return {"n": 0, "diff": None, "ci": None, "d": None}
    xs, ys = map(np.array, zip(*pairs))
    d = xs - ys
    return {"n": len(d), "diff": float(d.mean()), "ci": bootstrap_ci(d, B, seed), "d": cohens_d_paired(xs, ys)}


def excludes_zero(ci):
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def _num(v):
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def sentence(label, measure, p):
    """One plain line for a paired result, e.g. '+0.5 raised food words by 2.1 per 100 words (95% CI 0.9 to 3.4)
    versus no steering.'"""
    if p["diff"] is None:
        return f"{label}: no scores to compare."
    name, u = measure.get("name", "the measure"), unit(measure)
    diff, ci = p["diff"], p["ci"]
    if measure["type"] == "contains":  # a rate: report percentage points
        size = f"{abs(diff) * 100:.0f} percentage points"
        cis = f" (95% CI {ci[0] * 100:.0f} to {ci[1] * 100:.0f})" if ci else ""
    else:
        size = f"{_num(abs(diff))}{' ' + u if u else ''}"
        cis = f" (95% CI {_num(ci[0])} to {_num(ci[1])})" if ci else ""
    if diff == 0 and (ci is None or ci == [0.0, 0.0]):
        return f"{label} left {name} unchanged versus no steering."
    verb = "raised" if diff > 0 else "lowered"
    tail = "" if excludes_zero(ci) else "; the interval includes zero, so this is not distinguishable from no steering"
    return f"{label} {verb} {name} by {size}{cis} versus no steering{tail}."


# ---------- dose-response ----------
def window(points, toxic):
    """Effective dose, toxic dose and therapeutic window per direction, from a dose sweep.

    points: {dose: {"measure": [per-sample], "coherence": [...], "script": [...]}} with dose 0 present; samples
    paired by index. toxic: {"coherence_drop", "script_mix"}. ED = the smallest |dose| whose paired measure
    difference from dose 0 has a 95% interval excluding zero; TD = the smallest |dose| whose mean coherence falls
    more than coherence_drop below dose 0's, or whose mean script mix exceeds script_mix."""
    base = points[0]
    base_coh = summary(base["coherence"])["mean"]
    top = max(abs(d) for d in points)
    out = {}
    for name, sign in (("positive", 1), ("negative", -1)):
        doses = sorted((d for d in points if d * sign > 0), key=abs)
        ed = td = None
        for d in doses:
            p = points[d]
            if ed is None and excludes_zero(paired(p["measure"], base["measure"])["ci"]):
                ed = abs(d)
            coh, mix = summary(p["coherence"])["mean"], summary(p["script"])["mean"]
            if td is None and ((coh is not None and base_coh is not None and coh < base_coh - toxic["coherence_drop"])
                               or (mix is not None and mix > toxic["script_mix"])):
                td = abs(d)
        r = {"ed": ed, "td": td, "max_dose": top}
        if ed is None:
            r["note"] = "no effect in range"
        elif td is None:
            r["note"] = f"TD > {top:g}"
        else:
            r["window"] = round(td - ed, 6)
            r["index"] = round(td / ed, 6)
        out[name] = r
    return out


def best_index(w):
    """One sortable number for a neuron's window: the larger therapeutic index of its two directions (a direction
    with an effect and no toxic dose in range counts as its max dose over ED); None when neither has an effect."""
    vals = []
    for r in w.values():
        if r.get("ed"):
            vals.append(r["index"] if "index" in r else r["max_dose"] / r["ed"])
    return max(vals) if vals else None


def finite(x):
    """JSON-safe: NaN and inf become None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x
