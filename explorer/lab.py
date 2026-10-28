"""Lab: the records and games built on experiment mode. The notebook's one-line view of each saved experiment (what
was run, its result and the prediction made before it), blind rating's binomial test, challenge scoring and the
labels for guess the neuron. Pure python; storage and the model are in server.py.
"""
import collections
import math
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


# ---------- blind rating ----------
CHANCE = 1 / 3  # three columns


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p): the one-sided p-value of k hits in n trials at chance p."""
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(max(k, 0), n + 1)))


def blind_stats(trials):
    """Running accuracy of blind picks (a hit is picking the column pushed the way the trial expected) against
    chance, with an exact one-sided binomial p-value and a sentence."""
    n, hits = len(trials), sum(t["picked"] == t["target"] for t in trials)
    if not n:
        return {"n": 0, "hits": 0, "accuracy": None, "p": None, "chance": CHANCE, "sentence": "No blind trials yet."}
    p = binom_sf(hits, n, CHANCE)
    read = ("unlikely to be luck" if p < 0.05 else "could still be luck") if n >= 5 else "too few trials to tell yet"
    return {"n": n, "hits": hits, "accuracy": hits / n, "p": p, "chance": CHANCE,
            "sentence": f"{hits} of {n} blind pick{'s' * (n != 1)} found the steered column ({hits / n:.0%}; chance is "
                        f"33%), one-sided binomial p = {p:.3g}: {read}."}


def blind_summary(trials):
    """All of one neuron's trials together, and per property."""
    props = {}
    for t in trials:
        props.setdefault(t["property"].strip().lower(), []).append(t)
    return {**blind_stats(trials), "by_property": {k: blind_stats(v) for k, v in props.items()}}


# ---------- challenges ----------
def _value(measure, v):
    if measure["type"] == "contains":
        return f"{v:.0%} of replies"
    if measure["type"] == "words":
        return f"{v:.2g} per 100 words"
    return f"{v:.2g} per reply"


def challenge_result(ch, scores, coh_base, coh, script, toxic):
    """Scores one attempt: the steered replies' mean measure against the goal (ch["op"] ch["target"]), and their
    coherence against the unsteered replies of the same seeds. It counts only if the goal is met and the text stays
    under the Phase 1 toxic thresholds (mean coherence no more than coherence_drop below unsteered, script mix no
    more than script_mix)."""
    mean = lambda x: sum(x) / len(x) if x else 0.0
    value, drop, mix = mean(scores), mean(coh_base) - mean(coh), mean(script)
    met = value >= ch["target"] if ch["op"] == ">=" else value <= ch["target"]
    coherent = drop <= toxic["coherence_drop"] and mix <= toxic["script_mix"]
    m = ch["measure"]
    got = f"{m['name']}: {_value(m, value)} (goal {'≥' if ch['op'] == '>=' else '≤'} {_value(m, ch['target'])})"
    why = "Passed" if met and coherent else "Missed the goal" if not met else "Goal met, but the text broke"
    tail = f"coherence {abs(drop):.2f} {'below' if drop > 0.005 else 'above' if drop < -0.005 else 'from'} unsteered (limit {toxic['coherence_drop']:g} below)" + (
        f", {mix:.0%} other script (limit {toxic['script_mix']:.0%})" if mix > 0.01 else "")
    return {"value": value, "met": met, "coherence_drop": drop, "script": mix, "coherent": coherent,
            "passed": met and coherent, "sentence": f"{why}. {got}; {tail}."}


def rank_key(ch, a):
    """Lower is better: the smallest |dose| (then fewest neurons), or the fewest neurons (then the smallest |dose|)."""
    k, d = len(a["neurons"]), abs(a["dose"])
    return (k, d) if ch.get("minimize") == "neurons" else (d, k)


def best(ch, attempts, k=5):
    """The k best passing attempts at a challenge."""
    return sorted((a for a in attempts if a["challenge"] == ch["id"] and a["passed"]), key=lambda a: rank_key(ch, a))[:k]


# ---------- guess the neuron ----------
def lens_theme(tokens):
    """A neuron's lens label when its top logit-lens tokens agree on one word, else None. Clear means at least 6 of the
    tokens are plain words of 4+ letters and 4 of those share their first 4 letters; the label is that group's three
    most common forms (shortest first on ties), e.g. "class · classes · classe"."""
    clean = [t.strip().lower() for t in tokens if re.fullmatch(r"\s?[A-Za-z]{4,}", t)]
    if len(clean) < 6:
        return None
    stem, k = collections.Counter(w[:4] for w in clean).most_common(1)[0]
    if k < 4:
        return None
    forms = collections.Counter(w for w in clean if w[:4] == stem)
    return " · ".join(sorted(forms, key=lambda w: (-forms[w], len(w), w))[:3])


def _stems(label):
    return {w[:4] for w in re.findall(r"[a-z]{3,}", label.lower())}


def guess_options(truth, pool, rng, k=3):
    """The true label among k distractors from pool that share no 4-letter stem with it or with each other, shuffled.
    Returns (options, index of the truth)."""
    seen, out = _stems(truth), []
    for i in rng.permutation(len(pool)):
        lab = pool[i]
        st = _stems(lab)
        if lab != truth and st and not st & seen:
            out.append(lab)
            seen |= st
            if len(out) == k:
                break
    opts = out + [truth]
    order = list(rng.permutation(len(opts)))
    return [opts[j] for j in order], order.index(len(opts) - 1)


def guess_score(history):
    """Rounds played, correct, the current streak and the best streak."""
    streak = best_ = 0
    for h in history:
        streak = streak + 1 if h["correct"] else 0
        best_ = max(best_, streak)
    return {"played": len(history), "correct": sum(h["correct"] for h in history), "streak": streak, "best": best_}
