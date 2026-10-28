"""Behavioral task battery: decision-making paradigms stored as JSON in tasks/, expanded into counterbalanced prompts,
scored from the next token's probabilities (or sampled and parsed when an answer spans several tokens) and fitted to
one behavioral parameter per dose. Pure numpy; the model side is in runtime.py, the job in server.py.

A task file holds: name, kind ("choice" or "likert"), template, grid, fit, parameter and method (the note shown in the
UI). The grid is a product over its keys; an entry that is an object merges all its fields into the item, a plain value
is stored under the key. Template and option strings are filled from the item twice, so a grid entry can itself hold
{fields}.

choice: two options, x and y, shown as A and B in both orders. An item's value is P(y), averaged over the orders.
likert: a rating from scale[0] to scale[1], asked with the anchors in both directions. An item's value is the expected
rating, averaged over the directions (the reversed one flipped back).
"""
import itertools
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(ROOT, "tasks")


def load():
    """Every task in tasks/, keyed by file name, in file-name order."""
    out = []
    for f in sorted(os.listdir(TASKS_DIR)) if os.path.isdir(TASKS_DIR) else []:
        if f.endswith(".json"):
            with open(os.path.join(TASKS_DIR, f)) as fh:
                out.append({"id": f[:-5], **json.load(fh)})
    return out


def get(tid):
    return next((t for t in load() if t["id"] == tid), None)


# ---------- items and prompts ----------
def expand(task, statements=None):
    """The task's items: one dict per point of the grid. statements replace a likert task's grid of statements."""
    grid = dict(task["grid"])
    if statements is not None:
        grid["statement"] = [s for s in statements if s.strip()]
    keys = list(grid)
    items = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        it = {}
        for k, v in zip(keys, combo):
            it.update(v if isinstance(v, dict) else {k: v})
        items.append(it)
    return items


def fill(s, fields):
    for _ in range(2):
        s = s.format_map(fields)
    return s


def labels(task):
    if task["kind"] == "likert":
        lo, hi = task.get("scale", [1, 7])
        return [str(r) for r in range(lo, hi + 1)]
    return task.get("labels", ["A", "B"])


def prompts(task, item):
    """The item's two prompts: options in both orders (choice), or anchors in both directions (likert)."""
    if task["kind"] == "likert":
        lo, hi = task.get("anchors", ["strongly disagree", "strongly agree"])
        return [fill(task["template"], {**item, "low": lo, "high": hi}), fill(task["template"], {**item, "low": hi, "high": lo})]
    x, y = fill(task["options"]["x"], item), fill(task["options"]["y"], item)
    return [fill(task["template"], {**item, "A": x, "B": y}), fill(task["template"], {**item, "A": y, "B": x})]


def label_tokens(task, tok):
    """Per label, the single-token ways the reply can open with it (with and without a leading space, and in other
    cases for word labels). None when some label has no single-token form or two labels share one: then the task is
    sampled and parsed instead."""
    out, seen = [], set()
    for lab in labels(task):
        forms = {lab, " " + lab} | ({lab.lower(), lab.capitalize(), " " + lab.lower(), " " + lab.capitalize()} if len(lab) > 1 else set())
        ids = sorted({i[0] for i in (tok(f, add_special_tokens=False).input_ids for f in forms) if len(i) == 1})
        if not ids or seen & set(ids):
            return None
        seen |= set(ids)
        out.append(ids)
    return out


# ---------- parsers (the sampling fallback) ----------
def parse_choice(text, labs):
    """Index of the label the reply chose: the first label standing as a word of its own (case-sensitive for one-letter
    labels, so the article "a" is not an answer). None when no label appears."""
    best = None
    for i, lab in enumerate(labs):
        m = re.search(r"(?<![A-Za-z0-9])" + re.escape(lab) + r"(?![A-Za-z0-9])", text, 0 if len(lab) == 1 else re.I)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), i)
    return None if best is None else best[1]


def parse_likert(text, lo, hi):
    """The rating in a reply: its first whole number, if it lies in [lo, hi]. None otherwise."""
    m = re.search(r"(?<![\d.])-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    v = float(m.group())
    return int(v) if v.is_integer() and lo <= v <= hi else None


# ---------- scoring one item ----------
def score(task, p0, p1):
    """An item's value from its label probabilities in the two prompts (each a vector over labels; they need not sum
    to 1). Returns (value, validity, bias): validity is the mean log of the probability put on valid answers; bias is
    the share given to the option listed first (choice) or the mean rating before flipping minus the scale midpoint
    (likert). value None when a prompt got no valid answer at all."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    s0, s1 = p0.sum(), p1.sum()
    if s0 <= 0 or s1 <= 0:
        return None, None, None
    validity = float((np.log(s0) + np.log(s1)) / 2)
    if task["kind"] == "likert":
        lo, hi = task.get("scale", [1, 7])
        r = np.arange(lo, hi + 1)
        e0, e1 = float(r @ p0 / s0), float(r @ p1 / s1)
        return (e0 + lo + hi - e1) / 2, validity, (e0 + e1) / 2 - (lo + hi) / 2
    py0, py1 = p0[1] / s0, p1[0] / s1  # y is B in the first prompt, A in the second
    return float(py0 + py1) / 2, validity, float((1 - py0) + py1) / 2


# ---------- fits ----------
BETAS = np.geomspace(0.3, 300, 30)


def _logistic_grid(pred, p, theta):
    """Least cross-entropy fit of P(y) = sigmoid(beta * pred[theta]): pred [T, N] for each candidate theta, p [N] soft
    targets. A coarse pass over theta, then a fine one between the best point's neighbors. Returns (theta, beta,
    at_bound)."""
    def best(pr, th):
        z = BETAS[:, None, None] * pr[None]
        ce = (p * np.logaddexp(0, -z) + (1 - p) * np.logaddexp(0, z)).mean(-1)
        b, t = np.unravel_index(np.argmin(ce), ce.shape)
        return t, b
    t, b = best(pred(theta), theta)
    at_bound = t in (0, len(theta) - 1)
    fine = np.linspace(theta[max(t - 1, 0)], theta[min(t + 1, len(theta) - 1)], 41)
    t2, b2 = best(pred(fine), fine)
    return float(fine[t2]), float(BETAS[b2]), bool(at_bound)


def fit_hyperbolic(c, p, spec):
    """Delay discounting: V = A / (1 + kD), P(later) = sigmoid(beta * (V_later - A_now) / A_later). Returns log10 k
    (k per unit of delay)."""
    now, later, delay = (np.asarray(c[spec[k]], float) for k in ("now", "later", "delay"))
    pred = lambda lk: (later / (1 + 10 ** lk[:, None] * delay) - now) / later
    lk, beta, edge = _logistic_grid(pred, p, np.linspace(-5, 1, 121))
    return {"value": lk, "beta": beta, "at_bound": edge}


def fit_risk(c, p, spec):
    """Risk preference with power utility u(x) = x^rho: P(gamble) = sigmoid(beta * (p - (sure / prize)^rho)), the
    gamble paying prize with probability p, else nothing. Returns the certainty equivalent of the reference gamble
    (spec["reference"], e.g. a 50% chance at $100): prize * p^(1/rho). rho < 1 is risk averse, > 1 risk seeking."""
    sure, prize, prob = (np.asarray(c[spec[k]], float) for k in ("sure", "prize", "p"))
    pred = lambda lr: prob - (sure / prize) ** (10 ** lr[:, None])
    lr, beta, edge = _logistic_grid(pred, p, np.linspace(-1, 1, 81))
    ref = spec.get("reference", {"p": 0.5, "prize": 100})
    return {"value": ref["prize"] * ref["p"] ** (1 / 10 ** lr), "rho": 10 ** lr, "beta": beta, "at_bound": edge}


def fit_loss(c, p, spec):
    """Loss aversion on 50/50 bets (win gain or lose loss): P(accept) = sigmoid(beta * (1 - lambda * loss / gain)).
    Returns lambda, how many times a dollar lost outweighs a dollar won (1 = neutral)."""
    gain, loss = (np.asarray(c[spec[k]], float) for k in ("gain", "loss"))
    pred = lambda ll: 1 - 10 ** ll[:, None] * loss / gain
    ll, beta, edge = _logistic_grid(pred, p, np.linspace(-1, 1.3, 93))
    return {"value": 10 ** ll, "beta": beta, "at_bound": edge}


def fit_mean(c, p, spec):
    """No model: the mean item value (a choice rate, or a mean rating)."""
    return {"value": float(p.mean())}


FITS = {"hyperbolic": fit_hyperbolic, "risk": fit_risk, "loss": fit_loss, "mean": fit_mean}


def fit(task, c, p):
    spec = task["fit"]
    return FITS[spec["type"]](c, np.asarray(p, float), spec)


def columns(items):
    return {k: np.array([it.get(k) for it in items]) for k in items[0]} if items else {}


def bootstrap(task, items, values, B=500, seed=0):
    """The fitted parameter and its bootstrap distribution, resampling units (task["unit"]: items sharing that field
    resample together; default each item) with replacement. The draws depend only on the items and the seed, so two
    doses bootstrapped with the same seed share every draw and their difference is paired. Items with value None are
    left out. Returns (fit with "rate", the mean item value; {"value": [B], "rate": [B]} draws)."""
    keep = [i for i, v in enumerate(values) if v is not None]
    items, values = [items[i] for i in keep], np.array([values[i] for i in keep], float)
    c = columns(items)
    full = {**fit(task, c, values), "rate": float(values.mean())}
    ukey = task.get("unit")
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(it[ukey] if ukey else i, []).append(i)
    groups = list(groups.values())
    rng = np.random.default_rng(seed)
    boots = {"value": np.empty(B), "rate": np.empty(B)}
    for b in range(B):
        sel = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        boots["value"][b] = fit(task, {k: v[sel] for k, v in c.items()}, values[sel])["value"]
        boots["rate"][b] = values[sel].mean()
    return full, boots


def ci(draws, level=0.95):
    a = (1 - level) / 2
    return [float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))]
