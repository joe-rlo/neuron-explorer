"""Learn: the logic behind the Learn tab's chapters on how a language model works. The logit lens read layer by layer,
nearest tokens in embedding space, the attention head that best shows a word reading back, and the next-token
distribution at a temperature. Pure numpy; the model is in runtime.py, the endpoints in server.py.
"""
import numpy as np


def _p(x):
    return round(float(x), 4)


def lens_rows(lp, vocab, k=5):
    """The logit lens as rows, one per layer: lp [L, V] log-probs from each layer's residual read through the final norm
    and the unembedding. Each row has that layer's top k next tokens with their probabilities and `final`, the
    probability it gives the token the last layer ranks first (the model's real top guess)."""
    lp = np.asarray(lp, np.float32)
    want = int(lp[-1].argmax())
    rows = []
    for l in range(lp.shape[0]):
        top = np.argsort(-lp[l])[:k]
        rows.append({"top": [[vocab[t], _p(np.exp(lp[l, t]))] for t in top], "final": _p(np.exp(lp[l, want]))})
    return {"answer": vocab[want], "layers": rows}


def neighbors(query, hits, k=12):
    """Nearest tokens to `query` from hits [(token string, cosine)], best first, leaving out the query itself and any
    token that only differs from one already kept by case or surrounding spaces (" cat", "Cat", "cat")."""
    seen, out = {query.strip().lower()}, []
    for s, sim in hits:
        key = s.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append([s, round(float(sim), 3)])
        if len(out) == k:
            break
    return out


def best_heads(rows, q, k=3):
    """The heads whose attention at query unit q best shows reading back: rows [(layer, head, weights over units 0..q)],
    scored by the largest weight on an earlier unit, leaving out unit 0 (the attention sink, where heads park weight
    they don't need) and the query itself. Best first, ties to the earlier layer."""
    def score(w):
        return max(w[1:q]) if q > 1 else 0.0
    ranked = sorted(rows, key=lambda r: (-score(r[2]), r[0], r[1]))
    return [{"layer": l, "head": h, "score": round(float(score(w)), 3), "row": [round(float(x), 3) for x in w]}
            for l, h, w in ranked[:k]]


def at_temperature(lp, t):
    """Next-token probabilities after dividing the logits by temperature t: lp [V] log-probs (any constant offset
    cancels). t near 0 puts everything on the top token, t = 1 is the model's own distribution, higher flattens it."""
    z = np.asarray(lp, np.float64) / max(float(t), 1e-3)
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def top_next(p, vocab, k=10):
    """The k likeliest next tokens [[token, p]] and the probability left over for everything else."""
    top = np.argsort(-p)[:k]
    return {"top": [[vocab[t], _p(p[t])] for t in top], "rest": _p(max(0.0, 1 - p[top].sum()))}
