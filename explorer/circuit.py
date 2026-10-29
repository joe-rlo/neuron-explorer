"""Circuit: the open neuron's web cut down to a diagram someone can read. From the cone's links (server.cone_links),
keep the strongest neurons each way that still connect back to the open neuron, and every link among them. Pure
Python; the endpoint in server.py adds names and lens tokens, the drawing is in app.js.
"""


def pick(links, root, per_side=10):
    """links: rows (src layer, src n, dst layer, dst n, z * sign, hop), hop > 0 downstream of root and < 0 upstream.
    A neuron's score is its strongest link from a kept neighbour nearer the root, |z| times 0.8 per hop past the first.
    Neurons are kept best first, each only once a neighbour nearer the root is kept, up to per_side each way.
    Returns (nodes, edges): nodes are dicts with layer, n, hop, sign and score, the root first. sign is the pole of the
    neuron the chain from the root drives (signs multiplied along the kept path): downstream, + means the root firing
    pushes it up; upstream, + means it firing pushes the root up. edges are (i, j, z) between kept nodes, writer i to
    reader j."""
    near = {}  # neuron -> [(score, hop, nearer neighbour, z)]
    for sl, sn, dl, dn, z, hop in links:
        a, b = ((sl, sn), (dl, dn)) if hop > 0 else ((dl, dn), (sl, sn))
        if b == root:
            continue
        near.setdefault(b, []).append((abs(z) * 0.8 ** (abs(hop) - 1), hop, a, z))
    kept = {root: {"layer": root[0], "n": root[1], "hop": 0, "sign": 1, "score": 0.0}}
    count = {1: 0, -1: 0}
    order = sorted(near, key=lambda b: -max(c[0] for c in near[b]))
    grew = True
    while grew:
        grew = False
        for b in order:
            if b in kept:
                continue
            via = [c for c in near[b] if c[2] in kept]
            if not via:
                continue
            score, hop, a, z = max(via)
            side = 1 if hop > 0 else -1
            if count[side] >= per_side:
                continue
            count[side] += 1
            kept[b] = {"layer": b[0], "n": b[1], "hop": hop, "sign": kept[a]["sign"] * (1 if z > 0 else -1),
                       "score": round(score, 2)}
            grew = True
            break  # rescan from the top: a newly kept neuron can open a stronger one
    nodes = list(kept.values())
    at = {(d["layer"], d["n"]): i for i, d in enumerate(nodes)}
    edges, seen = [], set()
    for sl, sn, dl, dn, z, hop in links:
        i, j = at.get((sl, sn)), at.get((dl, dn))
        if i is None or j is None or (i, j) in seen:
            continue
        seen.add((i, j))
        edges.append((i, j, round(float(z), 2)))
    return nodes, edges
