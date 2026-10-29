import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import circuit  # noqa: E402

ROOT = (10, 0)
LINKS = [
    (10, 0, 12, 1, 40, 1),     # root -> A, pushes it up
    (10, 0, 13, 2, -30, 1),    # root -> B, pushes it down
    (12, 1, 15, 3, 50, 2),     # A -> C
    (13, 2, 15, 3, -20, 2),    # B -> C as well
    (20, 9, 21, 9, 99, 2),     # not connected to anything kept
    (8, 5, 10, 0, -25, -1),    # D -> root, holds it back
    (6, 4, 8, 5, 30, -2),      # E -> D
]


def at(nodes, l, n):
    return next(d for d in nodes if (d["layer"], d["n"]) == (l, n))


def test_keeps_connected_neurons_with_chain_signs():
    nodes, edges = circuit.pick(LINKS, ROOT)
    assert (nodes[0]["layer"], nodes[0]["n"], nodes[0]["hop"]) == (10, 0, 0)
    assert {(d["layer"], d["n"]) for d in nodes} == {ROOT, (12, 1), (13, 2), (15, 3), (8, 5), (6, 4)}
    assert at(nodes, 12, 1)["sign"] == 1 and at(nodes, 13, 2)["sign"] == -1
    assert at(nodes, 15, 3)["sign"] == 1  # through A, its strongest kept path (50 * 0.8 beats 20 * 0.8)
    assert at(nodes, 8, 5)["sign"] == -1 and at(nodes, 6, 4)["sign"] == -1  # E excites D, which holds the root back
    ids = {i: (d["layer"], d["n"]) for i, d in enumerate(nodes)}
    assert sorted((ids[i], ids[j], z) for i, j, z in edges) == sorted([
        (ROOT, (12, 1), 40), (ROOT, (13, 2), -30), ((12, 1), (15, 3), 50), ((13, 2), (15, 3), -20),
        ((8, 5), ROOT, -25), ((6, 4), (8, 5), 30)])


def test_per_side_limit_keeps_the_strongest_first():
    nodes, _ = circuit.pick(LINKS, ROOT, per_side=1)
    assert {(d["layer"], d["n"]) for d in nodes} == {ROOT, (12, 1), (8, 5)}
