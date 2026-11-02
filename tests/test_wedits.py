import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import wedits  # noqa: E402

H, I = 6, 10


def layers(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=torch.randn(H, I, generator=g))))
            for _ in range(3)]


def test_projection_removes_the_direction_and_undo_is_exact():
    ls = layers()
    before = [x.mlp.down_proj.weight.clone() for x in ls]
    d = [1.0, 2.0, 0.0, 0.0, -1.0, 0.5]
    saved = wedits.apply(ls, {"name": "p", "edits": [{"layer": 1, "neurons": [2, 7], "direction": d}]})
    W, u = ls[1].mlp.down_proj.weight, torch.tensor(d) / torch.tensor(d).norm()
    assert torch.allclose(u @ W[:, [2, 7]], torch.zeros(2), atol=1e-6)
    assert torch.equal(W[:, 3], before[1][:, 3]) and torch.equal(ls[0].mlp.down_proj.weight, before[0])
    wedits.restore(ls, saved)
    assert all(torch.equal(x.mlp.down_proj.weight, b) for x, b in zip(ls, before))


def test_alpha_two_reverses_and_scale_multiplies():
    ls = layers(1)
    W0 = ls[0].mlp.down_proj.weight.clone()
    d = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    wedits.apply(ls, {"name": "r", "edits": [{"layer": 0, "neurons": [4], "direction": d, "alpha": 2.0},
                                            {"layer": 2, "neurons": [0, 1], "scale": 0.0}]})
    assert torch.allclose(ls[0].mlp.down_proj.weight[1, 4], -W0[1, 4])
    assert torch.equal(ls[2].mlp.down_proj.weight[:, :2], torch.zeros(H, 2))


def test_overlapping_steps_restore_the_first_original():
    ls = layers(2)
    W0 = ls[0].mlp.down_proj.weight.clone()
    saved = wedits.apply(ls, {"name": "o", "edits": [{"layer": 0, "neurons": [3], "scale": 2.0},
                                                    {"layer": 0, "neurons": [3], "scale": 3.0}]})
    assert torch.allclose(ls[0].mlp.down_proj.weight[:, 3], 6 * W0[:, 3])
    wedits.restore(ls, saved)
    assert torch.equal(ls[0].mlp.down_proj.weight, W0)


@pytest.mark.parametrize("edit", [
    {"edits": [{"layer": 0, "neurons": [1], "scale": 0}]},                          # no name
    {"name": "x", "edits": []},                                                      # no steps
    {"name": "x", "edits": [{"layer": 3, "neurons": [1], "scale": 0}]},              # layer out of range
    {"name": "x", "edits": [{"layer": 0, "neurons": [I], "scale": 0}]},              # neuron out of range
    {"name": "x", "edits": [{"layer": 0, "neurons": [1]}]},                          # neither kind
    {"name": "x", "edits": [{"layer": 0, "neurons": [1], "direction": [1.0]}]},      # wrong length
])
def test_check_rejects_malformed_edits(edit):
    with pytest.raises(ValueError):
        wedits.check(edit, 3, I, H)


def test_listing_reads_the_folder(tmp_path):
    e = {"name": "Cut two", "note": "n", "edits": [{"layer": 1, "neurons": [1, 2], "scale": 0},
                                                   {"layer": 2, "neurons": [1], "direction": [1.0] * H}]}
    (tmp_path / "cut-two.json").write_text(json.dumps(e))
    (tmp_path / "broken.json").write_text("{")
    [x] = wedits.listing(str(tmp_path))
    assert x == {"id": "cut-two", "name": "Cut two", "note": "n", "layers": [1, 2], "columns": 3,
                 "kinds": ["project", "scale"], "result": None}
    assert wedits.slug("Stop sooner: L22+L26!") == "stop-sooner-l22-l26"
