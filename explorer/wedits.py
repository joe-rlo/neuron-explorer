"""Weight edits: changes written into the MLP down-projection weights themselves, not hooks on a forward pass
(weight-space programming, rikiyat.github.io/blog/weight-space-programming). An edit is a JSON file in
experiments/weight_edits/:

  {"name": ..., "note": ..., "edits": [step, ...]}

and each step changes some neurons' down_proj columns (the vectors they write into the residual stream) in one layer:

  {"layer": l, "neurons": [...], "direction": [hidden floats], "alpha": 1.0}   W[:, n] -= alpha * d (d . W[:, n])
  {"layer": l, "neurons": [...], "scale": s}                                    W[:, n] *= s   (0 = gone for good)

The direction is normalised first, so alpha 1 removes the neurons' reach along it and 2 reverses it. Applying keeps
the original columns so undo is exact (a projection can't be inverted from the edited weights alone).
"""
import json
import os
import re

import torch


def check(edit, n_layers, n_neurons, hidden):
    """Raises ValueError unless `edit` is a well-formed edit for this model."""
    if not isinstance(edit, dict) or not str(edit.get("name", "")).strip():
        raise ValueError("an edit needs a name")
    steps = edit.get("edits")
    if not isinstance(steps, list) or not steps:
        raise ValueError("an edit needs at least one step in 'edits'")
    for s in steps:
        l, ns = s.get("layer"), s.get("neurons")
        if not isinstance(l, int) or not 0 <= l < n_layers:
            raise ValueError(f"layer must be 0..{n_layers - 1}")
        if not isinstance(ns, list) or not ns or not all(isinstance(n, int) and 0 <= n < n_neurons for n in ns):
            raise ValueError(f"neurons must be a list of 0..{n_neurons - 1}")
        if ("direction" in s) == ("scale" in s):
            raise ValueError("each step has either a direction or a scale")
        if "direction" in s and len(s["direction"]) != hidden:
            raise ValueError(f"a direction has {hidden} numbers")


def size(edit):
    """How many weights the edit changes, counting each (layer, neuron) column once."""
    cols = {(s["layer"], n) for s in edit["edits"] for n in s["neurons"]}
    return len(cols)


def apply(layers, edit):
    """Writes the edit into layers[l].mlp.down_proj.weight. Returns the original columns, for restore()."""
    saved = {}
    with torch.no_grad():
        for s in edit["edits"]:
            W = layers[s["layer"]].mlp.down_proj.weight  # [hidden, intermediate]
            ns = torch.tensor(s["neurons"], device=W.device)
            for n in s["neurons"]:
                saved.setdefault((s["layer"], n), W[:, n].clone())
            cols = W[:, ns].float()
            if "scale" in s:
                cols *= float(s["scale"])
            else:
                d = torch.tensor(s["direction"], dtype=torch.float32, device=W.device)
                d = d / d.norm()
                cols -= float(s.get("alpha", 1.0)) * d[:, None] * (d @ cols)[None, :]
            W[:, ns] = cols.to(W.dtype)
    return saved


def restore(layers, saved):
    with torch.no_grad():
        for (l, n), col in saved.items():
            layers[l].mlp.down_proj.weight[:, n] = col


# ---------- the store ----------
def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "edit"


def listing(folder):
    out = []
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not f.endswith(".json"):
            continue
        try:
            e = json.load(open(os.path.join(folder, f)))
        except (OSError, ValueError):
            continue
        steps = e.get("edits") or []
        out.append({"id": f[:-5], "name": e.get("name", f[:-5]), "note": e.get("note", ""),
                    "layers": sorted({s.get("layer") for s in steps}), "columns": size(e) if steps else 0,
                    "kinds": sorted({"scale" if "scale" in s else "project" for s in steps}), "result": e.get("result")})
    return out
