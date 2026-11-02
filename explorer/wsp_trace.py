"""Weight-space programming, the trace: find the direction the MLPs write when the model has the answer to a
GSM8K problem but keeps deliberating (at a paragraph-ending full stop followed by more thinking) against when it stops
(the same kind of token followed by </think>), then build an edit (explorer/wedits.py format) that projects it out of
the neurons writing most of it. Reads a probe's replies (explorer/wsp_probe.py).

  uv run python explorer/wsp_trace.py trace experiments/wsp/base_train.json --out experiments/wsp/trace.npz
  uv run python explorer/wsp_trace.py make experiments/wsp/trace.npz --layers 22 26 --k 64 \\
      --out experiments/weight_edits/stop-sooner.json --name "stop sooner: L22+L26"
"""
import argparse
import json
import re

import numpy as np
import torch


def positions(tok, row, prompt_len):
    """Token positions (relative to the reply) where the model decided, at the full stop ending a paragraph of its
    thinking: 'close' = the one followed by '\\n' and </think>; 'cont' = ones after the right answer first appeared
    that were followed by '\\n\\n' and more deliberating. Both are the same kind of token, so the comparison isn't
    just token identity."""
    ids = tok(row["text"], add_special_tokens=False)["input_ids"]
    end = row["text"].find("</think>")
    think = row["text"][:end] if end >= 0 else row["text"]
    g = row["gold"]
    m = re.search(r"(?<![\d.])" + re.escape(g) + r"(?![\d])", think)
    if not m:
        return ids, [], []
    first = m.end()
    offs, pos = [], 0
    for i, t in enumerate(ids):  # char offset where each token ends
        pos += len(tok.decode([t]))
        offs.append(pos)
    shut = tok.convert_tokens_to_ids("</think>")
    ends = {".", ").", "}.", "$."}
    stop = ids.index(shut) - 1 if shut in ids else len(ids)
    close = [stop - 1] if stop < len(ids) and tok.decode([ids[stop - 1]]) in ends else []
    cont = [i - 1 for i in range(1, stop) if offs[i] >= first and tok.decode([ids[i]]) == "\n\n"
            and tok.decode([ids[i - 1]]) in ends]
    return ids, close, cont


def trace(probe_path, out):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wsp_probe
    tok, m, dev = wsp_probe.load()
    lm = m.model.language_model
    L = len(lm.layers)
    from datasets import load_dataset
    data = json.load(open(probe_path))
    ds = load_dataset("openai/gsm8k", "main", split=data["args"]["split"])
    want = {"ix": None}  # positions to keep: only those rows are copied out of each layer, so memory stays small
    acts, outs = {}, {}

    def hook(i):
        def h(mod, inp, o):
            acts[i] = inp[0][0, want["ix"]].to(torch.float16).cpu()
            outs[i] = o[0, want["ix"]].to(torch.float16).cpu()
        return h
    hs = [lm.layers[i].mlp.down_proj.register_forward_hook(hook(i)) for i in range(L)]
    keep = {"out": [], "act": [], "kind": [], "row": []}
    for row in data["rows"]:
        prompt = tok.apply_chat_template([{"role": "user", "content": wsp_probe.PROMPT.format(q=ds[row["idx"]]["question"])}],
                                         add_generation_prompt=True, enable_thinking=True, tokenize=False)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        ids, close, cont = positions(tok, row, len(p))
        cont = cont[-12:]  # the last few, nearest to where it could have stopped
        ps = close + cont
        if not ps:
            continue
        want["ix"] = torch.tensor([len(p) + i for i in ps], device=dev)
        with torch.no_grad():  # nothing after the last kept position matters
            m(input_ids=torch.tensor([(p + ids)[:len(p) + max(ps) + 1]], device=dev), use_cache=False)
        keep["out"].append(torch.stack([outs[i] for i in range(L)], 1).numpy())  # [pos, L, hidden]
        keep["act"].append(torch.stack([acts[i] for i in range(L)], 1).numpy())
        keep["kind"] += [1] * len(close) + [0] * len(cont)
        keep["row"] += [row["idx"]] * len(ps)
        if dev == "mps":
            torch.mps.empty_cache()
        print(row["idx"], "close", len(close), "cont", len(cont), flush=True)
    for h in hs:
        h.remove()
    O, A = np.concatenate(keep["out"]), np.concatenate(keep["act"])
    kind, rows = np.array(keep["kind"]), np.array(keep["row"])
    np.savez(out, out=O, act=A, kind=kind, row=rows)
    mc, mn = O[kind == 1].astype(np.float32).mean(0), O[kind == 0].astype(np.float32).mean(0)
    diff = mn - mc
    print("n close", int(kind.sum()), "n cont", int((kind == 0).sum()))
    for i in range(L):
        print(f"L{i:2d} |cont-close| {np.linalg.norm(diff[i]):7.2f}  |close| {np.linalg.norm(mc[i]):7.2f}")


def make(trace_path, layers, k, alpha, out, name):
    """For each layer: d = mean MLP write at 'cont' minus at 'close'; the k neurons that put the most of it there
    (activation gap times their column's reach along d) get d projected out of their columns."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wsp_probe
    t = np.load(trace_path)
    O, A, kind = t["out"].astype(np.float32), t["act"].astype(np.float32), t["kind"]
    t = {"out_close": O[kind == 1].mean(0), "out_cont": O[kind == 0].mean(0),
         "act_close": A[kind == 1].mean(0), "act_cont": A[kind == 0].mean(0)}
    _, m, _ = wsp_probe.load()
    lm = m.model.language_model
    edits = []
    for l in layers:
        d = t["out_cont"][l] - t["out_close"][l]
        d = d / np.linalg.norm(d)
        W = lm.layers[l].mlp.down_proj.weight.detach().float().cpu().numpy()
        gap = t["act_cont"][l] - t["act_close"][l]
        score = gap * (d @ W)
        ns = np.argsort(-score)[:k]
        edits.append({"layer": int(l), "neurons": [int(n) for n in ns], "alpha": alpha,
                      "share": float(score[ns].sum() / (d @ (t["out_cont"][l] - t["out_close"][l]))),
                      "direction": [float(x) for x in d]})
        print(f"L{l}: top {k} neurons carry {edits[-1]['share']:.0%} of the gap along d")
    json.dump({"name": name, "note": "project the 'keep deliberating' MLP direction out of the neurons that write it",
               "edits": edits}, open(out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("trace"); a1.add_argument("probe"); a1.add_argument("--out", required=True)
    a2 = sub.add_parser("make"); a2.add_argument("trace"); a2.add_argument("--layers", type=int, nargs="+", required=True)
    a2.add_argument("--k", type=int, default=64); a2.add_argument("--alpha", type=float, default=1.0)
    a2.add_argument("--out", required=True); a2.add_argument("--name", default="edit")
    a = ap.parse_args()
    if a.cmd == "trace":
        trace(a.probe, a.out)
    else:
        make(a.trace, a.layers, a.k, a.alpha, a.out, a.name)
