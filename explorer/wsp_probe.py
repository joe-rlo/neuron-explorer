"""Weight-space programming, step 1: how Qwen3.5-4B fails at grade-school math with thinking on.
Samples GSM8K problems, records each reply, whether it hit the token cap, whether it loops (a repeated chunk at the
end), and whether the final number is right. Optionally applies a weight edit (explorer/wedits.py format) first.

  uv run python explorer/wsp_probe.py --n 48 --max-new 2048 --out experiments/wsp/base.json
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")
PROMPT = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def final_number(text):
    box = re.findall(r"\\boxed\{([^{}]*)\}", text)
    s = box[-1] if box else (re.findall(r"-?\d[\d,]*\.?\d*", text) or [""])[-1]
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    return m.group(0).rstrip(".") if m else None


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def loops(ids, span=400, min_rep=3):
    """True when the last `span` tokens are some chunk of 8..100 tokens repeated at least min_rep times."""
    tail = ids[-span:]
    for k in range(8, 101):
        if len(tail) < k * min_rep:
            break
        chunk = tail[-k:]
        reps, i = 1, len(tail) - 2 * k
        while i >= 0 and tail[i:i + k] == chunk:
            reps, i = reps + 1, i - k
        if reps >= min_rep:
            return True
    return False


def load(edit=None):
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    if edit:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import wedits
        wedits.apply(m.model.language_model.layers, json.load(open(edit)))
    return tok, m, dev


def run(tok, m, dev, problems, max_new, batch, seed, log=print):
    out = []
    for b in range(0, len(problems), batch):
        rows = problems[b:b + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=r["question"])}],
                                         add_generation_prompt=True, enable_thinking=True, tokenize=False)
                 for r in rows]
        enc = tok(texts, return_tensors="pt", padding=True).to(dev)
        torch.manual_seed(seed + b)
        t0 = time.time()
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                           pad_token_id=tok.pad_token_id)
        g = g[:, enc["input_ids"].shape[1]:].tolist()
        for r, ids in zip(rows, g):
            ids = [t for t in ids if t != tok.pad_token_id]
            text = tok.decode(ids, skip_special_tokens=False)
            hit = len(ids) >= max_new
            ans = final_number(text.split("</think>")[-1]) if "</think>" in text else None
            out.append({"idx": r["idx"], "gold": gold(r["answer"]), "pred": ans, "correct": same(ans, gold(r["answer"])),
                        "n_tok": len(ids), "hit_cap": hit, "loops": loops(ids), "closed_think": "</think>" in text,
                        "text": text})
        done = out[-len(rows):]
        log(f"batch {b // batch}: {time.time() - t0:.0f}s, correct {sum(d['correct'] for d in done)}/{len(done)}, "
            f"cap {sum(d['hit_cap'] for d in done)}, loops {sum(d['loops'] for d in done)}", flush=True)
    return out


def summary(rows):
    n = len(rows)
    return {"n": n, "acc": sum(r["correct"] for r in rows) / n, "hit_cap": sum(r["hit_cap"] for r in rows) / n,
            "loops": sum(r["loops"] for r in rows) / n, "mean_tok": sum(r["n_tok"] for r in rows) / n}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = load_dataset("openai/gsm8k", "main", split=a.split)
    probs = [dict(ds[i], idx=i) for i in range(a.start, a.start + a.n)]
    tok, m, dev = load(a.edit)
    rows = run(tok, m, dev, probs, a.max_new, a.batch, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    s = summary(rows)
    json.dump({"args": vars(a), "summary": s, "rows": rows}, open(a.out, "w"), indent=1)
    print("SUMMARY", json.dumps(s), flush=True)
