import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explorer"))
import learn  # noqa: E402

VOCAB = ["a", " mat", " floor", " bed", " cat"]


def logp(*p):
    return np.log(np.array(p, float))


def test_lens_rows_follow_the_final_guess_down_the_layers():
    lp = np.stack([logp(0.6, 0.1, 0.1, 0.1, 0.1), logp(0.1, 0.4, 0.3, 0.1, 0.1), logp(0.05, 0.8, 0.05, 0.05, 0.05)])
    out = learn.lens_rows(lp, VOCAB, k=2)
    assert out["answer"] == " mat"
    assert [r["final"] for r in out["layers"]] == [0.1, 0.4, 0.8]
    assert out["layers"][0]["top"][0] == ["a", 0.6] and out["layers"][1]["top"] == [[" mat", 0.4], [" floor", 0.3]]


def test_neighbors_fold_case_and_spacing_and_skip_the_query():
    hits = [(" cat", 1.0), ("cat", 0.9), ("Cat", 0.8), (" cats", 0.7), (" ", 0.6), (" dog", 0.5), (" Dog", 0.4), ("kitten", 0.3)]
    assert learn.neighbors(" cat", hits, k=3) == [[" cats", 0.7], [" dog", 0.5], ["kitten", 0.3]]


def test_best_heads_skip_the_sink_and_the_token_itself():
    rows = [(3, 0, [0.9, 0.05, 0.05]),       # parks almost everything on the sink
            (7, 2, [0.2, 0.7, 0.1]),         # reads back to token 1
            (3, 5, [0.1, 0.1, 0.8]),         # attends to itself
            (11, 1, [0.3, 0.7, 0.0])]        # the same score as (7, 2): the earlier layer wins
    out = learn.best_heads(rows, 2, k=2)
    assert [(h["layer"], h["head"]) for h in out] == [(7, 2), (11, 1)] and out[0]["score"] == 0.7
    assert learn.best_heads([(3, 0, [1.0])], 0)[0]["score"] == 0.0  # the first token has nothing earlier to read


def test_temperature_sharpens_and_flattens():
    lp = logp(0.5, 0.3, 0.2)
    assert learn.at_temperature(lp, 1) == pytest.approx([0.5, 0.3, 0.2])
    assert learn.at_temperature(lp, 0.01)[0] == pytest.approx(1)
    hot = learn.at_temperature(lp, 100)
    assert hot == pytest.approx([1 / 3] * 3, abs=0.01)
    assert learn.at_temperature(lp + 7, 0.5) == pytest.approx(learn.at_temperature(lp, 0.5))  # offsets cancel


def test_top_next_reports_what_is_left():
    out = learn.top_next(np.array([0.1, 0.6, 0.3]), ["x", "y", "z"], k=2)
    assert out == {"top": [["y", 0.6], ["z", 0.3]], "rest": 0.1}
