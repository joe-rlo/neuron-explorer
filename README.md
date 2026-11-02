# neuron explorer

> This is a fork of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer). The sweep, the
> explorer and the caches are theirs; this fork adds experiment tools, a Lab, a Learn tab and map features, listed under
> [This fork](#this-fork). Upstream's MIT license applies to all of it.

Steering-first neuron explorer for Qwen3.5-4B. Every MLP neuron in a layer is pushed up and down on a few prompts, every
rollout is kept, and the explorer lays the results out over a 3D view of the model.

![neuron explorer](explorer.png)

## Explorer

    uv sync
    uv run python explorer/server.py        # http://localhost:8791

Reads sweep runs from `runs/` and per-layer stats from `stats/`, both produced by the sweep below; the caches in
`explorer/cache/` ship with the repo. Loads Qwen3.5-4B locally (MPS, CUDA or CPU) for live
steered chat, lens, connections and live inference over your own text.

- **map**: the model's layers in perspective, attention and MLP per layer, one point per neuron. Hover shows a neuron's
  steer lens and logit lens; click opens it; click a layer's label to view its sheet face-on. `writes` places
  neurons with similar down-projection columns next to each other.
- **list**: units ranked by word-cluster mass against random-direction controls; `lens` searches the steer and logit
  lenses for a token, `fires` runs your text through the model and filters neurons by activation.
- **neuron**: steered rollouts per prompt and dose, a three-column multi-turn steered chat (−, 0, +), lens and
  connections, and a name.
- **embeddings**: UMAP of a layer's rollouts, coloured by neuron, prompt, dose or kind.
- **language**: the interface follows the browser language on first visit; the menu at the right of the header
  switches between English and Simplified Chinese (the choice is kept in this browser). Translations live in
  `explorer/static/i18n.js`.
- **Claude** (fourth neuron-panel tab, or the right-edge handle): a chat running the `claude` CLI (installed and
  logged in) with the explorer as an MCP server (`explorer/agent_mcp.py`). It reads neurons, cones and lenses, runs the
  live model (steering, activations, next-token logprobs, ablations), moves the map, list and highlights, and has file
  tools and a shell in your home folder. Its prompt is `explorer/agent_prompt.md` plus `explorer/agent_methodology.md`, and
  `explorer/agent_prompt.local.md` when present (untracked, for your own additions);
  `EXPLORER_CLAUDE_MODEL` picks the model. The server answers only pages and tools on this machine.

## This fork

Everything here reads the same `runs/`, `stats/` and `explorer/cache/` as upstream and doesn't change them. What the
new tools save (experiment results, hypotheses, ratings) goes in `experiments/`, and names in `explorer/names.json`;
both are untracked.

- **experiment mode** (neuron panel): steered chats and dose sweeps with `n` samples on paired seeds (temperature 0.7 by
  default), outcome measures, bootstrap intervals, paired stats and a coherence check, so a steering effect comes with
  its uncertainty. Thresholds are in `explorer/experiment_config.json`.
- **lesion**: ablate a neuron or a pinned set (mean ablation by default) and compare against the intact model.
- **tuning curves** and **contrast search**: per-item responses over stimulus sets with selectivity and d', and fires
  mode with "minus" texts to find neurons that fire on one text but not another.
- **task battery**: behavioural tasks run under steering — delay discounting, risk preference and loss aversion,
  gain/loss framing, Likert items with your own statements.
- **Lab tab**: a hypothesis log (write a prediction before a run, give a verdict after), blind rating of shuffled
  columns, challenges scored by smallest dose or fewest neurons, "guess the neuron", and a field guide with a card per
  named neuron (PNG and Markdown export).
- **Learn tab**: chapters on tokens, embeddings, the layers, attention, neurons, choosing the next word and steering,
  each run live on a sentence you type.
- **map**: the model framed in the space the panels leave, a ring and label on the open neuron, a layer ruler along the
  bottom (what's on screen, where the hits are, the logit-lens guess while playing), hits readable from the overview,
  named neurons pinned with their names, and keys (← → step layers, F frames the open neuron, H goes home).
- **circuit view**: the open neuron's strongest links before and after it as a flat diagram, with the chain's sign and
  each neuron's lens words (`explorer/circuit.py`).
- **play a token** (run tab, fires mode): what fired at a token, one layer at a time, with the model's guess for the
  next word at each layer.
- **weight edits** (Lab tab): changes written into the down_proj weights themselves rather than hooks, after
  [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/). An edit scales some neurons'
  columns (× 0 removes them for good) or projects a direction out of them; it's applied to the live model one at a
  time, undone exactly, and compared against the original on paired seeds. A header badge shows while one is applied.
  Edits are files in `experiments/weight_edits/` (`explorer/wedits.py`). `explorer/wsp_probe.py` runs GSM8K with
  thinking on and scores accuracy and cap hits, with or without an edit; `explorer/wsp_trace.py` finds the direction
  the MLPs write when the model keeps deliberating and builds a projection edit from it. The first try at making the
  4B model stop overthinking didn't help; see `research_log.md` (2026-11-02).
- smaller things: a dismissable first-run tour, a map legend, top-bar tooltips, a notice when opening a neuron switches
  layers.

Tests: `uv run pytest tests`.

## Sweep

Runs on a CUDA 13 GPU pod with the steering fork of vLLM ([RhizoNymph/vllm](https://github.com/RhizoNymph/vllm)).

    bash sweep/setup_pod.sh                                   # fork, deps, model; writes /workspace/env.sh
    python sweep/neuron_stats.py --layers 22 --tokens 2000000 --out stats/q35-4b
    python sweep/steer_sweep.py --stats stats/q35-4b_L22.npz --design main_rel --dose-unit rel \
        --neurons all --random-controls 256 --run q35-4b-L22
    sweep/run_depth.sh "4 9 13 17" 384 64                     # random-subset sweep across several layers

`main_rel` doses are fractions of the median residual norm leaving the layer: {who, snake, orange} × {−0.7, +0.5, +0.8}
and {free} × {+3}, plus one unsteered baseline per prompt.

| script | output |
|---|---|
| `sweep/neuron_stats.py` | per-neuron activation extremes by sign regime, down columns, residual norms |
| `sweep/steer_sweep.py` | `runs/<run>.jsonl`: text, token ids, logprobs, top-k per step; optional activation captures |
| `sweep/rank.py` | `runs/<run>.units.jsonl`: distinctive words, cluster mass, percentile against controls |
| `sweep/steer_lens.py` | `stats/steer_lens_L{l}.npz`: next-token Δlogprob and KL for ±0.8 pushes on every neuron |
| `sweep/depth_report.py` | `runs/depth_report.json`: per-layer share of neurons above controls, coherence, top units |
| `sweep/shift.py` | word-shift score against controls, all words counted |
| `sweep/merge_runs.py` | joins sharded runs of one layer |
| `sweep/check_equivalence.py` | HF vs vLLM steering and capture agreement |
| `sweep/bench.py`, `sweep/peek.py` | throughput conditions; terminal views of a run |
| `explorer/build_embeddings.py` | `runs/<run>.umap.json` |
| `explorer/build_layout.py` | `explorer/cache/layout_writes.npz` |
| `explorer/build_connections.py` | `explorer/cache/connections.npz`: each neuron's 24 strongest direct-path links up and down |
| `sweep/capture_prefill.py` | `captures/<run>.mlp_in.npy`: MLP input at the last prompt token at the run's capture layers, steered |

`research_log.md` holds the findings and reasoning; `NOTES.md` the source notes.

## License

MIT, see [LICENSE](LICENSE).
