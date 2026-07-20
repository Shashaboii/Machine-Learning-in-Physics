# Hit-to-track association with a permutation-invariant transformer

Exam project for *Machine Learning in Particle Physics and Astronomy* (2026).

The problem: when charged particles fly through a tracking detector they leave
behind **hits**. Before you can measure anything physical, you first have to
figure out *which hits came from which particle*. That grouping step is what
this repo does, with a small transformer.

The full write-up is in [`paper/hit-to-track-association.pdf`](paper/hit-to-track-association.pdf)
and the original assignment brief is in [`docs/assignment-brief.pdf`](docs/assignment-brief.pdf).

---

## The idea in one minute

Two things make this harder than a normal classification problem:

1. **Variable size.** Every event has a different number of hits and a
   different number of tracks, so I can't use a fixed-size input or a fixed
   number of output classes.
2. **Labels are arbitrary.** "Track 1" and "track 2" are just numbering — the
   same event relabelled is the same physics. A model must never learn a fixed
   label; it should only learn *which hits belong together*.

So instead of classifying each hit into one of `K` classes, I treat the event
as a **set** and ask a pairwise question for every pair of hits *(i, j)*:
**do these two hits belong to the same track?**

- A **transformer encoder with no positional encoding** embeds the unordered
  hits. Because there's no positional encoding, self-attention is
  permutation-equivariant: shuffle the input hits and the output embeddings
  shuffle the same way. That's exactly the symmetry the physics has.
- A symmetric head turns each pair into a probability
  `p_ij = sigmoid(z_i · z_j / sqrt(d))`.
- Variable event sizes are handled by **padding + an attention mask**, so
  padded slots contribute nothing.
- At test time the `p_ij` form an affinity matrix, and **agglomerative
  clustering** turns them into `K` groups. Scoring uses the **Hungarian
  algorithm** to optimally match predicted groups to true tracks, which is how
  you fairly score a grouping when the labels don't mean anything.

Nothing in the network depends on the number or shape of the tracks, so the
*same* architecture is used for straight tracks, 50-track events, and the
curved magnetic-field data.

---

## Results

Fraction of hits correctly grouped (after Hungarian matching), on a sealed test
split that was never used for training or model selection. The baseline sorts
hits by azimuth and cuts at the `K` largest angular gaps.

| Setting               | Geometric baseline | Transformer |
|-----------------------|:------------------:|:-----------:|
| Straight, 10 tracks   | 0.993              | **0.991**   |
| Straight, 50 tracks   | 0.960              | 0.713       |
| Curved, 5–15 tracks   | 0.578              | **0.872**   |

The short story:

- On **straight tracks** the transformer matches a near-optimal geometric
  baseline — it hits the physical resolution limit (~1 mrad per hit) and can't
  really do better, which is the honest conclusion.
- At **50 tracks** both methods degrade for a real physical reason (tracks get
  angularly close, below detector resolution); the network additionally runs
  into a capacity limit.
- On **curved tracks** the geometric baseline *breaks* (its "hits share one
  angle" assumption is false once tracks bend), but the identical transformer
  keeps working and wins by a wide margin. This is the whole point: a learned,
  geometry-agnostic method earns its keep exactly where hand-built heuristics
  stop applying.

---

## Repository layout

```
.
├── notebooks/                     # one notebook per assignment task (a–e)
│   ├── task-a-simulator.ipynb           # 2D straight-line simulator + detector geometry
│   ├── task-b-detector-effects.ipynb    # 95% efficiency + Gaussian smearing, 10k events
│   ├── task-c-distributions.ipynb       # histograms of hits/tracks in x, y, phi
│   ├── task-d-association-straight.ipynb # the transformer, baseline, training + eval (10/50 tracks)
│   └── task-e-curved-tracks.ipynb       # same method applied to the curved dataset
├── models/                        # trained weights (self-describing checkpoints)
│   ├── model_d_10tracks.pt
│   └── model_e_curved.pt
├── predictions/                   # per-event predictions + fitted track params (see its README)
├── paper/                         # the 4-page PRL-style write-up
├── docs/                          # the original assignment brief
├── run_saved_model.py             # load a saved model and reproduce its test score
├── requirements.txt
└── README.md
```

Notebook ↔ assignment task mapping (original filenames in parentheses):

| Notebook                              | Task | Was            |
|---------------------------------------|:----:|----------------|
| `task-a-simulator.ipynb`              | a    | `task-1-mlp`   |
| `task-b-detector-effects.ipynb`       | b    | `task-2-mlp`   |
| `task-c-distributions.ipynb`          | c    | `task-3-mlp`   |
| `task-d-association-straight.ipynb`   | d    | `task-4-mlp`   |
| `task-e-curved-tracks.ipynb`          | e    | `task-5-mlpy`  |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU is enough to *run* the saved models. Training the larger (50-track / curved)
model is much more comfortable on a GPU.

---

## Reproducing the results

### Option 1 — the saved models (fast)

The checkpoints store their own architecture config, so they load without you
having to redefine anything.

**Straight-track model.** It re-creates its own sealed test set from a fixed
seed, so it needs no extra data:

```bash
python run_saved_model.py --model models/model_d_10tracks.pt
```

**Curved-track model.** This one needs the original curved hits file. ⚠️ Note:
the file it expects is the **tidy long-format** hits table from Brightspace
(columns `event_id, track_id, layer, x, y, ...`), e.g. `hits_FINAL.csv` — *not*
the wide prediction file in `predictions/`. Point `--data` at that file:

```bash
python run_saved_model.py --model models/model_e_curved.pt --data path/to/hits_FINAL.csv
```

Each run prints the Hungarian-matched hit accuracy, the Adjusted Rand Index, and
the fraction of perfectly reconstructed events.

### Option 2 — the notebooks (full pipeline)

Run them in order a → e. Task a builds the simulator, b generates the 10k-event
dataset with detector effects, c makes the distribution plots, d trains and
evaluates the transformer on straight tracks (and the 50-track scaling test),
and e applies the same model to the curved data.

---

## A few implementation notes

- **Inputs are just coordinates.** The network sees scaled `(x/10, y/10)` (and,
  for the larger model, the layer index `layer/4` as a third feature). Fitted
  track parameters like azimuth or curvature radius are used *only* for
  evaluation, never as inputs.
- **Class imbalance.** With `K` tracks only ~`1/K` of hit pairs are "same
  track", so the pairwise loss is swamped by easy negatives. Up-weighting the
  positive class by `K − 1` fixed a training stall.
- **Two model sizes.** The 10-track straight model is small
  (`d_model=64`, 3 layers, 4 heads, ~1.1×10⁵ params). The harder 50-track and
  curved settings use a bigger one (`d_model=160`, 6 layers, 8 heads,
  dropout 0.1, ~1.28×10⁶ params) with the layer index appended.
- **Smearing convention.** The brief says "0.1 percent"; I read this as 0.1% of
  the hit radius, σ = 0.001·R per coordinate. That sets the ~1 mrad angular
  resolution that shows up everywhere in the analysis. It's a single parameter
  and trivial to change.
- **Determinism.** Seeds are fixed (straight data seed 2026, 50-track seed 777);
  the test split is never touched during training or model selection.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017) — the transformer encoder.
- Zaheer et al., *Deep Sets* (2017) — why set symmetry matters here.
- Kuhn, *The Hungarian method* (1955) — optimal label matching for scoring.
- Hubert & Arabie, *Comparing partitions* (1985) — the Adjusted Rand Index.
- Amrouche et al., *The Tracking Machine Learning Challenge* (2020) — the
  physics context (TrackML / Kaggle).

Full reference list is in the paper.
