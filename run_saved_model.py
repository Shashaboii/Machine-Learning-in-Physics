"""
HOW TO RUN

  1. Keep this script in the same folder as my model files
     (model_d_10tracks.pt, model_e_curved.pt) and the curved data (hits_FINAL.csv).

  2. Straight-track model — re-creates its test set from a fixed seed
     then prints the score:
       python run_saved_model.py --model model_d_10tracks.pt

  3. Curved-track model — needs the data file passed with --data:
       python run_saved_model.py --model model_e_curved.pt --data hits_FINAL.csv

"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# --------------------------------------------------------------------------
# Simulator (identical to the notebooks, needed to regenerate the straight-
# track test set from its seed).
# --------------------------------------------------------------------------
COLUMNS = ["event_id", "track_id", "layer", "x", "y", "phi"]


def detector_radii(n_circles: int = 5, spacing: float = 2.0) -> np.ndarray:
    return spacing * np.arange(1, n_circles + 1)


def simulate_event(event_id, n_tracks=3, n_circles=5, spacing=2.0,
                   efficiency=1.0, smear=0.0, rng=None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()
    radii = detector_radii(n_circles, spacing)
    phis = rng.uniform(0.0, 2.0 * np.pi, size=n_tracks)
    rows = []
    for track_id, phi in enumerate(phis):
        hits = np.column_stack([radii * np.cos(phi), radii * np.sin(phi)])
        keep = rng.random(n_circles) < efficiency
        layers = np.nonzero(keep)[0]
        hits = hits[keep]
        if smear > 0.0 and len(hits):
            sigma = smear * radii[keep]
            hits = hits + rng.normal(0.0, 1.0, hits.shape) * sigma[:, None]
        for layer, (x, y) in zip(layers, hits):
            rows.append((event_id, track_id, layer, x, y, phi))
    return pd.DataFrame(rows, columns=COLUMNS)


def simulate_events(n_events, seed=None, **kwargs) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat([simulate_event(i, rng=rng, **kwargs)
                      for i in range(n_events)], ignore_index=True)


# --------------------------------------------------------------------------
# Model (identical class to the notebooks so the state_dict loads 1:1).
# --------------------------------------------------------------------------
class HitSetTransformer(nn.Module):
    def __init__(self, n_features=2, d_model=64, n_heads=4, n_layers=3,
                 d_ff=128, d_embed=32, dropout=0.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_features, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(d_model, d_embed)
        self.d_embed = d_embed

    def forward(self, hits, mask):
        h = self.embed(hits)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.head(h)

    def pair_logits(self, z):
        return torch.matmul(z, z.transpose(1, 2)) / np.sqrt(self.d_embed)


# --------------------------------------------------------------------------
# Inference + evaluation (identical logic to the notebooks).
# --------------------------------------------------------------------------
def cluster_affinity(prob, n_clusters):
    D = 1.0 - 0.5 * (prob + prob.T)
    np.fill_diagonal(D, 0.0)
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                  linkage="average")
    return agg.fit_predict(D)


def hungarian_hit_accuracy(true_labels, pred_labels):
    t_ids, t_inv = np.unique(true_labels, return_inverse=True)
    p_ids, p_inv = np.unique(pred_labels, return_inverse=True)
    C = np.zeros((len(t_ids), len(p_ids)), dtype=int)
    np.add.at(C, (t_inv, p_inv), 1)
    rows, cols = linear_sum_assignment(-C)
    return C[rows, cols].sum() / len(true_labels)


@torch.no_grad()
def predict_event(model, g, n_tracks, feature_cols, scales, device):
    feats = [g[c].to_numpy() / s for c, s in zip(feature_cols, scales)]
    x = torch.tensor(np.column_stack(feats), dtype=torch.float32,
                     device=device).unsqueeze(0)
    m = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    prob = torch.sigmoid(model.pair_logits(model(x, m)))[0].cpu().numpy()
    return cluster_affinity(prob, n_tracks)


# --------------------------------------------------------------------------
# Data loading for the curved dataset (same tolerant reader as task (e)).
# --------------------------------------------------------------------------
def read_curved_file(path):
    with open(path) as f:
        lines = f.readlines()
    hdr = next(i for i, l in enumerate(lines)
               if "event_id" in l and "track_id" in l)
    sep = "\t" if "\t" in lines[hdr] else ","
    return pd.read_csv(path, sep=sep, skiprows=hdr)


def curved_test_split(df, seed=2026):
    """Reproduce the exact 80/10/10 event split used in task (e)."""
    ev = df.event_id.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ev)
    n = len(ev)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    return df[df.event_id.isin(ev[n_tr + n_va:])]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="path to model_d_10tracks.pt or model_e_curved.pt")
    ap.add_argument("--data", default=None,
                    help="curved hits CSV (required for the curved model)")
    ap.add_argument("--max-events", type=int, default=None,
                    help="optionally limit the number of evaluated events")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"loaded {args.model}  config: {cfg}")

    is_curved = cfg.get("task") == "e_curved" or "extra_cols" in cfg

    model_kwargs = {k: cfg[k] for k in
                    ("d_model", "n_heads", "n_layers", "d_ff", "d_embed")
                    if k in cfg}
    model = HitSetTransformer(n_features=cfg.get("n_features", 2),
                              **model_kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if is_curved:
        if args.data is None:
            ap.error("the curved model needs --data <curved hits CSV>")
        df = read_curved_file(args.data)
        df_test = curved_test_split(df, seed=2026)
        feature_cols = ["x", "y"] + cfg.get("extra_cols", [])
        scales = [10.0, 10.0] + [cfg.get("extra_scale", 1.0)] * \
                 len(cfg.get("extra_cols", []))
        per_event_K = True
        label = "curved test split"
    else:
        seed, n_tracks = cfg["seed"], cfg["n_tracks"]
        print(f"regenerating straight-track dataset (seed {seed}) ...")
        df = simulate_events(10000, seed=seed, n_tracks=n_tracks,
                             efficiency=0.95, smear=0.001)
        df_test = df[df.event_id >= 9000]
        feature_cols, scales = ["x", "y"], [10.0, 10.0]
        per_event_K = False
        label = "straight-track sealed test split (events 9000-9999)"

    accs, aris = [], []
    for i, (eid, g) in enumerate(df_test.groupby("event_id")):
        if args.max_events is not None and i >= args.max_events:
            break
        K = g.track_id.nunique() if per_event_K else cfg["n_tracks"]
        pred = predict_event(model, g, K, feature_cols, scales, device)
        true = g.track_id.to_numpy()
        accs.append(hungarian_hit_accuracy(true, pred))
        aris.append(adjusted_rand_score(true, pred))

    print(f"\n{label}  ({len(accs)} events)")
    print(f"  hit accuracy (Hungarian-matched) : {np.mean(accs):.4f}")
    print(f"  adjusted Rand index              : {np.mean(aris):.4f}")
    print(f"  fraction of perfect events       : {np.mean(np.array(accs) == 1):.4f}")


if __name__ == "__main__":
    main()
