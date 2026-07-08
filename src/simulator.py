"""
simulator.py
------------
Monte-Carlo simulator for the simplified 2D tracking problem (tasks a & b).

Physics setup
-------------
* No magnetic field; every particle starts at (0, 0).
=> Each track is a straight RAY from the origin, described by ONE parameter,
   its azimuthal angle phi in [0, 2*pi):
       (x, y) = (r*cos phi, r*sin phi),  r >= 0.

Detector
--------
`n_circles` concentric circles of equally spaced radii, centred on the origin.
A hit is where a track crosses a circle. Because every track passes through the
common centre, the crossing with a circle of radius R is simply
(R*cos phi, R*sin phi) -- no line-circle quadratic needed, one hit per circle.

Detector effects (task b)
-------------------------
* Efficiency: each ideal hit is RECORDED with probability `efficiency` (0.95),
  independently. Missing hits => variable hit count per event.
* Smearing: each recorded hit's (x, y) is perturbed by independent Gaussian
  noise. We model the std as a FRACTION of the hit's radius: sigma = smear * R
  (smear = 0.001 => "0.1 percent"). See `simulate_event` for the interpretation.

Task a is recovered with the defaults efficiency=1.0, smear=0.0.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

COLUMNS = ["event_id", "track_id", "layer", "x", "y", "phi"]


# --- detector geometry -------------------------------------------------------

def detector_radii(n_circles: int = 5, spacing: float = 2.0) -> np.ndarray:
    """Radii of the concentric detector circles, equally spaced.

    n_circles=5, spacing=2 -> array([2., 4., 6., 8., 10.]).
    """
    return spacing * np.arange(1, n_circles + 1)


# --- track generation --------------------------------------------------------

def sample_track_angles(n_tracks: int,
                        rng: np.random.Generator,
                        min_sep: float = 0.0) -> np.ndarray:
    """Draw `n_tracks` track directions phi, uniform on [0, 2*pi).

    Set `min_sep` > 0 to guarantee a minimum angular gap between tracks
    (rejection sampling) -- useful when overlapping tracks would be ambiguous.
    """
    if min_sep <= 0.0:
        return rng.uniform(0.0, 2.0 * np.pi, size=n_tracks)

    phis: list[float] = []
    while len(phis) < n_tracks:
        cand = rng.uniform(0.0, 2.0 * np.pi)
        if all(_angular_gap(cand, p) >= min_sep for p in phis):
            phis.append(cand)
    return np.array(phis)


def _angular_gap(a: float, b: float) -> float:
    """Smallest absolute angle between two directions, in [0, pi]."""
    d = abs(a - b) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def track_hits(phi: float, radii: np.ndarray) -> np.ndarray:
    """Ideal (x, y) hits of one track (angle phi) on each circle.

    Shape (len(radii), 2), innermost first.
    """
    x = radii * np.cos(phi)
    y = radii * np.sin(phi)
    return np.column_stack([x, y])


# --- events ------------------------------------------------------------------

def simulate_event(event_id: int,
                   n_tracks: int = 3,
                   n_circles: int = 5,
                   spacing: float = 2.0,
                   efficiency: float = 1.0,
                   smear: float = 0.0,
                   min_sep: float = 0.0,
                   rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Simulate one event; return recorded hits as a tidy DataFrame.

    Columns: event_id, track_id, layer, x, y, phi. `x, y` are the RECORDED
    (possibly smeared) positions; `phi` is the TRUE track angle (the label).

    Detector effects
    ----------------
    efficiency : probability in [0, 1] that each ideal hit is recorded.
    smear      : Gaussian position-noise std as a fraction of the hit radius,
                 i.e. sigma_x = sigma_y = smear * R  (smear=0.001 -> 0.1%).
    """
    if rng is None:
        rng = np.random.default_rng()

    radii = detector_radii(n_circles, spacing)
    phis = sample_track_angles(n_tracks, rng, min_sep=min_sep)

    rows = []
    for track_id, phi in enumerate(phis):
        hits = track_hits(phi, radii)                 # (n_circles, 2), ideal

        # --- efficiency: keep each hit with prob `efficiency` ---
        keep = rng.random(n_circles) < efficiency
        layers = np.nonzero(keep)[0]
        hits = hits[keep]

        # --- smearing: Gaussian noise, std = smear * R (per coordinate) ---
        if smear > 0.0 and len(hits):
            sigma = smear * radii[keep]               # (m,)
            noise = rng.normal(0.0, 1.0, hits.shape) * sigma[:, None]
            hits = hits + noise

        for layer, (x, y) in zip(layers, hits):
            rows.append((event_id, track_id, layer, x, y, phi))

    return pd.DataFrame(rows, columns=COLUMNS)


def simulate_events(n_events: int, seed: int | None = None, **kwargs) -> pd.DataFrame:
    """Simulate many events and concatenate them into one tidy DataFrame."""
    rng = np.random.default_rng(seed)
    frames = [simulate_event(i, rng=rng, **kwargs) for i in range(n_events)]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # ideal (task a)
    print("Task a (ideal):")
    print(simulate_events(1, seed=0, n_tracks=3).to_string(index=False))
    # with detector effects (task b)
    print("\nTask b (efficiency + smearing), one 10-track event:")
    ev = simulate_event(0, n_tracks=10, efficiency=0.95, smear=0.001,
                        rng=np.random.default_rng(0))
    print(f"recorded hits: {len(ev)} (ideal would be 50)")
