"""
simulator.py
------------
Monte-Carlo simulator for the simplified 2D tracking problem.

Setup: no magnetic field, all particles start at the origin, so every track is a
straight ray. The detector is modeled as `n_circles` concentric circles of
equally spaced radii. A "hit" is where a track crosses a circle.

This is a SKELETON. The function bodies are left for you to implement as part of
task (a). Each docstring tells you exactly what the function should return.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# --- detector geometry -------------------------------------------------------

def detector_radii(n_circles: int = 5, spacing: float = 2.0) -> np.ndarray:
    """Return the radii of the concentric detector circles.

    Equal spacing between radii, e.g. n_circles=5, spacing=2 -> [2, 4, 6, 8, 10].

    Returns
    -------
    np.ndarray of shape (n_circles,)
    """
    # TODO (task a, decision 3): build the array of radii.
    raise NotImplementedError


# --- one track ---------------------------------------------------------------

def sample_track_angles(n_tracks: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n_tracks` distinct track directions (the single track parameter).

    Think: which distribution, over which range, and how do you make sure the
    angles are genuinely different?

    Returns
    -------
    np.ndarray of shape (n_tracks,) with the angle (phi) of each track.
    """
    # TODO (task a, decisions 1 & 2)
    raise NotImplementedError


def track_hits(phi: float, radii: np.ndarray) -> np.ndarray:
    """Return the (x, y) hit of one track (angle `phi`) on each detector circle.

    Key insight (task a, decision 4): the track passes through the origin, which
    is also the centre of every circle. So you do NOT need a general line-circle
    intersection. Work out the hit coordinates directly from phi and the radius.

    Returns
    -------
    np.ndarray of shape (len(radii), 2): the (x, y) of each hit, innermost first.
    """
    # TODO (task a, decision 4)
    raise NotImplementedError


# --- one event ---------------------------------------------------------------

def simulate_event(event_id: int,
                   n_tracks: int = 3,
                   n_circles: int = 5,
                   spacing: float = 2.0,
                   rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Simulate a single event and return its hits as a tidy DataFrame.

    One row per hit, with columns matching the provided sample dataset:
        event_id, track_id, layer, x, y, phi

    (Detector effects — 95% efficiency and Gaussian smearing — come in task b;
    leave them out for now.)
    """
    if rng is None:
        rng = np.random.default_rng()
    # TODO: combine detector_radii, sample_track_angles and track_hits,
    #       then assemble the rows into a DataFrame.
    raise NotImplementedError


def simulate_events(n_events: int, seed: int | None = None, **kwargs) -> pd.DataFrame:
    """Simulate many events and concatenate them into one DataFrame."""
    rng = np.random.default_rng(seed)
    frames = [simulate_event(i, rng=rng, **kwargs) for i in range(n_events)]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # Smoke test — will raise NotImplementedError until you implement the stubs.
    df = simulate_events(1, seed=0, n_tracks=3)
    print(df)
