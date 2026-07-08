"""Reusable plotting helpers for the tracking simulator."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .simulator import detector_radii

_COLORS = plt.cm.tab10.colors


def draw_detector(ax, radii):
    """Draw the concentric detector circles on `ax`."""
    for r in radii:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="tab:blue", lw=1.1))


def plot_event(event, ax=None, radii=None, show_rays=True, legend=True):
    """Plot one event's tracks (rays) and hits over the detector circles.

    `event` is a DataFrame as returned by simulate_event.
    """
    if radii is None:
        radii = detector_radii(5, 2.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    R = radii.max()
    draw_detector(ax, radii)

    for track_id, g in event.groupby("track_id"):
        c = _COLORS[track_id % 10]
        if show_rays:
            phi = g["phi"].iloc[0]
            ax.plot([0, R * np.cos(phi)], [0, R * np.sin(phi)],
                    color=c, lw=1, alpha=0.5)
        ax.scatter(g["x"], g["y"], color=c, s=35, zorder=3,
                   label=f"track {track_id}")

    lim = R * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if legend:
        ax.legend(loc="upper right", fontsize=8)
    return ax
