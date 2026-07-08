# Particle Tracking with Machine Learning

Simplified simulator and neural-network hit-to-track association for a
particle-physics tracking problem (Machine Learning in Particle Physics and
Astronomy, exam assignment 2026).

## Problem in one paragraph

When particles collide in a detector they leave "hits" as they cross detector
layers. In our simplified 2D setup there is no magnetic field and all particles
start from the origin, so each track is a straight ray described by a single
parameter (its angle). The detector is modeled as concentric circles. The goal
is first to *simulate* events (tracks + hits), and then to train a neural
network that assigns each hit back to the track that produced it.

## Repository layout

| Folder         | Contents                                                        |
|----------------|-----------------------------------------------------------------|
| `src/`         | Reusable code: the simulator, data utilities, models            |
| `notebooks/`   | Exploration & figures (`01_simulator.ipynb`, `02_detector_effects.ipynb`)|
| `data/`        | Generated events (large files are git-ignored)                  |
| `models/`      | Trained models in saved format (Deliverable 3)                  |
| `predictions/` | Comma-separated prediction files (Deliverable 4)                |
| `paper/`       | PRL-style letter describing the solution (Deliverable 1)        |

## Deliverables (as required by the assignment)

1. A max-4-page PRL-style paper (`paper/`)
2. The code, `.py` / `.ipynb` (`src/`, `notebooks/`)
3. The trained models in saved format (`models/`)
4. Prediction files, one event per line (`predictions/`)

## Tasks

- [x] a) Simulate events: 3 straight tracks from the origin, hits on 5 circles
- [x] b) Add detector effects: 95% hit efficiency + Gaussian position smearing; 10k events, 10 tracks
- [ ] c) Histograms of hits and tracks in x, y, phi
- [ ] d) Neural network for hit-to-track association (+ scaling to 50 tracks)
- [ ] e) Apply to the provided curved-track dataset
- [ ] f) (bonus) 3D with spheres

## Setup

This project uses a dedicated virtual environment named **`particle-tracking-ml`**
(directory `.venv-ptml/`), also registered as a Jupyter kernel of the same name.

```bash
# Create the environment (identifiable prompt + kernel name)
python -m venv --prompt particle-tracking-ml .venv-ptml
source .venv-ptml/Scripts/activate   # Windows (Git Bash); PowerShell: .venv-ptml\Scripts\Activate.ps1
pip install -r requirements.txt

# Register it as a Jupyter kernel so the notebooks can select it
python -m ipykernel install --user --name particle-tracking-ml \
    --display-name "Python (particle-tracking-ml)"
```

In VS Code / Jupyter, select the **Python (particle-tracking-ml)** kernel for the notebooks.

## Running

```bash
python -m src.simulator          # prints a sample ideal event and one task-b event
jupyter lab                      # open notebooks/01_simulator.ipynb and 02_detector_effects.ipynb
```

Run `01_simulator.ipynb` (task a) then `02_detector_effects.ipynb` (task b); the
latter regenerates `data/events_2d_10k.csv`.
