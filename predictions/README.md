# Prediction files

These are the "Element 4" deliverables: one line per event, no header.

## `task_d_10tracks.csv` and `task_e_curved.csv`

Each line is an event id followed by one `(x, y, trackID)` triple per hit:

```
event_id, hit1_x, hit1_y, hit1_trackID, hit2_x, hit2_y, hit2_trackID, ...
```

- `x, y` are the recorded (smeared) hit coordinates.
- `trackID` is the **predicted** group the model assigned that hit to.
  The absolute number is meaningless — only which hits share the same id matters
  (the task is grouping, not labelling). See the paper for how this is scored
  with Hungarian matching.

`task_d_10tracks.csv` covers the sealed straight-track test split (event ids
9000–9999). `task_e_curved.csv` covers the curved-track test split.

## `*_params.csv`

One fitted parameter per predicted track:

```
event_id, trackID_1, param_1, trackID_2, param_2, ...
```

- Straight tracks: `param` is the track **azimuth** phi (radians).
- Curved tracks: `param` is the fitted **radius of curvature** R_c
  (`nan` where a track had too few hits to fit reliably).

These parameters are only produced for evaluation/inspection — the network
itself never sees them, it only sees hit coordinates (and the layer index).
