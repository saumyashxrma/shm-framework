# Sample data (exported, not raw datasets)

| File | Content |
|---|---|
| `o1_recordings.npz` | ≤ 15 preprocessed recordings per damage tag (0–13): `waves` (n, 1000, 8) float16, `env` (temperature °C, humidity %), `tags`, `month`, `index`, `ref_level_A` (fold-A ensemble level computed on Kaggle) |
| `o1_stream.npz` | the time-ordered monitoring stream (all months except Oct 2022): `tags`, `month`, `env`, `t`, `level_A/B/C` (each fold's 3-seed ensemble, from K16) |
| `o2_specimens.json` | T1–T8: `cycles`, `cracks` (labelled inspections), `phys` (9 physics features); T7/T8 also `future_cycles`, `future_cracks` (challenge forecast points) |

Raw data are not redistributed here; see README.md, "Data sources".
