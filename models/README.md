# Model files (not produced by this repository's code; exported from the experiment runs)

| File | Objective | Source |
|---|---|---|
| `v3_level_fold{A,B,C}_seed{42,123,2024}.keras` | O1 Stage-2 damage-level CNN–BiLSTM (9 models, 3-seed ensemble per fold) | Kaggle `shm-final-figures`, cell K8 |
| `v3_stage1tc_fold-_seed42.keras` | O1 time-controlled Stage-1 detector (reported, not used by the decision layer) | Kaggle, cell K8 |
| `ENV_o1_export.json` | Python / TensorFlow / Keras versions of the export | `export_cells/KAGGLE_EXPORT.py` |
| `o2_model.json` | O2 diagnoser hyper-parameters, conformal half-widths, reference T7/T8 estimates | `export_cells/COLAB_EXPORT.py` |

The O2 gradient-boosting model is refitted at start-up from the exported T1–T6 features with the frozen
hyper-parameters (a few seconds, deterministic); `tests/test_pipeline.py` checks that the refit reproduces the
Colab estimates for T7/T8.
