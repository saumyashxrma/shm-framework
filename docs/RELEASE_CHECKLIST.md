# Release checklist (v1.0.0)

1. Kaggle committed run finished; numbers in the log match `results/` (K10 94.4 %, K16 83.8 %, K17 71.8/64.3, K18 KK 0.46/89.1 %).
2. `export_cells/KAGGLE_EXPORT.py` run -> `shm_export_o1.zip` unzipped into the repo root (fills `models/`, `data/sample/`, `results/o1/`, `figures/o1/`).
3. `export_cells/COLAB_EXPORT.py` run -> `shm_export_o2.zip` unzipped into the repo root.
4. Local: `pytest -v` -> all passed, the two data tests no longer skipped.
5. `pip freeze > requirements-lock.txt` committed.
6. Push to GitHub; Actions run green; note the run URL and test count.
7. Tag `v1.0.0` and publish the release with both export zips attached; note the commit hash.
8. Demo: `python app/app.py`; screenshots of the three tabs; 3–5 min backup recording.
