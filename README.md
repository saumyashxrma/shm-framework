# SHM framework v1.0.0

![tests](https://github.com/saumyashxrma/shm-framework/actions/workflows/tests.yml/badge.svg)

Decision-support software prototype for structural health monitoring (SHM), developed as Capstone Project-1 at
KL University (team Y23-HTE-PROJECT-62: Saumya Sharma, Abhigyan Sharma, Madhuhan Shekhar; guide
Dr. Dinesh Kumar Anguraj).

| Objective | What the software does |
|---|---|
| **O1** – damage detection and assessment (Utah long-term guided-wave data, outdoor aluminium plate, 2018–2022) | preprocesses 8-path guided-wave recordings, predicts a continuous damage level with a CNN–BiLSTM 3-seed ensemble, converts it to severity (Monitor / Inspect / Emergency) and mechanism (mass / dent / through-hole), tracks the level over time, raises change alarms and compensates temperature/humidity drift |
| **O2** – fatigue-crack diagnosis and prognosis (NASA / PHM 2019 lap-joint data) | extracts 9 physics features from Lamb-wave recordings, estimates crack length with gradient boosting (monotone in time), forecasts growth with a linear trend + Gaussian process, and attaches 90 % conformal intervals and an inspection rule |

> Not certified for operational use. Results come from two public laboratory/outdoor datasets.

## Results (leakage-free protocols)

| Result | Value |
|---|---|
| O1 known damage states (later recordings), severity accuracy / macro-F1 | 94.4 % / 0.958; Emergency recall 100 % (1250/1250) |
| O1 never-seen damage states, per recording | 72.2 % / 0.589 |
| O1 never-seen damage states, cumulative tracking (adopted by a pre-declared rule) | **83.8 % / 0.851** (folds A/B/C 100 / 51.5 / 100 %; target 85 %, narrowly missed) |
| O1 mechanism, never-seen states (balanced accuracy) | 92.4 % per recording, 95.6 % tracked (environment-only control 57.7 %, chance 50 %) |
| O1 environment-compensated detection, unseen healthy month | specificity 71.8 %, sensitivity 64.3 % |
| O2 diagnosis MAE, nested leave-one-specimen-out (5 seeds) | **1.047 ± 0.036 mm** (cycle-only 1.287, random-feature control 1.515) |
| O2 forecast RMSE from true history, T1–T6 | 1.15 mm (similarity 2.09, Paris law 2.99) |
| O2 90 % intervals and coverage on held-out T7/T8 | ±3.08 mm diagnosis, ±4.73 mm forecast; 88.9 % (16/18) |
| O2 PHM 2019 challenge RMSE T7 / T8 | 3.61 / 2.01 mm (oracle with true history 0.67 / 1.30) |

Known limitations: single-recording detection does not transfer reliably across seasons; never-seen damage states
near class boundaries are under-estimated (fold B); fold C under-calls tag 13 (32.5 % Emergency); tracking assumes
the start of permanent damage is confirmed; one structure per dataset. Full analysis: project report, Chapters 6–7.

## Repository structure

```
src/shm/            the package
  config.py         frozen parameters of v1.0.0
  preprocess.py     O1 band-pass, decimation, normalisation, DC/NaN guards
  splits.py         leave-damage-out folds, chronological validation, leakage checks
  model_o1.py       loads and runs the CNN–BiLSTM 3-seed ensembles (TensorFlow, lazy import)
  tracking.py       level -> severity/mechanism, causal median, cumulative tracking, change detection
  envcomp.py        temperature/humidity drift compensation for detection
  features_o2.py    NASA band-pass + 9 physics features
  model_o2.py       gradient-boosting crack diagnoser + causal monotone constraint
  forecast.py       linear trend + Gaussian-process forecaster
  conformal.py      split-conformal intervals
  pipeline.py       end-to-end O1 / O2 pipelines and console demo
  data.py           loaders for the exported sample data
app/app.py          Gradio demonstration (O1, O2, About)
tests/              pytest suite (runs without TensorFlow or data; data checks run when the samples exist)
export_cells/       Kaggle / Colab cells that export models and sample data from the experiment runs
models/ data/sample/ results/   exported artefacts (see the README in each folder)
.github/workflows/tests.yml     continuous integration
```

## Install and run (Python 3.11)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
pytest -v                         # automated tests
python -m shm.pipeline --demo     # console demonstration
python app/app.py                 # web demo at http://127.0.0.1:7860
```

Without the exported sample files the demo and app run on clearly labelled synthetic data.

## Reproducing the results

The experiments ran on Kaggle (Tesla T4, O1) and Google Colab (CPU, O2). The notebooks and the frozen outputs are
the evidence; this repository packages the inference logic and the trained artefacts.

| Step | Where | Output |
|---|---|---|
| O1 cache build and splits (cells B1, C1) | Colab | `shm-v2-cache` Kaggle dataset |
| O1 training and evaluation (K1–K18, KF2) | Kaggle notebook `shm-final-figures`, one committed run | models, `results/o1/*.json`, figures |
| O2 features, diagnosis, prognosis, intervals (Q-SETUP … Q-CONF) | Colab | `results/o2/*.json`, `results/MASTER_RESULTS_O2.csv`, figures |
| Export to this repository | `export_cells/KAGGLE_EXPORT.py`, `export_cells/COLAB_EXPORT.py` | `models/`, `data/sample/`, `results/` |

Seeds: O1 42, 123, 2024 (TensorFlow deterministic ops); O2 GBM seeds 0–4, random-feature control seed 2024.

## Data sources

* K. Yang et al., "Dataset on guided waves from long-term structural health monitoring under uncontrolled and
  dynamic conditions," *Scientific Data* 12, 991 (2025), doi:10.1038/s41597-025-05300-5; data on figshare,
  doi:10.6084/m9.figshare.28112504.
* NASA / PHM Society, "Fatigue Crack Growth in Aluminum Lap Joint" (PHM 2019 Data Challenge),
  https://data.nasa.gov/dataset/fatigue-crack-growth-in-aluminum-lap-joint

Use of the data follows the terms on those pages; the raw data are not redistributed here.

## Licence and citation

Code: MIT (see `LICENSE`). Citation metadata: `CITATION.cff`.
