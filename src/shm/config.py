"""Frozen configuration of release v1.0.0.

Every value here was fixed before the final evaluation runs and is identical to the values used in the
Kaggle (O1) and Colab (O2) experiment notebooks. Changing any of them changes the results and requires a new
release version.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "data" / "sample"
RESULTS_DIR = ROOT / "results"

# ---------------- O1: Utah long-term guided-wave dataset ----------------
FS_UTAH = 1e6                  # sampling rate of the raw recordings (Hz)
UTAH_BAND = (50e3, 240e3)      # Butterworth band-pass (Hz); below the post-decimation Nyquist of 250 kHz
FILTER_ORDER = 4               # order passed to scipy.signal.butter (applied forward-backward with filtfilt)
STRIDE = 2                     # 2000 raw samples -> 1000
RAW_SAMPLES = 2000
N_SAMP = RAW_SAMPLES // STRIDE
N_CH = 8                       # sensor paths
MAX_MEAN_ABS_DC = 0.02         # preprocessing fails loudly above this mean |DC| (leakage audit finding)

MONTHS = ["2018_04", "2019_01", "2019_07", "2021_04", "2021_06", "2021_08",
          "2021_10", "2022_01", "2022_07", "2022_08", "2022_09", "2022_10"]
HEALTHY_MONTHS = ["2018_04", "2019_01", "2019_07", "2021_04"]
EXTERNAL_MONTH = "2022_10"
TAG_FOLDS = {"A": [2, 7, 11], "B": [4, 5, 10], "C": [3, 8, 12]}   # leave-damage-out folds
SEEDS = [42, 123, 2024]
VAL_FRAC = 0.2                 # per tag: chronologically last 20 % -> validation

SEV_THR = (4.5, 9.5)           # damage level -> Monitor / Inspect / Emergency (tags 1-4 / 5-9 / 10-13)
SEV_NAMES = ["Monitor", "Inspect", "Emergency"]
MECH_THR = (1.5, 7.5)          # damage level -> mass / dent / through-hole
MECH_NAMES = ["mass", "dent", "hole"]

# Time-based monitoring (K16) - fixed before running
W_MED = 50                     # causal rolling-median window (recordings)
B_REF = 200                    # recordings used to (re)build a change-detection baseline
DELTA = 1.0                    # alarm threshold (tag units)
PERSIST = 50                   # consecutive recordings above DELTA before an alarm
D_WIN = 1000                   # detection window after a damage change (scoring only)

# Environment compensation (K17)
ENV_POLY_DEGREE = 2
ENV_RIDGE_ALPHA = 1.0
ENV_FA_QUANTILE = 0.95         # 5 % false-alarm budget on healthy reference recordings

# ---------------- O2: NASA / PHM 2019 lap-joint fatigue data ----------------
FS_NASA = 20e6
NASA_BAND = (50e3, 1000e3)
TRAIN_SPECS = ["T1", "T2", "T3", "T4", "T5", "T6"]
CHALLENGE_SPECS = ["T7", "T8"]
PHYS_NAMES = ["log_amp", "log_energy", "phase_lag", "aligned_decorr", "raw_decorr",
              "log_packet_E", "log_late_E", "peak_shift", "centroid_shift"]
MAXLAG = 200                   # +/- 10 us cross-correlation search at 20 MHz
PACKET_HALF_WIDTH = 150        # samples either side of the first-packet envelope peak
GB_GRID = [(d, n) for d in (1, 2) for n in (50, 150, 300)]
HIST_FRAC = 0.6                # forecasting protocol: first 60 % of a history -> forecast the rest
CONFORMAL_LEVEL = 0.90

# Final O2 interval half-widths (Q-CONF, deployment-matched calibration on T1-T6). Overwritten by
# models/o2_model.json when the exported file is present.
Q_DIAG_MM = 3.0831593004719973
Q_FORECAST_MM = 4.7323831775409335
