# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# COLAB EXPORT for the shm-framework repo (O2). CPU, ~5 min. Paste as ONE cell in a fresh Colab runtime and run.
# Part 1 below = the unchanged Q-SETUP, Q0 and Q1 cells of the final Part-2 notebook (reads the NASA data, builds
# the 9 physics features). Part 2 = export. Output: Drive SHM/SHM_v2/shm_export_o2.zip (download it from Drive).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# ── Q-SETUP. Settings for Part 2 v3 (run once per runtime, before Q0 / Q5) ──
from google.colab import drive
drive.mount('/content/drive')

import os, json
import numpy as np
import pandas as pd

BASE      = '/content/drive/MyDrive/SHM'
NASA_ROOT = f'{BASE}/PHMDC2019_Data/PHMDC2019_Data'
V2        = f'{BASE}/SHM_v2'
os.makedirs(f'{V2}/cache', exist_ok=True)

def _np(o):
    return o.tolist() if hasattr(o, 'tolist') else str(o)

def save_json(obj, name):
    path = os.path.join(V2, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_np)
    print('  saved', name)
    return path

def load_json(name):
    with open(os.path.join(V2, name)) as f:
        return json.load(f)

assert os.path.isdir(f'{NASA_ROOT}/training'), f'NASA data not found at {NASA_ROOT}'
assert os.path.exists(f'{V2}/cache/nasa_future_labels.json'), 'nasa_future_labels.json missing'
print('NASA specimens:', sorted(os.listdir(f'{NASA_ROOT}/training')), '+', sorted(os.listdir(f'{NASA_ROOT}/validation')))
print('OK')

# ── Q0. PART 2 v3 SETUP: raw NASA signals (amplitude kept), filtered at full 20 MHz. Run Cell 1 (settings) first. ──
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, correlate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

FS = 20e6
bQ, aQ = butter(4, [50e3 / (FS / 2), 1000e3 / (FS / 2)], btype='band')
TRAIN_SPECS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
FUT = load_json('cache/nasa_future_labels.json')['future_labels']

def parse_labels(desc_path):
    df = pd.read_excel(desc_path, sheet_name=0, header=None); labels = {}
    for i in range(len(df)):
        try: labels[int(float(df.iloc[i, 0]))] = float(df.iloc[i, 1])
        except (ValueError, TypeError): continue
    return labels

RAW = {}
for folder, names in [('training', TRAIN_SPECS), ('validation', ['T7', 'T8'])]:
    for spec in names:
        sdir = os.path.join(NASA_ROOT, folder, spec)
        labels = parse_labels(os.path.join(sdir, f'Description_{spec}.xlsx'))
        rec = []
        for d_ in os.listdir(sdir):
            if not os.path.isdir(os.path.join(sdir, d_)) or not d_.isdigit(): continue
            cyc = int(d_)
            if cyc not in labels: continue
            rx, tx = [], []
            for r in ['signal_1.csv', 'signal_2.csv']:
                p = os.path.join(sdir, d_, r)
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    rx.append(filtfilt(bQ, aQ, df['ch2'].values.astype(float)))
                    tx.append(np.abs(df['ch1'].values.astype(float)).max())
            if rx: rec.append((cyc, labels[cyc], np.mean(rx, 0), float(np.mean(tx))))
        rec.sort(key=lambda r: r[0])
        RAW[spec] = {'cycles': np.array([r[0] for r in rec], float), 'cracks': np.array([r[1] for r in rec], float),
                     'rx': np.stack([r[2] for r in rec]), 'tx_peak': np.array([r[3] for r in rec])}
        print(f'{spec}: {len(rec)} labelled recordings | excitation peak range {RAW[spec]["tx_peak"].min():.3f}–{RAW[spec]["tx_peak"].max():.3f}')

# ── Q1 (fixed). PHYSICS FEATURES relative to each specimen's own first recording ──
MAXLAG = 200                                          # ±10 µs at 20 MHz
EPS = 1e-12
def phys_features(rx, tx_peak):
    s0 = rx[0] / tx_peak[0]; e0 = np.abs(hilbert(s0)); k0 = int(np.argmax(e0)); L = len(s0)
    lo, hi = max(0, k0 - 150), min(L, k0 + 150)
    pk = np.zeros(L, bool); pk[lo:hi] = True; late = ~pk          # outside the first packet = scattered/reflected waves
    z0 = (s0 - s0.mean()) / s0.std()
    sp0 = np.abs(np.fft.rfft(s0)); f = np.arange(len(sp0)); c0 = (sp0 * f).sum() / sp0.sum()
    feats = []
    for s_raw, tp in zip(rx, tx_peak):
        s = s_raw / tp; e = np.abs(hilbert(s)); z = (s - s.mean()) / s.std()
        cc = correlate(z, z0, mode='full') / len(z); mid = len(z) - 1; w = cc[mid - MAXLAG: mid + MAXLAG + 1]
        lag = int(np.argmax(np.abs(w)) - MAXLAG)
        sp = np.abs(np.fft.rfft(s))
        feats.append([
            np.log((e.max() + EPS) / (e0.max() + EPS)),
            np.log(((s ** 2).sum() + EPS) / ((s0 ** 2).sum() + EPS)),
            lag,
            1.0 - np.max(np.abs(w)),
            1.0 - np.corrcoef(s, s0)[0, 1],
            np.log(((e[pk] ** 2).sum() + EPS) / ((e0[pk] ** 2).sum() + EPS)),
            np.log(((e[late] ** 2).sum() + EPS) / ((e0[late] ** 2).sum() + EPS)),
            int(np.argmax(e)) - k0,
            (sp * f).sum() / sp.sum() / c0 - 1.0,
        ])
    return np.nan_to_num(np.array(feats, float), nan=0.0, posinf=0.0, neginf=0.0)
PHYS_NAMES = ['log_amp', 'log_energy', 'phase_lag', 'aligned_decorr', 'raw_decorr', 'log_packet_E', 'log_late_E',
              'peak_shift', 'centroid_shift']

D = {}
for n, r in RAW.items():
    D[n] = {'cycles': r['cycles'], 'cracks': r['cracks'], 'phys': phys_features(r['rx'], r['tx_peak'])}
    print(f'{n}: first-packet peak at sample {int(np.argmax(np.abs(hilbert(r["rx"][0]))))} of {r["rx"].shape[1]}')
assert all(np.isfinite(D[n]['phys']).all() for n in D), 'still non-finite features'
print('\nfeature correlation with crack length, pooled T1–T6 (sanity check, not used for selection):')
Fp = np.vstack([D[n]['phys'] for n in TRAIN_SPECS]); Ap = np.concatenate([D[n]['cracks'] for n in TRAIN_SPECS])
for i, nm in enumerate(PHYS_NAMES):
    print(f'  {nm:15} r = {np.corrcoef(Fp[:, i], Ap)[0, 1]:+.3f}')


# ── EXPORT ──
import shutil, re, sklearn, platform
EXP = '/content/shm_export_o2'
shutil.rmtree(EXP, ignore_errors=True)
for p in ['models', 'data/sample', 'results/o2', 'figures/o2']:
    os.makedirs(f'{EXP}/{p}', exist_ok=True)

Q4 = load_json('Q4_challenge_v3.json'); QC = load_json('Q_CONF_intervals.json')
GBK = [k for k in Q4 if k.startswith('GBM+mono') and k.endswith('linear GP')][0]
hp = [int(v) for v in re.findall(r'\((\d+),\s*(\d+)\)', GBK)[0]]
print('final diagnoser key:', GBK, '-> hp', hp)

spec_out = {}
for n in sorted(D):
    d = {'cycles': D[n]['cycles'].tolist(), 'cracks': D[n]['cracks'].tolist(), 'phys': D[n]['phys'].tolist()}
    if n in FUT:
        d['future_cycles'] = [float(c) for c in FUT[n]]
        d['future_cracks'] = [float(v) for v in FUT[n].values()]
    spec_out[n] = d
json.dump(spec_out, open(f'{EXP}/data/sample/o2_specimens.json', 'w'))

ref = {n: {'estimate': [float(v) for v in Q4[GBK][n]['est']], 'forecast': [float(v) for v in Q4[GBK][n]['fut']]}
       for n in ['T7', 'T8']}
json.dump({'hp': hp, 'q_diag_mm': QC['q_d'], 'q_forecast_mm': QC['q_f_deployment'],
           'coverage_T7_T8': QC['coverage_all'], 'phys_names': PHYS_NAMES, 'reference': ref,
           'sklearn': sklearn.__version__, 'python': platform.python_version(),
           'source': 'Colab Part-2 final run (Q4 + Q-CONF)'}, open(f'{EXP}/models/o2_model.json', 'w'), indent=2)

for f in ['Q2_diagnosis_v3.json', 'Q3_prognosis_similarity_true_history.json', 'Q4_challenge_v3.json',
          'Q_CONF_intervals.json', 'Q7_paris_true_history.json']:
    if os.path.exists(f'{V2}/{f}'):
        shutil.copy(f'{V2}/{f}', f'{EXP}/results/o2/')
for f in sorted(os.listdir(f'{V2}/figures_O2')):
    shutil.copy(f'{V2}/figures_O2/{f}', f'{EXP}/' + ('results/o2/' if f.endswith('.csv') else 'figures/o2/'))

shutil.make_archive(f'{V2}/shm_export_o2', 'zip', EXP)
print('DONE -> Drive: SHM/SHM_v2/shm_export_o2.zip', f'({os.path.getsize(f"{V2}/shm_export_o2.zip")/1e6:.2f} MB)')
