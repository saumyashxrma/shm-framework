# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# KAGGLE EXPORT for the shm-framework repo (O1). Prediction only, ~5–10 min, GPU optional.
# Run in an INTERACTIVE session of the notebook "shm-final-figures" AFTER its committed run has finished:
#   right panel -> Input -> + Add Input -> Your Work / Notebooks -> shm-final-figures (latest version)  -> Add
#   keep the dataset "shm-v2-cache" attached as well.  Then paste this as ONE new cell and run it.
# Output: /kaggle/working/shm_export_o1.zip  -> right panel Output -> ⋮ -> Download
# ════════════════════════════════════════════════════════════════════════════════════════════════════════
import os, sys, json, shutil, glob, platform
import numpy as np

def find_dir(pred):
    for r, ds, fs in os.walk('/kaggle/input', followlinks=True):
        if pred(r, ds, fs):
            return r
    return None

DATA = find_dir(lambda r, ds, fs: 'utah_meta.npz' in fs)
OUT = find_dir(lambda r, ds, fs: 'models_v3' in ds and os.path.basename(r) == 'SHM_v2')
assert DATA, 'dataset shm-v2-cache is not attached'
assert OUT, 'committed output of shm-final-figures is not attached (Add Input -> Notebook output)'
print('data   :', DATA, '\noutput :', OUT)

MONTHS = ['2018_04', '2019_01', '2019_07', '2021_04', '2021_06', '2021_08',
          '2021_10', '2022_01', '2022_07', '2022_08', '2022_09', '2022_10']
SEEDS = [42, 123, 2024]
EXP = '/kaggle/working/shm_export_o1'
shutil.rmtree(EXP, ignore_errors=True)
for p in ['models', 'data/sample', 'results/o1', 'figures/o1']:
    os.makedirs(f'{EXP}/{p}', exist_ok=True)

# ---- 1. models (9 level models + time-controlled Stage 1) and training histories ----
mdir = f'{OUT}/models_v3'
models = sorted(f for f in os.listdir(mdir) if f.endswith('.keras'))
for f in models:
    shutil.copy(f'{mdir}/{f}', f'{EXP}/models/{f}')
print('models:', models)
need = [f'v3_level_fold{f}_seed{s}.keras' for f in 'ABC' for s in SEEDS]
assert all(n in models for n in need), 'a level model is missing'

# ---- 2. results JSON + final figures ----
for f in sorted(glob.glob(f'{OUT}/*.json')):
    shutil.copy(f, f'{EXP}/results/o1/')
for f in sorted(glob.glob(f'{OUT}/figures_final/*.png')) + sorted(glob.glob(f'{OUT}/figures/*.png')):
    shutil.copy(f, f'{EXP}/figures/o1/')
print('results:', sorted(os.listdir(f'{EXP}/results/o1')))

# ---- 3. metadata ----
mt = np.load(f'{DATA}/utah_meta.npz')
tags = mt['tags'].astype(int); month = mt['month'].astype(str)
env = np.column_stack([mt['temp'], mt['hum']]).astype(np.float32)
first = {m: int(np.argmax(month == m)) for m in MONTHS}

# ---- 4. level stream (all recordings except Oct 2022, time order; levels from each fold's 3-seed ensemble, K16) ----
Z = np.load(f'{OUT}/K16_stream_levels.npz')
st = Z['stream']
np.savez_compressed(f'{EXP}/data/sample/o1_stream.npz', tags=tags[st], month=month[st], env=env[st],
                    t=mt['t'][st], **{f'level_{f}': Z[f'level_{f}'].astype(np.float32) for f in 'ABC'})
print('stream:', len(st), 'recordings')

# ---- 5. sample recordings: up to 15 per tag, spread over months, fixed seed ----
rng = np.random.default_rng(0)
pick = []
for t in range(14):
    idx = np.where(tags == t)[0]
    if len(idx):
        pick += sorted(rng.choice(idx, size=min(15, len(idx)), replace=False).tolist())
pick = np.array(sorted(pick))
mm = {m: np.load(f'{DATA}/X_{m}.npy', mmap_mode='r') for m in MONTHS}
waves = np.stack([np.asarray(mm[month[i]][i - first[month[i]]]) for i in pick]).astype(np.float16)

import tensorflow as tf
ens = [tf.keras.models.load_model(f'{mdir}/v3_level_foldA_seed{s}.keras') for s in SEEDS]
ref = np.mean([np.asarray(m.predict({'wave_input': waves.astype(np.float32), 'env_input': env[pick]},
                                    batch_size=128, verbose=0)).ravel() for m in ens], axis=0)
np.savez_compressed(f'{EXP}/data/sample/o1_recordings.npz', waves=waves, env=env[pick], tags=tags[pick],
                    month=month[pick], index=pick, ref_level_A=ref.astype(np.float32))
print('sample recordings:', len(pick), '| tags', np.bincount(tags[pick]))

# ---- 6. environment record ----
import keras
info = {'python': platform.python_version(), 'tensorflow': tf.__version__, 'keras': keras.__version__,
        'numpy': np.__version__, 'source_notebook': 'shm-final-figures (committed version)', 'models': models}
json.dump(info, open(f'{EXP}/models/ENV_o1_export.json', 'w'), indent=2)
print(info)

shutil.make_archive('/kaggle/working/shm_export_o1', 'zip', EXP)
print('\nDONE -> /kaggle/working/shm_export_o1.zip',
      f'({os.path.getsize("/kaggle/working/shm_export_o1.zip")/1e6:.1f} MB). Download it from the Output panel.')
