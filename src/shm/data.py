"""Loading of the exported sample data and model metadata (see export_cells/ and data/sample/README.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C

O1_RECORDINGS = "o1_recordings.npz"
O1_STREAM = "o1_stream.npz"
O2_SPECIMENS = "o2_specimens.json"
O2_MODEL = "o2_model.json"


def has(name: str, folder: Path = C.SAMPLE_DIR) -> bool:
    return (Path(folder) / name).exists()


def load_o1_recordings(folder: Path = C.SAMPLE_DIR) -> dict:
    z = np.load(Path(folder) / O1_RECORDINGS, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_o1_stream(folder: Path = C.SAMPLE_DIR) -> dict:
    z = np.load(Path(folder) / O1_STREAM, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_o2_specimens(folder: Path = C.SAMPLE_DIR) -> dict:
    """{spec: {cycles, cracks, phys, (future_cycles, future_cracks)}} as numpy arrays."""
    raw = json.loads((Path(folder) / O2_SPECIMENS).read_text())
    out = {}
    for n, d in raw.items():
        out[n] = {k: np.asarray(v, float) for k, v in d.items()}
    return out


def load_o2_model_info(folder: Path = C.MODELS_DIR) -> dict:
    p = Path(folder) / O2_MODEL
    if p.exists():
        return json.loads(p.read_text())
    # hp = None -> the pipeline selects it by leave-one-specimen-out on T1-T6 (same rule as the notebook)
    return {"hp": None, "q_diag_mm": C.Q_DIAG_MM, "q_forecast_mm": C.Q_FORECAST_MM, "source": "config defaults"}
