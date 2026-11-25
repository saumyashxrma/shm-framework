"""Gradio demonstration of the SHM framework (release v1.0.0).

Run from the repository root:   python app/app.py      then open http://127.0.0.1:7860
Tabs: O1 damage assessment (single recording + monitoring stream), O2 crack diagnosis and prognosis, About.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr                     # noqa: E402
import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402

from shm import __version__             # noqa: E402
from shm import config as C             # noqa: E402
from shm import data as io              # noqa: E402
from shm import model_o1                # noqa: E402
from shm.envcomp import EnvCompensator  # noqa: E402
from shm.pipeline import O2Pipeline, decide, monitor_stream, synthetic_specimens, synthetic_o1_recordings, synthetic_o1_stream_full  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, LIGHT, ORANGE, GREEN = "#2F5597", "#9DC3E6", "#C55A11", "#548235"

# ---------------------------------------------------------------- data loaded once at start-up
SYN_REC, SYN_STREAM = not io.has(io.O1_RECORDINGS), not io.has(io.O1_STREAM)
REC = synthetic_o1_recordings() if SYN_REC else io.load_o1_recordings()
STREAM = synthetic_o1_stream_full() if SYN_STREAM else io.load_o1_stream()
SYN_NOTE = "**SYNTHETIC demo data** – the exported O1 recordings/models are not installed (see README, Phase 2)."
if io.has(io.O2_SPECIMENS):
    O2 = O2Pipeline(io.load_o2_specimens())
    O2_SRC = "NASA / PHM 2019 features exported from the Colab run"
else:
    O2 = O2Pipeline(synthetic_specimens())
    O2_SRC = "SYNTHETIC placeholder data (export data/sample/o2_specimens.json for the real specimens)"
_ENSEMBLES: dict = {}


def _ensemble(fold):
    if fold not in _ENSEMBLES:
        _ENSEMBLES[fold] = model_o1.LevelEnsemble(fold)
    return _ENSEMBLES[fold]


# ---------------------------------------------------------------- O1: single recording
def rec_choices():
    if REC is None:
        return []
    return [f"#{i}  tag {int(t)}  ({m})" for i, (t, m) in enumerate(zip(REC["tags"], REC["month"]))]


def o1_single(choice, fold):
    if REC is None or not choice:
        return None, "No sample recordings found. Copy `o1_recordings.npz` into `data/sample/` (see README, Phase 2)."
    i = int(choice.split()[0][1:])
    w, env, tag = REC["waves"][i].astype(np.float32), REC["env"][i], int(REC["tags"][i])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    t = np.arange(C.N_SAMP) * C.STRIDE / C.FS_UTAH * 1e6
    for ch in range(C.N_CH):
        ax.plot(t, w[:, ch] + 2.2 * ch, lw=0.6, color=BLUE)
    ax.set_yticks(2.2 * np.arange(C.N_CH))
    ax.set_yticklabels([f"path {c + 1}" for c in range(C.N_CH)])
    ax.set_xlabel("Time (µs)")
    ax.set_title(f"{'SYNTHETIC – ' if SYN_REC else ''}Preprocessed recording #{i} (band-pass 50–240 kHz, peak-normalised)", fontsize=9)
    plt.tight_layout()
    if model_o1.available(fold):
        lv = float(_ensemble(fold).predict(w[None], env[None])[0])
        src = f"computed now with the fold-{fold} 3-seed ensemble"
    elif f"ref_level_{fold}" in REC:
        lv = float(REC[f"ref_level_{fold}"][i])
        src = (SYN_NOTE if SYN_REC else f"fold-{fold} level computed in the Kaggle run (TensorFlow/models not loaded here)")
    else:
        return fig, "Models for this fold are not available. Put the exported `.keras` files in `models/`."
    d = decide([lv])
    hidden = tag in C.TAG_FOLDS[fold]
    truth = "healthy" if tag == 0 else f"{C.SEV_NAMES[int(np.digitize(tag, C.SEV_THR))]} / {C.MECH_NAMES[int(np.digitize(tag, C.MECH_THR))]}"
    md = (f"### Level {lv:.2f} → **{C.SEV_NAMES[d.severity[0]]}**, mechanism **{C.MECH_NAMES[d.mechanism[0]]}**\n"
          f"- True tag **{tag}** ({truth}); {'**never seen by this fold** (leave-damage-out)' if hidden else 'seen in training'}\n"
          f"- Temperature {env[0]:.1f} °C, humidity {env[1]:.0f} %\n- Source: {src}\n"
          f"- A single recording is weak evidence across seasons; the monitoring view uses the recording stream.")
    return fig, md


# ---------------------------------------------------------------- O1: monitoring stream
def o1_stream(fold, position):
    if STREAM is None:
        return None, "No level stream found. Copy `o1_stream.npz` into `data/sample/` (see README, Phase 2)."
    lv, tags, month, env = STREAM[f"level_{fold}"], STREAM["tags"], STREAM["month"], STREAM["env"]
    pos = int(np.clip(position, 1, len(lv) - 1))
    start = int(np.argmax(tags >= 2))                      # tracking start: first permanent damage (report assumption)
    r = monitor_stream(lv[:pos + 1], month[:pos + 1], tracking_start=start if pos >= start else None)
    healthy = tags == 0
    ec = EnvCompensator.fit(env[healthy], lv[healthy])     # demo: drift fitted on all healthy months of the stream
    flag = bool(ec.flag(env[pos:pos + 1], lv[pos:pos + 1])[0])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(pos + 1)
    k = max(1, (pos + 1) // 6000)
    ax.plot(x[::k], lv[:pos + 1:k], ".", ms=1, color=LIGHT, alpha=0.4, label="per-recording level")
    ax.plot(x[::k], r.tracked.level[::k], color=BLUE, lw=0.9, label="tracked level")
    ax.step(x[::k], tags[:pos + 1:k], where="post", color="black", lw=0.7, label="true tag")
    if len(r.alarms):
        ax.plot(r.alarms, r.tracked.level[r.alarms], "v", color="#C00000", ms=5, label="change alarm")
    for thr, nm in zip(C.SEV_THR, ["Monitor | Inspect", "Inspect | Emergency"]):
        ax.axhline(thr, ls="--", lw=0.6, color="grey")
        ax.text(pos, thr + 0.15, nm, fontsize=7, ha="right", color="grey")
    ax.set_ylim(-1, 15)
    ax.set_xlabel("Recording (time order)")
    ax.set_ylabel("Damage level")
    ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=4)
    ax.set_title(f"{'SYNTHETIC – ' if SYN_STREAM else ''}Fold {fold} monitoring stream up to recording {pos} ({month[pos]})", fontsize=9)
    plt.tight_layout()
    d = r.tracked
    md = (f"### At recording {pos} ({month[pos]}): **{C.SEV_NAMES[d.severity[-1]]}**, mechanism "
          f"**{C.MECH_NAMES[d.mechanism[-1]]}** (tracked level {d.level[-1]:.2f})\n"
          f"- Per-recording level {lv[pos]:.2f}; true tag {int(tags[pos])}"
          f"{' (never seen by this fold)' if int(tags[pos]) in C.TAG_FOLDS[fold] else ''}\n"
          f"- Change alarms so far: {len(r.alarms)}\n"
          f"- Environment-compensated detector: **{'damage suspected' if flag else 'no damage indication'}** "
          f"at {env[pos, 0]:.1f} °C / {env[pos, 1]:.0f} % RH (demo fit on all healthy months)\n"
          f"- Tracking starts at the first permanent damage (recording {start}); in operation this comes from a "
          f"confirmed alarm or an inspection."
          + (f"\n- {SYN_NOTE}" if SYN_STREAM else ""))
    return fig, md


# ---------------------------------------------------------------- O2
def o2_run(spec, n_obs, a_crit):
    d = O2.D[spec]
    o = O2.run(spec, n_obs=int(n_obs), a_crit=float(a_crit))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(o.cycles, o.est_lo, o.est_hi, color=LIGHT, alpha=0.5, label=f"90 % diagnosis interval (±{o.q_diag:.2f} mm)")
    ax.plot(o.cycles, o.estimate, "o-", color=BLUE, ms=4, label="estimated crack length")
    if len(o.forecast):
        ax.fill_between(o.future_cycles, o.fc_lo, o.fc_hi, color="#F4B183", alpha=0.4, label=f"90 % forecast interval (±{o.q_forecast:.2f} mm)")
        ax.plot(o.future_cycles, o.forecast, "s--", color=ORANGE, ms=4, label="forecast")
    truth_c, truth_a = d["cycles"], d["cracks"]
    if spec in C.CHALLENGE_SPECS and "future_cycles" in d and int(n_obs) >= len(d["cycles"]):
        truth_c, truth_a = np.r_[truth_c, d["future_cycles"]], np.r_[truth_a, d["future_cracks"]]
    ax.plot(truth_c, truth_a, "k^", mfc="none", ms=6, label="true (labels)")
    ax.axhline(a_crit, color="#C00000", lw=0.8, ls=":", label=f"critical length {a_crit:.1f} mm")
    ax.set_xlabel("Load cycles")
    ax.set_ylabel("Crack length (mm)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"{spec}: diagnosis from {len(o.cycles)} inspections and forecast", fontsize=9)
    plt.tight_layout()
    role = "held-out challenge specimen" if spec in C.CHALLENGE_SPECS else "training specimen (model refitted on all of T1–T6, so this is in-sample)"
    md = (f"### {spec}: {o.estimate[-1]:.2f} mm [{o.est_lo[-1]:.2f}, {o.est_hi[-1]:.2f}] at {o.cycles[-1]:.0f} cycles\n"
          f"- Inspection rule (flag when the 90 % upper bound reaches {a_crit:.1f} mm): "
          f"**{'INSPECT NOW' if o.inspect_now else 'no inspection needed yet'}**"
          + (f"; first flag at {o.first_flag_cycle:.0f} cycles" if o.first_flag_cycle is not None else "") +
          f"\n- Forecast at {o.future_cycles[-1]:.0f} cycles: {o.forecast[-1]:.2f} mm [{o.fc_lo[-1]:.2f}, {o.fc_hi[-1]:.2f}]\n"
          f"- {role}; diagnoser hp {O2.diagnoser.hp}; data: {O2_SRC}")
    return fig, md


ABOUT = f"""
## SHM framework v{__version__}
Capstone Project-1, KL University (team Y23-HTE-PROJECT-62). Decision-support software prototype; not certified
for operational use.

| Result (leakage-free protocols) | Value |
|---|---|
| O1 known damage states, severity accuracy / macro-F1 | 94.4 % / 0.958 (Emergency 1250/1250) |
| O1 never-seen damage states, per recording | 72.2 % / 0.589 |
| O1 never-seen damage states, cumulative tracking | 83.8 % / 0.851 (target 85 %: narrowly missed) |
| O1 mechanism, never-seen states (per recording / tracked) | 92.4 % / 95.6 % balanced accuracy |
| O1 environment-compensated detection, unseen healthy month | specificity 71.8 %, sensitivity 64.3 % |
| O2 diagnosis MAE (nested LOSO, 5 seeds) | 1.047 ± 0.036 mm (cycle-only 1.287, random control 1.515) |
| O2 forecast RMSE from true history (T1–T6) | 1.15 mm |
| O2 90 % intervals, coverage on T7/T8 | ±3.08 mm / ±4.73 mm, 88.9 % (16/18) |
| O2 PHM 2019 challenge RMSE T7 / T8 | 3.61 / 2.01 mm |

**Known limits:** single-recording detection does not transfer across seasons; never-seen damage states near class
boundaries are under-estimated; one structure per dataset; tracking start assumes a confirmed first damage.
Models: `v3_level_fold{{A,B,C}}_seed{{42,123,2024}}.keras` (O1), gradient boosting refitted from the exported
T1–T6 features (O2). TensorFlow models loaded: {'yes (fold A)' if model_o1.available('A') else 'no'}.
"""

with gr.Blocks(title="SHM framework v1.0") as demo:
    gr.Markdown(f"# Structural health monitoring – damage assessment and crack prognosis (v{__version__})")
    with gr.Tab("O1 · Damage assessment"):
        gr.Markdown("**Single recording** – CNN–BiLSTM damage level → severity and mechanism")
        with gr.Row():
            rec = gr.Dropdown(rec_choices(), label="Sample recording", value=(rec_choices() or [None])[0])
            fold1 = gr.Dropdown(list(C.TAG_FOLDS), value="A", label="Model fold")
            b1 = gr.Button("Assess", variant="primary")
        p1, m1 = gr.Plot(), gr.Markdown()
        b1.click(o1_single, [rec, fold1], [p1, m1])
        gr.Markdown("**Monitoring stream** – cumulative tracking, change alarms, environment compensation")
        n_stream = len(STREAM["tags"]) - 1 if STREAM is not None else 1
        with gr.Row():
            fold2 = gr.Dropdown(list(C.TAG_FOLDS), value="A", label="Model fold")
            pos = gr.Slider(1, n_stream, value=n_stream, step=1, label="Recording position")
            b2 = gr.Button("Show", variant="primary")
        p2, m2 = gr.Plot(), gr.Markdown()
        b2.click(o1_stream, [fold2, pos], [p2, m2])
    with gr.Tab("O2 · Crack diagnosis and prognosis"):
        with gr.Row():
            spec = gr.Dropdown(C.TRAIN_SPECS + C.CHALLENGE_SPECS, value="T7", label="Specimen")
            nobs = gr.Slider(2, 40, value=40, step=1, label="Inspections observed")
            acrit = gr.Slider(1.0, 8.0, value=5.0, step=0.5, label="Critical crack length (mm)")
            b3 = gr.Button("Run", variant="primary")
        p3, m3 = gr.Plot(), gr.Markdown()
        b3.click(o2_run, [spec, nobs, acrit], [p3, m3])
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()