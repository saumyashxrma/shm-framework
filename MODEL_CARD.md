# Model card – SHM framework v1.0.0

## O1 – damage-level CNN–BiLSTM (`v3_level_fold{A,B,C}_seed{42,123,2024}.keras`)

| Item | Description |
|---|---|
| Task | Regress a continuous damage level (tag scale 0–13) from one 8-path guided-wave recording plus temperature and humidity; thresholds 4.5 / 9.5 give severity, 1.5 / 7.5 give mechanism |
| Input | (1000, 8) band-passed (50–240 kHz), decimated, peak-normalised waveforms + (temperature °C, humidity %) |
| Architecture | Gaussian noise, spatial dropout, 2 × (Conv1D, BatchNorm, MaxPool), MaxPool, BiLSTM(64), dropout; environment branch Dense(16); Dense(64); linear output |
| Training | Huber loss (δ = 1), Adam 1e-3, class-balanced weights, time-stretch ±2 % and shift ±5 samples augmentation, early stopping on a chronological validation split, 3 seeds per fold |
| Training data | Utah outdoor plate, all months except October 2022; per fold three damage states are hidden (A: 2, 7, 11; B: 4, 5, 10; C: 3, 8, 12) |
| Evaluation | known states (later 20 % of recordings): 94.4 % severity accuracy; never-seen states: 72.2 % per recording, 83.8 % with cumulative tracking; mechanism 92.4 % / 95.6 %; October 2022 tag 13 Emergency recall A 99.96 %, B 94.7 %, C 32.5 % |
| Known failure modes | season/temperature shift (healthy January 2019 recordings predicted near level 6); under-estimation of never-seen states near class boundaries; extrapolation beyond the trained damage range depends on the fold |
| Intended use | decision support for inspection planning on comparable plate structures, together with an engineer; not for certification or unattended shutdown decisions |

## O2 – crack-length diagnoser and forecaster

| Item | Description |
|---|---|
| Diagnoser | Gradient-boosted trees (learning rate 0.05, subsample 0.8, min leaf 3) on [load cycles, 9 physics features]; depth and tree count chosen by leave-one-specimen-out on T1–T6; causal running maximum |
| Forecaster | linear trend + Gaussian process (Matérn ν = 1.5 + white noise) on the diagnosed history |
| Uncertainty | split-conformal 90 %: ±3.08 mm (diagnosis), ±4.73 mm (forecast, calibrated on forecasts from diagnosed T1–T6 histories) |
| Evaluation | nested LOSO MAE 1.047 ± 0.036 mm (5 seeds); forecast RMSE 1.15 mm from true history; T7/T8 challenge RMSE 3.61 / 2.01 mm; interval coverage 16/18 on T7/T8 |
| Limitations | six training specimens; one joint geometry and load spectrum; challenge accuracy limited by diagnosis error, not by the forecaster (oracle 0.67 / 1.30 mm) |

## Ethical and safety notes

Public datasets only; no personal data. A missed damage state or an over-confident interval can delay maintenance,
so outputs are advisory, intervals are reported with every estimate, and the inspection rule uses the upper bound.
