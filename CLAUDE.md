# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this project is

PhD dissertation: **"Computational EEG Densification: Zero-Shot Channel Reconstruction
from Consumer-Grade Wearables Using Foundation Models"**

Three methods compared:
- **SSI** — Spherical Spline Interpolation (Perrin 1989 classical baseline)
- **REVE** — Cross-subject ridge regression (ML baseline)
- **ZUNA** — 380M-param masked diffusion transformer (main contribution)

Three devices:
- **Device A — Emotiv EPOC** : inputs AF3, AF4, F3, F4  → targets C3, C4, P3, P4
- **Device B — Muse S**      : inputs AF7, AF8, T9, T10  → targets C3, C4, P3, P4
- **Device C — OpenBCI**     : inputs C3, C4, P3, P4, Fz, Cz → targets T7, T8, FC5, FC6

---

## Commands

All scripts run from the repo root (the folder containing `pipeline_v2/`).

```bash
# Install dependencies (once)
pip install -r pipeline_v2/requirements.txt

# Run SSI baseline for one device
python -m pipeline_v2.baselines.ssi_baseline --device emotiv_epoc
python -m pipeline_v2.baselines.ssi_baseline --device muse_s
python -m pipeline_v2.baselines.ssi_baseline --device openbci_cyton

# Run REVE baseline for one device
python -m pipeline_v2.baselines.reve_baseline --device emotiv_epoc

# Run ZUNA on Devices B + C (Priority 1 — overnight, ~8–12 h each)
python -m pipeline_v2.zuna.run_zuna_crossdevice --device muse_s
python -m pipeline_v2.zuna.run_zuna_crossdevice --device openbci_cyton
# Or both sequentially (~20 h total):
python -m pipeline_v2.zuna.run_zuna_crossdevice

# ZUNA scaling law (one point per night; start with n=4)
python -m pipeline_v2.experiments.run_scaling_zuna --n 4
# Then n=8, n=2, n=1, n=16, n=32 on subsequent nights.

# SSI + REVE scaling law (CPU-only, owner's PC)
python -m pipeline_v2.experiments.run_scaling_law_full

# Generate Figure 1 (after all scaling data is available)
python -m pipeline_v2.figures.scaling_law_figure
```

---

## Architecture

### Data flow (per subject, per method)

```
pipeline_v2/data/fif/S0XX_raw.fif   (64-ch, 256 Hz, PhysioNet EEGMMIDB)
         │
         ▼
  Device masking                     device_configs.py
  Keep 4–6 input channels,           Zero out the other 58–60 as "bad"
         │
    ┌────┴──────────────┐
    │ SSI               │ REVE                  ZUNA
    │ MNE interpolate   │ Pre-fitted Ridge      3-step diffusion:
    │ _bads()           │ models (1 per target  1) preprocessing() → .pt
    │ (zero-shot)       │ channel, trained on   2) inference() → .pt
    │                   │ 87 train subjects)    3) pt_to_fif() → .fif
    └────┬──────────────┘
         │
         ▼
  Extract target channels, align lengths (trim to min)
         │
         ▼
  eval/metrics.py  ─  compute_subject_metrics()
    ├── MSE + Pearson r (per target channel)
    ├── PSD band-power ratios (delta/theta/alpha/beta/gamma)
    ├── PLV phase coherence (F3↔C3, F4↔C4)
    └── BES (LDA accuracy on alpha+beta power, 5-fold CV)
             BES ≥ 0.85 = clinically equivalent; ≥ 0.90 = deployment-ready
         │
         ▼
  results/{method}/{device}/summary.json   (mean ± std across 22 test subjects)
```

### Key modules

| Module | Role |
|--------|------|
| `data/device_configs.py` | `DEVICE_CONFIGS` dict: input/target channels per device |
| `data/subject_split.py` | Frozen 87/22 train/test split (`TRAIN_SUBJECTS`, `TEST_SUBJECTS`) |
| `eval/metrics.py` | `compute_subject_metrics(pred, gt, ch_names, sfreq)` — all 4 metrics |
| `eval/bes_runner.py` | Bridges continuous arrays → epoch-level BES using T1/T2 event markers |
| `baselines/ssi_baseline.py` | `run_ssi_dataset(fif_dir, subjects, device)` — zero-shot |
| `baselines/reve_baseline.py` | `REVEModel.fit()` + `run_reve_dataset()` — supervised |
| `zuna/zuna_pipeline.py` | `run_zuna_subject()`, `run_zuna_dataset()` — foundation model |
| `zuna/run_zuna_crossdevice.py` | Entry point for Devices B + C with OOM-safe defaults |

### BES special case
BES requires epoch labels from `T1`/`T2` motor-imagery annotations baked into the .fif.
`bes_runner.py` segments *predicted continuous arrays* using the GT event timing, then
calls `compute_bes()`. ZUNA BES for Device A requires the per-subject `.fif` output from
`pipeline_v2/data/zuna_work/emotiv_epoc/S0XX_raw/4_fif_output/` (not just summary.json).

---

## Critical settings — DO NOT CHANGE

```
--tokens 1000          tokens_per_batch (8 GB VRAM safe)
--steps  10            diffusion_sample_steps
--crop_s 300           crop to 5 minutes (prevents GPU memory leak)
```

- **tokens=1000** : 3000 → SIGKILL after 112 min; 10000 → immediate SIGKILL
- **crop_s=300**  : Full recordings (322 epochs / 6 PT files) cause memory leak across batches → SIGKILL. 5 min = 1 PT file = 26 batches = safe.
- **steps=10**    : Paper-validated, 5x faster than default 50. Comparable quality.
- ZUNA internal constraints (hardcoded in model): 256 Hz, 5 s epochs (1280 samples), batch size 64, data normalised by dividing by 10.0.

### Warnings you can safely ignore
- "Channel positions out of bounds: 9 elements above max" — ZUNA montage mismatch, harmless
- "flex_attention called without torch.compile()" — harmless, slightly slower

### OOM diagnosis
Exit code 1 with no error message = SIGKILL = OOM. Retry with `--tokens 500`.

---

## Current status (as of 2026-04-22)

### DONE
- All 109 subjects downloaded → `pipeline_v2/data/fif/S001_raw.fif` … `S109_raw.fif`
- SSI on Device A (emotiv_epoc) → `results/ssi/emotiv_epoc/summary.json`  r=0.388, BES=0.929
- REVE on Device A              → `results/reve/emotiv_epoc/summary.json`  r=0.673, BES=0.942
- SSI zero-shot Device B (muse_s)      → `results/ssi/muse_s/summary.json`        r=0.542, BES=0.992
- REVE zero-shot Device B              → `results/reve/muse_s/summary.json`        r=0.549, BES=0.945
- SSI zero-shot Device C (openbci)     → `results/ssi/openbci_cyton/summary.json`  r=0.704, BES=1.083
- REVE zero-shot Device C              → `results/reve/openbci_cyton/summary.json` r=0.725, BES=1.040
- ZUNA Device A (emotiv_epoc)          → `results/zuna/emotiv_epoc/summary.json`   r=0.619, MSE=1886, beta=0.398
- ZUNA Device B (muse_s)               → `results/zuna/muse_s/summary.json`         r=0.446, BES=0.939
- ZUNA Device C (openbci_cyton)        → `results/zuna/openbci_cyton/summary.json`  r=0.659, BES=1.186
- ZUNA scaling law — all 6 points (1ch→32ch) → `results/scaling_law/zuna/`
- SSI + REVE scaling law               → `results/scaling_law/scaling_law_full.json`

### TODO

#### Remaining gap — ZUNA Device A BES
The `zuna_work/emotiv_epoc/` directory was cleaned up; per-subject `.fif` outputs are gone.
Device A BES is unrecoverable without re-running ZUNA on Device A.

#### PLV is NaN for all ZUNA runs
Expected for Devices B and C (F3/F4 not in those device configs). For Device A it is a bug —
F3 and C3 are both available but PLV comes out NaN. Investigate `metrics.py` PLV path.

---

## Confirmed results (paper Table 1 / Table 2)

| Method | Device     | r              | MSE (pV²)         | Beta ratio      | BES             |
|--------|-----------|----------------|-------------------|-----------------|-----------------|
| SSI    | A (train) | 0.388 ± 0.164  | 10,631 ± 8,684    | 10.82 ± 13.12   | 0.929 ± 0.109   |
| REVE   | A (train) | 0.673 ± 0.105  | 2,288 ± 2,403     | 0.0004 ± 0.0001 | 0.942 ± 0.116   |
| ZUNA   | A (train) | 0.619 ± 0.085  | 1,886 ± 2,581     | 0.398 ± 0.131   | [TODO — BES]    |
| SSI    | B (0-shot)| 0.542 ± 0.143  | 2,510 ± 2,630     | 0.685 ± 0.362   | 0.992 ± 0.126   |
| REVE   | B (0-shot)| 0.549 ± 0.121  | 2,330 ± 2,200     | 0.0004 ± 0.0001 | 0.945 ± 0.102   |
| ZUNA   | B (0-shot)| 0.446 ± 0.256  | 1,664 ± 2,417     | —               | 0.939 ± 0.315   |
| SSI    | C (0-shot)| 0.704 ± 0.206  | 3,440 ± 9,800     | 5.944 ± 22.08   | 1.083 ± 0.140   |
| REVE   | C (0-shot)| 0.725 ± 0.171  | 2,050 ± 1,100     | 0.0004 ± 0.0001 | 1.040 ± 0.116   |
| ZUNA   | C (0-shot)| 0.659 ± 0.151  | 930 ± 496         | —               | 1.186 ± 0.460   |

---

## After finishing all ZUNA runs

1. Zip `pipeline_v2/results/` and send to owner (roisabir@gmail.com).
2. Owner updates: Table 1 (ZUNA BES), Table 2 (ZUNA zero-shot rows), Abstract, Figure 1.
