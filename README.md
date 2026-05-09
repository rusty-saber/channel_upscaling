# Channel Upscaling — EEG Densification Pipeline

Code for the paper:

> **Zero-Shot Reconstruction of Task-Relevant EEG Channels from Simulated Wearable Montages Using a Pretrained Diffusion Model**
> Saber et al., 2025 — *[journal/venue TBD]*

---

## What this repo contains

| Directory | Contents |
|---|---|
| `pipeline_v2/baselines/` | SSI (spherical spline interpolation) and REVE (ridge regression) baselines |
| `pipeline_v2/zuna/` | ZUNA inference wrapper (zero-shot diffusion model reconstruction) |
| `pipeline_v2/eval/` | Metrics (Pearson r, MSE, band-power ratios, BES), BES runner, Phase 2 stats |
| `pipeline_v2/data/` | Download script for PhysioNet EEGMMIDB; device layout configs; subject split |
| `pipeline_v2/configs/` | `config.yaml` — all hyperparameters; seed is frozen for reproducibility |
| `pipeline_v2/results/` | Pre-computed JSON summaries for SSI, REVE, ZUNA; Phase 2 stats report |
| `pipeline_v2/gpu_rental/` | One-shot GPU rental kit for matched-epoch ZUNA rerun (see below) |

EEG data (`.fif` files) and ZUNA working files are not committed — they are large and re-downloadable. See **Data** below.

---

## Setup

```bash
pip install mne==1.6.1 scipy==1.11.4 scikit-learn==1.3.2 numpy==1.26.4 \
            tqdm==4.66.1 pyyaml==6.0.1 matplotlib==3.8.2
pip install git+https://github.com/Zyphra/zuna.git@7b6b858fd36808353bce1b2184ca93695cf68075
```

ZUNA inference additionally requires PyTorch with CUDA:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

---

## Data

The pipeline uses the **PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)**, publicly available at [https://physionet.org/content/eegmmidb/1.0.0/](https://physionet.org/content/eegmmidb/1.0.0/) under the Open Data Commons Attribution License v1.0.

Download and preprocess (109 subjects → `.fif` at 256 Hz):

```bash
python -m pipeline_v2.data.download_eegmmidb
```

This takes ~30–60 minutes and uses ~11 GB of disk space.

---

## Reproducing the results

### SSI and REVE baselines (CPU, ~30 min)

```bash
python -m pipeline_v2.run_zeroshot
```

### ZUNA zero-shot inference (GPU required, ~8–10 hr per device)

```bash
python -m pipeline_v2.zuna.zuna_pipeline \
    --fif_dir pipeline_v2/data/fif \
    --device emotiv_epoc \
    --results_dir pipeline_v2/results/zuna \
    --split test --gpu 0
```

Repeat for `muse_s` and `openbci_cyton`.

For the **matched-epoch rerun** (motor-imagery epochs only), see `pipeline_v2/gpu_rental/README.md`.

### Phase 2 statistics (CPU, seconds — uses pre-computed per-subject data)

```bash
python -m pipeline_v2.run_phase2_stats_fast
```

### Scaling-law refit (CPU, seconds)

```bash
python -m pipeline_v2.run_scaling_refit
```

---

## ZUNA model

ZUNA is a third-party EEG diffusion model developed by Zyphra. Model weights and code are publicly available at:
- Code: [https://github.com/Zyphra/zuna](https://github.com/Zyphra/zuna)
- Weights: [https://huggingface.co/Zyphra/ZUNA](https://huggingface.co/Zyphra/ZUNA)

This repo pins ZUNA to commit `7b6b858` for reproducibility. The authors of this paper have no affiliation with Zyphra.

---

## Key configuration

All hyperparameters are in `pipeline_v2/configs/config.yaml`. The subject split seed (`splits.seed: 42`) is frozen — do not change it, as it determines the 87/22 train/test partition used throughout the paper.

---

## Known limitations

The current results include a **task-epoch mismatch**: ZUNA inference runs on motor-execution epochs (5-minute continuous crop) while SSI and REVE are evaluated on motor-imagery epochs. A matched-epoch rerun is planned; see `pipeline_v2/gpu_rental/README.md`. Tables and figures will be updated after the rerun.

---

## Citation

```bibtex
@article{saber2025eeg,
  title={Zero-Shot Reconstruction of Task-Relevant EEG Channels from Simulated Wearable Montages Using a Pretrained Diffusion Model},
  author={Saber and others},
  journal={TBD},
  year={2025}
}
```

---

## License

MIT — see `LICENSE`. The EEGMMIDB dataset is governed by the Open Data Commons Attribution License v1.0; the ZUNA model is governed by Zyphra's license.
