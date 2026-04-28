"""
ZUNA inference wrapper for the channel densification pipeline.

ZUNA pinned commit: 7b6b858fd36808353bce1b2184ca93695cf68075
Install:  pip install zuna
Paper:    arxiv 2602.18478

How the masking simulation works
---------------------------------
ZUNA's preprocessing() accepts a bad_channels argument.  Those channels are
zeroed out in the .pt tensors before inference.  The model then reconstructs
them from the spatial context provided by the known (non-bad) channels.

For our experiment:
    - Known channels  : device input channels (e.g. AF3, AF4, F3, F4)
    - Zeroed channels : all other 60 channels  ← ZUNA reconstructs these
    - Ground truth    : original full 64-channel .fif  ← kept separately

ZUNA fixed constraints (cannot be changed):
    - Sampling rate : 256 Hz
    - Epoch length  : 5 s  (1 280 samples)
    - Batch size    : 64 epochs per .pt file
    - Normalization : data / data_norm  (expects std ≈ 0.1, data_norm=10.0)

Usage
-----
    python -m pipeline_v2.zuna.zuna_pipeline \\
        --fif_dir pipeline_v2/data/fif \\
        --device emotiv_epoc \\
        --results_dir pipeline_v2/results/zuna

Directory layout created per subject per device
-----------------------------------------------
    zuna_work/<device>/<S###>/
        1_fif_input/          .fif copied here for ZUNA
        1_fif_filtered/       ZUNA internal filtered copy
        2_pt_input/           .pt tensors (masked)
        3_pt_output/          .pt tensors (reconstructed)
        4_fif_output/         reconstructed .fif
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default I/O directories
FIF_DIR   = ROOT / "pipeline_v2" / "data" / "fif"
ZUNA_DIR  = ROOT / "pipeline_v2" / "data" / "zuna_work"
RESULTS_DIR = ROOT / "pipeline_v2" / "results" / "zuna"


# ─── ZUNA inference for one subject ──────────────────────────────────────────

def run_zuna_subject(
    fif_path: Path,
    device_name: str,
    work_dir: Path = ZUNA_DIR,
    # ZUNA inference hyper-parameters
    gpu_device: int = 0,
    tokens_per_batch: int = 100_000,
    data_norm: float = 10.0,
    diffusion_cfg: float = 1.0,
    diffusion_sample_steps: int = 50,
    overwrite: bool = False,
) -> Path:
    """
    Run the full ZUNA pipeline on one subject .fif file.

    Parameters
    ----------
    fif_path          : full 64-channel subject .fif @ 256 Hz
    device_name       : key in DEVICE_CONFIGS (determines which channels are known)
    work_dir          : root for intermediate ZUNA files
    gpu_device        : GPU index; pass -1 or '' for CPU
    overwrite         : re-run even if output .fif already exists

    Returns
    -------
    Path to the reconstructed output .fif file.
    """
    from zuna import preprocessing, inference, pt_to_fif
    from pipeline_v2.data.device_configs import (
        DEVICE_CONFIGS, validate_channels_in_recording
    )

    device_cfg      = DEVICE_CONFIGS[device_name]
    input_channels  = device_cfg["input_channels"]

    subject_stem = fif_path.stem                           # e.g. 'S001_raw'
    subject_work = work_dir / device_name / subject_stem

    fif_input_dir    = subject_work / "1_fif_input"
    fif_filtered_dir = subject_work / "1_fif_filtered"
    pt_input_dir     = subject_work / "2_pt_input"
    pt_output_dir    = subject_work / "3_pt_output"
    fif_output_dir   = subject_work / "4_fif_output"

    # Fast exit if already done
    existing = list(fif_output_dir.glob("*.fif")) if fif_output_dir.exists() else []
    if existing and not overwrite:
        print(f"  [SKIP] {subject_stem} — output already exists")
        return existing[0]

    # Create directories
    for d in [fif_input_dir, fif_filtered_dir, pt_input_dir,
              pt_output_dir, fif_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Validate device channels exist in recording
    import mne
    mne.set_log_level("WARNING")
    info = mne.io.read_info(str(fif_path), verbose=False)
    validate_channels_in_recording(device_name, info["ch_names"])

    # Determine bad channels (all except device inputs)
    bad_chs = [ch for ch in info["ch_names"] if ch not in set(input_channels)]

    # Copy .fif to ZUNA input directory
    dest = fif_input_dir / fif_path.name
    shutil.copy2(str(fif_path), str(dest))

    # ── Step 1: preprocessing ─────────────────────────────────────────────────
    # ZUNA handles: re-epoching into 5 s windows, normalization (data/data_norm)
    # We disable filtering — the .fif is already bandpass filtered.
    preprocessing(
        input_dir=str(fif_input_dir),
        output_dir=str(pt_input_dir),
        apply_notch_filter=False,
        apply_highpass_filter=False,   # already done
        apply_average_reference=False, # keep original reference
        bad_channels=bad_chs,
        preprocessed_fif_dir=str(fif_filtered_dir),
    )

    # ── Step 2: inference ─────────────────────────────────────────────────────
    gpu_arg = gpu_device if gpu_device >= 0 else ""
    inference(
        input_dir=str(pt_input_dir),
        output_dir=str(pt_output_dir),
        gpu_device=gpu_arg,
        tokens_per_batch=tokens_per_batch,
        data_norm=data_norm,
        diffusion_cfg=diffusion_cfg,
        diffusion_sample_steps=diffusion_sample_steps,
        plot_eeg_signal_samples=False,
    )

    # ── Step 3: convert .pt → .fif ────────────────────────────────────────────
    pt_to_fif(
        input_dir=str(pt_output_dir),
        output_dir=str(fif_output_dir),
    )

    # Locate output
    output_fifs = sorted(fif_output_dir.glob("*.fif"))
    if not output_fifs:
        raise RuntimeError(
            f"ZUNA produced no .fif in {fif_output_dir}.  "
            "Check ZUNA logs for errors."
        )

    out_path = output_fifs[0]
    print(f"  [OK] {subject_stem} → {out_path.name}")
    return out_path


# ─── Extract arrays from ZUNA output ─────────────────────────────────────────

def extract_reconstructed_channels(
    output_fif_path: Path,
    gt_fif_path: Path,
    target_channels: List[str],
    fs: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ZUNA's reconstructed .fif and the original ground-truth .fif,
    return aligned arrays for the target channels.

    ZUNA output is epoch-based (.fif epochs).  We concatenate and trim to
    the length of the ground truth for a fair comparison.

    Parameters
    ----------
    output_fif_path : ZUNA reconstructed .fif
    gt_fif_path     : original full-channel .fif (ground truth)
    target_channels : e.g. ['C3', 'C4', 'P3', 'P4']
    fs              : expected sampling rate

    Returns
    -------
    pred_data : (n_targets, n_samples)
    gt_data   : (n_targets, n_samples)   — same n_samples
    """
    import mne
    mne.set_log_level("WARNING")

    # Load ground truth (continuous)
    raw_gt = mne.io.read_raw_fif(str(gt_fif_path), preload=True, verbose=False)
    gt_data = raw_gt.get_data(picks=target_channels)          # (T, n_samples)

    # Load ZUNA output — could be Raw or Epochs
    try:
        recon = mne.io.read_raw_fif(str(output_fif_path), preload=True, verbose=False)
        pred_data = recon.get_data(picks=target_channels)
    except Exception:
        # Fallback: try loading as Epochs
        epochs = mne.read_epochs(str(output_fif_path), preload=True, verbose=False)
        # Concatenate epochs along time axis
        # epochs.get_data() → (n_epochs, n_channels, n_times)
        ep_data = epochs.get_data(picks=target_channels)     # (E, T, n_ep_samples)
        pred_data = ep_data.transpose(1, 0, 2).reshape(len(target_channels), -1)

    # Trim to minimum length (guard against rounding differences)
    n = min(pred_data.shape[-1], gt_data.shape[-1])
    return pred_data[:, :n], gt_data[:, :n]


# ─── Dataset-level runner ─────────────────────────────────────────────────────

def run_zuna_dataset(
    fif_dir: Path,
    subject_ids: List[int],
    device_name: str,
    target_channels: Optional[List[str]] = None,
    work_dir: Path = ZUNA_DIR,
    gpu_device: int = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run ZUNA + evaluate across a list of subjects.

    Returns dict keyed by subject ID string with 'metrics' inside.
    """
    from pipeline_v2.data.device_configs import INITIAL_TARGETS
    from pipeline_v2.eval.metrics import compute_subject_metrics

    if target_channels is None:
        target_channels = INITIAL_TARGETS

    results = {}

    for sid in subject_ids:
        fif_path = fif_dir / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            print(f"  [MISS] S{sid:03d} — .fif not found, skipping")
            continue
        try:
            out_fif = run_zuna_subject(
                fif_path, device_name, work_dir=work_dir,
                gpu_device=gpu_device, overwrite=overwrite,
            )
            pred, gt = extract_reconstructed_channels(
                out_fif, fif_path, target_channels
            )
            metrics = compute_subject_metrics(pred, gt, target_channels, fs=256)
            results[f"S{sid:03d}"] = {"metrics": metrics}
            if verbose:
                r  = metrics["pearson_mean"]
                mse = metrics["mse_mean"]
                print(f"    → r={r:.3f}  MSE={mse:.4f}")
        except Exception as e:
            print(f"  [FAIL] S{sid:03d}: {e}")

    return results


def summarise(results: Dict[str, Dict]) -> Dict:
    """Aggregate per-subject metrics → dataset-level mean ± std."""
    if not results:
        return {}
    all_metrics: Dict[str, List[float]] = {}
    for data in results.values():
        for k, v in data["metrics"].items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, []).append(float(v))
    summary = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"]  = float(arr.std())
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Run ZUNA inference pipeline.")
    parser.add_argument("--fif_dir",     type=str, default=str(FIF_DIR))
    parser.add_argument("--device",      type=str, default="emotiv_epoc",
                        choices=["emotiv_epoc", "muse_s", "openbci_cyton"])
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--work_dir",    type=str, default=str(ZUNA_DIR))
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS
    split_map = {"train": TRAIN_SUBJECTS, "test": TEST_SUBJECTS,
                 "all": TRAIN_SUBJECTS + TEST_SUBJECTS}
    subjects = split_map[args.split]

    results = run_zuna_dataset(
        fif_dir=Path(args.fif_dir),
        subject_ids=subjects,
        device_name=args.device,
        work_dir=Path(args.work_dir),
        gpu_device=args.gpu,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    summary = summarise(results)

    out_dir = Path(args.results_dir) / args.device
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"ZUNA — {args.device}  ({args.split} set, n={len(results)})")
    print(f"{'─'*50}")
    for k, v in sorted(summary.items()):
        print(f"  {k:35s}: {v:.4f}")
    print(f"\nSaved → {summary_path}")
