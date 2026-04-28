"""
Spherical Spline Interpolation (SSI) baseline.

This is the wall everything must beat.

MNE's interpolate_bads() implements spherical spline interpolation
(Perrin et al., 1989) — the standard in the EEG field.

Procedure per subject
---------------------
1.  Load the full 64-channel .fif file (ground truth, 256 Hz)
2.  Save the target channels (e.g. C3/C4/P3/P4) as ground truth
3.  Mark all channels EXCEPT the device's input channels as bad
4.  Run raw.interpolate_bads()  — uses only the 4 known channels
5.  Extract the reconstructed target channels
6.  Compute metrics vs ground truth

Usage
-----
    python -m pipeline_v2.baselines.ssi_baseline \\
        --fif_dir pipeline_v2/data/fif \\
        --device emotiv_epoc \\
        --results_dir pipeline_v2/results/ssi
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── Core function ────────────────────────────────────────────────────────────

def run_ssi_subject(
    fif_path: Path,
    input_channels: List[str],
    target_channels: List[str],
    fs: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SSI on one subject .fif file.

    Parameters
    ----------
    fif_path       : path to the full 64-channel .fif file (ground truth)
    input_channels : channels kept as known (device layout)
    target_channels: channels to evaluate (e.g. C3, C4, P3, P4)
    fs             : sampling rate (must match the .fif; default 256)

    Returns
    -------
    pred_data : np.ndarray  (n_targets, n_samples)  — SSI reconstruction
    gt_data   : np.ndarray  (n_targets, n_samples)  — ground truth
    """
    import mne
    mne.set_log_level("WARNING")

    # Load ground truth
    raw_gt = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

    if raw_gt.info["sfreq"] != fs:
        raise ValueError(
            f"Expected {fs} Hz but {fif_path.name} is at {raw_gt.info['sfreq']} Hz"
        )

    # Extract ground-truth target data before any modification
    gt_data = raw_gt.get_data(picks=target_channels)   # (n_targets, n_samples)

    # Build masked copy: mark everything except input channels as bad
    raw_masked = raw_gt.copy()
    input_set = set(input_channels)
    bad_channels = [ch for ch in raw_masked.ch_names if ch not in input_set]
    raw_masked.info["bads"] = bad_channels

    # Spherical spline interpolation
    raw_interp = raw_masked.copy().interpolate_bads(reset_bads=True, verbose=False)

    # Extract reconstructed targets
    pred_data = raw_interp.get_data(picks=target_channels)   # (n_targets, n_samples)

    return pred_data, gt_data


def run_ssi_dataset(
    fif_dir: Path,
    subject_ids: List[int],
    device_name: str,
    target_channels: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run SSI across a list of subjects for a given device layout.

    Returns
    -------
    results : dict keyed by subject ID string (e.g. 'S001') with keys:
        'pred' : np.ndarray (n_targets, n_samples)
        'gt'   : np.ndarray (n_targets, n_samples)
        'metrics': dict from compute_subject_metrics()
    """
    from pipeline_v2.data.device_configs import (
        DEVICE_CONFIGS, INITIAL_TARGETS, validate_channels_in_recording
    )
    from pipeline_v2.eval.metrics import compute_subject_metrics

    input_channels = DEVICE_CONFIGS[device_name]["input_channels"]
    if target_channels is None:
        target_channels = INITIAL_TARGETS

    results = {}

    for sid in subject_ids:
        fif_path = fif_dir / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            print(f"  [MISS] S{sid:03d} .fif not found — skipping")
            continue
        try:
            validate_channels_in_recording(device_name, _get_channel_names(fif_path))
            pred, gt = run_ssi_subject(fif_path, input_channels, target_channels)
            metrics = compute_subject_metrics(pred, gt, target_channels, fs=256)
            results[f"S{sid:03d}"] = {"pred": pred, "gt": gt, "metrics": metrics}
            if verbose:
                r_mean = metrics["pearson_mean"]
                mse    = metrics["mse_mean"]
                print(f"  S{sid:03d}  r={r_mean:.3f}  MSE={mse:.4f}")
        except Exception as e:
            print(f"  [FAIL] S{sid:03d}: {e}")

    return results


def summarise(results: Dict[str, Dict]) -> Dict:
    """
    Aggregate per-subject metrics into dataset-level statistics.

    Returns mean ± std over subjects for every scalar metric.
    """
    if not results:
        return {}

    # Collect per-subject scalars
    all_metrics: Dict[str, List[float]] = {}
    for subj_data in results.values():
        for k, v in subj_data["metrics"].items():
            if isinstance(v, (int, float)):
                all_metrics.setdefault(k, []).append(float(v))

    summary = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"]  = float(arr.std())

    return summary


# ─── Internal helper ─────────────────────────────────────────────────────────

def _get_channel_names(fif_path: Path) -> List[str]:
    import mne
    info = mne.io.read_info(str(fif_path), verbose=False)
    return info["ch_names"]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run SSI baseline on test subjects."
    )
    parser.add_argument("--fif_dir",     type=str,
                        default=str(ROOT / "pipeline_v2" / "data" / "fif"))
    parser.add_argument("--device",      type=str, default="emotiv_epoc",
                        choices=["emotiv_epoc", "muse_s", "openbci_cyton"])
    parser.add_argument("--results_dir", type=str,
                        default=str(ROOT / "pipeline_v2" / "results" / "ssi"))
    parser.add_argument("--split",       type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS
    split_map = {"train": TRAIN_SUBJECTS, "test": TEST_SUBJECTS,
                 "all": TRAIN_SUBJECTS + TEST_SUBJECTS}
    subjects = split_map[args.split]

    results = run_ssi_dataset(
        fif_dir=Path(args.fif_dir),
        subject_ids=subjects,
        device_name=args.device,
        verbose=args.verbose,
    )

    summary = summarise(results)

    # Save results
    results_dir = Path(args.results_dir) / args.device
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"SSI Baseline — {args.device}  ({args.split} set, n={len(results)})")
    print(f"{'─'*50}")
    for k, v in sorted(summary.items()):
        print(f"  {k:35s}: {v:.4f}")
    print(f"\nSaved → {summary_path}")
