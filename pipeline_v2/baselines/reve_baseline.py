"""
Ridge rEgression Venue Estimation (REVE) baseline.

Cross-subject supervised ridge regression: one Ridge model per target channel,
trained on 87 train subjects' continuous data, evaluated on 22 test subjects.

Unlike SSI (zero-shot), REVE requires per-device training data.

Procedure
---------
Train phase:
    1. Load each train subject's .fif file.
    2. Extract input channel data  (n_inputs,  n_samples) and
       target channel data         (n_targets, n_samples).
    3. Concatenate across subjects along the sample axis.
    4. Fit one Ridge(alpha) regressor per target channel:
           X = (n_total_samples, n_inputs)  →  y = (n_total_samples,)

Eval phase (per subject):
    1. Load test subject's .fif.
    2. Extract input channels → predict each target channel.
    3. Compute metrics vs ground-truth target channels.

Usage
-----
    python -m pipeline_v2.baselines.reve_baseline \\
        --fif_dir pipeline_v2/data/fif \\
        --device emotiv_epoc \\
        --results_dir pipeline_v2/results/reve
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Model -------------------------------------------------------------------

class REVEModel:
    """Ridge regression spatial filter, one model per target channel."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._models: List = []          # one sklearn Ridge per target channel
        self._target_channels: List[str] = []

    def fit(
        self,
        fif_paths: List[Path],
        input_channels: List[str],
        target_channels: List[str],
        verbose: bool = False,
        max_samples_per_subject: Optional[int] = 50000,
    ) -> None:
        """
        Fit one Ridge regressor per target channel across all training .fif files.

        Parameters
        ----------
        fif_paths               : list of paths to train-set .fif files
        input_channels          : device input channel names (features)
        target_channels         : channel names to reconstruct (labels)
        verbose                 : print progress
        max_samples_per_subject : if set, randomly subsample each subject's data
                                  to at most this many time points before
                                  concatenating.  Reduces peak RAM usage.
                                  Default 50,000 ≈ 3.3 min @ 256 Hz — enough
                                  for a stable ridge fit while keeping total
                                  memory under ~1 GB for 87 subjects × 4 ch.
                                  Set to None to use all samples.
        """
        from sklearn.linear_model import Ridge
        import mne
        mne.set_log_level("WARNING")

        self._target_channels = list(target_channels)

        # Accumulate data across subjects
        X_parts: List[np.ndarray] = []
        Y_parts: List[np.ndarray] = []

        rng = np.random.default_rng(42)   # fixed seed for reproducibility

        for fif_path in fif_paths:
            try:
                raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
                x = raw.get_data(picks=input_channels)   # (n_inputs, n_samples)
                y = raw.get_data(picks=target_channels)  # (n_targets, n_samples)
                n_samp = x.shape[1]

                # Optional subsampling to cap memory usage
                if max_samples_per_subject is not None and n_samp > max_samples_per_subject:
                    idx = rng.choice(n_samp, size=max_samples_per_subject, replace=False)
                    idx.sort()       # keep temporal order for potential future use
                    x = x[:, idx]
                    y = y[:, idx]

                X_parts.append(x.T)   # → (n_samples, n_inputs)
                Y_parts.append(y.T)   # → (n_samples, n_targets)
                if verbose:
                    print(f"    loaded {fif_path.name}  "
                          f"({x.shape[1]} samples{' [subsampled]' if max_samples_per_subject and n_samp > max_samples_per_subject else ''})")
            except Exception as e:
                print(f"    [WARN] skipping {fif_path.name}: {e}")

        if not X_parts:
            raise RuntimeError("No training data was loaded — cannot fit REVE model.")

        X = np.concatenate(X_parts, axis=0)  # (N_total, n_inputs)
        Y = np.concatenate(Y_parts, axis=0)  # (N_total, n_targets)

        if verbose:
            print(f"  Fitting Ridge(alpha={self.alpha}) on {X.shape[0]} samples "
                  f"× {X.shape[1]} inputs → {Y.shape[1]} targets …")

        self._models = []
        for ch_idx in range(Y.shape[1]):
            model = Ridge(alpha=self.alpha)
            model.fit(X, Y[:, ch_idx])
            self._models.append(model)

        if verbose:
            print(f"  Fitted {len(self._models)} ridge models.")

    def predict(
        self,
        fif_path: Path,
        input_channels: List[str],
        target_channels: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict target channels for one subject.

        Returns
        -------
        pred_data : np.ndarray  (n_targets, n_samples) — ridge reconstruction
        gt_data   : np.ndarray  (n_targets, n_samples) — ground truth
        """
        if not self._models:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        import mne
        mne.set_log_level("WARNING")

        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        x = raw.get_data(picks=input_channels)   # (n_inputs, n_samples)
        gt_data = raw.get_data(picks=target_channels)  # (n_targets, n_samples)

        X = x.T  # (n_samples, n_inputs)
        pred_rows = [model.predict(X) for model in self._models]  # each (n_samples,)
        pred_data = np.stack(pred_rows, axis=0)  # (n_targets, n_samples)

        return pred_data, gt_data


# --- Training helper ---------------------------------------------------------

def train_reve(
    fif_dir: Path,
    train_subjects: List[int],
    device_name: str,
    target_channels: Optional[List[str]] = None,
    alpha: float = 1.0,
    max_samples_per_subject: Optional[int] = 50000,
    verbose: bool = True,
) -> REVEModel:
    """
    Load all train .fif files, fit a REVEModel, and return it.

    Parameters
    ----------
    fif_dir        : directory containing S###_raw.fif files
    train_subjects : list of integer subject IDs to train on
    device_name    : key into DEVICE_CONFIGS
    target_channels: channels to reconstruct; defaults to INITIAL_TARGETS
    alpha          : Ridge regularisation strength
    verbose        : print progress

    Returns
    -------
    Fitted REVEModel instance.
    """
    from pipeline_v2.data.device_configs import DEVICE_CONFIGS, INITIAL_TARGETS

    input_channels = DEVICE_CONFIGS[device_name]["input_channels"]
    if target_channels is None:
        target_channels = INITIAL_TARGETS

    fif_paths = []
    for sid in train_subjects:
        p = fif_dir / f"S{sid:03d}_raw.fif"
        if p.exists():
            fif_paths.append(p)
        else:
            if verbose:
                print(f"  [MISS] S{sid:03d} .fif not found — skipping")

    if verbose:
        print(f"Training REVE on {len(fif_paths)} subjects "
              f"(device={device_name}, alpha={alpha}) …")

    model = REVEModel(alpha=alpha)
    model.fit(fif_paths, input_channels, target_channels,
              verbose=verbose,
              max_samples_per_subject=max_samples_per_subject)
    return model


# --- Dataset runner ----------------------------------------------------------

def run_reve_dataset(
    fif_dir: Path,
    subject_ids: List[int],
    device_name: str,
    model: REVEModel,
    target_channels: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run REVE evaluation across a list of subjects.

    Returns
    -------
    results : dict keyed by subject ID string (e.g. 'S001') with keys:
        'pred'    : np.ndarray (n_targets, n_samples)
        'gt'      : np.ndarray (n_targets, n_samples)
        'metrics' : dict from compute_subject_metrics()
    """
    from pipeline_v2.data.device_configs import (
        DEVICE_CONFIGS, INITIAL_TARGETS, validate_channels_in_recording
    )
    from pipeline_v2.eval.metrics import compute_subject_metrics

    input_channels = DEVICE_CONFIGS[device_name]["input_channels"]
    if target_channels is None:
        target_channels = INITIAL_TARGETS

    results: Dict[str, Dict] = {}

    for sid in subject_ids:
        fif_path = fif_dir / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            print(f"  [MISS] S{sid:03d} .fif not found — skipping")
            continue
        try:
            validate_channels_in_recording(device_name, _get_channel_names(fif_path))
            pred, gt = model.predict(fif_path, input_channels, target_channels)
            metrics = compute_subject_metrics(pred, gt, target_channels, fs=256)
            results[f"S{sid:03d}"] = {"pred": pred, "gt": gt, "metrics": metrics}
            if verbose:
                r_mean = metrics["pearson_mean"]
                mse    = metrics["mse_mean"]
                print(f"  S{sid:03d}  r={r_mean:.3f}  MSE={mse:.4f}")
        except Exception as e:
            print(f"  [FAIL] S{sid:03d}: {e}")

    return results


# --- Summarise ---------------------------------------------------------------

def summarise(results: Dict[str, Dict]) -> Dict:
    """
    Aggregate per-subject metrics into dataset-level statistics.

    Returns mean ± std over subjects for every scalar metric.
    """
    if not results:
        return {}

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


# --- Internal helper ---------------------------------------------------------

def _get_channel_names(fif_path: Path) -> List[str]:
    import mne
    info = mne.io.read_info(str(fif_path), verbose=False)
    return info["ch_names"]


# --- CLI ---------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run REVE (ridge regression) baseline on test subjects."
    )
    parser.add_argument("--fif_dir",      type=str,
                        default=str(ROOT / "pipeline_v2" / "data" / "fif"))
    parser.add_argument("--device",       type=str, default="emotiv_epoc",
                        choices=["emotiv_epoc", "muse_s", "openbci_cyton"])
    parser.add_argument("--results_dir",  type=str,
                        default=str(ROOT / "pipeline_v2" / "results" / "reve"))
    parser.add_argument("--split",        type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--train_split",  type=str, default="train",
                        choices=["train", "test", "all"],
                        help="Which split to use for fitting the ridge models.")
    parser.add_argument("--alpha",        type=float, default=1.0,
                        help="Ridge regularisation strength (default: 1.0).")
    parser.add_argument("--max_samples",  type=int,   default=50000,
                        help="Max samples per subject during training (0=unlimited).")
    parser.add_argument("--verbose",      action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS
    split_map = {
        "train": TRAIN_SUBJECTS,
        "test":  TEST_SUBJECTS,
        "all":   TRAIN_SUBJECTS + TEST_SUBJECTS,
    }
    train_subjects = split_map[args.train_split]
    eval_subjects  = split_map[args.split]

    fif_dir = Path(args.fif_dir)

    # -- Train --------------------------------------------------------------
    max_samp = None if args.max_samples == 0 else args.max_samples
    model = train_reve(
        fif_dir=fif_dir,
        train_subjects=train_subjects,
        device_name=args.device,
        alpha=args.alpha,
        max_samples_per_subject=max_samp,
        verbose=args.verbose,
    )

    # -- Evaluate -----------------------------------------------------------
    if args.verbose:
        print(f"\nEvaluating on {args.split} split ({len(eval_subjects)} subjects) …")

    results = run_reve_dataset(
        fif_dir=fif_dir,
        subject_ids=eval_subjects,
        device_name=args.device,
        model=model,
        verbose=args.verbose,
    )

    summary = summarise(results)

    # -- Save ---------------------------------------------------------------
    results_dir = Path(args.results_dir) / args.device
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'-'*50}")
    print(f"REVE Baseline — {args.device}  ({args.split} set, n={len(results)})")
    print(f"{'-'*50}")
    for k, v in sorted(summary.items()):
        print(f"  {k:35s}: {v:.4f}")
    print(f"\nSaved → {summary_path}")
