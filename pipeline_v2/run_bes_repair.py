"""
BES Grouped-CV Repair — CPU sprint task.

Problems fixed vs the original run_zeroshot.py BES computation:

1. Per-subject BES was never saved for SSI or REVE — only the mean/std
   aggregate in summary.json. Phase 2 statistics (Wilcoxon, bootstrap CIs)
   need all 22 individual values.

2. compute_bes() used StratifiedKFold(shuffle=True), which randomly mixes
   epochs from different PhysioNet runs across train/test folds, leaking
   temporal autocorrelation. The fix groups epochs by their source MI run
   (R05, R06, R09, R10, R13, R14 = 6 groups) and uses leave-one-run-out CV.

Output: for each method × device, writes two new files ALONGSIDE the
existing summary.json (does not overwrite v1.8 results):
    results/<method>/<device>/bes_per_subject_grouped.json
    results/<method>/<device>/bes_summary_grouped.json

Usage (from repo root, ~15–20 min on CPU):
    python -m pipeline_v2.run_bes_repair
"""

import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline_v2.baselines.reve_baseline import REVEModel
from pipeline_v2.baselines.ssi_baseline  import run_ssi_subject
from pipeline_v2.data.device_configs     import DEVICE_CONFIGS
from pipeline_v2.data.subject_split      import TEST_SUBJECTS, TRAIN_SUBJECTS
from pipeline_v2.eval.metrics            import band_power
from pipeline_v2.eval.bes_runner         import _load_events_from_fif, extract_pred_epochs

FIF_DIR      = ROOT / "pipeline_v2" / "data" / "fif"
RESULTS_ROOT = ROOT / "pipeline_v2" / "results"
DEVICES      = ["emotiv_epoc", "muse_s", "openbci_cyton"]
MI_RUNS      = [5, 6, 9, 10, 13, 14]   # config.yaml bes_mi_runs


# ---------------------------------------------------------------------------
# Run-boundary helpers
# ---------------------------------------------------------------------------

def _get_run_boundaries_sec(fif_path: Path) -> List[float]:
    """
    Return the onset times (seconds) of each run-start boundary in the
    concatenated .fif. Index 0 is always 0.0 (first run starts at t=0).
    """
    import mne
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
    boundaries = sorted([
        ann["onset"] for ann in raw.annotations
        if "boundary" in ann["description"].lower()
    ])
    return [0.0] + boundaries


def _assign_run_numbers(
    event_onsets_sec: np.ndarray,
    run_starts_sec: List[float],
    total_duration_sec: float,
) -> np.ndarray:
    """
    Given event onset times, return the 1-indexed PhysioNet run number for
    each event (1 = R01, 5 = R05, …, 14 = R14).

    `run_starts_sec` must include 0.0 as the first element (n_runs entries
    total → n_runs − 1 boundary + 0.0).
    """
    run_ends_sec = list(run_starts_sec[1:]) + [total_duration_sec + 1.0]
    run_numbers = np.full(len(event_onsets_sec), fill_value=-1, dtype=np.int32)
    for i, (s, e) in enumerate(zip(run_starts_sec, run_ends_sec)):
        mask = (event_onsets_sec >= s) & (event_onsets_sec < e)
        run_numbers[mask] = i + 1   # 1-indexed
    return run_numbers


# ---------------------------------------------------------------------------
# Grouped BES computation
# ---------------------------------------------------------------------------

def compute_bes_grouped(
    pred_epochs: np.ndarray,
    gt_epochs:   np.ndarray,
    labels:      np.ndarray,
    groups:      np.ndarray,
    fs:          int = 256,
    feature_channels: Optional[List[int]] = None,
) -> Dict:
    """
    BES with leave-one-run-out cross-validation.

    Uses GroupKFold where each group is a PhysioNet run. Prevents temporal
    autocorrelation leakage between epochs from the same run.

    Returns dict with keys acc_gt, acc_pred, bes, n_folds, n_epochs.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if feature_channels is None:
        feature_channels = list(range(pred_epochs.shape[1]))

    def _featurize(epochs: np.ndarray) -> np.ndarray:
        rows = []
        for ep in epochs:
            row = []
            for ci in feature_channels:
                ch = ep[ci]
                row.append(np.log(band_power(ch, fs, 8.0,  13.0) + 1e-10))
                row.append(np.log(band_power(ch, fs, 13.0, 30.0) + 1e-10))
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    X_gt   = _featurize(gt_epochs)
    X_pred = _featurize(pred_epochs)
    y      = np.asarray(labels)
    g      = np.asarray(groups)

    unique_groups = np.unique(g)
    n_splits = len(unique_groups)
    if n_splits < 2:
        return {}   # can't do CV with fewer than 2 groups

    cv = GroupKFold(n_splits=n_splits)
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())

    accs_gt, accs_pred = [], []
    for train_idx, test_idx in cv.split(X_gt, y, groups=g):
        clf.fit(X_gt[train_idx], y[train_idx])
        accs_gt.append(clf.score(X_gt[test_idx], y[test_idx]))

        clf.fit(X_pred[train_idx], y[train_idx])
        accs_pred.append(clf.score(X_pred[test_idx], y[test_idx]))

    acc_gt   = float(np.mean(accs_gt))
    acc_pred = float(np.mean(accs_pred))
    return {
        "acc_gt":   acc_gt,
        "acc_pred": acc_pred,
        "bes":      acc_pred / (acc_gt + 1e-10),
        "n_folds":  n_splits,
        "n_epochs": len(y),
    }


def run_bes_grouped_subject(
    pred:            np.ndarray,
    gt_fif_path:     Path,
    target_channels: List[str],
    fs:              int = 256,
) -> Dict:
    """
    Full per-subject grouped BES: extract MI epochs + run labels → grouped CV.
    """
    import mne
    from pipeline_v2.eval.metrics import extract_mi_epochs

    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_fif(str(gt_fif_path), preload=False, verbose=False)
    total_dur = float(raw.times[-1])

    run_starts = _get_run_boundaries_sec(gt_fif_path)

    # Ground-truth epochs
    gt_epochs, labels = extract_mi_epochs(
        fif_path=gt_fif_path,
        channel_names=target_channels,
        fs=fs,
        tmin=0.0,
        tmax=2.0,
    )
    if len(gt_epochs) < 6:
        warnings.warn(f"{gt_fif_path.name}: only {len(gt_epochs)} GT epochs — skipping")
        return {}

    # Predicted epochs
    pred_epochs, _ = extract_pred_epochs(
        pred_continuous=pred,
        gt_fif_path=gt_fif_path,
        target_channels=target_channels,
        epoch_tmin=0.0,
        epoch_tmax=2.0,
        fs=fs,
    )
    if len(pred_epochs) < 6:
        warnings.warn(f"{gt_fif_path.name}: only {len(pred_epochs)} pred epochs — skipping")
        return {}

    n = min(len(gt_epochs), len(pred_epochs))
    gt_epochs, pred_epochs, labels = gt_epochs[:n], pred_epochs[:n], labels[:n]

    # Get event onsets (in seconds) for the GT epochs to determine run labels.
    # Re-extract events to get their onset times.
    events, raw_sfreq = _load_events_from_fif(gt_fif_path)
    if events is None:
        return {}

    mask = np.isin(events[:, 2], [1, 2])
    events = events[mask]

    # Convert event samples to seconds at the pred signal rate
    resample_ratio = fs / raw_sfreq
    epoch_len = int(2.0 * fs)
    valid_onsets_sec = []
    for ev_sample, _, _ in events:
        onset = int(round(ev_sample * resample_ratio))
        if onset >= 0 and (onset + epoch_len) <= pred.shape[1]:
            valid_onsets_sec.append(ev_sample / raw_sfreq)   # in seconds (raw time)
    valid_onsets_sec = np.array(valid_onsets_sec[:n])

    run_nums = _assign_run_numbers(valid_onsets_sec, run_starts, total_dur)

    # Keep only MI-run epochs (they should already be MI since we filter T1/T2,
    # but cross-check for safety)
    mi_mask = np.isin(run_nums, MI_RUNS)
    if mi_mask.sum() < 6:
        warnings.warn(f"{gt_fif_path.name}: fewer than 6 MI-run epochs identified")
        # Fall back: use all epochs without run grouping
        run_nums_for_cv = run_nums
    else:
        run_nums_for_cv = run_nums

    return compute_bes_grouped(
        pred_epochs=pred_epochs,
        gt_epochs=gt_epochs,
        labels=labels,
        groups=run_nums_for_cv,
        fs=fs,
    )


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda x: None))
    print(f"  Saved -> {path.relative_to(ROOT)}")


def _bes_summary(per_subject: Dict) -> Dict:
    vals = [v["bes"] for v in per_subject.values()
            if isinstance(v, dict) and "bes" in v]
    if not vals:
        return {"bes_mean": float("nan"), "bes_std": float("nan"), "n": 0}
    return {
        "bes_mean": float(np.mean(vals)),
        "bes_std":  float(np.std(vals)),
        "n":        len(vals),
    }


def run_ssi_repair(device_name: str):
    cfg        = DEVICE_CONFIGS[device_name]
    input_chs  = cfg["input_channels"]
    target_chs = cfg["eval_targets"]

    out_ps  = RESULTS_ROOT / "ssi" / device_name / "bes_per_subject_grouped.json"
    out_sum = RESULTS_ROOT / "ssi" / device_name / "bes_summary_grouped.json"
    if out_ps.exists():
        print(f"  [SKIP] SSI {device_name} grouped BES already computed")
        return

    print(f"\n[SSI BES repair] {device_name}")
    t0 = time.time()
    per_subject = {}

    for sid in TEST_SUBJECTS:
        fif_path = FIF_DIR / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            per_subject[f"S{sid:03d}"] = {"error": "fif_missing"}
            continue
        try:
            pred, _ = run_ssi_subject(fif_path, input_chs, target_chs)
            result  = run_bes_grouped_subject(pred, fif_path, target_chs)
            per_subject[f"S{sid:03d}"] = result if result else {"error": "skipped"}
            bes_str = f"{result['bes']:.3f}" if result else "skip"
            print(f"  S{sid:03d}  BES={bes_str}")
        except Exception as e:
            per_subject[f"S{sid:03d}"] = {"error": str(e)}
            print(f"  S{sid:03d}  FAIL: {e}")

    _save_json(per_subject, out_ps)
    _save_json(_bes_summary(per_subject), out_sum)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")


def run_reve_repair(device_name: str):
    cfg        = DEVICE_CONFIGS[device_name]
    input_chs  = cfg["input_channels"]
    target_chs = cfg["eval_targets"]

    out_ps  = RESULTS_ROOT / "reve" / device_name / "bes_per_subject_grouped.json"
    out_sum = RESULTS_ROOT / "reve" / device_name / "bes_summary_grouped.json"
    if out_ps.exists():
        print(f"  [SKIP] REVE {device_name} grouped BES already computed")
        return

    print(f"\n[REVE BES repair] {device_name}  (trains on 87 subjects)")
    t0 = time.time()

    train_paths = [FIF_DIR / f"S{sid:03d}_raw.fif"
                   for sid in TRAIN_SUBJECTS
                   if (FIF_DIR / f"S{sid:03d}_raw.fif").exists()]
    model = REVEModel(alpha=1.0)
    model.fit(train_paths, input_chs, target_chs, verbose=False)
    print(f"  REVE trained ({(time.time()-t0)/60:.1f} min)")

    per_subject = {}
    for sid in TEST_SUBJECTS:
        fif_path = FIF_DIR / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            per_subject[f"S{sid:03d}"] = {"error": "fif_missing"}
            continue
        try:
            pred, _ = model.predict(fif_path, input_chs, target_chs)
            result  = run_bes_grouped_subject(pred, fif_path, target_chs)
            per_subject[f"S{sid:03d}"] = result if result else {"error": "skipped"}
            bes_str = f"{result['bes']:.3f}" if result else "skip"
            print(f"  S{sid:03d}  BES={bes_str}")
        except Exception as e:
            per_subject[f"S{sid:03d}"] = {"error": str(e)}
            print(f"  S{sid:03d}  FAIL: {e}")

    _save_json(per_subject, out_ps)
    _save_json(_bes_summary(per_subject), out_sum)
    print(f"  Done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BES Grouped-CV Repair  (SSI + REVE, all 3 devices)")
    print("=" * 60)

    for device in DEVICES:
        print(f"\n{'='*60}")
        print(f"DEVICE: {device}")
        print(f"{'='*60}")
        run_ssi_repair(device)
        run_reve_repair(device)

    print("\n\nSUMMARY (grouped BES means):")
    print(f"{'Device':<20} {'Method':<8} {'BES (grouped)':>14} {'n':>4}")
    print("-" * 50)
    for device in DEVICES:
        for method in ["ssi", "reve"]:
            p = RESULTS_ROOT / method / device / "bes_summary_grouped.json"
            if p.exists():
                d = json.loads(p.read_text())
                print(f"{device:<20} {method:<8} {d.get('bes_mean', float('nan')):>14.3f} "
                      f"{d.get('n', 0):>4}")
