"""
BES Runner — BCI Equivalence Score evaluation pipeline.

Bridges the gap between continuous reconstruction outputs (SSI, REVE, ZUNA)
and the epoch-based `compute_bes()` function in metrics.py.

The core problem: reconstruction models produce continuous arrays
(n_targets, n_samples), but BES needs epoch-segmented data with MI class
labels derived from PhysioNet event structure.  This module handles the
segmentation step by re-reading the original .fif file for event timing,
then applying the same epoch windows to the predicted signal.

BES interpretation (from paper):
    BES >= 0.85  →  clinically equivalent (paper threshold)
    BES >= 0.90  →  acceptable for consumer BCI deployment
    BES < 0.70   →  insufficient for practical use

Note on ZUNA results:
    ZUNA's run_zuna_dataset() does not store reconstructed numpy arrays in
    memory — it writes reconstructed .fif files to disk.  To run BES on
    ZUNA output, load the reconstructed .fif, extract the target channels
    as a numpy array, and call run_bes_subject() directly.  run_bes_dataset()
    will silently skip any subject entry that lacks a "pred" key.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Path setup ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_v2.eval.metrics import compute_bes  # noqa: E402


# --- Minimum epoch count for a reliable 5-fold CV ----------------------------
MIN_EPOCHS = 10


# --- Event extraction helpers -------------------------------------------------

def _load_events_from_fif(
    fif_path: Path,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Load a raw .fif file and return (events, sfreq).

    `events` is a NumPy array of shape (n_events, 3) in the standard MNE
    format [sample, prev_id, event_id].  Returns None if no usable events
    are found.

    Tries mne.events_from_annotations first (PhysioNet EEGMMIDB stores events
    as annotations).  Falls back to mne.find_events if annotations yield
    nothing.
    """
    import mne
    mne.set_log_level("WARNING")

    raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
    sfreq = raw.info["sfreq"]

    # Attempt 1: annotation-based events (standard for EEGMMIDB .fif files)
    try:
        events, _ = mne.events_from_annotations(
            raw,
            event_id={"T1": 1, "T2": 2},
            verbose=False,
        )
        if len(events) > 0:
            return events, sfreq
    except Exception:
        pass

    # Attempt 2: stim-channel events
    try:
        events = mne.find_events(
            raw,
            stim_channel="auto",
            verbose=False,
        )
        # Keep only T1 (id=1) and T2 (id=2)
        mask = np.isin(events[:, 2], [1, 2])
        events = events[mask]
        if len(events) > 0:
            return events, sfreq
    except Exception:
        pass

    return None, sfreq


# --- Core segmentation --------------------------------------------------------

def extract_pred_epochs(
    pred_continuous: np.ndarray,
    gt_fif_path: Path,
    target_channels: List[str],
    epoch_tmin: float = 0.0,
    epoch_tmax: float = 2.0,
    fs: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a continuous reconstructed signal into MI epochs.

    Uses the event markers from the ground-truth .fif to determine epoch
    onset times, then slices pred_continuous at the same sample offsets.
    This ensures pred epochs are temporally aligned with gt epochs from
    extract_mi_epochs().

    Parameters
    ----------
    pred_continuous : (n_targets, n_samples)
        Continuous reconstructed signal for the target channels.
    gt_fif_path : Path
        Original .fif file — used only for event timing, not signal data.
    target_channels : List[str]
        Names of the target channels (used for error messages; the order
        must match the first axis of pred_continuous).
    epoch_tmin : float
        Epoch start time relative to event onset in seconds (default 0.0).
    epoch_tmax : float
        Epoch end time relative to event onset in seconds (default 2.0).
    fs : int
        Sampling rate of pred_continuous in Hz.

    Returns
    -------
    pred_epochs : (n_epochs, n_channels, n_epoch_samples)
    labels      : (n_epochs,)  — 0 for T1, 1 for T2

    Notes
    -----
    Epochs that would extend beyond the boundaries of pred_continuous are
    silently skipped to avoid index errors on short recordings.
    """
    events, raw_sfreq = _load_events_from_fif(gt_fif_path)

    if events is None or len(events) == 0:
        warnings.warn(
            f"No T1/T2 events found in {gt_fif_path}. "
            "Returning empty epoch arrays."
        )
        n_targets = pred_continuous.shape[0]
        epoch_len = int((epoch_tmax - epoch_tmin) * fs)
        return (
            np.empty((0, n_targets, epoch_len), dtype=pred_continuous.dtype),
            np.empty(0, dtype=np.int64),
        )

    # Filter to T1 and T2 only
    mask = np.isin(events[:, 2], [1, 2])
    events = events[mask]

    # Convert raw event samples to pred_continuous sample indices.
    # The raw file may have a different sfreq than the reconstructed signal
    # (e.g., raw at 160 Hz, pred at 256 Hz), so we resample the onset.
    resample_ratio = fs / raw_sfreq

    n_targets, n_total_samples = pred_continuous.shape
    epoch_start_offset = int(epoch_tmin * fs)
    epoch_end_offset   = int(epoch_tmax * fs)
    epoch_len          = epoch_end_offset - epoch_start_offset

    epochs_list: List[np.ndarray] = []
    labels_list: List[int]        = []

    for event_sample, _, event_id in events:
        # Translate onset sample to reconstructed signal sample space
        onset = int(round(event_sample * resample_ratio))

        start = onset + epoch_start_offset
        end   = onset + epoch_end_offset

        # Skip out-of-bounds epochs
        if start < 0 or end > n_total_samples:
            continue

        epoch = pred_continuous[:, start:end]   # (n_targets, epoch_len)

        # Guard against variable-length slices at edge of array
        if epoch.shape[1] != epoch_len:
            continue

        epochs_list.append(epoch)
        labels_list.append(int(event_id) - 1)   # 0-indexed: T1→0, T2→1

    if len(epochs_list) == 0:
        return (
            np.empty((0, n_targets, epoch_len), dtype=pred_continuous.dtype),
            np.empty(0, dtype=np.int64),
        )

    pred_epochs = np.stack(epochs_list, axis=0)   # (n_epochs, n_channels, T)
    labels      = np.array(labels_list, dtype=np.int64)

    return pred_epochs, labels


# --- Per-subject BES ----------------------------------------------------------

def run_bes_subject(
    pred: np.ndarray,
    gt_fif_path: Path,
    target_channels: List[str],
    fs: int = 256,
    epoch_tmin: float = 0.0,
    epoch_tmax: float = 2.0,
) -> Dict:
    """
    Compute BES for a single subject.

    Extracts MI epochs from both the ground-truth .fif and the continuous
    pred array, then calls compute_bes().

    Parameters
    ----------
    pred : (n_targets, n_samples)
        Continuous reconstructed signal.
    gt_fif_path : Path
        Original .fif — provides both ground-truth signal and event timing.
    target_channels : List[str]
        Target channel names, ordered to match the first axis of pred.
    fs : int
        Sampling rate in Hz.
    epoch_tmin, epoch_tmax : float
        Epoch window in seconds relative to event onset.

    Returns
    -------
    dict with keys acc_gt, acc_pred, bes — or an empty dict if there are
    fewer than MIN_EPOCHS (10) epochs (not enough for stable 5-fold CV).
    """
    from pipeline_v2.eval.metrics import extract_mi_epochs

    # Ground-truth epochs
    gt_epochs, labels = extract_mi_epochs(
        fif_path=gt_fif_path,
        channel_names=target_channels,
        fs=fs,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
    )

    if len(gt_epochs) < MIN_EPOCHS:
        warnings.warn(
            f"{gt_fif_path.name}: only {len(gt_epochs)} GT epochs found "
            f"(need >= {MIN_EPOCHS}). Skipping BES."
        )
        return {}

    # Predicted epochs — use same event timing from the .fif
    pred_epochs, pred_labels = extract_pred_epochs(
        pred_continuous=pred,
        gt_fif_path=gt_fif_path,
        target_channels=target_channels,
        epoch_tmin=epoch_tmin,
        epoch_tmax=epoch_tmax,
        fs=fs,
    )

    if len(pred_epochs) < MIN_EPOCHS:
        warnings.warn(
            f"{gt_fif_path.name}: only {len(pred_epochs)} pred epochs "
            f"extracted (need >= {MIN_EPOCHS}). Skipping BES."
        )
        return {}

    # Align epoch counts (in case of minor boundary mismatches between
    # extract_mi_epochs and extract_pred_epochs)
    n = min(len(gt_epochs), len(pred_epochs))
    gt_epochs   = gt_epochs[:n]
    pred_epochs = pred_epochs[:n]
    labels      = labels[:n]

    return compute_bes(
        pred_epochs=pred_epochs,
        gt_epochs=gt_epochs,
        labels=labels,
        fs=fs,
    )


# --- Dataset-level BES --------------------------------------------------------

def run_bes_dataset(
    results: Dict[str, Dict],
    fif_dir: Path,
    target_channels: List[str],
    fs: int = 256,
    epoch_tmin: float = 0.0,
    epoch_tmax: float = 2.0,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Compute BES for every subject in a results dictionary.

    Designed to work with the output format of run_ssi_dataset() and
    run_reve_dataset(), where each subject entry contains a "pred" numpy
    array.

    Parameters
    ----------
    results : Dict[str, Dict]
        {subject_id: {"pred": np.ndarray, "gt": np.ndarray, "metrics": dict}}
        Subjects missing a "pred" key are skipped with a warning (see the
        ZUNA note in the module docstring).
    fif_dir : Path
        Directory that contains per-subject .fif files.  Files are expected
        to follow the naming convention used by download_eegmmidb.py,
        e.g. S001_runs_5_6_9_10_13_14.fif or S001.fif.  The function tries
        several common patterns and warns if no file is found.
    target_channels : List[str]
        Channel names that correspond to the first axis of each pred array.
    fs : int
        Sampling rate of the reconstructed signals.
    epoch_tmin, epoch_tmax : float
        Epoch window in seconds.
    verbose : bool
        If True, print per-subject BES results to stdout.

    Returns
    -------
    Augmented results dict — each subject entry gains a "bes" key containing
    the dict returned by compute_bes() (keys: acc_gt, acc_pred, bes), or an
    empty dict if BES could not be computed for that subject.

    Note on ZUNA:
        ZUNA results do not carry a "pred" array in the results dict.
        To evaluate ZUNA with BES, load the reconstructed .fif produced by
        run_zuna_dataset(), extract target channels as a numpy array, and
        call run_bes_subject() directly.  Subjects without "pred" will be
        skipped here with a warning.
    """
    fif_dir = Path(fif_dir)

    for subject_id, subj_data in results.items():
        # -- Check for pred array -----------------------------------------------
        if "pred" not in subj_data:
            warnings.warn(
                f"Subject {subject_id}: no 'pred' key in results dict. "
                "Skipping BES (ZUNA? load reconstructed .fif and call "
                "run_bes_subject() directly)."
            )
            subj_data["bes"] = {}
            continue

        pred = subj_data["pred"]

        # -- Locate the source .fif ---------------------------------------------
        fif_path = _find_fif(fif_dir, subject_id)
        if fif_path is None:
            warnings.warn(
                f"Subject {subject_id}: could not find .fif in {fif_dir}. "
                "Skipping BES."
            )
            subj_data["bes"] = {}
            continue

        # -- Run BES ------------------------------------------------------------
        bes_result = run_bes_subject(
            pred=pred,
            gt_fif_path=fif_path,
            target_channels=target_channels,
            fs=fs,
            epoch_tmin=epoch_tmin,
            epoch_tmax=epoch_tmax,
        )

        subj_data["bes"] = bes_result

        if verbose:
            if bes_result:
                print(
                    f"  {subject_id}: BES={bes_result['bes']:.3f}  "
                    f"acc_gt={bes_result['acc_gt']:.3f}  "
                    f"acc_pred={bes_result['acc_pred']:.3f}"
                )
            else:
                print(f"  {subject_id}: BES skipped (insufficient epochs)")

    return results


def _find_fif(fif_dir: Path, subject_id: str) -> Optional[Path]:
    """
    Locate a .fif file for `subject_id` inside `fif_dir`.

    Tries several naming patterns used by download_eegmmidb.py and common
    preprocessing conventions.  Returns None if nothing matches.
    """
    candidates = [
        fif_dir / f"{subject_id}_runs_5_6_9_10_13_14.fif",
        fif_dir / f"{subject_id}_mi.fif",
        fif_dir / f"{subject_id}.fif",
        fif_dir / subject_id / f"{subject_id}.fif",
        fif_dir / subject_id / f"{subject_id}_mi.fif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# --- Summary statistics -------------------------------------------------------

def summarise_bes(results: Dict[str, Dict]) -> Dict:
    """
    Aggregate BES metrics across all subjects in a results dict.

    Subjects whose "bes" entry is an empty dict (BES could not be computed)
    are excluded from the aggregation.

    Parameters
    ----------
    results : Dict[str, Dict]
        Output of run_bes_dataset() — each subject entry should have a "bes"
        sub-dict with keys acc_gt, acc_pred, bes.

    Returns
    -------
    {
        "bes_mean":      float,
        "bes_std":       float,
        "acc_gt_mean":   float,
        "acc_pred_mean": float,
        "n_subjects":    int,   # number of subjects included
    }
    """
    bes_vals:      List[float] = []
    acc_gt_vals:   List[float] = []
    acc_pred_vals: List[float] = []

    for subject_id, subj_data in results.items():
        bes_entry = subj_data.get("bes", {})
        if not bes_entry:
            continue
        bes_vals.append(bes_entry["bes"])
        acc_gt_vals.append(bes_entry["acc_gt"])
        acc_pred_vals.append(bes_entry["acc_pred"])

    if len(bes_vals) == 0:
        warnings.warn("summarise_bes: no valid BES results found.")
        return {
            "bes_mean":      float("nan"),
            "bes_std":       float("nan"),
            "acc_gt_mean":   float("nan"),
            "acc_pred_mean": float("nan"),
            "n_subjects":    0,
        }

    return {
        "bes_mean":      float(np.mean(bes_vals)),
        "bes_std":       float(np.std(bes_vals)),
        "acc_gt_mean":   float(np.mean(acc_gt_vals)),
        "acc_pred_mean": float(np.mean(acc_pred_vals)),
        "n_subjects":    len(bes_vals),
    }
