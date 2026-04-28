"""
Full evaluation suite for the EEG Densification paper.

All four metrics must be reported for every experimental condition:

    1. MSE + Pearson r  — per reconstructed channel
    2. Per-band PSD     — δ/θ/α/β/γ band power ratio (pred / GT)
    3. Phase coherence  — PLV between input and reconstructed channels
    4. BES              — downstream MI classifier accuracy ratio

Results table format (per paper):  SSI | ZUNA | REVE-frozen | REVE-LoRA

Usage
-----
    from pipeline_v2.eval.metrics import compute_subject_metrics, compute_bes

    metrics = compute_subject_metrics(pred, gt, channel_names, fs=256)
    bes     = compute_bes(pred_epochs, gt_epochs, labels)
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal


# ─── Band definitions ─────────────────────────────────────────────────────────

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ─── 1. MSE + Pearson r ───────────────────────────────────────────────────────

def mse_per_channel(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Mean squared error per channel.

    Parameters
    ----------
    pred, gt : (n_channels, n_samples)

    Returns
    -------
    mse : (n_channels,)
    """
    return np.mean((pred - gt) ** 2, axis=-1)


def pearson_per_channel(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Pearson r per channel.

    Parameters
    ----------
    pred, gt : (n_channels, n_samples)

    Returns
    -------
    r : (n_channels,)  — values in [-1, 1]
    """
    n_ch = pred.shape[0]
    r = np.zeros(n_ch, dtype=np.float64)
    for i in range(n_ch):
        p, g = pred[i], gt[i]
        pm, gm = p - p.mean(), g - g.mean()
        denom = np.sqrt((pm ** 2).sum() * (gm ** 2).sum())
        r[i] = (pm * gm).sum() / (denom + 1e-10)
    return r


# ─── 2. Per-band PSD ─────────────────────────────────────────────────────────

def band_power(x: np.ndarray, fs: int, low: float, high: float) -> float:
    """
    Mean PSD (μV²/Hz) in [low, high] Hz via Welch's method.

    Parameters
    ----------
    x   : 1-D signal array
    fs  : sampling rate
    low, high : band edges in Hz
    """
    nperseg = min(fs * 2, len(x))
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(psd[mask]))


def psd_band_comparison(
    pred: np.ndarray,
    gt: np.ndarray,
    fs: int = 256,
    channel_names: Optional[List[str]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    For each band and each channel compute the band-power ratio pred/gt.

    A ratio of 1.0 = perfect spectral fidelity.
    > 1.0 = over-powered,  < 1.0 = under-powered.

    Parameters
    ----------
    pred, gt       : (n_channels, n_samples)
    fs             : sampling rate in Hz
    channel_names  : list of channel labels; defaults to ['ch0', 'ch1', …]
    bands          : dict of {band_name: (low_hz, high_hz)}; defaults to BANDS

    Returns
    -------
    Flat dict with keys like 'beta_C3_ratio', 'beta_C4_ratio', …
    and aggregate keys like 'beta_mean_ratio'.
    """
    if bands is None:
        bands = BANDS
    n_ch = pred.shape[0]
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_ch)]

    out: Dict[str, float] = {}

    for band_name, (low, high) in bands.items():
        ratios = []
        for i, ch in enumerate(channel_names):
            p_pow = band_power(pred[i], fs, low, high)
            g_pow = band_power(gt[i],   fs, low, high)
            ratio = p_pow / (g_pow + 1e-12)
            out[f"{band_name}_{ch}_ratio"] = float(ratio)
            ratios.append(ratio)
        out[f"{band_name}_mean_ratio"] = float(np.nanmean(ratios))
        out[f"{band_name}_std_ratio"]  = float(np.nanstd(ratios))

    return out


# ─── 3. Phase coherence (PLV) ────────────────────────────────────────────────

def phase_locking_value(x: np.ndarray, y: np.ndarray) -> float:
    """
    Phase Locking Value (PLV) between two EEG channels.

    PLV = |mean(exp(i * (φ_x − φ_y)))|  ∈ [0, 1]

    1.0 = perfect phase-locking
    0.0 = uniformly distributed phase difference (random)

    Uses the Hilbert transform to extract instantaneous phase.

    Parameters
    ----------
    x, y : 1-D arrays of the same length
    """
    phi_x = np.angle(sp_signal.hilbert(x))
    phi_y = np.angle(sp_signal.hilbert(y))
    plv = np.abs(np.mean(np.exp(1j * (phi_x - phi_y))))
    return float(plv)


def phase_coherence_pairs(
    pred: np.ndarray,
    gt: np.ndarray,
    channel_names: List[str],
    pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    """
    Compute PLV for each specified channel pair, comparing:
        - PLV between pred[ch_a] and pred[ch_b]        (reconstructed coherence)
        - PLV between gt[ch_a] and gt[ch_b]            (ground truth coherence)
        - ratio = reconstructed / ground truth          (fidelity score)

    Parameters
    ----------
    pred, gt       : (n_channels, n_samples)
    channel_names  : list mapping index → channel name
    pairs          : list of (ch_a, ch_b) tuples, e.g. [('F3', 'C3'), ('F4', 'C4')]
                     Note: both channels must be in channel_names.

    Returns
    -------
    Dict with keys like 'plv_F3_C3_pred', 'plv_F3_C3_gt', 'plv_F3_C3_ratio'
    """
    ch_idx = {ch: i for i, ch in enumerate(channel_names)}
    out: Dict[str, float] = {}

    for ch_a, ch_b in pairs:
        if ch_a not in ch_idx or ch_b not in ch_idx:
            # One of the pair channels is not in this channel set — skip
            out[f"plv_{ch_a}_{ch_b}_pred"]  = np.nan
            out[f"plv_{ch_a}_{ch_b}_gt"]    = np.nan
            out[f"plv_{ch_a}_{ch_b}_ratio"] = np.nan
            continue
        ia, ib = ch_idx[ch_a], ch_idx[ch_b]
        plv_pred = phase_locking_value(pred[ia], pred[ib])
        plv_gt   = phase_locking_value(gt[ia],   gt[ib])
        ratio    = plv_pred / (plv_gt + 1e-10)
        out[f"plv_{ch_a}_{ch_b}_pred"]  = plv_pred
        out[f"plv_{ch_a}_{ch_b}_gt"]    = plv_gt
        out[f"plv_{ch_a}_{ch_b}_ratio"] = float(ratio)

    return out


# ─── 4. BCI Equivalence Score (BES) ──────────────────────────────────────────

def compute_bes(
    pred_epochs: np.ndarray,
    gt_epochs:   np.ndarray,
    labels:      np.ndarray,
    fs:          int = 256,
    feature_channels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    BCI Equivalence Score (BES) — the paper's novel metric.

        BES = accuracy(LDA on reconstructed channels)
              ─────────────────────────────────────────
              accuracy(LDA on real channels)

    Both accuracies are evaluated with 5-fold stratified cross-validation.
    Features: log alpha (8–13 Hz) + log beta (13–30 Hz) band power
              per specified channel → n_features = 2 × n_channels.

    BES ≥ 0.90  → acceptable for consumer BCI deployment
    BES ≥ 0.80  → useful but limited
    BES < 0.70  → insufficient for practical use

    Parameters
    ----------
    pred_epochs      : (n_epochs, n_channels, n_samples) — reconstructed
    gt_epochs        : (n_epochs, n_channels, n_samples) — ground truth
    labels           : (n_epochs,) — integer class labels (e.g. 0=left, 1=right)
    fs               : sampling rate
    feature_channels : indices of channels to use for features
                       (defaults to all channels)

    Returns
    -------
    dict with keys:
        'acc_gt'   : float — ground truth classifier accuracy
        'acc_pred' : float — reconstructed classifier accuracy
        'bes'      : float — BES = acc_pred / acc_gt
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if feature_channels is None:
        feature_channels = list(range(pred_epochs.shape[1]))

    def extract_features(epochs: np.ndarray) -> np.ndarray:
        """epochs: (N, C, T) → features: (N, 2 × len(feature_channels))"""
        N = epochs.shape[0]
        feats = []
        for ep in epochs:                               # ep: (C, T)
            row = []
            for ci in feature_channels:
                ch_sig = ep[ci]
                a_pow = band_power(ch_sig, fs, 8.0,  13.0)
                b_pow = band_power(ch_sig, fs, 13.0, 30.0)
                row += [np.log(a_pow + 1e-10), np.log(b_pow + 1e-10)]
            feats.append(row)
        return np.array(feats, dtype=np.float32)

    X_gt   = extract_features(gt_epochs)
    X_pred = extract_features(pred_epochs)
    y      = np.asarray(labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_gt   = make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis())
    clf_pred = make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis())

    accs_gt, accs_pred = [], []

    for train_idx, test_idx in cv.split(X_gt, y):
        clf_gt.fit(X_gt[train_idx], y[train_idx])
        accs_gt.append(clf_gt.score(X_gt[test_idx], y[test_idx]))

        clf_pred.fit(X_pred[train_idx], y[train_idx])
        accs_pred.append(clf_pred.score(X_pred[test_idx], y[test_idx]))

    acc_gt   = float(np.mean(accs_gt))
    acc_pred = float(np.mean(accs_pred))
    bes      = acc_pred / (acc_gt + 1e-10)

    return {
        "acc_gt":   acc_gt,
        "acc_pred": acc_pred,
        "bes":      bes,
    }


def extract_mi_epochs(
    fif_path,
    channel_names: List[str],
    fs: int = 256,
    tmin: float = 0.5,   # s after event onset (avoid motor prep artifact)
    tmax: float = 4.5,   # s after event onset
    event_ids: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract motor-imagery epochs from a PhysioNet EEGMMIDB .fif file.

    Looks for events T1 (left hand) and T2 (right hand) in the annotations.
    Returns only epochs for subjects where both classes are present.

    Parameters
    ----------
    fif_path     : path to .fif  (continuous recording with embedded events)
    channel_names: channels to extract, e.g. ['C3', 'C4']
    tmin, tmax   : epoch window (seconds, relative to event onset)
    event_ids    : defaults to {'T1': 1, 'T2': 2}

    Returns
    -------
    epochs : (n_epochs, n_channels, n_samples)
    labels : (n_epochs,) — 0=T1, 1=T2
    """
    import mne
    mne.set_log_level("WARNING")

    if event_ids is None:
        event_ids = {"T1": 1, "T2": 2}

    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

    # Extract events from annotations
    try:
        events, ev_id_map = mne.events_from_annotations(
            raw, event_id=event_ids, verbose=False
        )
    except Exception as e:
        warnings.warn(f"Could not extract events from {fif_path}: {e}")
        return np.empty((0, len(channel_names), 0)), np.empty(0, dtype=int)

    if len(events) == 0:
        return np.empty((0, len(channel_names), 0)), np.empty(0, dtype=int)

    # Build Epochs
    picks = mne.pick_channels(raw.ch_names, include=channel_names, ordered=True)
    epochs = mne.Epochs(
        raw, events, event_id=event_ids,
        tmin=tmin, tmax=tmax,
        picks=picks, baseline=None,
        preload=True, verbose=False,
    )
    if len(epochs) == 0:
        return np.empty((0, len(channel_names), 0)), np.empty(0, dtype=int)

    data   = epochs.get_data()                           # (N, C, T)
    labels = epochs.events[:, 2] - 1                     # 0-indexed

    return data, labels


# ─── Composite subject-level evaluation ──────────────────────────────────────

def compute_subject_metrics(
    pred: np.ndarray,
    gt:   np.ndarray,
    channel_names: List[str],
    fs: int = 256,
    phase_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """
    Run the full metric suite for one subject.

    Parameters
    ----------
    pred, gt       : (n_channels, n_samples)
    channel_names  : list of channel labels for pred/gt rows
    fs             : sampling rate
    phase_pairs    : pairs for phase coherence; defaults to [('F3','C3'),('F4','C4')]

    Returns
    -------
    Flat dict with all scalar metrics.
    """
    if phase_pairs is None:
        phase_pairs = [("F3", "C3"), ("F4", "C4")]

    metrics: Dict[str, float] = {}

    # ── 1. MSE + Pearson r ────────────────────────────────────────────────────
    mse = mse_per_channel(pred, gt)
    r   = pearson_per_channel(pred, gt)

    metrics["mse_mean"]     = float(mse.mean())
    metrics["pearson_mean"] = float(r.mean())
    metrics["pearson_std"]  = float(r.std())

    for i, ch in enumerate(channel_names):
        metrics[f"mse_{ch}"]     = float(mse[i])
        metrics[f"pearson_{ch}"] = float(r[i])

    # ── 2. Per-band PSD ───────────────────────────────────────────────────────
    psd_metrics = psd_band_comparison(pred, gt, fs=fs, channel_names=channel_names)
    metrics.update(psd_metrics)

    # ── 3. Phase coherence ────────────────────────────────────────────────────
    # phase_pairs reference input channels (e.g. F3) which may not be in
    # the target-only pred array.  We compute coherence only for pairs
    # where BOTH channels are present.
    plv_metrics = phase_coherence_pairs(pred, gt, channel_names, phase_pairs)
    metrics.update(plv_metrics)

    return metrics


# ─── Results table helper ─────────────────────────────────────────────────────

def print_results_table(
    results: Dict[str, Dict[str, float]],
    conditions: Optional[List[str]] = None,
    metrics_to_show: Optional[List[str]] = None,
) -> None:
    """
    Print the paper's results table:  SSI | ZUNA | REVE-frozen | REVE-LoRA

    Parameters
    ----------
    results    : {condition_name: {metric_name: value}}
    conditions : ordered list of condition names for columns
    """
    if conditions is None:
        conditions = list(results.keys())
    if metrics_to_show is None:
        metrics_to_show = [
            "pearson_mean", "mse_mean",
            "beta_mean_ratio", "alpha_mean_ratio",
            "plv_F3_C3_ratio", "plv_F4_C4_ratio",
        ]

    col_w = 14
    header = f"{'Metric':<30}" + "".join(f"{c:>{col_w}}" for c in conditions)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))

    for m in metrics_to_show:
        row = f"{m:<30}"
        for cond in conditions:
            val = results.get(cond, {}).get(m, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("─" * len(header) + "\n")
