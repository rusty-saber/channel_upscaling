"""
CSP+LDA Analysis — Tests the Utility-Fidelity Decoupling Hypothesis.

Hypothesis:
    Methods with equivalent LDA-BES (utility) diverge sharply on CSP+LDA
    because CSP is spectrally sensitive while LDA on log-band-power is not.

    Predicted outcome:
        Ground truth CSP:  ≈ 65-75% accuracy (baseline)
        ZUNA CSP:          close to ground truth (β ratio = 0.611)
        SSI CSP:           degraded (β ratio = 10.82, inflated covariance)
        REVE CSP:          near-chance (β ratio = 0.0004, no beta signal)

    If correct: same BES but very different CSP → the dissociation is real.

CSP approach:
    - Bandpass 8-30 Hz (motor band, same for all)
    - Extract 0-2 s epochs from MI events (matching BES window)
    - CSP with n_components=4 (2 per class)
    - LDA classifier with leave-one-run-out CV (matching BES grouped-CV)
    - Metric: CSP accuracy ratio vs ground truth (analogous to BES)

ZUNA note:
    ZUNA reconstructed signals are not available locally (GPU instance destroyed).
    ZUNA result is estimated from its beta ratio and flagged as "predicted."
    A future GPU run will confirm. The SSI/REVE divergence alone demonstrates
    the dissociation.

Output:
    results/csp_analysis/csp_report.json
    results/csp_analysis/csp_report_readable.txt

Usage (CPU, ~25 min):
    python -m pipeline_v2.run_csp_analysis
"""

import json, sys, time, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline_v2.baselines.ssi_baseline  import run_ssi_subject
from pipeline_v2.baselines.reve_baseline import REVEModel
from pipeline_v2.data.device_configs     import DEVICE_CONFIGS
from pipeline_v2.data.subject_split      import TEST_SUBJECTS, TRAIN_SUBJECTS
from pipeline_v2.eval.bes_runner         import (
    _load_events_from_fif, extract_pred_epochs
)
from pipeline_v2.eval.metrics            import extract_mi_epochs
from pipeline_v2.run_bes_repair          import (
    _get_run_boundaries_sec, _assign_run_numbers
)

FIF_MI_DIR   = ROOT / "pipeline_v2" / "data" / "fif_mi"
FIF_DIR      = ROOT / "pipeline_v2" / "data" / "fif"
OUT_DIR      = ROOT / "pipeline_v2" / "results" / "csp_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CHS   = DEVICE_CONFIGS["emotiv_epoc"]["eval_targets"]   # C3, C4, P3, P4
INPUT_CHS    = DEVICE_CONFIGS["emotiv_epoc"]["input_channels"]
FS           = 256
MI_RUNS      = [5, 6, 9, 10, 13, 14]

# ── Motor band bandpass ──────────────────────────────────────────────────────

def bandpass_motor(signal_2d: np.ndarray, fs: int = FS,
                   low: float = 8.0, high: float = 30.0) -> np.ndarray:
    """Bandpass filter (n_channels, n_samples) in the motor band."""
    b, a = sp_signal.butter(4, [low / (fs/2), high / (fs/2)], btype="band")
    return sp_signal.filtfilt(b, a, signal_2d, axis=1)


# ── CSP ──────────────────────────────────────────────────────────────────────

def fit_csp(epochs_class1: np.ndarray, epochs_class2: np.ndarray,
            n_components: int = 4) -> np.ndarray:
    """
    Fit Common Spatial Patterns.

    Parameters
    ----------
    epochs_class1, epochs_class2 : (n_epochs, n_channels, n_samples)

    Returns
    -------
    W : (n_components, n_channels)  — spatial filter matrix (rows = filters)
    """
    def cov(epochs):
        # Average normalised covariance across epochs
        covs = []
        for ep in epochs:
            c = ep @ ep.T
            covs.append(c / np.trace(c))
        return np.mean(covs, axis=0)

    C1 = cov(epochs_class1)
    C2 = cov(epochs_class2)

    # Solve generalised eigenvalue problem C1 w = λ (C1+C2) w
    eigenvalues, eigenvectors = eigh(C1, C1 + C2)

    # Take extreme components: first n//2 and last n//2
    half = n_components // 2
    idx = np.concatenate([np.arange(half),
                          np.arange(len(eigenvalues) - half, len(eigenvalues))])
    W = eigenvectors[:, idx].T          # (n_components, n_channels)
    return W


def csp_features(epochs: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Apply CSP filters and return log-variance features.

    Parameters
    ----------
    epochs : (n_epochs, n_channels, n_samples)
    W      : (n_components, n_channels)

    Returns
    -------
    features : (n_epochs, n_components)
    """
    filtered = np.einsum("fc,ecs->efs", W, epochs)   # (n_epochs, n_comp, n_samples)
    variances = np.var(filtered, axis=-1)             # (n_epochs, n_comp)
    return np.log(variances + 1e-10)


# ── Run-group helper (reused from run_bes_repair) ────────────────────────────

def get_run_groups(fif_path: Path, event_onsets_sec: np.ndarray) -> np.ndarray:
    import mne
    mne.set_log_level("WARNING")
    raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
    total_dur = float(raw.times[-1])
    run_starts = _get_run_boundaries_sec(fif_path)
    return _assign_run_numbers(event_onsets_sec, run_starts, total_dur)


# ── Per-subject CSP accuracy ─────────────────────────────────────────────────

def csp_accuracy_subject(
    pred:      np.ndarray,   # (n_targets, n_samples)
    gt_sig:    np.ndarray,   # (n_targets, n_samples)
    fif_path:  Path,
    n_components: int = 4,
) -> Dict:
    """
    Compute CSP+LDA accuracy for both pred and gt signals.
    Returns dict with acc_gt, acc_pred, csp_ratio.
    """
    import mne
    mne.set_log_level("WARNING")

    # Motor-band bandpass
    pred_bp = bandpass_motor(pred)
    gt_bp   = bandpass_motor(gt_sig)

    # Load events
    events, raw_sfreq = _load_events_from_fif(fif_path)
    if events is None or len(events) == 0:
        return {}

    mask = np.isin(events[:, 2], [1, 2])
    events = events[mask]
    if len(events) < 6:
        return {}

    resample_ratio = FS / raw_sfreq
    epoch_len      = int(2.0 * FS)
    epoch_tmin_s   = 0.0
    epoch_tmax_s   = 2.0

    def slice_epochs(sig):
        eps, labs, onsets_sec = [], [], []
        for ev_sample, _, ev_id in events:
            onset = int(round(ev_sample * resample_ratio))
            if onset < 0 or (onset + epoch_len) > sig.shape[1]:
                continue
            ep = sig[:, onset:onset + epoch_len]
            if ep.shape[1] != epoch_len:
                continue
            eps.append(ep)
            labs.append(int(ev_id) - 1)
            onsets_sec.append(ev_sample / raw_sfreq)
        if not eps:
            return None, None, None
        return np.stack(eps), np.array(labs), np.array(onsets_sec)

    gt_epochs,   labels,   onsets_sec = slice_epochs(gt_bp)
    pred_epochs, labels_p, _          = slice_epochs(pred_bp)

    if gt_epochs is None or len(gt_epochs) < 6:
        return {}

    n = min(len(gt_epochs), len(pred_epochs))
    gt_epochs   = gt_epochs[:n]
    pred_epochs = pred_epochs[:n]
    labels      = labels[:n]
    onsets_sec  = onsets_sec[:n]

    # Run groups for leave-one-run-out CV
    groups = get_run_groups(fif_path, onsets_sec)
    mi_mask = np.isin(groups, MI_RUNS)
    groups_cv = groups  # use all available groups

    unique_groups = np.unique(groups_cv)
    n_splits = max(2, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)

    accs_gt, accs_pred = [], []

    for train_idx, test_idx in cv.split(gt_epochs, labels, groups=groups_cv):
        tr_gt, te_gt = gt_epochs[train_idx], gt_epochs[test_idx]
        tr_pr, te_pr = pred_epochs[train_idx], pred_epochs[test_idx]
        y_tr, y_te   = labels[train_idx], labels[test_idx]

        # Only train/evaluate if both classes present
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        # GT CSP
        try:
            W_gt = fit_csp(tr_gt[y_tr == 0], tr_gt[y_tr == 1], n_components)
            X_tr_gt = csp_features(tr_gt, W_gt)
            X_te_gt = csp_features(te_gt, W_gt)
            clf_gt  = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            clf_gt.fit(X_tr_gt, y_tr)
            accs_gt.append(clf_gt.score(X_te_gt, y_te))
        except Exception:
            pass

        # Pred CSP
        try:
            W_pr = fit_csp(tr_pr[y_tr == 0], tr_pr[y_tr == 1], n_components)
            X_tr_pr = csp_features(tr_pr, W_pr)
            X_te_pr = csp_features(te_pr, W_pr)
            clf_pr  = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
            clf_pr.fit(X_tr_pr, y_tr)
            accs_pred.append(clf_pr.score(X_te_pr, y_te))
        except Exception:
            pass

    if not accs_gt:
        return {}

    acc_gt   = float(np.mean(accs_gt))
    acc_pred = float(np.mean(accs_pred)) if accs_pred else float("nan")
    return {
        "acc_gt":    acc_gt,
        "acc_pred":  acc_pred,
        "csp_ratio": acc_pred / (acc_gt + 1e-10) if acc_pred == acc_pred else float("nan"),
        "n_folds":   len(accs_gt),
        "n_epochs":  n,
    }


# ── Per-method runners ────────────────────────────────────────────────────────

def run_csp_method(method_name: str, pred_fn) -> Dict[str, Dict]:
    """Run CSP analysis for one method across all test subjects."""
    results = {}
    t0 = time.time()
    for sid in TEST_SUBJECTS:
        fif_mi   = FIF_MI_DIR / f"S{sid:03d}_raw.fif"
        fif_orig = FIF_DIR    / f"S{sid:03d}_raw.fif"
        if not fif_mi.exists():
            results[f"S{sid:03d}"] = {"error": "fif_missing"}
            continue
        try:
            pred, gt = pred_fn(fif_mi)
            r = csp_accuracy_subject(pred, gt, fif_mi)
            results[f"S{sid:03d}"] = r if r else {"error": "no_epochs"}
            ratio_str = f"{r['csp_ratio']:.3f}" if r and "csp_ratio" in r else "skip"
            print(f"  {method_name} S{sid:03d}  "
                  f"gt={r.get('acc_gt',0):.3f}  "
                  f"pred={r.get('acc_pred',0):.3f}  "
                  f"ratio={ratio_str}")
        except Exception as e:
            results[f"S{sid:03d}"] = {"error": str(e)}
            print(f"  {method_name} S{sid:03d}  FAIL: {e}")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")
    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(results: Dict[str, Dict]) -> Dict:
    gt_vals   = [v["acc_gt"]    for v in results.values()
                 if isinstance(v, dict) and "acc_gt" in v]
    pred_vals = [v["acc_pred"]  for v in results.values()
                 if isinstance(v, dict) and "acc_pred" in v
                 and v["acc_pred"] == v["acc_pred"]]
    ratio_vals = [v["csp_ratio"] for v in results.values()
                  if isinstance(v, dict) and "csp_ratio" in v
                  and v["csp_ratio"] == v["csp_ratio"]]
    return {
        "acc_gt_mean":    float(np.mean(gt_vals))    if gt_vals    else float("nan"),
        "acc_pred_mean":  float(np.mean(pred_vals))  if pred_vals  else float("nan"),
        "csp_ratio_mean": float(np.mean(ratio_vals)) if ratio_vals else float("nan"),
        "csp_ratio_std":  float(np.std(ratio_vals))  if ratio_vals else float("nan"),
        "n": len(ratio_vals),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train REVE once (needed for all subjects)
    print("Training REVE...")
    train_paths = [FIF_DIR / f"S{sid:03d}_raw.fif" for sid in TRAIN_SUBJECTS
                   if (FIF_DIR / f"S{sid:03d}_raw.fif").exists()]
    reve_model = REVEModel(alpha=1.0)
    reve_model.fit(train_paths, INPUT_CHS, TARGET_CHS, verbose=False)
    print("REVE trained.")

    def ssi_pred(fif_path):
        return run_ssi_subject(fif_path, INPUT_CHS, TARGET_CHS)

    def reve_pred(fif_path):
        return reve_model.predict(fif_path, INPUT_CHS, TARGET_CHS)

    all_results = {}

    for name, fn in [("SSI", ssi_pred), ("REVE", reve_pred)]:
        print(f"\n{'='*55}\nCSP analysis: {name}\n{'='*55}")
        res = run_csp_method(name, fn)
        summ = summarise(res)
        all_results[name] = {"per_subject": res, "summary": summ}
        print(f"  {name} CSP ratio: {summ['csp_ratio_mean']:.3f} ± {summ['csp_ratio_std']:.3f}")

    # Load existing BES grouped results for comparison
    bes_ref = {}
    for method in ["ssi", "reve"]:
        p = ROOT / "pipeline_v2/results/bes_matched" / method / "emotiv_epoc" / "bes_summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            bes_ref[method.upper()] = d.get("bes_mean", float("nan"))

    # Save JSON
    output = {
        "methods": all_results,
        "bes_reference": bes_ref,
        "note_zuna": (
            "ZUNA CSP not computed locally (GPU required for reconstructed signals). "
            "Predicted: CSP ratio > REVE (0.252) based on beta ratio 0.611 >> 0.0004. "
            "Confirmation deferred to future GPU run."
        )
    }
    out_json = OUT_DIR / "csp_report.json"
    out_json.write_text(json.dumps(output, indent=2, default=lambda x: None))

    # Readable report
    lines = [
        "CSP+LDA ANALYSIS — Utility-Fidelity Decoupling",
        "=" * 65,
        "",
        "Metric: CSP ratio = acc_pred / acc_gt  (analogous to BES)",
        "CSP bandpass: 8-30 Hz (motor band)  |  Leave-one-run-out CV",
        "",
        f"{'Method':<8} {'BES (LDA)':<14} {'CSP ratio':<14} {'Dissociation?'}",
        "-" * 55,
    ]
    bes_lda = {"SSI": bes_ref.get("SSI", float("nan")),
               "REVE": bes_ref.get("REVE", float("nan")),
               "ZUNA": 0.858}

    for method in ["SSI", "REVE"]:
        if method in all_results:
            s = all_results[method]["summary"]
            lda = bes_lda.get(method, float("nan"))
            csp = s["csp_ratio_mean"]
            gap = lda - csp
            dissoc = "YES ★" if abs(gap) > 0.10 else "no"
            lines.append(f"{method:<8} {lda:<14.3f} {csp:<14.3f} {dissoc}")

    lines += [
        f"{'ZUNA':<8} {'0.858':<14} {'predicted>REVE':<14} (GPU run pending)",
        "",
        "KEY FINDING:",
        "If CSP ratio << BES for any method, the dissociation is confirmed:",
        "LDA utility and spectral fidelity are decoupled.",
    ]

    report = "\n".join(lines)
    (OUT_DIR / "csp_report_readable.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
