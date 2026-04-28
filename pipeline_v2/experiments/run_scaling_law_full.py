"""
Full Expansion Scaling Law — SSI + REVE (CPU-only, no GPU needed).

Runs both SSI and REVE at n ∈ {1, 2, 4, 8, 16, 32} input channels,
fits the power law r(n) = a·n^b for each method, and saves results to
pipeline_v2/results/scaling_law/scaling_law_full.json.

Run this on your own PC before visiting the GPU machine.
The GPU machine only needs to run ZUNA scaling points (run_scaling_zuna.py).

Usage (from repo root)
----------------------
    python -m pipeline_v2.experiments.run_scaling_law_full

Optional flags
--------------
    --fif_dir       path/to/fif           (default: pipeline_v2/data/fif)
    --results_dir   path/to/results       (default: pipeline_v2/results/scaling_law)
    --verbose                             print per-subject progress
    --overwrite                           re-run even if output already exists

Expected runtime
----------------
    SSI  : ~3-8 min  (22 subjects x 6 scale points, pure CPU)
    REVE : ~5-12 min (trains 4 ridge models per scale point on 87 subjects)
    Total: ~10-20 min
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline_v2.baselines.ssi_baseline import run_ssi_subject
from pipeline_v2.baselines.reve_baseline import REVEModel
from pipeline_v2.eval.metrics import compute_subject_metrics
from pipeline_v2.data.subject_split import TEST_SUBJECTS, TRAIN_SUBJECTS

FIF_DIR     = ROOT / "pipeline_v2" / "data" / "fif"
RESULTS_DIR = ROOT / "pipeline_v2" / "results" / "scaling_law"

# ── Channel sets (nested, progressively denser) ───────────────────────────────
NESTED_INPUT_SETS: Dict[int, List[str]] = {
    1:  ["Fz"],
    2:  ["F3", "F4"],
    4:  ["AF3", "AF4", "F3", "F4"],                          # Device A baseline
    8:  ["AF7", "AF3", "AFz", "AF4", "AF8", "F3", "Fz", "F4"],
    16: [
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
        "FT7", "FT8",
    ],
    32: [
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
        "FT7", "FT8",
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "T7",  "T8",  "T9",  "T10",
        "O1",  "Oz",  "O2",
        "TP7", "TP8",
    ],
}

TARGET_CHANNELS = ["C3", "C4", "P3", "P4"]
SCALE_POINTS    = [1, 2, 4, 8, 16, 32]


# ── Power-law fit ─────────────────────────────────────────────────────────────

def fit_power_law(ns: List[int], rs: List[float]) -> Dict:
    nan_result = {"a": float("nan"), "b": float("nan"), "r2": float("nan")}
    try:
        n_arr = np.array(ns, dtype=float)
        r_arr = np.array(rs, dtype=float)
        valid = r_arr > 0
        if valid.sum() < 2:
            return nan_result
        log_n = np.log(n_arr[valid])
        log_r = np.log(r_arr[valid])
        coeffs = np.polyfit(log_n, log_r, 1)
        b = float(coeffs[0])
        a = float(math.exp(coeffs[1]))
        log_r_pred = np.polyval(coeffs, log_n)
        ss_res = float(np.sum((log_r - log_r_pred) ** 2))
        ss_tot = float(np.sum((log_r - log_r.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return {"a": a, "b": b, "r2": r2}
    except Exception:
        return nan_result


def _summarise(per_subject: Dict) -> Dict:
    all_m: Dict[str, List[float]] = {}
    for v in per_subject.values():
        for k, val in v.get("metrics", {}).items():
            if isinstance(val, (int, float)):
                all_m.setdefault(k, []).append(float(val))
    out = {}
    for k, vals in all_m.items():
        arr = np.array(vals)
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"]  = float(arr.std())
    return out


# ── SSI at one scale point ────────────────────────────────────────────────────

# Max samples to use per subject for SSI (to keep spherical-spline RAM low).
# Shape loaded by MNE = (64, n_samples) float64  → 64*n*8 bytes.
# 46080 samples ≈ 3 min @ 256 Hz  →  64*46080*8 ≈ 23 MB per subject.
_SSI_MAX_SAMPLES = 46_080   # 3 min @ 256 Hz


def _run_ssi_subject_cropped(
    fif_path: Path,
    input_channels: List[str],
    target_channels: List[str],
    max_samples: int = _SSI_MAX_SAMPLES,
) -> tuple:
    """Run SSI on a (optionally cropped) recording to limit peak RAM."""
    import mne
    mne.set_log_level("WARNING")

    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    n_samp = raw.n_times
    if n_samp > max_samples:
        raw.crop(tmax=max_samples / raw.info["sfreq"])

    gt_data = raw.get_data(picks=target_channels)

    raw_masked = raw.copy()
    input_set = set(input_channels)
    raw_masked.info["bads"] = [ch for ch in raw_masked.ch_names if ch not in input_set]
    raw_interp = raw_masked.copy().interpolate_bads(reset_bads=True, verbose=False)
    pred_data = raw_interp.get_data(picks=target_channels)
    return pred_data, gt_data


def run_ssi_at_n(
    n: int,
    fif_dir: Path,
    verbose: bool = False,
) -> Dict:
    input_chs = NESTED_INPUT_SETS[n]
    per_subject: Dict[str, Dict] = {}

    for sid in TEST_SUBJECTS:
        fif_path = fif_dir / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            continue
        try:
            pred, gt = _run_ssi_subject_cropped(fif_path, input_chs, TARGET_CHANNELS)
            metrics  = compute_subject_metrics(pred, gt, TARGET_CHANNELS, fs=256)
            per_subject[f"S{sid:03d}"] = {"metrics": metrics}
            if verbose:
                print(f"    S{sid:03d}  r={metrics['pearson_mean']:.3f}")
        except Exception as e:
            if verbose:
                print(f"    S{sid:03d} FAIL: {e}")

    return _summarise(per_subject)


# ── REVE at one scale point ───────────────────────────────────────────────────

def run_reve_at_n(
    n: int,
    fif_dir: Path,
    verbose: bool = False,
) -> Dict:
    input_chs = NESTED_INPUT_SETS[n]

    # Scale down samples per subject to avoid OOM at high n.
    # Target total training matrix ≤ 50 MB:
    #   budget_bytes = 50e6
    #   max_samp = budget / (8 bytes * n_channels * 87 subjects)
    budget_bytes = 50_000_000
    max_samp = max(2_000, min(20_000, int(budget_bytes / (8 * max(1, n) * 87))))
    print(f"    max_samples_per_subject={max_samp} (n={n} → matrix ~"
          f"{87 * max_samp * n * 8 / 1e6:.0f} MB)", flush=True)

    # Build train paths
    train_paths = [
        fif_dir / f"S{sid:03d}_raw.fif"
        for sid in TRAIN_SUBJECTS
        if (fif_dir / f"S{sid:03d}_raw.fif").exists()
    ]

    # Train
    print(f"    Training REVE on {len(train_paths)} subjects...")
    t0 = time.time()
    model = REVEModel(alpha=1.0)
    model.fit(train_paths, input_chs, TARGET_CHANNELS, verbose=False,
              max_samples_per_subject=max_samp)
    print(f"    Training done ({(time.time()-t0)/60:.1f} min)")

    # Evaluate
    per_subject: Dict[str, Dict] = {}
    for sid in TEST_SUBJECTS:
        fif_path = fif_dir / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            continue
        try:
            pred, gt = model.predict(fif_path, input_chs, TARGET_CHANNELS)
            metrics  = compute_subject_metrics(pred, gt, TARGET_CHANNELS, fs=256)
            per_subject[f"S{sid:03d}"] = {"metrics": metrics}
            if verbose:
                print(f"    S{sid:03d}  r={metrics['pearson_mean']:.3f}")
        except Exception as e:
            if verbose:
                print(f"    S{sid:03d} FAIL: {e}")

    return _summarise(per_subject)


# ── Main ──────────────────────────────────────────────────────────────────────

def _save_checkpoint(results_dir: Path, ssi: Dict, reve: Dict) -> None:
    """Save partial results after each scale point (crash-safe)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "ssi":  {str(k): v for k, v in ssi.items()},
        "reve": {str(k): v for k, v in reve.items()},
    }
    with open(results_dir / "_checkpoint.json", "w") as f:
        json.dump(ckpt, f, indent=2)


def _load_checkpoint(results_dir: Path):
    ckpt_path = results_dir / "_checkpoint.json"
    if not ckpt_path.exists():
        return {}, {}
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    # Convert string keys back to int
    ssi  = {int(k): v for k, v in ckpt.get("ssi",  {}).items()}
    reve = {int(k): v for k, v in ckpt.get("reve", {}).items()}
    return ssi, reve


def run(args):
    fif_dir     = Path(args.fif_dir)
    results_dir = Path(args.results_dir)
    out_path    = results_dir / "scaling_law_full.json"

    if out_path.exists() and not args.overwrite:
        print(f"[SKIP] {out_path} already exists. Use --overwrite to re-run.")
        with open(out_path) as f:
            results = json.load(f)
        _print_table(results)
        return

    # Resume from checkpoint if available
    ssi_results, reve_results = _load_checkpoint(results_dir)
    if ssi_results or reve_results:
        done_ns = sorted(set(ssi_results) & set(reve_results))
        print(f"Resuming from checkpoint — already done: n={done_ns}")

    for n in SCALE_POINTS:
        if n not in NESTED_INPUT_SETS:
            continue

        # Skip if both methods already done for this n
        if n in ssi_results and n in reve_results and not args.overwrite:
            r_s = ssi_results[n].get("pearson_mean_mean", float("nan"))
            r_r = reve_results[n].get("pearson_mean_mean", float("nan"))
            print(f"  [SKIP] n={n}  SSI r={r_s:.4f}  REVE r={r_r:.4f}  (checkpoint)")
            continue

        print(f"\n{'─'*55}")
        print(f"Scale point n={n}  inputs={NESTED_INPUT_SETS[n]}")
        print(f"{'─'*55}")

        # SSI (skip if already done)
        if n not in ssi_results or args.overwrite:
            print(f"  [SSI] n={n}")
            t0 = time.time()
            ssi_sum = run_ssi_at_n(n, fif_dir, verbose=args.verbose)
            r_ssi = ssi_sum.get("pearson_mean_mean", float("nan"))
            print(f"  SSI done ({(time.time()-t0)/60:.1f} min)  r={r_ssi:.4f}")
            ssi_results[n] = ssi_sum
            _save_checkpoint(results_dir, ssi_results, reve_results)
        else:
            print(f"  [SKIP SSI] n={n} already in checkpoint")

        # REVE
        if n not in reve_results or args.overwrite:
            print(f"  [REVE] n={n}")
            t0 = time.time()
            reve_sum = run_reve_at_n(n, fif_dir, verbose=args.verbose)
            r_reve = reve_sum.get("pearson_mean_mean", float("nan"))
            print(f"  REVE done ({(time.time()-t0)/60:.1f} min)  r={r_reve:.4f}")
            reve_results[n] = reve_sum
            _save_checkpoint(results_dir, ssi_results, reve_results)
        else:
            print(f"  [SKIP REVE] n={n} already in checkpoint")

    # Power-law fits
    valid_ns = sorted(n for n in SCALE_POINTS if n in ssi_results and n in reve_results)
    pl_ssi  = fit_power_law(valid_ns, [ssi_results[n].get("pearson_mean_mean", 0) for n in valid_ns])
    pl_reve = fit_power_law(valid_ns, [reve_results[n].get("pearson_mean_mean", 0) for n in valid_ns])

    def _str_key(d):
        return {str(k): v for k, v in d.items()}

    results = {
        "scale_points": valid_ns,
        "ssi":  _str_key(ssi_results),
        "reve": _str_key(reve_results),
        "power_law": {
            "ssi":  pl_ssi,
            "reve": pl_reve,
        },
    }

    # Save final output
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    _print_table(results)


def _print_table(results: Dict) -> None:
    ns    = results.get("scale_points", [])
    ssi   = results.get("ssi",  {})
    reve  = results.get("reve", {})
    pl_s  = results.get("power_law", {}).get("ssi",  {})
    pl_r  = results.get("power_law", {}).get("reve", {})

    print(f"\n{'='*60}")
    print(f"{'n':>4}  {'SSI r':>8}  {'REVE r':>8}")
    print(f"{'─'*26}")
    for n in ns:
        r_s = ssi.get(str(n),  {}).get("pearson_mean_mean", float("nan"))
        r_r = reve.get(str(n), {}).get("pearson_mean_mean", float("nan"))
        print(f"{n:>4}  {r_s:>8.4f}  {r_r:>8.4f}")
    print(f"{'─'*26}")
    print(f"Power law SSI : r(n) = {pl_s.get('a', float('nan')):.4f} * n^{pl_s.get('b', float('nan')):.4f}  "
          f"R2={pl_s.get('r2', float('nan')):.3f}")
    print(f"Power law REVE: r(n) = {pl_r.get('a', float('nan')):.4f} * n^{pl_r.get('b', float('nan')):.4f}  "
          f"R2={pl_r.get('r2', float('nan')):.3f}")
    print(f"{'='*60}\n")


def _parse():
    p = argparse.ArgumentParser(
        description="Scaling law experiment — SSI + REVE (CPU only)"
    )
    p.add_argument("--fif_dir",
                   default=str(ROOT / "pipeline_v2/data/fif"))
    p.add_argument("--results_dir",
                   default=str(ROOT / "pipeline_v2/results/scaling_law"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
