"""
Expansion Scaling Law experiment.

Hypothesis: reconstruction quality follows a power law as a function of input
channel density:  r(n) ≈ a · n^b

We vary the number of input channels across nested subsets that correspond to
progressively denser consumer-grade EEG layouts (1 → 2 → 4 → 8 → 16 → 32)
and measure Pearson r, MSE, and beta band-power ratio for each density.

SSI is run directly; ZUNA results are optionally loaded from pre-computed
summary JSON files (--include_zuna flag).

Usage
-----
    python -m pipeline_v2.experiments.scaling_law \\
        --fif_dir pipeline_v2/data/fif \\
        --results_dir pipeline_v2/results/scaling_law \\
        --split test \\
        --verbose

    # Also load pre-computed ZUNA results if available:
    python -m pipeline_v2.experiments.scaling_law \\
        --fif_dir pipeline_v2/data/fif \\
        --results_dir pipeline_v2/results/scaling_law \\
        --include_zuna --verbose
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_v2.baselines.ssi_baseline import run_ssi_subject, summarise
from pipeline_v2.eval.metrics import compute_subject_metrics
from pipeline_v2.data.subject_split import TEST_SUBJECTS, TRAIN_SUBJECTS


# --- Channel set definitions --------------------------------------------------

NESTED_INPUT_SETS: Dict[int, List[str]] = {
    1:  ["Fz"],
    2:  ["F3", "F4"],
    4:  ["AF3", "AF4", "F3", "F4"],
    8:  ["AF7", "AF3", "AFz", "AF4", "AF8", "F3", "Fz", "F4"],
    16: [
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
        "FT7", "FT8",
    ],
    32: [
        # Frontal
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
        # Fronto-central (no C3/C4/P3/P4 or their immediate neighbours)
        "FT7", "FT8",
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        # Temporal
        "T7",  "T8",  "T9",  "T10",
        # Occipital (far from targets, spatial diversity)
        "O1",  "Oz",  "O2",
        # Temporo-parietal
        "TP7", "TP8",
    ],
}

TARGET_CHANNELS: List[str] = ["C3", "C4", "P3", "P4"]

DEFAULT_SCALE_POINTS: List[int] = [1, 2, 4, 8, 16, 32]


# --- Core experiment functions ------------------------------------------------

def run_ssi_scale_point(
    fif_dir: Path,
    subject_ids: List[int],
    input_channels: List[str],
    target_channels: List[str],
    verbose: bool = False,
) -> Dict:
    """
    Run SSI at one scale point (one input-channel density).

    Iterates over all subjects, runs spherical spline interpolation with the
    given input_channels and evaluates against target_channels, then aggregates
    the per-subject metrics into a dataset-level summary.

    Parameters
    ----------
    fif_dir         : directory containing S001_raw.fif … files
    subject_ids     : list of integer subject IDs to process
    input_channels  : channels available at this density level
    target_channels : channels to reconstruct and evaluate
    verbose         : print per-subject progress

    Returns
    -------
    Summarised metrics dict, e.g.
    {"pearson_mean_mean": 0.72, "pearson_mean_std": 0.05, ...}
    """
    per_subject_results: Dict[str, Dict] = {}

    for sid in subject_ids:
        fif_path = fif_dir / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            if verbose:
                print(f"  [MISS] S{sid:03d} not found — skipping")
            continue
        try:
            pred, gt = run_ssi_subject(fif_path, input_channels, target_channels)
            metrics = compute_subject_metrics(pred, gt, target_channels, fs=256)
            per_subject_results[f"S{sid:03d}"] = {"pred": pred, "gt": gt, "metrics": metrics}
            if verbose:
                print(
                    f"  S{sid:03d}  r={metrics['pearson_mean']:.3f}"
                    f"  MSE={metrics['mse_mean']:.4f}"
                )
        except Exception as exc:
            print(f"  [FAIL] S{sid:03d}: {exc}")

    return summarise(per_subject_results)


def fit_power_law(n_values: List[int], r_values: List[float]) -> Dict:
    """
    Fit r(n) = a * n^b via log-linear regression on the log-log transform.

    Uses np.polyfit(log(n), log(r), 1).  Only scale points where r > 0 are
    included in the fit; if fewer than two valid points remain, or if the fit
    raises any exception, a and b are set to NaN.

    Parameters
    ----------
    n_values : list of input channel counts, e.g. [1, 2, 4, 8, 16, 32]
    r_values : corresponding mean Pearson r values

    Returns
    -------
    dict with keys:
        "a"  : float  — amplitude coefficient
        "b"  : float  — scaling exponent  (b > 0 means quality improves with n)
        "r2" : float  — coefficient of determination on the log-log fit

    Notes
    -----
    b > 0  → quality improves with more channels (expected)
    Larger b → steeper scaling (more benefit per added electrode)
    """
    nan_result = {"a": float("nan"), "b": float("nan"), "r2": float("nan")}

    try:
        n_arr = np.array(n_values, dtype=float)
        r_arr = np.array(r_values, dtype=float)

        # Keep only points where r > 0 (log is undefined otherwise)
        valid = r_arr > 0
        if valid.sum() < 2:
            return nan_result

        log_n = np.log(n_arr[valid])
        log_r = np.log(r_arr[valid])

        # Degree-1 polyfit on log-log: log(r) = b*log(n) + log(a)
        coeffs = np.polyfit(log_n, log_r, 1)
        b = float(coeffs[0])
        a = float(math.exp(coeffs[1]))

        # R² on the log-log scale
        log_r_pred = np.polyval(coeffs, log_n)
        ss_res = float(np.sum((log_r - log_r_pred) ** 2))
        ss_tot = float(np.sum((log_r - log_r.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        return {"a": a, "b": b, "r2": r2}

    except Exception:
        return nan_result


def run_scaling_experiment(
    fif_dir: Path,
    subject_ids: List[int] = None,
    scale_points: List[int] = None,
    verbose: bool = True,
    results_dir: Path = None,
    include_zuna: bool = False,
) -> Dict:
    """
    Run SSI at all scale points and optionally load pre-computed ZUNA results.

    ZUNA is NOT run here because it requires a GPU and is too slow for quick
    iteration.  Pass --include_zuna to load results from
    results_dir/zuna/<device>/summary.json if those files exist.

    Parameters
    ----------
    fif_dir      : directory containing .fif files
    subject_ids  : subjects to evaluate (default: TEST_SUBJECTS)
    scale_points : channel counts to sweep (default: [1,2,4,8,16,32])
    verbose      : print progress
    results_dir  : directory for saving output and loading ZUNA results
    include_zuna : attempt to load pre-computed ZUNA summaries

    Returns
    -------
    {
        "scale_points": [1, 2, 4, 8, 16, 32],
        "ssi": {
            1:  {"pearson_mean_mean": ..., "mse_mean_mean": ...,
                 "beta_mean_ratio_mean": ...},
            2:  {...},
            ...
        },
        "zuna": {   # only present when include_zuna=True and files exist
            1:  {...}, ...
        },
        "power_law": {
            "ssi":  {"a": ..., "b": ..., "r2": ...},
            "zuna": {"a": ..., "b": ..., "r2": ...}   # if zuna loaded
        }
    }
    """
    if subject_ids is None:
        subject_ids = TEST_SUBJECTS
    if scale_points is None:
        scale_points = DEFAULT_SCALE_POINTS

    ssi_results: Dict[int, Dict] = {}

    for n in scale_points:
        if n not in NESTED_INPUT_SETS:
            print(f"  [WARN] No channel set defined for n={n}, skipping.")
            continue

        input_channels = NESTED_INPUT_SETS[n]

        if verbose:
            print(f"\n{'-'*55}")
            print(f"Scale point  n={n:2d}  ({len(input_channels)} input channels)")
            print(f"  Input : {input_channels}")
            print(f"  Target: {TARGET_CHANNELS}")
            print(f"{'-'*55}")

        summary = run_ssi_scale_point(
            fif_dir=fif_dir,
            subject_ids=subject_ids,
            input_channels=input_channels,
            target_channels=TARGET_CHANNELS,
            verbose=verbose,
        )

        # Store a compact view of the three headline metrics
        ssi_results[n] = {
            "pearson_mean_mean":     summary.get("pearson_mean_mean",     float("nan")),
            "mse_mean_mean":         summary.get("mse_mean_mean",         float("nan")),
            "beta_mean_ratio_mean":  summary.get("beta_mean_ratio_mean",  float("nan")),
            # carry full summary for serialisation
            "_full": summary,
        }

        if verbose:
            r = ssi_results[n]["pearson_mean_mean"]
            mse = ssi_results[n]["mse_mean_mean"]
            beta = ssi_results[n]["beta_mean_ratio_mean"]
            print(f"  --> SSI  r={r:.4f}  MSE={mse:.5f}  beta_ratio={beta:.4f}")

    # -- Power law fit for SSI -------------------------------------------------
    valid_ns = [n for n in scale_points if n in ssi_results]
    r_vals_ssi = [ssi_results[n]["pearson_mean_mean"] for n in valid_ns]
    power_law_ssi = fit_power_law(valid_ns, r_vals_ssi)

    if verbose:
        print(f"\nPower-law fit (SSI): "
              f"a={power_law_ssi['a']:.4f}  "
              f"b={power_law_ssi['b']:.4f}  "
              f"R²={power_law_ssi['r2']:.4f}")

    # -- Optional ZUNA loading -------------------------------------------------
    zuna_results: Optional[Dict[int, Dict]] = None
    power_law_zuna: Optional[Dict] = None

    if include_zuna and results_dir is not None:
        zuna_results = _load_zuna_results(results_dir, scale_points, verbose=verbose)
        if zuna_results:
            r_vals_zuna = [
                zuna_results[n]["pearson_mean_mean"]
                for n in valid_ns
                if n in zuna_results
            ]
            zuna_ns = [n for n in valid_ns if n in zuna_results]
            power_law_zuna = fit_power_law(zuna_ns, r_vals_zuna)
            if verbose:
                print(f"Power-law fit (ZUNA): "
                      f"a={power_law_zuna['a']:.4f}  "
                      f"b={power_law_zuna['b']:.4f}  "
                      f"R²={power_law_zuna['r2']:.4f}")

    # -- Assemble output -------------------------------------------------------
    # Strip internal _full keys for the top-level scale dict
    ssi_compact = {
        n: {k: v for k, v in d.items() if k != "_full"}
        for n, d in ssi_results.items()
    }

    output: Dict = {
        "scale_points": valid_ns,
        "ssi": ssi_compact,
        "power_law": {
            "ssi": power_law_ssi,
        },
    }

    if zuna_results:
        output["zuna"] = {
            n: {k: v for k, v in d.items() if k != "_full"}
            for n, d in zuna_results.items()
        }
        output["power_law"]["zuna"] = power_law_zuna

    return output


def _load_zuna_results(
    results_dir: Path,
    scale_points: List[int],
    verbose: bool = False,
) -> Dict[int, Dict]:
    """
    Try to load pre-computed ZUNA summaries from
    results_dir/zuna/<n>ch/summary.json  (one file per scale point).

    Falls back gracefully if a file is missing.
    """
    zuna: Dict[int, Dict] = {}

    for n in scale_points:
        # Convention: results_dir/zuna/<n>ch/summary.json
        candidate = results_dir / "zuna" / f"{n}ch" / "summary.json"
        if not candidate.exists():
            if verbose:
                print(f"  [ZUNA] No summary found for n={n} at {candidate}")
            continue
        try:
            with open(candidate) as fh:
                summary = json.load(fh)
            zuna[n] = {
                "pearson_mean_mean":    summary.get("pearson_mean_mean",    float("nan")),
                "mse_mean_mean":        summary.get("mse_mean_mean",        float("nan")),
                "beta_mean_ratio_mean": summary.get("beta_mean_ratio_mean", float("nan")),
                "_full": summary,
            }
            if verbose:
                print(f"  [ZUNA] Loaded n={n}  r={zuna[n]['pearson_mean_mean']:.4f}")
        except Exception as exc:
            print(f"  [ZUNA] Failed to load {candidate}: {exc}")

    return zuna


# --- Output helpers -----------------------------------------------------------

def print_scaling_table(results: Dict) -> None:
    """
    Print a formatted table of scaling results.

    Columns: n_channels | SSI r | SSI MSE | SSI beta_ratio | ZUNA r (if present)
    """
    scale_points = results.get("scale_points", [])
    ssi = results.get("ssi", {})
    zuna = results.get("zuna", None)
    pl_ssi = results.get("power_law", {}).get("ssi", {})
    pl_zuna = results.get("power_law", {}).get("zuna", None)

    has_zuna = zuna is not None and len(zuna) > 0

    # -- Header ----------------------------------------------------------------
    sep = "-" * (66 if has_zuna else 52)
    print(f"\n{sep}")
    if has_zuna:
        print(f"{'n_ch':>6}  {'SSI r':>8}  {'SSI MSE':>10}  {'SSI beta':>10}  {'ZUNA r':>8}")
    else:
        print(f"{'n_ch':>6}  {'SSI r':>8}  {'SSI MSE':>10}  {'SSI beta':>10}")
    print(sep)

    # -- Rows ------------------------------------------------------------------
    for n in scale_points:
        ssi_r     = ssi.get(n, {}).get("pearson_mean_mean",    float("nan"))
        ssi_mse   = ssi.get(n, {}).get("mse_mean_mean",        float("nan"))
        ssi_beta  = ssi.get(n, {}).get("beta_mean_ratio_mean", float("nan"))

        row = f"{n:>6}  {ssi_r:>8.4f}  {ssi_mse:>10.5f}  {ssi_beta:>11.4f}"

        if has_zuna:
            zuna_r = zuna.get(n, {}).get("pearson_mean_mean", float("nan"))
            row += f"  {zuna_r:>8.4f}"

        print(row)

    # -- Power-law summary row -------------------------------------------------
    print(sep)
    pl_line = (
        f"  Power law (SSI):  r(n) = {pl_ssi.get('a', float('nan')):.4f} "
        f"* n^{pl_ssi.get('b', float('nan')):.4f}  "
        f"(R2={pl_ssi.get('r2', float('nan')):.4f})"
    )
    print(pl_line)
    if has_zuna and pl_zuna:
        pl_line_z = (
            f"  Power law (ZUNA): r(n) = {pl_zuna.get('a', float('nan')):.4f} "
            f"* n^{pl_zuna.get('b', float('nan')):.4f}  "
            f"(R2={pl_zuna.get('r2', float('nan')):.4f})"
        )
        print(pl_line_z)
    print(sep + "\n")


# --- CLI ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expansion scaling law: SSI quality vs. input channel count."
    )
    parser.add_argument(
        "--fif_dir",
        type=str,
        default=str(ROOT / "pipeline_v2" / "data" / "fif"),
        help="Directory containing S001_raw.fif … files.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(ROOT / "pipeline_v2" / "results" / "scaling_law"),
        help="Directory for saving scaling_law.json (and loading ZUNA results).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Subject split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-subject progress.",
    )
    parser.add_argument(
        "--include_zuna",
        action="store_true",
        help=(
            "Load pre-computed ZUNA results from "
            "results_dir/zuna/<n>ch/summary.json if they exist."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    split_map = {
        "train": TRAIN_SUBJECTS,
        "test":  TEST_SUBJECTS,
        "all":   TRAIN_SUBJECTS + TEST_SUBJECTS,
    }
    subjects = split_map[args.split]

    fif_dir     = Path(args.fif_dir)
    results_dir = Path(args.results_dir)

    print(f"\nExpansion Scaling Law Experiment")
    print(f"  Split      : {args.split}  (n={len(subjects)} subjects)")
    print(f"  fif_dir    : {fif_dir}")
    print(f"  results_dir: {results_dir}")
    print(f"  Scale points: {DEFAULT_SCALE_POINTS}")
    print(f"  Targets    : {TARGET_CHANNELS}")
    if args.include_zuna:
        print(f"  ZUNA       : loading pre-computed results if available")

    results = run_scaling_experiment(
        fif_dir=fif_dir,
        subject_ids=subjects,
        scale_points=DEFAULT_SCALE_POINTS,
        verbose=args.verbose,
        results_dir=results_dir,
        include_zuna=args.include_zuna,
    )

    print_scaling_table(results)

    # -- Save results ----------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "scaling_law.json"

    # Convert int keys to strings for JSON serialisation
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    with open(out_path, "w") as fh:
        json.dump(_json_safe(results), fh, indent=2)

    print(f"Saved → {out_path}")
