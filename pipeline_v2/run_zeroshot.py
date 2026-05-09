"""
Zero-shot evaluation runner — Tasks 1, 2, 3.

Runs SSI and REVE on all three devices (emotiv_epoc, muse_s, openbci_cyton),
computes BES for each method × device combination.

Usage (from repo root):
    python -m pipeline_v2.run_zeroshot
"""

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline_v2.baselines.reve_baseline import REVEModel, summarise as reve_summarise
from pipeline_v2.baselines.ssi_baseline  import run_ssi_subject, summarise as ssi_summarise
from pipeline_v2.data.device_configs     import DEVICE_CONFIGS
from pipeline_v2.data.subject_split      import TEST_SUBJECTS, TRAIN_SUBJECTS
from pipeline_v2.eval.metrics            import compute_subject_metrics
from pipeline_v2.eval.bes_runner         import run_bes_subject

FIF_DIR      = ROOT / "pipeline_v2" / "data" / "fif"
RESULTS_ROOT = ROOT / "pipeline_v2" / "results"

DEVICES = ["emotiv_epoc", "muse_s", "openbci_cyton"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path.relative_to(ROOT)}")


def _bes(pred: np.ndarray, fif_path: Path, target_chs: list) -> float:
    try:
        result = run_bes_subject(pred, fif_path, target_chs)
        return result.get("bes", float("nan"))
    except Exception as e:
        print(f"      BES error: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# SSI runner
# ---------------------------------------------------------------------------

def run_ssi_device(device_name: str):
    cfg        = DEVICE_CONFIGS[device_name]
    input_chs  = cfg["input_channels"]
    target_chs = cfg["eval_targets"]

    # Skip if already done
    out_path = RESULTS_ROOT / "ssi" / device_name / "summary.json"
    if out_path.exists():
        print(f"  [SKIP] SSI {device_name} already done")
        return

    print(f"\n[SSI] {device_name}")
    print(f"  inputs  : {input_chs}")
    print(f"  targets : {target_chs}")
    t0 = time.time()

    results = {}
    for sid in TEST_SUBJECTS:
        fif_path = FIF_DIR / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            continue
        try:
            pred, gt = run_ssi_subject(fif_path, input_chs, target_chs)
            metrics  = compute_subject_metrics(pred, gt, target_chs, fs=256)
            bes      = _bes(pred, fif_path, target_chs)
            metrics["bes"] = bes
            results[f"S{sid:03d}"] = {"metrics": metrics}
            print(f"  S{sid:03d}  r={metrics['pearson_mean']:.3f}  "
                  f"beta={metrics.get('beta_mean_ratio', float('nan')):.3f}  BES={bes:.3f}")
        except Exception as e:
            print(f"  S{sid:03d} FAIL: {e}")
            traceback.print_exc()

    summary = ssi_summarise({"k": {"metrics": v["metrics"]} for k, v in results.items()}
                             if results else {})
    # Re-do summary correctly
    summary = _summarise_results(results)
    _save(summary, out_path)
    print(f"  Done in {(time.time()-t0)/60:.1f} min | "
          f"r={summary.get('pearson_mean_mean', float('nan')):.3f}  "
          f"BES={summary.get('bes_mean', float('nan')):.3f}")


# ---------------------------------------------------------------------------
# REVE runner
# ---------------------------------------------------------------------------

def run_reve_device(device_name: str):
    cfg        = DEVICE_CONFIGS[device_name]
    input_chs  = cfg["input_channels"]
    target_chs = cfg["eval_targets"]

    out_path = RESULTS_ROOT / "reve" / device_name / "summary.json"
    if out_path.exists():
        print(f"  [SKIP] REVE {device_name} already done")
        return

    print(f"\n[REVE] {device_name}")
    print(f"  inputs  : {input_chs}")
    print(f"  targets : {target_chs}")
    t0 = time.time()

    # Train
    train_paths = [FIF_DIR / f"S{sid:03d}_raw.fif"
                   for sid in TRAIN_SUBJECTS
                   if (FIF_DIR / f"S{sid:03d}_raw.fif").exists()]
    print(f"  Training on {len(train_paths)} subjects...")
    model = REVEModel(alpha=1.0)
    model.fit(train_paths, input_chs, target_chs, verbose=False)
    print(f"  Training done ({(time.time()-t0)/60:.1f} min)")

    # Evaluate
    results = {}
    for sid in TEST_SUBJECTS:
        fif_path = FIF_DIR / f"S{sid:03d}_raw.fif"
        if not fif_path.exists():
            continue
        try:
            pred, gt = model.predict(fif_path, input_chs, target_chs)
            metrics  = compute_subject_metrics(pred, gt, target_chs, fs=256)
            bes      = _bes(pred, fif_path, target_chs)
            metrics["bes"] = bes
            results[f"S{sid:03d}"] = {"metrics": metrics}
            print(f"  S{sid:03d}  r={metrics['pearson_mean']:.3f}  "
                  f"beta={metrics.get('beta_mean_ratio', float('nan')):.3f}  BES={bes:.3f}")
        except Exception as e:
            print(f"  S{sid:03d} FAIL: {e}")
            traceback.print_exc()

    summary = _summarise_results(results)
    _save(summary, out_path)
    print(f"  Done in {(time.time()-t0)/60:.1f} min | "
          f"r={summary.get('pearson_mean_mean', float('nan')):.3f}  "
          f"BES={summary.get('bes_mean', float('nan')):.3f}")


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------

def _summarise_results(results: dict) -> dict:
    if not results:
        return {}
    all_metrics: dict = {}
    for v in results.values():
        for k, val in v["metrics"].items():
            if isinstance(val, (int, float)):
                all_metrics.setdefault(k, []).append(float(val))
    summary = {}
    for k, vals in all_metrics.items():
        arr = np.array(vals)
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"]  = float(arr.std())
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Zero-shot evaluation: SSI + REVE on all 3 devices")
    print("=" * 60)

    for device in DEVICES:
        print(f"\n{'='*60}")
        print(f"DEVICE: {device}")
        print(f"{'='*60}")
        run_ssi_device(device)
        run_reve_device(device)

    # Print final summary table
    print("\n\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Device':<20} {'Method':<8} {'r':>6} {'BES':>6}")
    print("-" * 45)
    for device in DEVICES:
        for method in ["ssi", "reve"]:
            p = RESULTS_ROOT / method / device / "summary.json"
            if p.exists():
                d = json.load(open(p))
                r   = d.get("pearson_mean_mean", float("nan"))
                bes = d.get("bes_mean", float("nan"))
                print(f"{device:<20} {method:<8} {r:>6.3f} {bes:>6.3f}")
            else:
                print(f"{device:<20} {method:<8} {'N/A':>6} {'N/A':>6}")
