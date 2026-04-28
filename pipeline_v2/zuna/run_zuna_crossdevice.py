"""
Run ZUNA zero-shot inference on Device B (Muse S) and Device C (OpenBCI Cyton).

This script is the follow-up to run_zuna_local.py (which did Device A / Emotiv EPOC).
It assumes the .fif files are already present in fif_dir.

Tasks
-----
1. Device B — Muse S   : inputs AF7, AF8, T9, T10  → targets C3, C4, P3, P4
2. Device C — OpenBCI  : inputs C3, C4, P3, P4, Fz, Cz → targets T7, T8, FC5, FC6

Usage (from repo root)
----------------------
    python -m pipeline_v2.zuna.run_zuna_crossdevice

Optional flags
--------------
    --fif_dir   path/to/fif          (default: pipeline_v2/data/fif)
    --work_dir  path/to/work         (default: pipeline_v2/data/zuna_work)
    --results_dir path/to/results    (default: pipeline_v2/results/zuna)
    --device    muse_s | openbci_cyton | all   (default: all)
    --tokens    tokens_per_batch     (default: 1000, safe for 8 GB VRAM)
    --steps     diffusion steps      (default: 10)
    --overwrite                      re-run completed subjects

Memory note
-----------
Same settings as run_zuna_local.py:
  - Crop each .fif to first 5 minutes to prevent GPU memory leak
  - tokens_per_batch = 1000  (safe for 8 GB VRAM)
  - diffusion_sample_steps = 10  (paper-validated, 5x faster than default)
"""

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline_v2.data.device_configs import DEVICE_CONFIGS
from pipeline_v2.data.subject_split import TEST_SUBJECTS
from pipeline_v2.eval.metrics import compute_subject_metrics
from pipeline_v2.zuna.zuna_pipeline import (
    extract_reconstructed_channels,
    run_zuna_subject,
    summarise,
)

mne.set_log_level("WARNING")

CROSS_DEVICE_TARGETS = {
    "muse_s":         DEVICE_CONFIGS["muse_s"]["eval_targets"],
    "openbci_cyton":  DEVICE_CONFIGS["openbci_cyton"]["eval_targets"],
}


# ---------------------------------------------------------------------------

def crop_fif(fif_path: Path, out_dir: Path, crop_s: float = 300.0) -> Path:
    """Crop .fif to first crop_s seconds and save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / fif_path.name
    if dest.exists():
        return dest
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    tmax = min(crop_s, raw.times[-1])
    raw.crop(tmax=tmax)
    raw.save(str(dest), overwrite=True, verbose=False)
    del raw
    gc.collect()
    print(f"    Cropped to {tmax:.0f} s")
    return dest


def run_device(device_name: str, args) -> None:
    import torch

    cfg        = DEVICE_CONFIGS[device_name]
    input_chs  = cfg["input_channels"]
    target_chs = cfg["eval_targets"]

    fif_dir     = Path(args.fif_dir)
    work_dir    = Path(args.work_dir)
    results_dir = Path(args.results_dir) / device_name
    crop_dir    = work_dir / "cropped"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_subject_path = results_dir / "per_subject_results.json"
    summary_path     = results_dir / "summary.json"

    print(f"\n{'='*60}")
    print(f"DEVICE : {device_name}")
    print(f"inputs : {input_chs}")
    print(f"targets: {target_chs}")
    print(f"{'='*60}")

    # GPU check
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({vram:.1f} GB VRAM)")
    else:
        print("WARNING: no CUDA GPU found — running on CPU (will be slow)")

    # Subject list
    test_fifs = [
        fif_dir / f"S{sid:03d}_raw.fif"
        for sid in TEST_SUBJECTS
        if (fif_dir / f"S{sid:03d}_raw.fif").exists()
    ]
    print(f"Test subjects found: {len(test_fifs)}/{len(TEST_SUBJECTS)}")

    # Resume
    if per_subject_path.exists():
        with open(per_subject_path) as f:
            results = json.load(f)
        done = {k for k, v in results.items() if v.get("status") == "ok"}
        print(f"Resuming: {len(done)} subjects already done")
    else:
        results, done = {}, set()

    if args.overwrite:
        done = set()

    remaining = [f for f in test_fifs if f.stem not in done]
    print(f"To process: {len(remaining)} subjects\n")

    session_start = time.time()

    for i, fif_path in enumerate(remaining, 1):
        sid = fif_path.stem
        print(f"[{i:2d}/{len(remaining)}] {sid}", flush=True)
        t0 = time.time()

        try:
            # Crop to 5 min
            cropped = crop_fif(fif_path, crop_dir, crop_s=args.crop_s)

            # ZUNA inference
            out = run_zuna_subject(
                fif_path               = cropped,
                device_name            = device_name,
                work_dir               = work_dir,
                gpu_device             = 0,
                tokens_per_batch       = args.tokens,
                diffusion_sample_steps = args.steps,
                data_norm              = 10.0,
                overwrite              = True,
            )

            # Evaluate
            pred, gt = extract_reconstructed_channels(out, cropped, target_chs)
            metrics  = compute_subject_metrics(pred, gt, target_chs, fs=256)
            dt = (time.time() - t0) / 60

            results[sid] = {"status": "ok", "metrics": metrics, "time_min": dt}
            print(
                f"    OK  r={metrics['pearson_mean']:.3f}  "
                f"MSE={metrics['mse_mean']:.2e}  ({dt:.1f} min)",
                flush=True,
            )

        except Exception as e:
            dt = (time.time() - t0) / 60
            results[sid] = {"status": "fail", "error": str(e), "time_min": dt}
            print(f"    FAIL ({dt:.1f} min): {e}", flush=True)
            traceback.print_exc()

        # Checkpoint
        with open(per_subject_path, "w") as f:
            json.dump(results, f, indent=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    summary = summarise(ok)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_h = (time.time() - session_start) / 3600
    print(f"\n{'='*60}")
    print(f"Completed: {len(ok)}/{len(test_fifs)} subjects in {total_h:.2f} h")
    print(f"  pearson_mean: {summary.get('pearson_mean_mean', float('nan')):.3f}"
          f" +/- {summary.get('pearson_mean_std', float('nan')):.3f}")
    print(f"  mse_mean    : {summary.get('mse_mean_mean', float('nan')):.2e}")
    print(f"  beta_ratio  : {summary.get('beta_mean_ratio_mean', float('nan')):.4f}")
    print(f"\nResults saved to: {summary_path}")


# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="ZUNA zero-shot cross-device inference (Muse S + OpenBCI Cyton)"
    )
    p.add_argument("--fif_dir",
                   default=str(ROOT / "pipeline_v2/data/fif"))
    p.add_argument("--work_dir",
                   default=str(ROOT / "pipeline_v2/data/zuna_work"))
    p.add_argument("--results_dir",
                   default=str(ROOT / "pipeline_v2/results/zuna"))
    p.add_argument("--device", default="all",
                   choices=["muse_s", "openbci_cyton", "all"],
                   help="Which device to run (default: all)")
    p.add_argument("--crop_s", type=float, default=300.0,
                   help="Seconds to keep per subject (default 300 = 5 min)")
    p.add_argument("--tokens", type=int, default=1_000,
                   help="tokens_per_batch (default 1000, safe for 8 GB VRAM)")
    p.add_argument("--steps", type=int, default=10,
                   help="diffusion_sample_steps (default 10)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if subject already done")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    devices = (
        ["muse_s", "openbci_cyton"]
        if args.device == "all"
        else [args.device]
    )

    for device in devices:
        run_device(device, args)

    # Final summary table
    print("\n\n" + "=" * 60)
    print("ZERO-SHOT RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Device':<20} {'r':>6} {'MSE':>12} {'Beta':>8}")
    print("-" * 50)
    for device in devices:
        p = Path(args.results_dir) / device / "summary.json"
        if p.exists():
            d = json.load(open(p))
            r    = d.get("pearson_mean_mean", float("nan"))
            mse  = d.get("mse_mean_mean", float("nan"))
            beta = d.get("beta_mean_ratio_mean", float("nan"))
            print(f"{device:<20} {r:>6.3f} {mse:>12.1f} {beta:>8.4f}")
        else:
            print(f"{device:<20} {'N/A':>6}")
