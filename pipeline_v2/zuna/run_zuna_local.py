"""
Run ZUNA inference locally on RTX 3050 (8 GB VRAM).

Fixes applied
-------------
- Crop each .fif to first 5 min  →  1 PT file per subject (no memory leak)
- tokens_per_batch = 1_000       →  safe for 8 GB VRAM
- diffusion_sample_steps = 10    →  fast enough, paper-validated
- Checkpoint after every subject →  restart-safe

Usage (from repo root)
----------------------
    python -m pipeline_v2.zuna.run_zuna_local

Optional flags
--------------
    --fif_dir   path/to/fif        (default: pipeline_v2/data/fif)
    --work_dir  path/to/work       (default: pipeline_v2/data/zuna_work)
    --results_dir path/to/results  (default: pipeline_v2/results/zuna)
    --crop_s    seconds to keep    (default: 300)
    --tokens    tokens_per_batch   (default: 1000)
    --steps     diffusion steps    (default: 10)
    --overwrite                    re-run completed subjects
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

from pipeline_v2.data.device_configs import INITIAL_TARGETS
from pipeline_v2.data.subject_split import TEST_SUBJECTS
from pipeline_v2.eval.metrics import compute_subject_metrics
from pipeline_v2.zuna.zuna_pipeline import (
    extract_reconstructed_channels,
    run_zuna_subject,
    summarise,
)

mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------

def crop_fif(fif_path: Path, out_dir: Path, crop_s: float = 300.0) -> Path:
    """Copy .fif cropped to first crop_s seconds into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / fif_path.name
    if dest.exists():
        return dest                        # already cropped in a previous run
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    tmax = min(crop_s, raw.times[-1])
    raw.crop(tmax=tmax)
    n_ep = int(tmax / 5)
    raw.save(str(dest), overwrite=True, verbose=False)
    del raw
    gc.collect()
    print(f"    Cropped to {tmax:.0f} s ({n_ep} epochs)")
    return dest


def run(args):
    import torch

    # ── Paths ─────────────────────────────────────────────────────────────────
    fif_dir     = Path(args.fif_dir)
    work_dir    = Path(args.work_dir)
    results_dir = Path(args.results_dir) / "emotiv_epoc"
    crop_dir    = work_dir / "cropped"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_subject_path = results_dir / "per_subject_results.json"
    summary_path     = results_dir / "summary.json"

    # ── GPU check ─────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU : {name}  ({vram:.1f} GB VRAM)")
    else:
        print("WARNING: no CUDA GPU found — running on CPU (will be very slow)")

    # ── Subject list ──────────────────────────────────────────────────────────
    test_fifs = [
        fif_dir / f"S{sid:03d}_raw.fif"
        for sid in TEST_SUBJECTS
        if (fif_dir / f"S{sid:03d}_raw.fif").exists()
    ]
    print(f"Test subjects found : {len(test_fifs)}/{len(TEST_SUBJECTS)}")
    if len(test_fifs) == 0:
        print(f"No .fif files found in {fif_dir}. Aborting.")
        return

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if per_subject_path.exists():
        with open(per_subject_path) as f:
            results = json.load(f)
        done = {k for k, v in results.items() if v.get("status") == "ok"}
        print(f"Resuming  : {len(done)} subjects already done")
    else:
        results, done = {}, set()

    if args.overwrite:
        done = set()

    remaining = [f for f in test_fifs if f.stem not in done]
    print(f"To process: {len(remaining)} subjects\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    session_start = time.time()

    for i, fif_path in enumerate(remaining, 1):
        sid = fif_path.stem
        print(f"[{i:2d}/{len(remaining)}] {sid}", flush=True)
        t0 = time.time()

        try:
            # 1. Crop .fif to first 5 min  (avoids memory leak in subprocess)
            cropped = crop_fif(fif_path, crop_dir, crop_s=args.crop_s)

            # 2. ZUNA inference
            out = run_zuna_subject(
                fif_path               = cropped,
                device_name            = "emotiv_epoc",
                work_dir               = work_dir,
                gpu_device             = 0,
                tokens_per_batch       = args.tokens,
                diffusion_sample_steps = args.steps,
                data_norm              = 10.0,
                overwrite              = True,
            )

            # 3. Evaluate
            pred, gt = extract_reconstructed_channels(
                out, cropped, INITIAL_TARGETS
            )
            metrics = compute_subject_metrics(pred, gt, INITIAL_TARGETS, fs=256)
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

        # Checkpoint to disk after every subject
        with open(per_subject_path, "w") as f:
            json.dump(results, f, indent=2)

        # Free GPU memory before next subject
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    summary = summarise(ok)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_h = (time.time() - session_start) / 3600
    print(f"\n{'='*60}")
    print(f"Completed : {len(ok)}/{len(test_fifs)} subjects in {total_h:.1f} h")
    print(f"{'='*60}")
    print(f"  pearson_mean : {summary.get('pearson_mean_mean', float('nan')):.3f}"
          f" +/- {summary.get('pearson_mean_std', float('nan')):.3f}")
    print(f"  mse_mean     : {summary.get('mse_mean_mean', float('nan')):.2e}")
    print(f"  beta_ratio   : {summary.get('beta_mean_ratio_mean', float('nan')):.4f}")
    print(f"\nResults saved to: {summary_path}")


# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="ZUNA local inference (RTX 3050)")
    p.add_argument("--fif_dir",     default=str(ROOT / "pipeline_v2/data/fif"))
    p.add_argument("--work_dir",    default=str(ROOT / "pipeline_v2/data/zuna_work"))
    p.add_argument("--results_dir", default=str(ROOT / "pipeline_v2/results/zuna"))
    p.add_argument("--crop_s",  type=float, default=300.0,
                   help="Seconds of recording to use per subject (default 300 = 5 min)")
    p.add_argument("--tokens",  type=int,   default=1_000,
                   help="tokens_per_batch for ZUNA inference (default 1000, safe for 8 GB)")
    p.add_argument("--steps",   type=int,   default=10,
                   help="diffusion_sample_steps (default 10)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if subject already done")
    return p.parse_args()


if __name__ == "__main__":
    run(_parse())
