"""
ZUNA Scaling Law — run ZUNA at n ∈ {1, 2, 4, 8, 16, 32} input channels.

Run this on the GPU machine (friend's RTX 3050) AFTER run_zuna_crossdevice.py.

Each scale point uses a different nested input channel set (frontal-only,
progressively denser) and evaluates against the same 4 targets: C3, C4, P3, P4.

Results are saved to:
    pipeline_v2/results/scaling_law/zuna/<n>ch/summary.json

These files are then loaded by scaling_law_full.json and scaling_law_figure.py
to add the ZUNA curve to Figure 1.

Usage (from repo root)
----------------------
    python -m pipeline_v2.experiments.run_scaling_zuna

Optional flags
--------------
    --fif_dir       path/to/fif
    --work_dir      path/to/zuna_work
    --results_dir   path/to/scaling_law/results   (default: pipeline_v2/results/scaling_law)
    --tokens        tokens_per_batch               (default: 1000)
    --steps         diffusion_sample_steps         (default: 10)
    --n             specific n to run (default: all)
    --overwrite

Expected runtime
----------------
    ~29 min per subject x 22 subjects x 6 scale points = ~64 hours total.
    Run one scale point at a time: use --n 4 first (fastest validation),
    then --n 1, --n 2, --n 8, --n 16, --n 32 on subsequent nights.

Memory note
-----------
    Same settings as run_zuna_local.py.
    Crop to 5 min + tokens=1000 = safe for 8 GB VRAM.
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

from pipeline_v2.data.subject_split import TEST_SUBJECTS
from pipeline_v2.eval.metrics import compute_subject_metrics
from pipeline_v2.zuna.zuna_pipeline import (
    extract_reconstructed_channels,
    run_zuna_subject,
    summarise,
)

mne.set_log_level("WARNING")

NESTED_INPUT_SETS = {
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


def crop_fif(fif_path: Path, out_dir: Path, crop_s: float = 300.0) -> Path:
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


def run_at_n(n: int, args) -> None:
    import torch

    input_chs   = NESTED_INPUT_SETS[n]
    fif_dir     = Path(args.fif_dir)
    work_dir    = Path(args.work_dir)
    crop_dir    = work_dir / "cropped"
    results_dir = Path(args.results_dir) / "zuna" / f"{n}ch"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_subject_path = results_dir / "per_subject_results.json"
    summary_path     = results_dir / "summary.json"

    if summary_path.exists() and not args.overwrite:
        print(f"  [SKIP] n={n} already done ({summary_path})")
        return

    print(f"\n{'='*60}")
    print(f"ZUNA scaling law  n={n}  inputs={input_chs}")
    print(f"targets           : {TARGET_CHANNELS}")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name}  ({vram:.1f} GB VRAM)")

    test_fifs = [
        fif_dir / f"S{sid:03d}_raw.fif"
        for sid in TEST_SUBJECTS
        if (fif_dir / f"S{sid:03d}_raw.fif").exists()
    ]

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
            cropped = crop_fif(fif_path, crop_dir, crop_s=args.crop_s)

            # Use a temporary device name hack: create an inline config
            # that maps our custom input_chs as the device's input channels.
            # We use the zuna_pipeline's bad_channels mechanism directly.
            import mne as _mne
            _info = _mne.io.read_info(str(cropped), verbose=False)
            _all_chs = _info["ch_names"]
            _bad_chs = [ch for ch in _all_chs if ch not in set(input_chs)]

            from zuna import preprocessing, inference, pt_to_fif
            import shutil

            subject_stem = cropped.stem
            subject_work = work_dir / f"scaling_{n}ch" / subject_stem
            fif_in_dir   = subject_work / "1_fif_input"
            fif_filt_dir = subject_work / "1_fif_filtered"
            pt_in_dir    = subject_work / "2_pt_input"
            pt_out_dir   = subject_work / "3_pt_output"
            fif_out_dir  = subject_work / "4_fif_output"
            for d in [fif_in_dir, fif_filt_dir, pt_in_dir, pt_out_dir, fif_out_dir]:
                d.mkdir(parents=True, exist_ok=True)

            shutil.copy2(str(cropped), str(fif_in_dir / cropped.name))

            preprocessing(
                input_dir=str(fif_in_dir),
                output_dir=str(pt_in_dir),
                apply_notch_filter=False,
                apply_highpass_filter=False,
                apply_average_reference=False,
                bad_channels=_bad_chs,
                preprocessed_fif_dir=str(fif_filt_dir),
            )
            inference(
                input_dir=str(pt_in_dir),
                output_dir=str(pt_out_dir),
                gpu_device=0,
                tokens_per_batch=args.tokens,
                data_norm=10.0,
                diffusion_cfg=1.0,
                diffusion_sample_steps=args.steps,
                plot_eeg_signal_samples=False,
            )
            pt_to_fif(
                input_dir=str(pt_out_dir),
                output_dir=str(fif_out_dir),
            )

            out_fifs = sorted(fif_out_dir.glob("*.fif"))
            if not out_fifs:
                raise RuntimeError("ZUNA produced no .fif output")
            out_fif = out_fifs[0]

            pred, gt = extract_reconstructed_channels(out_fif, cropped, TARGET_CHANNELS)
            metrics  = compute_subject_metrics(pred, gt, TARGET_CHANNELS, fs=256)
            dt = (time.time() - t0) / 60

            results[sid] = {"status": "ok", "metrics": metrics, "time_min": dt}
            print(
                f"    OK  r={metrics['pearson_mean']:.3f}  ({dt:.1f} min)",
                flush=True,
            )

        except Exception as e:
            dt = (time.time() - t0) / 60
            results[sid] = {"status": "fail", "error": str(e), "time_min": dt}
            print(f"    FAIL ({dt:.1f} min): {e}", flush=True)
            traceback.print_exc()

        with open(per_subject_path, "w") as f:
            json.dump(results, f, indent=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    summary = summarise(ok)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_h = (time.time() - session_start) / 3600
    print(f"\nCompleted {len(ok)}/{len(test_fifs)} subjects in {total_h:.2f} h")
    print(f"  r = {summary.get('pearson_mean_mean', float('nan')):.3f} +/- "
          f"{summary.get('pearson_mean_std', float('nan')):.3f}")
    print(f"Saved -> {summary_path}")


def _parse():
    p = argparse.ArgumentParser(
        description="ZUNA scaling law: one scale point at a time"
    )
    p.add_argument("--fif_dir",
                   default=str(ROOT / "pipeline_v2/data/fif"))
    p.add_argument("--work_dir",
                   default=str(ROOT / "pipeline_v2/data/zuna_work"))
    p.add_argument("--results_dir",
                   default=str(ROOT / "pipeline_v2/results/scaling_law"))
    p.add_argument("--n", type=int, default=None,
                   help="Which scale point to run (1/2/4/8/16/32). Default: all.")
    p.add_argument("--crop_s", type=float, default=300.0)
    p.add_argument("--tokens", type=int, default=1_000)
    p.add_argument("--steps",  type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    scale_points = [args.n] if args.n is not None else [1, 2, 4, 8, 16, 32]
    for n in scale_points:
        if n not in NESTED_INPUT_SETS:
            print(f"[WARN] n={n} not defined in NESTED_INPUT_SETS. Skipping.")
            continue
        run_at_n(n, args)
