"""
Month 2 Orchestrator — Cross-Device Zero-Shot Generalisation

Extends Month 1 by:
  1. Training the REVE supervised baseline on Device A (train subjects)
  2. Running SSI + REVE + ZUNA on all three devices (test subjects)
  3. Building the cross-device comparison table

Devices
-------
  A: emotiv_epoc   — AF3, AF4, F3, F4          (train device)
  B: muse_s        — AF7, AF8, T9, T10          (zero-shot)
  C: openbci_cyton — C3, C4, P3, P4, Fz, Cz    (zero-shot)

Key question answered
---------------------
  Can ZUNA (zero-shot, no per-device training) match REVE
  (supervised, trained on Device A data) on Devices B and C?

Usage
-----
    # Full run: train REVE + evaluate all devices
    python pipeline_v2/run_month2.py

    # Skip REVE training (load from disk if saved)
    python pipeline_v2/run_month2.py --skip_reve_train

    # Only evaluate a single device
    python pipeline_v2/run_month2.py --device emotiv_epoc

    # Only SSI (no REVE, no ZUNA) — fast sanity check
    python pipeline_v2/run_month2.py --only_ssi

    # Dry run
    python pipeline_v2/run_month2.py --dry_run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
_project_root = ROOT.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ─── Paths ────────────────────────────────────────────────────────────────────

FIF_DIR     = ROOT / "data" / "fif"
ZUNA_DIR    = ROOT / "data" / "zuna_work"
RESULTS_DIR = ROOT / "results"

TARGETS  = ["C3", "C4", "P3", "P4"]
DEVICES  = ["emotiv_epoc", "muse_s", "openbci_cyton"]
DEVICE_LABELS = {
    "emotiv_epoc":   "Device A (Emotiv EPOC)",
    "muse_s":        "Device B (Muse S)",
    "openbci_cyton": "Device C (OpenBCI Cyton)",
}


# ─── Step 1: Train REVE ──────────────────────────────────────────────────────

def step1_train_reve(device_name: str, alpha: float = 1.0):
    """
    Train REVE ridge regression on train subjects for a given device.
    Returns the fitted REVEModel.
    """
    print(f"\n{'='*60}")
    print(f"STEP 1 — Train REVE  [{DEVICE_LABELS[device_name]}]")
    print(f"{'='*60}")

    from pipeline_v2.baselines.reve_baseline import train_reve
    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS

    model = train_reve(
        fif_dir=FIF_DIR,
        train_subjects=TRAIN_SUBJECTS,
        device_name=device_name,
        target_channels=TARGETS,
        alpha=alpha,
        verbose=True,
    )
    return model


# ─── Step 2: SSI on one device ───────────────────────────────────────────────

def step2_ssi(device_name: str, split: str = "test") -> Dict:
    """Run SSI baseline for one device. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"STEP 2 — SSI  [{DEVICE_LABELS[device_name]}]")
    print(f"{'='*60}")

    from pipeline_v2.baselines.ssi_baseline import run_ssi_dataset, summarise
    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS

    subjects = {"train": TRAIN_SUBJECTS, "test": TEST_SUBJECTS,
                "all": TRAIN_SUBJECTS + TEST_SUBJECTS}[split]

    results = run_ssi_dataset(
        fif_dir=FIF_DIR,
        subject_ids=subjects,
        device_name=device_name,
        target_channels=TARGETS,
        verbose=True,
    )
    summary = summarise(results)

    out_dir = RESULTS_DIR / "ssi" / device_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"summary_{split}.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary("SSI", summary)
    return summary


# ─── Step 3: REVE evaluation on one device ───────────────────────────────────

def step3_reve(device_name: str, model, split: str = "test") -> Dict:
    """Evaluate REVE on test subjects for one device. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"STEP 3 — REVE  [{DEVICE_LABELS[device_name]}]")
    print(f"{'='*60}")

    from pipeline_v2.baselines.reve_baseline import run_reve_dataset, summarise
    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS

    subjects = {"train": TRAIN_SUBJECTS, "test": TEST_SUBJECTS,
                "all": TRAIN_SUBJECTS + TEST_SUBJECTS}[split]

    results = run_reve_dataset(
        fif_dir=FIF_DIR,
        subject_ids=subjects,
        device_name=device_name,
        model=model,
        target_channels=TARGETS,
        verbose=True,
    )
    summary = summarise(results)

    out_dir = RESULTS_DIR / "reve" / device_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"summary_{split}.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary("REVE", summary)
    return summary


# ─── Step 4: ZUNA on one device ──────────────────────────────────────────────

def step4_zuna(device_name: str, split: str = "test",
               gpu: int = 0, overwrite: bool = False) -> Dict:
    """Run ZUNA on test subjects for one device. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"STEP 4 — ZUNA  [{DEVICE_LABELS[device_name]}]")
    print(f"{'='*60}")

    from pipeline_v2.zuna.zuna_pipeline import run_zuna_dataset, summarise
    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS

    subjects = {"train": TRAIN_SUBJECTS, "test": TEST_SUBJECTS,
                "all": TRAIN_SUBJECTS + TEST_SUBJECTS}[split]

    results = run_zuna_dataset(
        fif_dir=FIF_DIR,
        subject_ids=subjects,
        device_name=device_name,
        target_channels=TARGETS,
        work_dir=ZUNA_DIR,
        gpu_device=gpu,
        overwrite=overwrite,
        verbose=True,
    )
    summary = summarise(results)

    out_dir = RESULTS_DIR / "zuna" / device_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"summary_{split}.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary("ZUNA", summary)
    return summary


# ─── Step 5: Cross-device results table ─────────────────────────────────────

def step5_results_table(all_results: Dict) -> None:
    """
    Print the cross-device comparison table and save combined JSON.
    all_results: {device: {method: summary_dict}}
    """
    print(f"\n{'='*60}")
    print("STEP 5 — Cross-Device Results Table")
    print(f"{'='*60}\n")

    metric_keys = {
        "Pearson r":   "pearson_mean_mean",
        "Beta ratio":  "beta_mean_ratio_mean",
    }

    header = f"{'Device':<20} {'Method':<8}"
    for label in metric_keys:
        header += f"  {label:>12}"
    print(header)
    print("-" * len(header))

    for device, methods in all_results.items():
        label = DEVICE_LABELS.get(device, device)
        for method, summary in methods.items():
            if not summary:
                continue
            row = f"{label:<20} {method:<8}"
            for key in metric_keys.values():
                val = summary.get(key, float("nan"))
                row += f"  {val:>12.3f}"
            print(row)
        print()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "month2_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results -> {out_path}")


# ─── Helper ──────────────────────────────────────────────────────────────────

def _print_summary(method: str, summary: Dict) -> None:
    r   = summary.get("pearson_mean_mean", float("nan"))
    mse = summary.get("mse_mean_mean",     float("nan"))
    b   = summary.get("beta_mean_ratio_mean", float("nan"))
    print(f"\n  {method} results:")
    print(f"    Pearson r  : {r:.3f}")
    print(f"    MSE        : {mse:.3e} V^2")
    print(f"    Beta ratio : {b:.3f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Month 2: cross-device zero-shot evaluation"
    )
    parser.add_argument("--device", type=str, default=None,
                        choices=DEVICES,
                        help="Evaluate only this device (default: all three)")
    parser.add_argument("--split",  type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--alpha",  type=float, default=1.0,
                        help="Ridge regression regularisation for REVE")
    parser.add_argument("--only_ssi",    action="store_true",
                        help="Run only SSI (skip REVE and ZUNA)")
    parser.add_argument("--skip_zuna",   action="store_true",
                        help="Skip ZUNA inference (useful if no GPU)")
    parser.add_argument("--skip_reve_train", action="store_true",
                        help="Skip REVE training (load from disk if available)")
    parser.add_argument("--overwrite",   action="store_true")
    parser.add_argument("--dry_run",     action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    devices = [args.device] if args.device else DEVICES

    if args.dry_run:
        print("\n[DRY RUN] Steps that would execute:")
        for d in devices:
            print(f"\n  {DEVICE_LABELS[d]}:")
            print(f"    - SSI baseline")
            if not args.only_ssi:
                if not args.skip_reve_train:
                    print(f"    - Train REVE (cross-subject ridge regression)")
                print(f"    - REVE evaluation")
            if not args.only_ssi and not args.skip_zuna:
                print(f"    - ZUNA inference (GPU {args.gpu})")
        return

    all_results: Dict[str, Dict] = {}
    reve_model = None

    # Train REVE once (on Device A layout) — used for all devices
    if not args.only_ssi and not args.skip_reve_train:
        reve_model = step1_train_reve("emotiv_epoc", alpha=args.alpha)

    for device in devices:
        all_results[device] = {}

        # SSI
        ssi_summary = step2_ssi(device, split=args.split)
        all_results[device]["SSI"] = ssi_summary

        if args.only_ssi:
            continue

        # REVE (needs model trained on Device A; different model per device ideally
        # but for now Device A model is the cross-device baseline)
        if reve_model is not None:
            reve_summary = step3_reve(device, reve_model, split=args.split)
            all_results[device]["REVE"] = reve_summary

        # ZUNA
        if not args.skip_zuna:
            zuna_summary = step4_zuna(
                device, split=args.split,
                gpu=args.gpu, overwrite=args.overwrite
            )
            all_results[device]["ZUNA"] = zuna_summary

    step5_results_table(all_results)
    print("\n[DONE] Month 2 pipeline complete.")


if __name__ == "__main__":
    main()
