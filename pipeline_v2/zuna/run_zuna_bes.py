"""
ZUNA BES — BCI Equivalence Score for ZUNA zero-shot device reconstructions.

Loads reconstructed .fif files from pipeline_v2/data/zuna_work/ and computes
BES against ground-truth .fif files from pipeline_v2/data/fif/.

Usage
-----
    # All devices (default):
    python -m pipeline_v2.zuna.run_zuna_bes

    # Single device:
    python -m pipeline_v2.zuna.run_zuna_bes --device muse_s
    python -m pipeline_v2.zuna.run_zuna_bes --device openbci_cyton
    python -m pipeline_v2.zuna.run_zuna_bes --device emotiv_epoc

Results saved to:
    pipeline_v2/results/zuna/<device>/bes_summary.json
    pipeline_v2/results/zuna/<device>/bes_per_subject.json

Note on emotiv_epoc:
    The original Device A .fif files were not retained after the initial run.
    We use scaling_4ch reconstructions (AF3, AF4, F3, F4 → C3, C4, P3, P4)
    as a proxy — identical input/target channels, 20/22 subjects available.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_v2.data.device_configs  import DEVICE_CONFIGS
from pipeline_v2.data.subject_split   import TEST_SUBJECTS, subject_id_to_str
from pipeline_v2.eval.bes_runner      import run_bes_subject

import mne
mne.set_log_level("WARNING")


# ── Paths ─────────────────────────────────────────────────────────────────────

ZUNA_WORK_DIR = ROOT / "pipeline_v2" / "data"  / "zuna_work"
FIF_DIR       = ROOT / "pipeline_v2" / "data"  / "fif"
RESULTS_DIR   = ROOT / "pipeline_v2" / "results" / "zuna"


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_recon_fif(device: str, subject_id: int) -> Path:
    """
    Return the path to the ZUNA-reconstructed .fif for a subject.
    Pattern: zuna_work/<device>/S0XX_raw/4_fif_output/S0XX_raw.fif

    emotiv_epoc uses scaling_4ch as a proxy (same input/target channels).
    """
    sid = subject_id_to_str(subject_id)  # e.g. "S088"
    folder = "emotiv_epoc_scaling4" if device == "emotiv_epoc" else device
    return ZUNA_WORK_DIR / folder / f"{sid}_raw" / "4_fif_output" / f"{sid}_raw.fif"


def get_gt_fif(subject_id: int) -> Path:
    """
    Return the ground-truth .fif path for a subject.
    Pattern: data/fif/S0XX_raw.fif
    """
    sid = subject_id_to_str(subject_id)
    return FIF_DIR / f"{sid}_raw.fif"


def load_target_channels(fif_path: Path, target_channels: list) -> np.ndarray:
    """
    Load a .fif and return the target channels as (n_targets, n_samples).
    """
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    picks = mne.pick_channels(raw.ch_names, include=target_channels, ordered=True)
    if len(picks) != len(target_channels):
        found = [raw.ch_names[p] for p in picks]
        missing = [c for c in target_channels if c not in found]
        raise ValueError(
            f"{fif_path.name}: target channels not found: {missing}\n"
            f"Available: {raw.ch_names[:10]}..."
        )
    return raw.get_data(picks=picks)   # (n_targets, n_samples)


# ── Per-device BES runner ─────────────────────────────────────────────────────

def run_device_bes(device: str) -> None:
    """
    Compute BES for all 22 test subjects on one device and save results.
    """
    cfg            = DEVICE_CONFIGS[device]
    target_channels = cfg["eval_targets"]

    print(f"\n{'='*55}")
    print(f" ZUNA BES — {device}  ({cfg['description']})")
    print(f" Targets: {target_channels}")
    print(f"{'='*55}")

    per_subject: dict = {}
    bes_vals:    list = []
    gt_vals:     list = []
    pred_vals:   list = []

    for sid_int in TEST_SUBJECTS:
        sid = subject_id_to_str(sid_int)

        recon_fif = get_recon_fif(device, sid_int)
        gt_fif    = get_gt_fif(sid_int)

        # ── Check files exist ──────────────────────────────────────────────────
        if not recon_fif.exists():
            warnings.warn(f"  {sid}: reconstructed .fif not found — {recon_fif}")
            per_subject[sid] = {"error": "recon_fif_missing"}
            continue

        if not gt_fif.exists():
            warnings.warn(f"  {sid}: ground-truth .fif not found — {gt_fif}")
            per_subject[sid] = {"error": "gt_fif_missing"}
            continue

        # ── Load reconstructed target channels ─────────────────────────────────
        try:
            pred = load_target_channels(recon_fif, target_channels)
        except Exception as e:
            warnings.warn(f"  {sid}: failed to load recon .fif — {e}")
            per_subject[sid] = {"error": str(e)}
            continue

        # ── Compute BES ────────────────────────────────────────────────────────
        try:
            result = run_bes_subject(
                pred            = pred,
                gt_fif_path     = gt_fif,
                target_channels = target_channels,
                fs              = 256,
                epoch_tmin      = 0.0,
                epoch_tmax      = 2.0,
            )
        except Exception as e:
            warnings.warn(f"  {sid}: BES computation failed — {e}")
            per_subject[sid] = {"error": str(e)}
            continue

        if result:
            bes_vals.append(result["bes"])
            gt_vals.append(result["acc_gt"])
            pred_vals.append(result["acc_pred"])
            per_subject[sid] = result
            print(
                f"  {sid}  BES={result['bes']:.3f}  "
                f"acc_gt={result['acc_gt']:.3f}  "
                f"acc_pred={result['acc_pred']:.3f}"
            )
        else:
            per_subject[sid] = {"error": "insufficient_epochs"}
            print(f"  {sid}  BES=skipped (insufficient epochs)")

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(bes_vals)
    if n == 0:
        print("\n  No valid BES results. Check errors above.")
        return

    summary = {
        "bes_mean":      float(np.mean(bes_vals)),
        "bes_std":       float(np.std(bes_vals)),
        "acc_gt_mean":   float(np.mean(gt_vals)),
        "acc_pred_mean": float(np.mean(pred_vals)),
        "n_subjects":    n,
        "device":        device,
        "target_channels": target_channels,
    }

    print(f"\n  BES  = {summary['bes_mean']:.3f} ± {summary['bes_std']:.3f}  (n={n})")
    print(f"  A_gt = {summary['acc_gt_mean']:.3f}")
    print(f"  A_pred = {summary['acc_pred_mean']:.3f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir = RESULTS_DIR / device
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "bes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "bes_per_subject.json", "w") as f:
        json.dump(per_subject, f, indent=2)

    print(f"\n  Saved -> {out_dir / 'bes_summary.json'}")
    print(f"  Saved -> {out_dir / 'bes_per_subject.json'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Compute ZUNA BES for zero-shot devices.")
    p.add_argument(
        "--device",
        choices=["emotiv_epoc", "muse_s", "openbci_cyton", "all"],
        default="all",
        help="Which device to evaluate (default: all).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    devices = (
        ["emotiv_epoc", "muse_s", "openbci_cyton"]
        if args.device == "all"
        else [args.device]
    )

    for device in devices:
        run_device_bes(device)

    print("\nDone.")
