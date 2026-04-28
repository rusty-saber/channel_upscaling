"""
ZUNA BES Scaling Law — BES at each input channel count n.

Computes BES for ZUNA reconstructions at n=1,2,4,8,16,32 input channels.
Reconstructed .fif files come from pipeline_v2/data/zuna_work/scaling_<n>ch/.
Targets are always C3, C4, P3, P4 (same as Device A / emotiv_epoc).

Usage
-----
    python -m pipeline_v2.experiments.run_bes_scaling_zuna

Results saved to:
    pipeline_v2/results/scaling_law/zuna/<n>ch/bes_summary.json
    pipeline_v2/results/scaling_law/zuna/<n>ch/bes_per_subject.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_v2.data.subject_split import TEST_SUBJECTS, subject_id_to_str
from pipeline_v2.eval.bes_runner    import run_bes_subject
from pipeline_v2.zuna.run_zuna_bes  import load_target_channels, get_gt_fif

import mne
mne.set_log_level("WARNING")


# ── Config ────────────────────────────────────────────────────────────────────

ZUNA_WORK_DIR = ROOT / "pipeline_v2" / "data"   / "zuna_work"
RESULTS_DIR   = ROOT / "pipeline_v2" / "results" / "scaling_law" / "zuna"

TARGET_CHANNELS = ["C3", "C4", "P3", "P4"]   # same targets for all n

N_VALUES = [1, 2, 4, 8, 16, 32]


# ── Per-n BES ─────────────────────────────────────────────────────────────────

def run_bes_at_n(n: int) -> None:
    work_dir    = ZUNA_WORK_DIR / f"scaling_{n}ch"
    out_dir     = RESULTS_DIR  / f"{n}ch"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f" BES scaling  n={n}  targets={TARGET_CHANNELS}")
    print(f"{'='*55}")

    if not work_dir.exists():
        print(f"  [SKIP] {work_dir} not found.")
        return

    per_subject: dict = {}
    bes_vals, gt_vals, pred_vals = [], [], []

    for sid_int in TEST_SUBJECTS:
        sid       = subject_id_to_str(sid_int)
        recon_fif = work_dir / f"{sid}_raw" / "4_fif_output" / f"{sid}_raw.fif"
        gt_fif    = get_gt_fif(sid_int)

        if not recon_fif.exists():
            per_subject[sid] = {"error": "recon_fif_missing"}
            continue

        if not gt_fif.exists():
            per_subject[sid] = {"error": "gt_fif_missing"}
            continue

        try:
            pred = load_target_channels(recon_fif, TARGET_CHANNELS)
        except Exception as e:
            warnings.warn(f"  {sid}: load failed — {e}")
            per_subject[sid] = {"error": str(e)}
            continue

        try:
            result = run_bes_subject(
                pred            = pred,
                gt_fif_path     = gt_fif,
                target_channels = TARGET_CHANNELS,
                fs              = 256,
                epoch_tmin      = 0.0,
                epoch_tmax      = 2.0,
            )
        except Exception as e:
            warnings.warn(f"  {sid}: BES failed — {e}")
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
            print(f"  {sid}  BES=skipped")

    n_valid = len(bes_vals)
    if n_valid == 0:
        print(f"  No valid BES results for n={n}.")
        return

    summary = {
        "n_channels":    n,
        "bes_mean":      float(np.mean(bes_vals)),
        "bes_std":       float(np.std(bes_vals)),
        "acc_gt_mean":   float(np.mean(gt_vals)),
        "acc_pred_mean": float(np.mean(pred_vals)),
        "n_subjects":    n_valid,
        "target_channels": TARGET_CHANNELS,
    }

    print(f"\n  n={n:2d}  BES={summary['bes_mean']:.3f} ± {summary['bes_std']:.3f}"
          f"  (n={n_valid} subjects)")

    with open(out_dir / "bes_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "bes_per_subject.json", "w") as f:
        json.dump(per_subject, f, indent=2)

    print(f"  Saved -> {out_dir / 'bes_summary.json'}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ZUNA BES Scaling Law")
    print(f"Targets: {TARGET_CHANNELS}")

    for n in N_VALUES:
        run_bes_at_n(n)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'n':>4}  {'BES mean':>10}  {'BES std':>9}  {'n_subj':>7}")
    print(f"{'='*55}")
    for n in N_VALUES:
        p = RESULTS_DIR / f"{n}ch" / "bes_summary.json"
        if p.exists():
            d = json.load(open(p))
            print(f"  {n:2d}    {d['bes_mean']:.3f}       ±{d['bes_std']:.3f}      {d['n_subjects']}")
        else:
            print(f"  {n:2d}    [missing]")

    print(f"{'='*55}")
    print("\nDone.")
