"""
Month 1 Orchestrator — Computational EEG Densification

Runs the four steps of Month 1 in order.  Each step has a shape/sanity gate
that must pass before the next step begins.

Steps
-----
    1. Data preparation
       Download all 109 PhysioNet EEGMMIDB subjects, resample to 256 Hz,
       set standard_1005 montage, save as .fif.
       Gate: >= 87 train .fif + 22 test .fif exist, all are 64 ch @ 256 Hz.

    2. SSI baseline  (Device A: Emotiv EPOC — AF3/AF4/F3/F4)
       Spherical spline interpolation of C3/C4/P3/P4 from only the 4 frontal
       channels.  This is the wall everything must beat.
       Gate: outputs MSE and Pearson r; logs to results/ssi/.

    3. ZUNA inference  (Device A, test set)
       Run ZUNA on all 22 test subjects.
       Gate: outputs .fif for every test subject in zuna_work/.

    4. Side-by-side comparison
       Print the results table:  SSI | ZUNA
       Save to results/month1_summary.json.

Usage
-----
    # Full run (steps 1-4)
    python pipeline_v2/run_month1.py

    # Skip data preparation if .fif files already exist
    python pipeline_v2/run_month1.py --skip_download

    # Only run baseline (step 2) — useful for quick iteration
    python pipeline_v2/run_month1.py --only_baseline

    # Only run ZUNA (step 3+4) — assumes .fif files exist
    python pipeline_v2/run_month1.py --only_zuna

    # Dry-run: check what would be done without executing
    python pipeline_v2/run_month1.py --dry_run
"""

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows (avoids CP1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent

# ── Make `pipeline_v2` importable regardless of how this script is invoked ───
# When run as  `python pipeline_v2/run_month1.py`  Python adds pipeline_v2/ to
# sys.path, not the project root.  We need the project root (parent of pipeline_v2)
# so that `from pipeline_v2.xxx import ...` resolves correctly.
_project_root = ROOT.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ─── Paths (mirrors config.yaml) ─────────────────────────────────────────────

RAW_DIR     = ROOT / "data" / "raw"
FIF_DIR     = ROOT / "data" / "fif"
ZUNA_DIR    = ROOT / "data" / "zuna_work"
RESULTS_DIR = ROOT / "results"

DEVICE     = "emotiv_epoc"    # Month 1: Device A only
TARGETS    = ["C3", "C4", "P3", "P4"]


# ─── Step 1: Data preparation ─────────────────────────────────────────────────

def step1_prepare_data(overwrite: bool = False, quick: bool = False) -> bool:
    """
    Download + preprocess PhysioNet EEGMMIDB → .fif files.

    Parameters
    ----------
    quick : if True, process only 5 subjects (for debugging)

    Returns True if gate passes.
    """
    print("\n" + "═" * 60)
    print("STEP 1 — Data Preparation")
    print("═" * 60)
    print(f"  Target: 64 ch × N samples @ 256 Hz, standard_1005 montage")
    print(f"  Output: {FIF_DIR}")

    from pipeline_v2.data.download_eegmmidb import process_all
    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS

    subjects = (TRAIN_SUBJECTS + TEST_SUBJECTS)[:5] if quick else None
    process_all(subjects=subjects, fif_dir=FIF_DIR, raw_dir=RAW_DIR,
                overwrite=overwrite, cleanup_raw=True)

    return _gate_step1(quick)


def _gate_step1(quick: bool = False) -> bool:
    """
    Shape gate for step 1.
    Checks every existing .fif: 64 ch, 256 Hz, montage positions set.
    Returns True if enough files pass.
    """
    import mne
    mne.set_log_level("WARNING")

    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS

    fif_files = sorted(FIF_DIR.glob("S*_raw.fif"))
    expected = 5 if quick else (len(TRAIN_SUBJECTS) + len(TEST_SUBJECTS))

    n_ok, n_fail = 0, 0
    for f in fif_files:
        try:
            info = mne.io.read_info(str(f), verbose=False)
            assert len(info["ch_names"]) == 64,  f"Expected 64 ch, got {len(info['ch_names'])}"
            assert info["sfreq"]         == 256, f"Expected 256 Hz, got {info['sfreq']}"
            # Check montage positions exist
            pos = [
                info["chs"][i]["loc"][:3]
                for i in range(len(info["ch_names"]))
            ]
            n_with_pos = sum(1 for p in pos if not all(v == 0 for v in p))
            assert n_with_pos >= 60, f"Too few channels with 3D positions: {n_with_pos}"
            n_ok += 1
        except Exception as e:
            print(f"  [GATE FAIL] {f.name}: {e}")
            n_fail += 1

    print(f"\n  Gate 1: {n_ok}/{len(fif_files)} files pass  "
          f"(need >= {expected})")
    if n_ok >= expected:
        print("  ✓ GATE PASSED\n")
        return True
    else:
        print("  ✗ GATE FAILED — fix errors above before continuing\n")
        return False


# ─── Step 2: SSI baseline ─────────────────────────────────────────────────────

def step2_ssi_baseline(split: str = "test") -> Dict:
    """
    Run SSI baseline on test subjects.  Returns aggregate metrics dict.
    """
    print("\n" + "═" * 60)
    print("STEP 2 — SSI Baseline  (Device A: Emotiv EPOC)")
    print("═" * 60)
    print(f"  Input : {', '.join(['AF3','AF4','F3','F4'])}")
    print(f"  Targets: {', '.join(TARGETS)}")
    print(f"  Method : MNE interpolate_bads() (spherical spline)\n")

    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS
    from pipeline_v2.baselines.ssi_baseline import run_ssi_dataset, summarise

    split_map = {"train": TRAIN_SUBJECTS, "test": TEST_SUBJECTS,
                 "all": TRAIN_SUBJECTS + TEST_SUBJECTS}
    subjects = split_map[split]

    results = run_ssi_dataset(
        fif_dir=FIF_DIR,
        subject_ids=subjects,
        device_name=DEVICE,
        target_channels=TARGETS,
        verbose=True,
    )
    summary = summarise(results)

    # Save
    ssi_dir = RESULTS_DIR / "ssi" / DEVICE
    ssi_dir.mkdir(parents=True, exist_ok=True)
    with open(ssi_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  SSI gate values:")
    print(f"    Pearson r (mean)  : {summary.get('pearson_mean_mean', float('nan')):.3f}")
    mse_v2 = summary.get('mse_mean_mean', float('nan'))
    print(f"    MSE       (mean)  : {mse_v2 * 1e12:.4f} pV^2  ({mse_v2:.3e} V^2)")
    print(f"    Beta ratio (mean) : {summary.get('beta_mean_ratio_mean', float('nan')):.3f}")
    print(f"\n  → Saved to {ssi_dir / 'summary.json'}")

    return summary


# ─── Step 3: ZUNA inference ───────────────────────────────────────────────────

def step3_zuna_inference(split: str = "test", gpu: int = 0,
                         overwrite: bool = False) -> Dict:
    """
    Run ZUNA on test subjects and evaluate.  Returns aggregate metrics dict.
    """
    print("\n" + "═" * 60)
    print("STEP 3 — ZUNA Inference  (Device A: Emotiv EPOC)")
    print("═" * 60)
    print(f"  Commit : 7b6b858fd36808353bce1b2184ca93695cf68075")
    print(f"  Input  : {', '.join(['AF3','AF4','F3','F4'])}")
    print(f"  Targets: {', '.join(TARGETS)}")
    print(f"  GPU    : {gpu}\n")

    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS
    from pipeline_v2.zuna.zuna_pipeline import run_zuna_dataset, summarise

    split_map = {"train": TRAIN_SUBJECTS, "test": TEST_SUBJECTS,
                 "all": TRAIN_SUBJECTS + TEST_SUBJECTS}
    subjects = split_map[split]

    results = run_zuna_dataset(
        fif_dir=FIF_DIR,
        subject_ids=subjects,
        device_name=DEVICE,
        target_channels=TARGETS,
        work_dir=ZUNA_DIR,
        gpu_device=gpu,
        overwrite=overwrite,
        verbose=True,
    )
    summary = summarise(results)

    # Save
    zuna_dir = RESULTS_DIR / "zuna" / DEVICE
    zuna_dir.mkdir(parents=True, exist_ok=True)
    with open(zuna_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ZUNA values:")
    print(f"    Pearson r (mean)  : {summary.get('pearson_mean_mean', float('nan')):.3f}")
    mse_v2 = summary.get('mse_mean_mean', float('nan'))
    print(f"    MSE       (mean)  : {mse_v2 * 1e12:.4f} pV^2  ({mse_v2:.3e} V^2)")
    print(f"    Beta ratio (mean) : {summary.get('beta_mean_ratio_mean', float('nan')):.3f}")
    print(f"\n  → Saved to {zuna_dir / 'summary.json'}")

    return summary


# ─── Step 4: Results table ────────────────────────────────────────────────────

def step4_results_table(ssi_summary: Dict, zuna_summary: Dict) -> None:
    """
    Print the Month 1 results table and save the combined summary.
    """
    print("\n" + "═" * 60)
    print("STEP 4 — Month 1 Results  (Device A / Test set)")
    print("═" * 60)

    from pipeline_v2.eval.metrics import print_results_table

    # Flatten aggregated summaries into per-metric dicts for the table
    # (summarise() returns keys like 'pearson_mean_mean'; we strip one _mean)
    def flatten(d: Dict, prefix: str = "") -> Dict:
        return {k.replace("_mean", "", 1): v for k, v in d.items()
                if k.endswith("_mean")}

    table_data = {
        "SSI":  flatten(ssi_summary),
        "ZUNA": flatten(zuna_summary),
    }

    print_results_table(table_data, conditions=["SSI", "ZUNA"])

    # Combined save
    combined = {"SSI": ssi_summary, "ZUNA": zuna_summary}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "month1_summary_v2.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Full results → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Month 1 pipeline: download → SSI baseline → ZUNA → compare"
    )
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip step 1 (assume .fif files already exist)")
    parser.add_argument("--only_baseline", action="store_true",
                        help="Run only step 2 (SSI baseline)")
    parser.add_argument("--only_zuna",     action="store_true",
                        help="Run only steps 3+4 (ZUNA + table)")
    parser.add_argument("--split",   type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick",   action="store_true",
                        help="Process only 5 subjects (debugging)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without running")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-subject metrics as they are computed")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        print("\n[DRY RUN] Steps that would execute:")
        if not (args.skip_download or args.only_baseline or args.only_zuna):
            print("  1. Download + preprocess 109 subjects → .fif @ 256 Hz")
        if not args.only_zuna:
            print("  2. SSI baseline (Emotiv EPOC, test set)")
        if not args.only_baseline:
            print("  3. ZUNA inference (Emotiv EPOC, test set)")
        if not args.only_baseline:
            print("  4. Results table → month1_summary_v2.json")
        return

    ssi_summary, zuna_summary = {}, {}

    # Step 1: download + preprocess (always run unless --skip_download)
    if not args.skip_download:
        ok = step1_prepare_data(overwrite=args.overwrite, quick=args.quick)
        if not ok:
            print("\n[ABORT] Step 1 gate failed.  Fix data issues before continuing.")
            sys.exit(1)
    else:
        # --skip_download: just gate-check what already exists
        print("\n[INFO] Checking existing .fif files (download skipped) ...")
        ok = _gate_step1(quick=args.quick)
        if not ok:
            print("\n[ABORT] Gate 1 failed on existing .fif files.")
            sys.exit(1)

    # Step 2: SSI baseline (skip only if --only_zuna)
    if not args.only_zuna:
        ssi_summary = step2_ssi_baseline(split=args.split)
        if not ssi_summary:
            print("\n[WARN] SSI baseline produced no results.")

    # Step 3 + 4: ZUNA (skip if --only_baseline)
    if not args.only_baseline:
        zuna_summary = step3_zuna_inference(
            split=args.split, gpu=args.gpu, overwrite=args.overwrite
        )
        if not zuna_summary:
            print("\n[WARN] ZUNA produced no results.")
        else:
            # Load SSI summary from disk if we skipped step 2
            if not ssi_summary:
                ssi_path = RESULTS_DIR / "ssi" / DEVICE / "summary.json"
                if ssi_path.exists():
                    with open(ssi_path) as f:
                        ssi_summary = json.load(f)
            step4_results_table(ssi_summary, zuna_summary)

    print("\n[DONE] Month 1 pipeline complete.")


if __name__ == "__main__":
    main()
