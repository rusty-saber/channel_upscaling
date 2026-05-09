"""
Download PhysioNet EEGMMIDB and convert every subject to a single .fif file.

Pipeline per subject
--------------------
1.  Download all 14 run .edf files via mne.datasets.eegbci  (skipped if cached)
2.  Load + concatenate all runs into one continuous Raw object
3.  Rename channels: strip PhysioNet dot-padding, remap to standard_1005 case
4.  Bandpass filter 1–45 Hz (before resample to avoid aliasing)
5.  Resample 160 Hz → 256 Hz  (ZUNA hard requirement)
6.  Set standard_1005 montage  (required by ZUNA for 3-D electrode positions)
7.  Save as  <fif_dir>/S<NNN>_raw.fif   (overwrite=False by default)

Usage
-----
    python -m pipeline_v2.data.download_eegmmidb            # all 109 subjects
    python -m pipeline_v2.data.download_eegmmidb --subjects 1-5   # quick test
    python -m pipeline_v2.data.download_eegmmidb --subjects 1,2,3 # explicit IDs

Output shape check (printed after each subject)
    Raw: 64 ch x N samples @ 256 Hz  -- must pass to continue
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# ─── Configuration ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]   # …/channel upscaling/

RAW_DIR = ROOT / "pipeline_v2" / "data" / "raw"
FIF_DIR = ROOT / "pipeline_v2" / "data" / "fif"

TARGET_FS     = 256      # Hz — ZUNA requirement
BANDPASS_LOW  = 1.0      # Hz
BANDPASS_HIGH = 45.0     # Hz
ALL_RUNS      = list(range(1, 15))   # R01 … R14

# Every PhysioNet EEGMMIDB channel in the order MNE loads them, mapped to
# the corresponding standard_1005 name.  Key = dot-stripped+capitalized form,
# Value = correct standard_1005 name.
# PhysioNet stores e.g. 'Fc5.' (capitalize → 'Fc5') but standard_1005 = 'FC5'.
_PHYSIONET_TO_STD1005 = {
    "Fc5":  "FC5",  "Fc3":  "FC3",  "Fc1":  "FC1",  "Fcz":  "FCz",
    "Fc2":  "FC2",  "Fc4":  "FC4",  "Fc6":  "FC6",
    "C5":   "C5",   "C3":   "C3",   "C1":   "C1",   "Cz":   "Cz",
    "C2":   "C2",   "C4":   "C4",   "C6":   "C6",
    "Cp5":  "CP5",  "Cp3":  "CP3",  "Cp1":  "CP1",  "Cpz":  "CPz",
    "Cp2":  "CP2",  "Cp4":  "CP4",  "Cp6":  "CP6",
    "Fp1":  "Fp1",  "Fpz":  "Fpz",  "Fp2":  "Fp2",
    "Af7":  "AF7",  "Af3":  "AF3",  "Afz":  "AFz",  "Af4":  "AF4",
    "Af8":  "AF8",
    "F7":   "F7",   "F5":   "F5",   "F3":   "F3",   "F1":   "F1",
    "Fz":   "Fz",   "F2":   "F2",   "F4":   "F4",   "F6":   "F6",
    "F8":   "F8",
    "Ft7":  "FT7",  "Ft8":  "FT8",
    "T7":   "T7",   "T8":   "T8",   "T9":   "T9",   "T10":  "T10",
    "Tp7":  "TP7",  "Tp8":  "TP8",
    "P7":   "P7",   "P5":   "P5",   "P3":   "P3",   "P1":   "P1",
    "Pz":   "Pz",   "P2":   "P2",   "P4":   "P4",   "P6":   "P6",
    "P8":   "P8",
    "Po7":  "PO7",  "Po3":  "PO3",  "Poz":  "POz",  "Po4":  "PO4",
    "Po8":  "PO8",
    "O1":   "O1",   "Oz":   "Oz",   "O2":   "O2",
    "Iz":   "Iz",
}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _strip_physionet_name(raw_name: str) -> str:
    """'Fz..' → 'Fz',  'Af3.' → 'Af3',  'Fp1.' → 'Fp1'."""
    return raw_name.rstrip(".")


def _build_rename_map(ch_names: List[str]) -> dict:
    """
    Build {old_name: new_name} mapping from PhysioNet raw channel names
    to standard_1005 names.  Raises ValueError for any unrecognised channel.
    """
    rename = {}
    for raw_name in ch_names:
        stripped = _strip_physionet_name(raw_name)
        std_name = _PHYSIONET_TO_STD1005.get(stripped)
        if std_name is None:
            # Try case-insensitive fallback
            std_name = next(
                (v for k, v in _PHYSIONET_TO_STD1005.items()
                 if k.lower() == stripped.lower()),
                None
            )
        if std_name is None:
            raise ValueError(
                f"Channel '{raw_name}' (stripped: '{stripped}') not in "
                f"PhysioNet→standard_1005 mapping.  Update _PHYSIONET_TO_STD1005."
            )
        if raw_name != std_name:
            rename[raw_name] = std_name
    return rename


# ─── Main processing function ─────────────────────────────────────────────────

def process_subject(
    subject_id: int,
    fif_dir: Path = FIF_DIR,
    raw_dir: Path = RAW_DIR,
    overwrite: bool = False,
    verbose: bool = False,
    cleanup_raw: bool = False,
) -> Path:
    """
    Download (if needed) + preprocess one subject → save as .fif.

    Returns
    -------
    Path to the saved .fif file.

    Shape gate
    ----------
    Asserts output Raw has exactly 64 channels @ 256 Hz before saving.
    """
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("WARNING" if not verbose else "INFO")

    fif_dir.mkdir(parents=True, exist_ok=True)
    out_path = fif_dir / f"S{subject_id:03d}_raw.fif"

    if out_path.exists() and not overwrite:
        print(f"  [SKIP] S{subject_id:03d} — .fif already exists: {out_path}")
        return out_path

    # ── 1. Download via MNE ───────────────────────────────────────────────────
    raw_fnames = eegbci.load_data(
        subject_id, ALL_RUNS,
        path=str(raw_dir),
        update_path=False,
        verbose=False,
    )
    if not raw_fnames:
        raise RuntimeError(f"No files downloaded for subject {subject_id}.")

    # ── 2. Load + concatenate all runs ────────────────────────────────────────
    # Some subjects (e.g. S088) have mixed sample rates across runs.
    # Resample all runs to the modal (most common) rate before concatenating.
    raws = []
    for fname in raw_fnames:
        r = mne.io.read_raw_edf(fname, preload=True, verbose=False)
        raws.append(r)

    freqs = [r.info["sfreq"] for r in raws]
    modal_freq = max(set(freqs), key=freqs.count)
    for i, r in enumerate(raws):
        if r.info["sfreq"] != modal_freq:
            raws[i] = r.resample(modal_freq, npad="auto", verbose=False)

    raw = mne.concatenate_raws(raws, verbose=False)

    # ── 3. Rename channels to standard_1005 convention ───────────────────────
    rename_map = _build_rename_map(raw.ch_names)
    if rename_map:
        raw.rename_channels(rename_map)

    # Verify we have 64 EEG channels
    if len(raw.ch_names) != 64:
        raise RuntimeError(
            f"S{subject_id:03d}: expected 64 channels after rename, "
            f"got {len(raw.ch_names)}: {raw.ch_names}"
        )

    # Set channel type (eegbci sometimes marks some as misc)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    # ── 4. Bandpass filter 1–45 Hz ────────────────────────────────────────────
    raw.filter(
        l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
        fir_window="hamming", verbose=False,
    )

    # ── 5. Resample to 256 Hz ─────────────────────────────────────────────────
    if raw.info["sfreq"] != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto", verbose=False)

    # ── 6. Set standard_1005 montage with 3-D positions ──────────────────────
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn", verbose=False)

    # ── Shape gate ────────────────────────────────────────────────────────────
    assert len(raw.ch_names) == 64, (
        f"SHAPE GATE FAILED: expected 64 ch, got {len(raw.ch_names)}"
    )
    assert raw.info["sfreq"] == TARGET_FS, (
        f"SHAPE GATE FAILED: expected {TARGET_FS} Hz, got {raw.info['sfreq']}"
    )

    # ── 7. Save ───────────────────────────────────────────────────────────────
    raw.save(out_path, overwrite=True, verbose=False)
    duration_min = raw.times[-1] / 60
    print(
        f"  [OK] S{subject_id:03d} → {out_path.name}  "
        f"| 64 ch @ {TARGET_FS} Hz | {duration_min:.1f} min"
    )

    # ── 8. Optionally delete raw EDFs to free disk space ─────────────────────
    if cleanup_raw:
        subject_raw_dir = (
            raw_dir / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
            / f"S{subject_id:03d}"
        )
        if subject_raw_dir.exists():
            import shutil as _shutil
            _shutil.rmtree(str(subject_raw_dir))
            print(f"  [CLEAN] Deleted raw EDFs for S{subject_id:03d}")

    return out_path


# ─── Batch processing ─────────────────────────────────────────────────────────

def process_all(
    subjects: Optional[List[int]] = None,
    fif_dir: Path = FIF_DIR,
    raw_dir: Path = RAW_DIR,
    overwrite: bool = False,
    cleanup_raw: bool = False,
) -> List[Path]:
    """
    Process all (or a subset of) subjects.

    Parameters
    ----------
    subjects : list of int, optional
        Subject IDs to process.  Defaults to all 109.

    Returns
    -------
    List of .fif paths for successfully processed subjects.
    """
    from pipeline_v2.data.subject_split import TRAIN_SUBJECTS, TEST_SUBJECTS

    if subjects is None:
        subjects = TRAIN_SUBJECTS + TEST_SUBJECTS

    print(f"\nProcessing {len(subjects)} subjects -> {fif_dir}\n")
    ok, failed = [], []

    for sid in subjects:
        try:
            path = process_subject(sid, fif_dir=fif_dir, raw_dir=raw_dir,
                                   overwrite=overwrite, cleanup_raw=cleanup_raw)
            ok.append(path)
        except Exception as e:
            print(f"  [FAIL] S{sid:03d}: {e}")
            failed.append(sid)

    print(f"\n{'─'*50}")
    print(f"Done: {len(ok)}/{len(subjects)} succeeded.")
    if failed:
        print(f"Failed subjects: {failed}")
    return ok


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download + convert PhysioNet EEGMMIDB to .fif @ 256 Hz"
    )
    parser.add_argument(
        "--subjects", type=str, default=None,
        help=(
            "Subjects to process.  "
            "Range: '1-5'  |  Explicit: '1,2,3'  |  Omit for all 109."
        ),
    )
    parser.add_argument("--fif_dir", type=str, default=str(FIF_DIR))
    parser.add_argument("--raw_dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process even if .fif already exists.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _parse_subjects(subject_str: Optional[str]) -> Optional[List[int]]:
    if subject_str is None:
        return None
    if "-" in subject_str and "," not in subject_str:
        lo, hi = subject_str.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s.strip()) for s in subject_str.split(",")]


if __name__ == "__main__":
    args = _parse_args()
    subjects = _parse_subjects(args.subjects)
    process_all(
        subjects=subjects,
        fif_dir=Path(args.fif_dir),
        raw_dir=Path(args.raw_dir),
        overwrite=args.overwrite,
    )
