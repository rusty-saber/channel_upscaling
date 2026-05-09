"""
Build matched-epoch .fif files containing only motor-imagery runs.

PhysioNet EEGMMIDB run codes:
    R01-R02 : baseline eyes open / closed
    R03,R07,R11 : motor execution — left/right fist
    R04,R08,R12 : motor imagery — left/right fist  (NOT used here either)
    R05,R09,R13 : motor execution — both fists / both feet
    R06,R10,R14 : motor imagery — both fists / both feet

Per the project config (`bes_mi_runs`), motor imagery = [5, 6, 9, 10, 13, 14].
Note: this list mixes execution and imagery runs as the BES classifier targets
both. We use the project's defined list for consistency.

Output goes to `data/fif_mi/S<NNN>_mi_raw.fif` so the original `data/fif/` is
untouched and the existing v1.8 motor-execution-cropped runs remain reproducible.

Usage (on rental instance, after `pip install -r requirements_pinned.txt`):
    python -m pipeline_v2.gpu_rental.preprocess_motor_imagery --subjects 88-109
"""
import argparse
from pathlib import Path

# Reuse the project's existing download / rename / filter / resample logic.
from pipeline_v2.data.download_eegmmidb import (
    process_subject as _process_subject_full,
    _build_rename_map,
    TARGET_FS,
    BANDPASS_LOW,
    BANDPASS_HIGH,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "pipeline_v2" / "data" / "raw"
FIF_MI_DIR = ROOT / "pipeline_v2" / "data" / "fif_mi"

MI_RUNS = [5, 6, 9, 10, 13, 14]  # config.yaml `bes_mi_runs`


def process_subject_mi(subject_id: int, overwrite: bool = False) -> Path:
    """Download MI runs only, concatenate, filter, resample, save."""
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("WARNING")

    FIF_MI_DIR.mkdir(parents=True, exist_ok=True)
    # NOTE: saved with the project's standard filename so zuna_pipeline.py's
    # hardcoded `S{sid:03d}_raw.fif` loader works when --fif_dir is pointed
    # at fif_mi/. The MI-only content is what makes this directory different.
    out_path = FIF_MI_DIR / f"S{subject_id:03d}_raw.fif"

    if out_path.exists() and not overwrite:
        print(f"  [SKIP] S{subject_id:03d} — exists: {out_path.name}")
        return out_path

    # Download only motor-imagery EDFs
    raw_fnames = eegbci.load_data(
        subject_id, MI_RUNS,
        path=str(RAW_DIR),
        update_path=False,
        verbose=False,
    )
    if not raw_fnames:
        raise RuntimeError(f"No EDFs downloaded for S{subject_id:03d}")

    # Load + handle mixed sample rates + concatenate
    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False)
            for f in raw_fnames]
    freqs = [r.info["sfreq"] for r in raws]
    modal = max(set(freqs), key=freqs.count)
    for i, r in enumerate(raws):
        if r.info["sfreq"] != modal:
            raws[i] = r.resample(modal, npad="auto", verbose=False)
    raw = mne.concatenate_raws(raws, verbose=False)

    # Channel rename to standard_1005 (reuse project mapping)
    rename = _build_rename_map(raw.ch_names)
    if rename:
        raw.rename_channels(rename)
    if len(raw.ch_names) != 64:
        raise RuntimeError(
            f"S{subject_id:03d}: got {len(raw.ch_names)} channels, expected 64"
        )
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    # Bandpass 1-45 Hz, then resample to 256 Hz
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
               fir_window="hamming", verbose=False)
    if raw.info["sfreq"] != TARGET_FS:
        raw.resample(TARGET_FS, npad="auto", verbose=False)

    # Set 3D montage required by ZUNA
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn", verbose=False)

    # Shape gate
    assert len(raw.ch_names) == 64
    assert raw.info["sfreq"] == TARGET_FS

    raw.save(out_path, overwrite=True, verbose=False)
    duration_min = raw.times[-1] / 60
    print(f"  [OK] S{subject_id:03d} -> {out_path.name} "
          f"| {duration_min:.1f} min MI-only")
    return out_path


def _parse_subjects(s: str | None):
    if s is None:
        from pipeline_v2.data.subject_split import TEST_SUBJECTS
        return TEST_SUBJECTS  # default: 22 test subjects only
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Build MI-only .fif files")
    ap.add_argument("--subjects", default=None,
                    help="e.g. '88-109' or '88,89,90'. Default: 22 test subjects.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subjects = _parse_subjects(args.subjects)
    print(f"Processing {len(subjects)} subjects -> {FIF_MI_DIR}\n")
    ok, failed = [], []
    for sid in subjects:
        try:
            process_subject_mi(sid, overwrite=args.overwrite)
            ok.append(sid)
        except Exception as e:
            print(f"  [FAIL] S{sid:03d}: {e}")
            failed.append(sid)
    print(f"\nDone: {len(ok)}/{len(subjects)} subjects.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
