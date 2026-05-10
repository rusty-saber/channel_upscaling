#!/usr/bin/env bash
# Minimal GPU rental: ZUNA Device A inference + BES computation only.
# Runtime: ~4-5 hours. Cost: ~$0.25 on RTX 3060.
#
# Run from channel_upscaling repo root:
#   bash pipeline_v2/gpu_rental/instance_bes_device_a.sh

set -euo pipefail

LOG="pipeline_v2/gpu_rental/logs/bes_device_a.log"
mkdir -p pipeline_v2/gpu_rental/logs pipeline_v2/results/zuna_bes

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== Stage 1: install deps ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || pip install --quiet torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install --quiet -r pipeline_v2/gpu_rental/requirements_pinned.txt
python -c "import zuna, mne; print('deps OK')"

log "=== Stage 2: download + preprocess MI-only EDFs (S088-S109) ==="
python -m pipeline_v2.gpu_rental.preprocess_motor_imagery --subjects 88-109 \
    2>&1 | tee -a "$LOG"

N=$(ls pipeline_v2/data/fif_mi/S*_raw.fif 2>/dev/null | wc -l)
log "fif_mi files ready: $N / 22"
[ "$N" -lt 20 ] && { log "ERROR: too few fif_mi files. Aborting."; exit 1; }

log "=== Stage 3: ZUNA inference — emotiv_epoc only ==="
python -m pipeline_v2.zuna.zuna_pipeline \
    --fif_dir pipeline_v2/data/fif_mi \
    --device emotiv_epoc \
    --results_dir pipeline_v2/results/zuna_bes \
    --work_dir pipeline_v2/data/zuna_work_bes \
    --split test \
    --gpu 0 \
    --verbose \
    2>&1 | tee -a "$LOG"

log "=== Stage 4: compute BES per subject from reconstructed .fif ==="
python - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, warnings, numpy as np
from pathlib import Path

ROOT      = Path(".").resolve()
WORK_DIR  = ROOT / "pipeline_v2/data/zuna_work_bes/emotiv_epoc"
FIF_MI    = ROOT / "pipeline_v2/data/fif_mi"
OUT_FILE  = ROOT / "pipeline_v2/results/zuna_bes/emotiv_epoc_bes_per_subject.json"

import sys
sys.path.insert(0, str(ROOT))
from pipeline_v2.eval.bes_runner import run_bes_subject
from pipeline_v2.run_bes_repair  import (
    _get_run_boundaries_sec, _assign_run_numbers, compute_bes_grouped
)
from pipeline_v2.eval.metrics    import extract_mi_epochs, band_power
from pipeline_v2.eval.bes_runner import extract_pred_epochs, _load_events_from_fif

TARGET_CHS = ["C3", "C4", "P3", "P4"]
MI_RUNS    = [5, 6, 9, 10, 13, 14]

per_subject = {}
for subj_dir in sorted(WORK_DIR.iterdir()):
    sid = subj_dir.name  # e.g. S088_raw
    recon_fif = subj_dir / "4_fif_output" / f"{sid}_recon.fif"
    if not recon_fif.exists():
        # try alternative name pattern
        candidates = list((subj_dir / "4_fif_output").glob("*.fif"))
        if candidates:
            recon_fif = candidates[0]
        else:
            print(f"  [MISS] {sid} — no reconstructed .fif")
            per_subject[sid] = {"error": "recon_fif_missing"}
            continue

    gt_fif = FIF_MI / f"{sid}.fif"
    if not gt_fif.exists():
        # standard naming: S088_raw.fif
        gt_fif = FIF_MI / f"{sid.replace('_raw','')}_raw.fif"
    if not gt_fif.exists():
        print(f"  [MISS] {sid} — no GT fif")
        per_subject[sid] = {"error": "gt_fif_missing"}
        continue

    try:
        import mne
        mne.set_log_level("WARNING")
        recon_raw = mne.io.read_raw_fif(str(recon_fif), preload=True, verbose=False)
        target_idx = [recon_raw.ch_names.index(ch) for ch in TARGET_CHS
                      if ch in recon_raw.ch_names]
        if len(target_idx) < len(TARGET_CHS):
            print(f"  [WARN] {sid} — only {len(target_idx)} target channels in recon")
        pred = recon_raw.get_data(picks=target_idx)  # (n_targets, n_samples)

        result = run_bes_subject(pred, gt_fif, TARGET_CHS)
        per_subject[sid] = result if result else {"error": "bes_failed"}
        bes_str = f"{result['bes']:.3f}" if result else "skip"
        print(f"  [OK] {sid}  BES={bes_str}")
    except Exception as e:
        print(f"  [FAIL] {sid}: {e}")
        per_subject[sid] = {"error": str(e)}

# summary
vals = [v["bes"] for v in per_subject.values() if isinstance(v,dict) and "bes" in v]
summary = {"bes_mean": float(np.mean(vals)), "bes_std": float(np.std(vals)), "n": len(vals)}
OUT_FILE.write_text(json.dumps({"per_subject": per_subject, "summary": summary}, indent=2))
print(f"\nBES summary: mean={summary['bes_mean']:.3f} std={summary['bes_std']:.3f} n={summary['n']}")
print(f"Saved -> {OUT_FILE}")
PYEOF

log "=== Stage 5: pack output ==="
tar -czf pipeline_v2/results/zuna_bes_device_a.tar.gz \
    pipeline_v2/results/zuna_bes/
log "Done. Output: pipeline_v2/results/zuna_bes_device_a.tar.gz"
log "Cat the BES JSON directly:"
cat pipeline_v2/results/zuna_bes/emotiv_epoc_bes_per_subject.json | \
    python3 -c "import sys,json; d=json.load(sys.stdin)['summary']; print('BES mean:', d['bes_mean'], 'n:', d['n'])"
log "=== COMPLETE — terminate instance and retrieve JSON ==="
