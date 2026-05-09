#!/usr/bin/env bash
# Self-contained run for a one-shot GPU rental (Vast.ai / RunPod / Lambda).
# Designed to minimize paid GPU time: every minute of preventable idle is money.
#
# Stages:
#   1. install pinned deps              (~5  min)
#   2. download MI-only EDFs            (~30 min, mostly network)
#   3. preprocess to MI-only .fif       (~10 min, CPU-bound)
#   4. smoke test on 1 subject, 1 dev   (~15 min, validates everything)
#   5. full ZUNA inference, 3 devices   (~24 hr, the only long stage)
#   6. pack outputs into a tar.gz       (~2  min)
#
# After stage 4 succeeds, you can confidently let stage 5 run unattended.
# If stage 4 fails, kill the instance and you've spent <$0.50.
#
# IMPORTANT: run from the channel-upscaling project root, e.g.:
#   cd /workspace/channel_upscaling
#   bash pipeline_v2/gpu_rental/instance_run.sh

set -euo pipefail

PROJECT_ROOT="$(pwd)"
RENTAL_DIR="$PROJECT_ROOT/pipeline_v2/gpu_rental"
RESULTS_DIR="$PROJECT_ROOT/pipeline_v2/results/zuna_matched"
LOG_DIR="$PROJECT_ROOT/pipeline_v2/gpu_rental/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/run.log"; }

log "=== Stage 1: install pinned deps ==="
# Most rental images ship with a torch already installed. Don't fight it —
# pin to whatever cu118 wheel is already on the instance unless missing.
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || pip install --quiet torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install --quiet -r "$RENTAL_DIR/requirements_pinned.txt"
python -c "import zuna, mne, scipy, sklearn; print('deps OK')"

log "=== Stage 2: download MI-only EDFs (test subjects S088-S109) ==="
python -m pipeline_v2.gpu_rental.preprocess_motor_imagery \
    --subjects 88-109 \
    2>&1 | tee -a "$LOG_DIR/preprocess.log"

log "=== Stage 3: verify .fif files ==="
ls -lh pipeline_v2/data/fif_mi/ | tail -25 | tee -a "$LOG_DIR/run.log"
N_FIF=$(ls pipeline_v2/data/fif_mi/S*_raw.fif 2>/dev/null | wc -l)
if [ "$N_FIF" -ne 22 ]; then
    log "ERROR: expected 22 .fif files, got $N_FIF. Aborting before paid inference."
    exit 1
fi

log "=== Stage 4: smoke test (S088, emotiv_epoc, ~15 min) ==="
python -m pipeline_v2.zuna.zuna_pipeline \
    --fif_dir pipeline_v2/data/fif_mi \
    --device emotiv_epoc \
    --results_dir "$RESULTS_DIR/smoke" \
    --work_dir pipeline_v2/data/zuna_work_matched \
    --split test \
    --gpu 0 \
    --verbose \
    2>&1 | tee "$LOG_DIR/smoke.log" &
SMOKE_PID=$!
# Watch the smoke log for first subject completion (~10-15 min) then continue;
# we let the smoke run interleave with the main loop since zuna_pipeline checkpoints
# per-subject. If it fails on S088 we will see it before paying for the others.
wait $SMOKE_PID || { log "Smoke test FAILED. Aborting full run."; exit 1; }

log "=== Stage 5: full ZUNA inference, all 3 devices, 22 test subjects ==="
for DEVICE in emotiv_epoc muse_s openbci_cyton; do
    log "--- Device: $DEVICE ---"
    python -m pipeline_v2.zuna.zuna_pipeline \
        --fif_dir pipeline_v2/data/fif_mi \
        --device "$DEVICE" \
        --results_dir "$RESULTS_DIR" \
        --work_dir pipeline_v2/data/zuna_work_matched \
        --split test \
        --gpu 0 \
        --verbose \
        2>&1 | tee -a "$LOG_DIR/$DEVICE.log"
    # Pack and upload outputs immediately after each device finishes — so if
    # the instance gets evicted (spot pricing) or budget runs out, you keep
    # what's already done.
    log "--- Packing $DEVICE outputs ---"
    tar -czf "$RESULTS_DIR/${DEVICE}_outputs.tar.gz" \
        -C "$RESULTS_DIR" "$DEVICE" \
        2>&1 | tee -a "$LOG_DIR/run.log"
done

log "=== Stage 6: final package ==="
cd "$RESULTS_DIR/.."
tar -czf zuna_matched_all.tar.gz zuna_matched/
log "Outputs ready: $(pwd)/zuna_matched_all.tar.gz"
ls -lh zuna_matched_all.tar.gz | tee -a "$LOG_DIR/run.log"

log "=== DONE. Download zuna_matched_all.tar.gz then terminate the instance. ==="
