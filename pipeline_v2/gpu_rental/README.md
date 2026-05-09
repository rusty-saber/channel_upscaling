# GPU Rental Kit — Matched-Epoch ZUNA Rerun

One-shot rental procedure for the Phase 1 fix: rerun ZUNA inference on motor-imagery-only epochs across all three simulated wearable montages so the comparison against SSI and REVE is no longer confounded by task mismatch.

**Cost target: under $5 total.** Realistic range $3–7 depending on GPU availability and spot pricing. If you blow past $5 the script is structured so you can stop after Device A and still have a defensible primary result.

---

## Provider recommendation

**First choice: Vast.ai** with an RTX 3060 (12 GB VRAM) or RTX 3090 spot listing at $0.15–0.30/hr. ZUNA is 380M params — 8 GB VRAM is enough; you're paying for compute speed, not memory. Filter for "interruptible" instances; the script checkpoints per-subject so an eviction loses at most one subject.

**Backup: RunPod community cloud.** Slightly more expensive, slightly more reliable. RTX 3090 is typically $0.20–0.40/hr on community.

**Avoid:** A100, H100, Lambda Cloud on-demand. You're paying for hardware you don't need.

**Free alternatives to try first:** ask your supervisor at LMSC about lab GPU/HPC. Then Google Colab Pro ($10/month, ~T4) or Kaggle (30 hr/week free P100). Colab session timeouts make 24-hour runs awkward but doable if you stage outputs to Google Drive after each device.

---

## Pre-flight (do at home, zero GPU cost)

1. **Push the `pipeline_v2/` source tree to a public or private GitHub repo.** Excluding `data/raw/`, `data/fif/`, `data/zuna_work*/`, `results/` (.gitignore them). This is what the rental instance clones — keep the upload small.
2. **Smoke-test the preprocessing script locally** for a single subject (CPU only, no GPU needed):
   ```bash
   python -m pipeline_v2.gpu_rental.preprocess_motor_imagery --subjects 88
   ```
   Verify `data/fif_mi/S088_raw.fif` exists and is roughly 30–40% the size of `data/fif/S088_raw.fif` (only 6 of 14 runs).
3. **Decide the budget ceiling** before opening the rental dashboard. Stick to it.

---

## Rental procedure

### Step 1 — pick instance (~5 min, no clock yet)

On Vast.ai, filter:
- GPU: RTX 3060 / 3070 / 3090
- VRAM: ≥ 8 GB
- Disk: ≥ 50 GB (data + intermediate ZUNA work files take ~30 GB)
- CUDA: 11.8+
- Image: `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` or similar
- Sort by `$/hr` ascending; prefer interruptible

### Step 2 — provision and clone (~5 min, ~$0.05)

SSH in, then:
```bash
cd /workspace
git clone <your-repo-url> channel_upscaling
cd channel_upscaling
chmod +x pipeline_v2/gpu_rental/instance_run.sh
```

### Step 3 — fire the script (one command, ~24 hr unattended)

```bash
nohup bash pipeline_v2/gpu_rental/instance_run.sh > run.out 2>&1 &
disown
tail -f run.out  # detach with Ctrl-C, the run continues
```

The script runs six stages in order. The first four take ~1 hour combined and are your sanity checks; if anything fails before stage 5 starts, you've spent under $0.50 and can debug or kill the instance. Stage 5 is the long one.

### Step 4 — collect outputs as devices finish

The script tars and saves outputs after **each device completes** (not just at the end). So even if the instance gets evicted partway, whatever finished is already packaged at `pipeline_v2/results/zuna_matched/<device>_outputs.tar.gz`.

From your local machine (or Colab/Drive sync):
```bash
scp -P <port> root@<host>:/workspace/channel_upscaling/pipeline_v2/results/zuna_matched/zuna_matched_all.tar.gz .
```

### Step 5 — terminate the instance immediately after download

Vast.ai keeps charging until you destroy the instance, not until you stop the job. **Destroy it from the web console** as soon as the tarball is downloaded.

---

## Cost ceiling guardrails

The script's stage-by-stage structure lets you bail out at three checkpoints:

| Bail-out point | Spent so far | Got |
|---|---|---|
| Stage 1 fails (deps don't install) | <$0.10 | nothing — kill instance, check requirements |
| Stage 4 fails (smoke test) | <$0.50 | nothing — debug on a tiny instance, retry |
| After Device A finishes | ~$1.50 | the headline in-distribution result for the abstract |
| After Devices A + B | ~$3.50 | both Device A and the first cross-device test |
| Full run | $4.50–7.00 | all three devices, complete dataset |

If the rental clock crosses $5 and Device C hasn't started, **stop**. Device C is the one with the geometric BES > 1 confound (already discussed in §5.2 of v1.9), so its scientific value is lower than A and B. You can defer it.

---

## Files in this directory

- `preprocess_motor_imagery.py` — builds `data/fif_mi/` with motor-imagery runs only (R05, R06, R09, R10, R13, R14)
- `requirements_pinned.txt` — locked dependency versions
- `instance_run.sh` — the unattended end-to-end script
- `README.md` — this file

---

## After the run — back to v1.9 → v1.10

Once `zuna_matched_all.tar.gz` is on your local machine:
1. Extract per-device `summary.json` files
2. Run the existing eval scripts to compute matched-epoch BES, MSE, Pearson r, beta-band ratio
3. Swap the new numbers into v1.10 of the manuscript (abstract, Tables 1–2, §4 results)
4. Drop the §3.5 task-mismatch caveat and rewrite the corresponding §5.4 limitation

I'll prepare the v1.9 → v1.10 number-swap script when you're ready to execute the rental.
