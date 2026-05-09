# Next Steps — Research Recommendations (2026-04-22)

Priority order, highest impact first.

---

## 1. Downstream Task Validation (HIGH IMPACT — ~1 week)

The single strongest addition. Prove spectral fidelity matters in practice.

**Task:** Motor imagery classification on EEGMMIDB (labels already in the dataset).

**Pipeline:**
1. Reconstruct C3/C4 using SSI, REVE, and ZUNA on the 22 test subjects
2. Train CSP + LDA classifier on each reconstruction
3. Compare classification accuracy: Ground Truth (64-ch) vs SSI vs REVE vs ZUNA

**Expected result:** ZUNA accuracy ≈ ground truth >> REVE (REVE kills beta, which is
exactly the frequency band used for motor imagery). This closes the argument definitively —
REVE wins on Pearson r but loses on every task that matters.

**Key claim:** "A method that achieves r=0.673 but eliminates the beta band is clinically
useless for BCI. ZUNA (r=0.619, beta ratio=0.398) is the only method suitable for
downstream neural decoding."

---

## 2. Statistical Significance Tests (LOW EFFORT — ~2 hours)

Reviewers will ask for p-values. Add paired Wilcoxon signed-rank tests (22 subjects,
non-parametric, appropriate for this sample size).

**Run:**
- ZUNA vs SSI on Pearson r → expect p < 0.001
- ZUNA vs REVE on beta ratio → expect p < 0.001
- ZUNA vs SSI on MSE → expect p < 0.001

Report in Table 1 with significance markers (*, **, ***).

---

## 3. Investigate S109 Anomaly (~2 hours)

S109 achieved r=0.393 — nearly at SSI baseline level. This will be flagged by reviewers.

**Check:**
- Power spectrum of S109 vs other subjects (MNE raw.compute_psd())
- Channel impedance or clipping artifacts
- Whether S109 is an outlier in the original 64-ch data too

If bad recording: flag as artifact subject, report metrics with/without.
If not: explain in limitations why ZUNA struggles (e.g. unusual spatial distribution).

---

## 4. Write the Paper Now

You have everything needed. Don't wait for more experiments before drafting.

**Recommended target venues:**
| Venue | Type | Notes |
|-------|------|-------|
| Journal of Neural Engineering | Journal | Strong EEG community, ~3 month review |
| IEEE Trans. Neural Sys. Rehab. Eng. | Journal | Good fit for BCI/EEG work |
| IEEE EMBC 2027 | Conference | Deadline ~Jan 2027 |
| NeurIPS 2026 Workshops | Workshop | Good for visibility |

**Framing (lead with this):**
> "Regression-based methods achieve high waveform correlation at the cost of complete
> spectral suppression, rendering them unsuitable for oscillation-dependent applications.
> ZUNA is the first zero-shot method to reconstruct both waveform shape and spectral
> content from a 14-channel consumer headset — the only method clinically viable for
> BCI deployment."

---

## Current Results Summary (for reference)

| Method | Pearson r | MSE (×1e-9) | Beta ratio | Spectral viable? |
|--------|-----------|-------------|------------|------------------|
| SSI    | 0.388 ± 0.164 | 10,631 ± 8,684 | 10.82 ± 13.12 | No (over-amplifies) |
| REVE   | 0.673 ± 0.105 | 2,288 ± 2,403 | 0.0004 ± 0.0001 | No (kills HF content) |
| ZUNA   | 0.619 ± 0.085 | 1.89 ± 2.58 | 0.398 ± 0.131 | **Yes** |

**ZUNA is the only method you would actually deploy.**

---

## What to Skip (for now)

- Additional consumer headsets (save for v2 / follow-up paper)
- Stronger ML baseline beyond REVE (reviewers accept it as standard)
- More subjects (22 is acceptable for this study design)
