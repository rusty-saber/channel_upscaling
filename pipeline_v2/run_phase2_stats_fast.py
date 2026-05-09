"""
Phase 2 statistics — fast version.

Reads pre-computed per_subject_<device>.json files (written by run_phase2_stats.py)
instead of re-running inference. Computes Wilcoxon + Holm + bootstrap CI + Cliff's delta.

Usage (seconds, no GPU/training needed):
    python -m pipeline_v2.run_phase2_stats_fast
"""

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_ROOT = ROOT / "pipeline_v2" / "results"
OUT_DIR      = RESULTS_ROOT / "phase2_stats"
DEVICES      = ["emotiv_epoc", "muse_s", "openbci_cyton"]
METRICS      = ["pearson_mean", "mse_mean", "beta_mean_ratio", "bes"]
N_BOOT       = 10_000
RNG_SEED     = 42


# ---------------------------------------------------------------------------
# Stats helpers (same as run_phase2_stats.py)
# ---------------------------------------------------------------------------

def bootstrap_ci(diff, n_boot=N_BOOT, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2)))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    dom = sum(1 if ai > bi else (-1 if ai < bi else 0) for ai in a for bi in b)
    return dom / (len(a) * len(b))


def interpret_delta(d):
    ad = abs(d)
    if ad < 0.147: return "negligible"
    if ad < 0.330: return "small"
    if ad < 0.474: return "medium"
    return "large"


def holm_bonferroni(pvalues):
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def compare_pair(data_a, data_b, metric, label_a, label_b, rng):
    # Normalise ZUNA keys (S088_raw -> S088)
    def clean(d):
        return {k.replace("_raw","").replace("_mi",""): v for k,v in d.items()}
    data_a, data_b = clean(data_a), clean(data_b)

    common = sorted(set(data_a) & set(data_b))
    a_vals = np.array([data_a[s].get(metric, float("nan")) for s in common], dtype=float)
    b_vals = np.array([data_b[s].get(metric, float("nan")) for s in common], dtype=float)

    valid = np.isfinite(a_vals) & np.isfinite(b_vals)
    a_vals, b_vals = a_vals[valid], b_vals[valid]
    n = len(a_vals)
    if n < 4:
        return {"n": n, "error": "too few valid pairs"}

    diff = b_vals - a_vals

    try:
        stat, p = scipy_stats.wilcoxon(a_vals, b_vals, alternative="two-sided")
    except Exception:
        stat, p = float("nan"), float("nan")

    ci_lo, ci_hi = bootstrap_ci(diff, rng=rng)
    delta = cliffs_delta(b_vals, a_vals)

    return {
        "comparison":   f"{label_b} vs {label_a}",
        "metric":       metric,
        "n":            n,
        "mean_a":       float(np.mean(a_vals)),
        "mean_b":       float(np.mean(b_vals)),
        "mean_diff":    float(np.mean(diff)),
        "ci_95_lo":     ci_lo,
        "ci_95_hi":     ci_hi,
        "wilcoxon_W":   float(stat) if np.isfinite(stat) else None,
        "p_raw":        float(p)    if np.isfinite(p) else None,
        "cliffs_delta": float(delta),
        "effect_size":  interpret_delta(delta),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)
    all_results = {}

    for device in DEVICES:
        ps_file = OUT_DIR / f"per_subject_{device}.json"
        if not ps_file.exists():
            print(f"[MISSING] {ps_file} — run run_phase2_stats.py first")
            continue

        method_data = json.loads(ps_file.read_text())
        print(f"\n{device.upper()}")

        device_results = {}
        comparison_keys = []

        for metric in METRICS:
            for (label_a, data_a), (label_b, data_b) in combinations(
                method_data.items(), 2
            ):
                result = compare_pair(data_a, data_b, metric, label_a, label_b, rng)
                key = f"{metric}__{label_b}_vs_{label_a}"
                device_results[key] = result
                comparison_keys.append(key)

        # Holm-Bonferroni across all comparisons for this device
        raw_pvals = [device_results[k].get("p_raw") or 1.0 for k in comparison_keys]
        adjusted  = holm_bonferroni(raw_pvals)
        for k, adj_p in zip(comparison_keys, adjusted):
            device_results[k]["p_adjusted"] = adj_p
            device_results[k]["significant_05"] = adj_p < 0.05

        all_results[device] = device_results

    # Save JSON
    (OUT_DIR / "stats_report.json").write_text(
        json.dumps(all_results, indent=2, default=lambda x: None)
    )

    # Build readable report
    sig_symbol = lambda p: "***" if p < 0.001 else ("**" if p < 0.01
                           else ("*" if p < 0.05 else "ns"))

    lines = [
        "PHASE 2 STATISTICS REPORT", "=" * 80, "",
        "Paired Wilcoxon signed-rank test (two-sided), n=22 subjects per cell",
        "Holm-Bonferroni corrected p-values across all comparisons per device",
        "Bootstrap 95% CI on mean difference (10 000 resamples, seed=42)",
        "Effect size: Cliff's delta  (|d|<0.147 negligible, <0.33 small, <0.474 medium, else large)",
        "",
    ]

    for device, dresults in all_results.items():
        lines.append(f"\n{device.upper()}")
        lines.append("-" * 80)
        lines.append(
            f"  {'Comparison':<24} {'Metric':<18} {'mean_a':>7} {'mean_b':>7} "
            f"{'Δmean':>7} {'95% CI':>22} {'p_adj':>8} {'sig':>4} {'delta':>7} {'eff':>12}"
        )
        lines.append("  " + "-" * 115)
        for key, r in dresults.items():
            if "error" in r:
                continue
            comp  = r["comparison"][:23]
            met   = r["metric"][:17]
            ma    = r["mean_a"]
            mb    = r["mean_b"]
            dm    = r["mean_diff"]
            ci    = f"[{r['ci_95_lo']:+.3f}, {r['ci_95_hi']:+.3f}]"
            padj  = r.get("p_adjusted", float("nan"))
            sig   = sig_symbol(padj) if np.isfinite(padj) else "?"
            delta = r["cliffs_delta"]
            eff   = r["effect_size"]
            lines.append(
                f"  {comp:<24} {met:<18} {ma:7.3f} {mb:7.3f} {dm:+7.3f} "
                f"{ci:>24} {padj:8.4f} {sig:>4} {delta:+7.3f} {eff:>12}"
            )

    report = "\n".join(lines)
    (OUT_DIR / "stats_report_readable.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved to {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
