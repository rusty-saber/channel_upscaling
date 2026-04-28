"""
Figure 1 — Expansion Scaling Law
=================================
Plots mean Pearson r vs. input channel count for SSI, REVE, and ZUNA,
with power-law fits overlaid.

Usage
-----
    # Basic — SSI only (from scaling_law.json)
    python -m pipeline_v2.figures.scaling_law_figure

    # With ZUNA and REVE results available
    python -m pipeline_v2.figures.scaling_law_figure \\
        --scaling_json pipeline_v2/results/scaling_law/scaling_law.json \\
        --zuna_json    pipeline_v2/results/zuna/emotiv_epoc/summary.json \\
        --reve_json    pipeline_v2/results/reve/emotiv_epoc/summary.json \\
        --out_dir      pipeline_v2/results/figures

Design notes
------------
- SSI is shown as individual points + dotted line (non-monotonic, poor fit)
- ZUNA/REVE are shown as connected curves with shaded ±1 SD bands
- Power-law fits shown as dashed curves (SSI fit is shown greyed to signal
  poor R²; ZUNA fit is prominent)
- Reference lines: r=0.85 (approx BES>=0.85 threshold), r=0.5 (SSI typical)
- All aesthetics follow a clean, paper-ready style (no seaborn dependency)
"""

import sys
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Colour palette (colourblind-safe) ────────────────────────────────────────
C_SSI  = "#E76F51"   # warm orange-red
C_REVE = "#2A9D8F"   # teal
C_ZUNA = "#264653"   # dark navy
C_REF  = "#AAAAAA"   # reference-line grey


# ── Power-law fitting ─────────────────────────────────────────────────────────

def fit_power_law(ns: List[float], rs: List[float]) -> Optional[Dict]:
    """
    Fit r(n) = a * n^b via log-log linear regression.
    Returns {"a", "b", "r2"} or None if fit fails.
    """
    ns_arr = np.array(ns, dtype=float)
    rs_arr = np.array(rs, dtype=float)
    valid = (rs_arr > 0) & np.isfinite(rs_arr)
    if valid.sum() < 2:
        return None
    log_n = np.log(ns_arr[valid])
    log_r = np.log(rs_arr[valid])
    try:
        coeffs = np.polyfit(log_n, log_r, 1)
        b = float(coeffs[0])
        a = float(math.exp(coeffs[1]))
        pred = np.polyval(coeffs, log_n)
        ss_res = float(np.sum((log_r - pred) ** 2))
        ss_tot = float(np.sum((log_r - log_r.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        return {"a": a, "b": b, "r2": r2}
    except Exception:
        return None


def power_law_curve(a: float, b: float, n_range: np.ndarray) -> np.ndarray:
    return a * (n_range ** b)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_scaling_json(path: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Load SSI curve from scaling_law_full.json (or legacy scaling_law.json).
    Returns (n_values, r_means, r_stds).
    """
    with open(path) as f:
        data = json.load(f)

    ssi = data.get("ssi", {})
    ns = sorted(int(k) for k in ssi.keys())
    rs_mean = [ssi[str(n)].get("pearson_mean_mean", float("nan")) for n in ns]
    rs_std  = [ssi[str(n)].get("pearson_mean_std",  0.0)          for n in ns]
    return ns, rs_mean, rs_std


def load_reve_scaling_json(path: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Load full REVE scaling curve from scaling_law_full.json.
    Returns (n_values, r_means, r_stds).  Empty lists if no REVE key in file.
    """
    with open(path) as f:
        data = json.load(f)

    reve = data.get("reve", {})
    if not reve:
        return [], [], []
    ns = sorted(int(k) for k in reve.keys())
    rs_mean = [reve[str(n)].get("pearson_mean_mean", float("nan")) for n in ns]
    rs_std  = [reve[str(n)].get("pearson_mean_std",  0.0)          for n in ns]
    return ns, rs_mean, rs_std


def load_method_json(path: Path, n_channels: int) -> Tuple[float, float]:
    """
    Load a single-point method summary JSON (e.g. ZUNA at n=4).
    Returns (r_mean, r_std).
    """
    with open(path) as f:
        data = json.load(f)
    r_mean = data.get("pearson_mean_mean", float("nan"))
    r_std  = data.get("pearson_mean_std",  0.0)
    return r_mean, r_std


def load_bes_scaling_curve(zuna_scaling_dir: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Load ZUNA BES scaling curve from results/scaling_law/zuna/<n>ch/bes_summary.json.
    Returns (n_values, bes_means, bes_stds).
    """
    ns, bes_mean, bes_std = [], [], []
    for ch_dir in sorted(zuna_scaling_dir.iterdir(), key=lambda p: int(p.name.replace("ch", ""))):
        summary = ch_dir / "bes_summary.json"
        if not summary.exists():
            continue
        n = int(ch_dir.name.replace("ch", ""))
        with open(summary) as f:
            data = json.load(f)
        bes = data.get("bes_mean", float("nan"))
        std = data.get("bes_std",  0.0)
        ns.append(n)
        bes_mean.append(bes)
        bes_std.append(std)
    return ns, bes_mean, bes_std


def load_zuna_scaling_curve(zuna_scaling_dir: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Load full ZUNA scaling curve from results/scaling_law/zuna/<n>ch/summary.json.
    Returns (n_values, r_means, r_stds).
    """
    ns, rs_mean, rs_std = [], [], []
    for ch_dir in sorted(zuna_scaling_dir.iterdir(), key=lambda p: int(p.name.replace("ch", ""))):
        summary = ch_dir / "summary.json"
        if not summary.exists():
            continue
        n = int(ch_dir.name.replace("ch", ""))
        with open(summary) as f:
            data = json.load(f)
        r = data.get("pearson_mean_mean", float("nan"))
        s = data.get("pearson_mean_std",  0.0)
        ns.append(n)
        rs_mean.append(r)
        rs_std.append(s)
    return ns, rs_mean, rs_std


# ── Main figure function ───────────────────────────────────────────────────────

def make_scaling_figure(
    scaling_json:       Optional[Path] = None,
    zuna_json:          Optional[Path] = None,
    reve_json:          Optional[Path] = None,
    zuna_n:             int = 4,
    zuna_scaling_dir:   Optional[Path] = None,   # NEW: full ZUNA scaling curve
    out_dir:            Path = Path("."),
    show:               bool = False,
) -> None:
    """
    Build Figure 1 and save to out_dir as scaling_law.pdf + scaling_law.png.

    Parameters
    ----------
    scaling_json : path to scaling_law/scaling_law.json  (SSI across all n)
    zuna_json    : path to zuna/emotiv_epoc/summary.json  (single n point)
    reve_json    : path to reve/emotiv_epoc/summary.json  (single n point)
    zuna_n       : the input channel count ZUNA was evaluated at
    out_dir      : output directory for figure files
    show         : if True, display interactively (requires display)
    """

    # ── Load data ──────────────────────────────────────────────────────────────
    ssi_ns:   List[int]   = []
    ssi_mean: List[float] = []
    ssi_std:  List[float] = []

    reve_ns:   List[int]   = []
    reve_mean: List[float] = []
    reve_std:  List[float] = []

    if scaling_json and scaling_json.exists():
        ssi_ns,  ssi_mean,  ssi_std  = load_scaling_json(scaling_json)
        reve_ns, reve_mean, reve_std = load_reve_scaling_json(scaling_json)
        print(f"[fig] Loaded SSI  data: n={ssi_ns},  r={[f'{r:.3f}' for r in ssi_mean]}")
        if reve_ns:
            print(f"[fig] Loaded REVE data: n={reve_ns}, r={[f'{r:.3f}' for r in reve_mean]}")
    else:
        ssi_ns   = [1, 2, 4, 8, 16, 32]
        ssi_mean = [0.693, 0.691, 0.421, 0.280, 0.297, 0.764]
        ssi_std  = [0.116, 0.104, 0.141, 0.178, 0.192, 0.117]
        reve_ns  = [1, 2, 4, 8, 16, 32]
        reve_mean = [0.737, 0.725, 0.672, 0.644, 0.671, 0.753]
        reve_std  = [0.104, 0.098, 0.105, 0.105, 0.105, 0.096]
        print("[fig] WARNING: No scaling_law_full.json found — using hardcoded fallback values")

    zuna_r, zuna_s = None, None
    if zuna_json and zuna_json.exists():
        zuna_r, zuna_s = load_method_json(zuna_json, zuna_n)
        print(f"[fig] Loaded ZUNA single-point: n={zuna_n}  r={zuna_r:.3f} ± {zuna_s:.3f}")

    # Load full ZUNA scaling curve if directory provided
    zuna_scale_ns:   List[int]   = []
    zuna_scale_mean: List[float] = []
    zuna_scale_std:  List[float] = []
    if zuna_scaling_dir and zuna_scaling_dir.exists():
        zuna_scale_ns, zuna_scale_mean, zuna_scale_std = load_zuna_scaling_curve(zuna_scaling_dir)
        print(f"[fig] Loaded ZUNA scaling: n={zuna_scale_ns}, r={[f'{r:.3f}' for r in zuna_scale_mean]}")
        # Use n=4 point from scaling curve for bar chart
        if 4 in zuna_scale_ns:
            idx = zuna_scale_ns.index(4)
            zuna_r, zuna_s = zuna_scale_mean[idx], zuna_scale_std[idx]

    # ── Power-law fits ─────────────────────────────────────────────────────────
    pl_ssi  = fit_power_law(ssi_ns,  ssi_mean)
    pl_reve = fit_power_law(reve_ns, reve_mean) if reve_ns else None
    pl_zuna = fit_power_law(zuna_scale_ns, zuna_scale_mean) if zuna_scale_ns else None
    print(f"[fig] SSI  power law: a={pl_ssi['a']:.3f}  b={pl_ssi['b']:.3f}  R2={pl_ssi['r2']:.3f}"
          if pl_ssi else "[fig] SSI power law: fit failed")
    if pl_reve:
        print(f"[fig] REVE power law: a={pl_reve['a']:.3f}  b={pl_reve['b']:.3f}  R2={pl_reve['r2']:.3f}")

    # Load BES scaling curve
    bes_ns:   List[int]   = []
    bes_mean: List[float] = []
    bes_std:  List[float] = []
    if zuna_scaling_dir and zuna_scaling_dir.exists():
        bes_ns, bes_mean, bes_std = load_bes_scaling_curve(zuna_scaling_dir)
        if bes_ns:
            print(f"[fig] Loaded BES scaling: n={bes_ns}, BES={[f'{b:.3f}' for b in bes_mean]}")

    # ── Figure layout ──────────────────────────────────────────────────────────
    n_panels = 3 if bes_ns else 2
    width_ratios = [2, 1, 1] if bes_ns else [2, 1]
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(14 if bes_ns else 10, 5.0),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.38},
    )
    ax_main = axes[0]   # scaling curve (Pearson r)
    ax_bar  = axes[1]   # bar chart: single operating point (n=4)
    ax_bes  = axes[2] if bes_ns else None   # BES scaling curve

    n_smooth = np.logspace(np.log10(0.8), np.log10(36), 200)

    # ── LEFT PANEL: Scaling curves ─────────────────────────────────────────────

    # SSI — plot as individual points + thin connecting line
    ssi_arr = np.array(ssi_mean)
    ssi_std_arr = np.array(ssi_std)
    ax_main.plot(ssi_ns, ssi_mean, "o-", color=C_SSI, lw=1.2, ms=6,
                 label="SSI (Perrin 1989)", zorder=3)
    ax_main.fill_between(
        ssi_ns,
        ssi_arr - ssi_std_arr,
        ssi_arr + ssi_std_arr,
        color=C_SSI, alpha=0.15, zorder=2,
    )

    # SSI power-law fit (greyed, dotted — to signal poor fit)
    if pl_ssi:
        y_fit_ssi = power_law_curve(pl_ssi["a"], pl_ssi["b"], n_smooth)
        r2_label = f"$r(n)={pl_ssi['a']:.2f}\\cdot n^{{{pl_ssi['b']:.2f}}}$  ($R^2={pl_ssi['r2']:.2f}$)"
        ax_main.plot(n_smooth, y_fit_ssi, "--", color=C_SSI, lw=1.0, alpha=0.55,
                     label=f"SSI fit: {r2_label}", zorder=2)

    # REVE full curve (if available)
    if reve_ns:
        reve_arr     = np.array(reve_mean)
        reve_std_arr = np.array(reve_std)
        ax_main.plot(reve_ns, reve_mean, "s-", color=C_REVE, lw=1.2, ms=6,
                     label="REVE (ridge regression)", zorder=3)
        ax_main.fill_between(
            reve_ns,
            reve_arr - reve_std_arr,
            reve_arr + reve_std_arr,
            color=C_REVE, alpha=0.12, zorder=2,
        )
        if pl_reve:
            y_fit_reve = power_law_curve(pl_reve["a"], pl_reve["b"], n_smooth)
            r2_lbl = (f"$r(n)={pl_reve['a']:.2f}\\cdot n^{{{pl_reve['b']:.3f}}}$"
                      f"  ($R^2={pl_reve['r2']:.2f}$)")
            ax_main.plot(n_smooth, y_fit_reve, "--", color=C_REVE, lw=1.0, alpha=0.5,
                         label=f"REVE fit: {r2_lbl}", zorder=2)

    # ZUNA full scaling curve (if available)
    if zuna_scale_ns:
        zuna_arr     = np.array(zuna_scale_mean)
        zuna_std_arr = np.array(zuna_scale_std)
        ax_main.plot(zuna_scale_ns, zuna_scale_mean, "D-", color=C_ZUNA, lw=1.8, ms=7,
                     label="ZUNA (masked diffusion transformer)", zorder=5)
        ax_main.fill_between(
            zuna_scale_ns,
            zuna_arr - zuna_std_arr,
            zuna_arr + zuna_std_arr,
            color=C_ZUNA, alpha=0.15, zorder=2,
        )
        if pl_zuna:
            y_fit_zuna = power_law_curve(pl_zuna["a"], pl_zuna["b"], n_smooth)
            r2_lbl = (f"$r(n)={pl_zuna['a']:.2f}\\cdot n^{{{pl_zuna['b']:.3f}}}$"
                      f"  ($R^2={pl_zuna['r2']:.2f}$)")
            ax_main.plot(n_smooth, y_fit_zuna, "--", color=C_ZUNA, lw=1.2, alpha=0.7,
                         label=f"ZUNA fit: {r2_lbl}", zorder=3)
            print(f"[fig] ZUNA power law: a={pl_zuna['a']:.3f}  b={pl_zuna['b']:.3f}  R2={pl_zuna['r2']:.3f}")
    elif zuna_r is not None:
        # Fallback: single point
        ax_main.errorbar(
            [zuna_n], [zuna_r], yerr=[zuna_s],
            fmt="D", color=C_ZUNA, ms=9, lw=2, capsize=4, capthick=2,
            label=f"ZUNA (n={zuna_n}, Device A)", zorder=5,
        )

    # Reference lines
    ax_main.axhline(0.85, color=C_REF, lw=0.9, ls=":", zorder=1)
    # BES label on the LEFT edge so it doesn't clash with the n=32 annotation
    ax_main.text(0.85, 0.862, "BES>=0.85 target", fontsize=7.5, color=C_REF,
                 ha="left", va="bottom", transform=ax_main.get_yaxis_transform())

    # Non-monotonic annotation — SSI dip at n=4–16
    ax_main.annotate(
        "SSI collapses\n(sparse frontal\nlayout)",
        xy=(4, 0.421), xytext=(2.2, 0.28),
        fontsize=7.5, color=C_SSI,
        arrowprops=dict(arrowstyle="->", color=C_SSI, lw=1.0),
        ha="center",
    )
    # n=32 recovery annotation — text placed bottom-right, arrow points up-left
    ax_main.annotate(
        "SSI recovers\nat n=32",
        xy=(32, 0.764), xytext=(32, 0.60),
        fontsize=7.5, color=C_SSI,
        arrowprops=dict(arrowstyle="->", color=C_SSI, lw=1.0),
        ha="center",
    )

    ax_main.set_xscale("log")
    ax_main.set_xlim(0.8, 40)
    ax_main.set_ylim(0.0, 1.05)
    ax_main.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: str(int(round(x))) if x >= 1 else f"{x:.1f}"
    ))
    ax_main.set_xlabel("Input channel count  $n$", fontsize=10)
    ax_main.set_ylabel("Mean Pearson $r$  (test set)", fontsize=10)
    ax_main.set_title("Expansion Scaling Law", fontsize=11, fontweight="bold")
    # Legend placed BELOW the axes so it never covers data
    ax_main.legend(
        fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3, framealpha=0.9, borderpad=0.8,
    )
    ax_main.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
    ax_main.tick_params(labelsize=9)

    # ── RIGHT PANEL: Bar chart at n=4 (Device A operating point) ──────────────
    methods, means, stds, colours = [], [], [], []

    # SSI at n=4
    if 4 in ssi_ns:
        idx = ssi_ns.index(4)
        methods.append("SSI")
        means.append(ssi_mean[idx])
        stds.append(ssi_std[idx])
        colours.append(C_SSI)

    # REVE at n=4
    if 4 in reve_ns:
        idx = reve_ns.index(4)
        methods.append("REVE")
        means.append(reve_mean[idx])
        stds.append(reve_std[idx])
        colours.append(C_REVE)

    # ZUNA at n=4
    if zuna_r is not None:
        methods.append("ZUNA")
        means.append(zuna_r)
        stds.append(zuna_s)
        colours.append(C_ZUNA)

    if methods:
        x_pos = np.arange(len(methods))
        bars = ax_bar.bar(x_pos, means, yerr=stds, color=colours,
                          width=0.55, capsize=5, error_kw={"lw": 1.5},
                          zorder=3)
        ax_bar.axhline(0.85, color=C_REF, lw=0.9, ls=":", zorder=1)
        ax_bar.text(len(methods) - 0.6, 0.87, "target", fontsize=7.5,
                    color=C_REF, va="bottom")
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(methods, fontsize=9)
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_ylabel("Mean Pearson $r$", fontsize=10)
        ax_bar.set_title(f"Device A  ($n=4$)", fontsize=11, fontweight="bold")
        ax_bar.tick_params(labelsize=9)
        ax_bar.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
        # Value labels on bars
        for bar, val in zip(bars, means):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold")
    else:
        ax_bar.text(0.5, 0.5, "[ZUNA / REVE\nresults pending]",
                    ha="center", va="center", transform=ax_bar.transAxes,
                    fontsize=9, color="grey", style="italic")
        ax_bar.set_title(f"Device A  ($n=4$)", fontsize=11, fontweight="bold")
        ax_bar.set_ylim(0.0, 1.05)

    # ── RIGHT-MOST PANEL: BES scaling curve ───────────────────────────────────
    if ax_bes is not None and bes_ns:
        bes_arr     = np.array(bes_mean)
        bes_std_arr = np.array(bes_std)
        ax_bes.plot(bes_ns, bes_mean, "D-", color=C_ZUNA, lw=1.8, ms=7,
                    label="ZUNA BES", zorder=4)
        ax_bes.fill_between(
            bes_ns,
            bes_arr - bes_std_arr,
            bes_arr + bes_std_arr,
            color=C_ZUNA, alpha=0.15, zorder=2,
        )
        ax_bes.axhline(0.85, color=C_REF, lw=0.9, ls=":", zorder=1)
        ax_bes.axhline(1.00, color="#CCCCCC", lw=0.7, ls="--", zorder=1)
        ax_bes.text(0.02, 0.872, "BES=0.85 threshold", fontsize=7.5,
                    color=C_REF, va="bottom", transform=ax_bes.get_yaxis_transform())
        ax_bes.set_xscale("log")
        ax_bes.set_xlim(0.8, 40)
        ax_bes.set_ylim(0.0, 1.6)
        ax_bes.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: str(int(round(x))) if x >= 1 else f"{x:.1f}"
        ))
        ax_bes.set_xlabel("Input channel count  $n$", fontsize=10)
        ax_bes.set_ylabel("BES (A_pred / A_gt)", fontsize=10)
        ax_bes.set_title("BES Scaling (ZUNA)", fontsize=11, fontweight="bold")
        ax_bes.tick_params(labelsize=9)
        ax_bes.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)

    # ── Footnote ───────────────────────────────────────────────────────────────
    note = ("Error bars = ±1 SD across test subjects.  "
            "Preliminary data: 12 train-split subjects."
            if not (scaling_json and scaling_json.exists())
            else "Error bars = ±1 SD across 22 test subjects.")
    fig.text(0.5, -0.01, note, ha="center", fontsize=7.5, color="grey",
             style="italic")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"scaling_law.{ext}"
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        print(f"[fig] Saved -> {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    print("[fig] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 1: Expansion Scaling Law")
    p.add_argument(
        "--scaling_json",
        type=str,
        default=str(ROOT / "pipeline_v2" / "results" / "scaling_law" / "scaling_law_full.json"),
        help="Path to scaling_law_full.json (SSI + REVE across all n).",
    )
    p.add_argument(
        "--zuna_json",
        type=str,
        default=str(ROOT / "pipeline_v2" / "results" / "zuna" / "emotiv_epoc" / "summary.json"),
        help="Path to ZUNA summary.json for the Device A operating point.",
    )
    p.add_argument(
        "--reve_json",
        type=str,
        default=str(ROOT / "pipeline_v2" / "results" / "reve" / "emotiv_epoc" / "summary.json"),
        help="Path to REVE summary.json for the Device A operating point.",
    )
    p.add_argument(
        "--zuna_n",
        type=int,
        default=4,
        help="Input channel count for the ZUNA/REVE single-point result (default: 4).",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "pipeline_v2" / "results" / "figures"),
        help="Output directory for figure files.",
    )
    p.add_argument(
        "--zuna_scaling_dir",
        type=str,
        default=str(ROOT / "pipeline_v2" / "results" / "scaling_law" / "zuna"),
        help="Directory containing ZUNA scaling results (<n>ch/summary.json).",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display figure interactively (requires a display).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_scaling_figure(
        scaling_json     = Path(args.scaling_json),
        zuna_json        = Path(args.zuna_json),
        reve_json        = Path(args.reve_json),
        zuna_n           = args.zuna_n,
        zuna_scaling_dir = Path(args.zuna_scaling_dir),
        out_dir          = Path(args.out_dir),
        show             = args.show,
    )
