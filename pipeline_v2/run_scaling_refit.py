"""
Scaling-law refit — CPU sprint.

Fits a power-law curve r(n) = a * n^b + c to the existing per-n Pearson r
results and produces the updated figure + JSON parameters for §4.

Drops n=1 from the fit (r=-0.02, degenerate — a single input channel is
below the interpolation threshold and breaks the monotonic assumption).

Output:
    results/scaling_law/power_law_fit.json
    results/scaling_law/scaling_law_figure.png

Usage:
    python -m pipeline_v2.run_scaling_refit
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCALING_DIR = ROOT / "pipeline_v2" / "results" / "scaling_law" / "zuna"
OUT_DIR     = ROOT / "pipeline_v2" / "results" / "scaling_law"


def load_data():
    points = {}
    for d in sorted(SCALING_DIR.iterdir()):
        f = d / "summary.json"
        if not f.exists():
            continue
        n_str = d.name.replace("ch", "")
        try:
            n = int(n_str)
        except ValueError:
            continue
        s = json.loads(f.read_text())
        r = s.get("pearson_mean_mean")
        r_std = s.get("pearson_mean_std")
        if r is not None:
            points[n] = {"r_mean": r, "r_std": r_std or 0.0}
    return points


def power_law(n, a, b, c):
    """r(n) = a * n^b + c"""
    return a * np.power(n, b) + c


def fit_power_law(n_arr, r_arr):
    p0 = [0.5, 0.3, 0.0]
    bounds = ([0, 0, -0.5], [2, 1, 0.5])
    try:
        popt, pcov = curve_fit(power_law, n_arr, r_arr, p0=p0, bounds=bounds,
                               maxfev=10_000)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None


def main():
    data = load_data()
    print("Loaded data points:")
    for n, v in sorted(data.items()):
        print(f"  n={n:>3}: r={v['r_mean']:.4f} ± {v['r_std']:.4f}")

    # Exclude n=1 (degenerate — r < 0)
    fit_data = {n: v for n, v in data.items() if n > 1 and v["r_mean"] > 0}
    n_arr = np.array(sorted(fit_data.keys()), dtype=float)
    r_arr = np.array([fit_data[int(n)]["r_mean"] for n in n_arr])
    r_std = np.array([fit_data[int(n)]["r_std"]  for n in n_arr])

    popt, perr = fit_power_law(n_arr, r_arr)

    if popt is not None:
        a, b, c = popt
        a_err, b_err, c_err = perr
        print(f"\nPower-law fit: r(n) = {a:.4f} × n^{b:.4f} + {c:.4f}")
        print(f"  a = {a:.4f} ± {a_err:.4f}")
        print(f"  b = {b:.4f} ± {b_err:.4f}  (scaling exponent)")
        print(f"  c = {c:.4f} ± {c_err:.4f}")

        # R² on fit points
        r_pred = power_law(n_arr, *popt)
        ss_res = np.sum((r_arr - r_pred) ** 2)
        ss_tot = np.sum((r_arr - r_arr.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  R² = {r2:.4f} (fit quality on n ∈ {{2,4,8,16,32}})")

        # Extrapolate
        n_extrap = np.array([48.0, 64.0])
        r_extrap = power_law(n_extrap, *popt)
        print(f"\nExtrapolation:")
        for n_e, r_e in zip(n_extrap, r_extrap):
            print(f"  n={int(n_e)}: predicted r={r_e:.4f}")

        result = {
            "model": "r(n) = a * n^b + c",
            "fit_points_n": list(map(int, n_arr)),
            "fit_points_r": list(r_arr),
            "a": float(a), "a_std": float(a_err),
            "b": float(b), "b_std": float(b_err),
            "c": float(c), "c_std": float(c_err),
            "r_squared": float(r2),
            "n1_excluded": True,
            "n1_r_mean": data.get(1, {}).get("r_mean"),
            "extrapolation": {str(int(n_e)): float(r_e)
                              for n_e, r_e in zip(n_extrap, r_extrap)},
            "note": "n=1 excluded from fit (r<0, degenerate single-channel input)"
        }
    else:
        result = {"error": "fit_failed"}

    # Save JSON
    out_json = OUT_DIR / "power_law_fit.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_json.relative_to(ROOT)}")

    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))

    all_n  = np.array(sorted(data.keys()), dtype=float)
    all_r  = np.array([data[int(n)]["r_mean"] for n in all_n])
    all_s  = np.array([data[int(n)]["r_std"]  for n in all_n])

    ax.errorbar(all_n, all_r, yerr=all_s, fmt="o", color="#1f77b4",
                capsize=4, label="Observed (mean ± SD)", zorder=3)

    if popt is not None:
        n_curve = np.linspace(1.5, 35, 200)
        ax.plot(n_curve, power_law(n_curve, *popt), "-", color="#ff7f0e",
                label=f"Power-law fit: $r(n) = {a:.3f}\\,n^{{{b:.3f}}} + {c:.3f}$\n($R^2={r2:.3f}$)")
        ax.scatter(n_extrap, r_extrap, marker="^", color="#2ca02c", s=80,
                   zorder=4, label="Extrapolated (n=48, 64)")

    ax.axhline(0.85, ls="--", color="gray", lw=0.8, label="BES operating threshold (0.85)")
    ax.scatter([1], [data[1]["r_mean"]], marker="x", color="red", s=80, zorder=4,
               label="n=1 (excluded from fit, r<0)")

    ax.set_xlabel("Number of input channels (n)")
    ax.set_ylabel("Mean Pearson r (reconstructed vs ground truth)")
    ax.set_title("Scaling Law: ZUNA Channel Reconstruction vs Input Density")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 35)
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_fig = OUT_DIR / "scaling_law_figure.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
