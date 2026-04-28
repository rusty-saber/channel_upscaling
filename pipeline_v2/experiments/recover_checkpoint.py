"""
Recovery script — saves partial scaling law results to checkpoint.

Run this ONCE before re-running run_scaling_law_full.py.
It encodes the per-subject Pearson r values that were already printed
to console before the OOM crash, so the re-run skips n=1,2,4,8 (and n=16 SSI).

Usage:
    python -m pipeline_v2.experiments.recover_checkpoint
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "pipeline_v2" / "results" / "scaling_law"

# ── Per-subject r values from the console output ──────────────────────────────

PER_SUBJECT_R = {
    # (n, method) -> list of 22 r values (S088..S109)
    (1, "ssi"):  [0.856,0.765,0.745,0.838,0.656,0.670,0.677,0.919,0.716,0.781,
                  0.604,0.753,0.632,0.679,0.807,0.746,0.826,0.852,0.613,0.834,0.785,0.458],
    (1, "reve"): [0.856,0.764,0.745,0.838,0.656,0.670,0.677,0.919,0.716,0.781,
                  0.604,0.753,0.632,0.679,0.806,0.746,0.826,0.852,0.613,0.834,0.785,0.458],
    (2, "ssi"):  [0.843,0.766,0.723,0.827,0.669,0.678,0.686,0.878,0.707,0.779,
                  0.610,0.735,0.623,0.665,0.767,0.727,0.823,0.871,0.632,0.688,0.791,0.470],
    (2, "reve"): [0.852,0.757,0.717,0.821,0.655,0.664,0.672,0.902,0.698,0.774,
                  0.594,0.729,0.620,0.680,0.762,0.725,0.823,0.868,0.620,0.752,0.784,0.480],
    (4, "ssi"):  [0.424,0.496,0.208,0.405,0.423,0.501,0.072,0.468,0.294,0.173,
                  0.487,0.303,0.236,0.338,0.444,0.586,0.436,0.868,0.415,0.449,0.333,0.180],
    (4, "reve"): [0.790,0.693,0.657,0.761,0.606,0.594,0.600,0.893,0.623,0.734,
                  0.543,0.699,0.574,0.617,0.693,0.684,0.742,0.825,0.554,0.759,0.738,0.420],
    (8, "ssi"):  [0.390,0.434,-0.004,0.197,0.334,0.373,0.062,0.401,0.162,-0.488,
                  0.310,0.243,0.220,0.256,0.329,0.499,0.448,0.798,0.336,0.251,0.322,0.140],
    (8, "reve"): [0.757,0.657,0.635,0.727,0.578,0.560,0.584,0.889,0.583,0.724,
                  0.528,0.647,0.554,0.585,0.663,0.673,0.713,0.782,0.514,0.743,0.701,0.406],
    (16, "ssi"): [0.398,0.368,0.028,-0.254,0.472,0.250,0.023,0.592,0.137,-0.323,
                  0.443,0.254,0.262,0.315,0.350,0.525,0.419,0.717,0.343,0.363,0.104,0.212],
}

def _make_summary(r_list):
    arr = np.array(r_list)
    # Only store fields we actually have — omit mse/beta (not in console output).
    # run_scaling_law_full.py uses .get("pearson_mean_mean", nan) so missing
    # fields default gracefully. No NaN in JSON (invalid for json.load).
    return {
        "pearson_mean_mean": float(arr.mean()),
        "pearson_mean_std":  float(arr.std()),
    }


def main():
    ssi_ckpt  = {}
    reve_ckpt = {}

    for (n, method), r_list in PER_SUBJECT_R.items():
        summary = _make_summary(r_list)
        if method == "ssi":
            ssi_ckpt[n] = summary
        else:
            reve_ckpt[n] = summary
        print(f"  n={n:2d} {method.upper():4s}  r={summary['pearson_mean_mean']:.4f}"
              f" +/- {summary['pearson_mean_std']:.4f}")

    checkpoint = {
        "ssi":  {str(k): v for k, v in ssi_ckpt.items()},
        "reve": {str(k): v for k, v in reve_ckpt.items()},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS_DIR / "_checkpoint.json"
    with open(ckpt_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\nCheckpoint saved -> {ckpt_path}")
    print("\nNow run:")
    print("  python -m pipeline_v2.experiments.run_scaling_law_full --verbose")
    print("It will skip n=1,2,4,8 (and n=16 SSI) and only compute the missing pieces.")


if __name__ == "__main__":
    main()
