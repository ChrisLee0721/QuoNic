"""Fit backend constant-factor profiles from benchmark data.

Generates ~/.quonic/profiles.json from either exp13 data or benchmarks.json.

Usage:
    python scripts/fit_profiles.py --source exp13
    python scripts/fit_profiles.py --source benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EXP13_PATH = Path(__file__).parent.parent / "experiments" / "paper" / "results" / "exp13_local_overnight.json"
BENCHMARKS_PATH = Path(__file__).parent.parent / "src" / "quonic" / "scheduler" / "data" / "benchmarks.json"
OUTPUT_PATH = os.path.join(os.path.expanduser("~"), ".quonic", "profiles.json")


def main():
    parser = argparse.ArgumentParser(description="Fit backend profiles")
    parser.add_argument(
        "--source",
        choices=["exp13", "benchmarks"],
        default="exp13",
        help="Data source for fitting",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help="Output path for profiles.json",
    )
    args = parser.parse_args()

    from quonic.scheduler.profiles import (
        fit_profiles_from_benchmarks,
        fit_profiles_from_exp13,
    )

    if args.source == "exp13":
        print(f"Fitting profiles from {EXP13_PATH}")
        profiles = fit_profiles_from_exp13(str(EXP13_PATH))
    else:
        print(f"Fitting profiles from {BENCHMARKS_PATH}")
        with open(BENCHMARKS_PATH, encoding="utf-8") as f:
            benchmarks = json.load(f)
        profiles = fit_profiles_from_benchmarks(benchmarks)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    data = {
        "version": profiles.version,
        "updated_at": profiles.updated_at,
        "profiles": {
            k: {
                "backend_method": v.backend_method,
                "startup_ms": round(v.startup_ms, 4),
                "per_gate_us": round(v.per_gate_us, 4),
                "scaling": v.scaling,
                "n_samples": v.n_samples,
                "r_squared": round(v.r_squared, 4),
                "ema_alpha": v.ema_alpha,
            }
            for k, v in profiles.profiles.items()
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nWrote {len(profiles.profiles)} profiles to {args.output}")
    print("\nProfiles:")
    for key, p in sorted(profiles.profiles.items()):
        print(
            f"  {key:30s}  startup={p.startup_ms:8.2f}ms  "
            f"per_gate={p.per_gate_us:8.2f}us  "
            f"scaling={p.scaling:6s}  "
            f"R2={p.r_squared:.3f}  "
            f"n={p.n_samples}"
        )


if __name__ == "__main__":
    main()
