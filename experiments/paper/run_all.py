"""Run all paper experiments in sequence.

Execution order (from plan):
1. Exp1: Cross-platform consistency (P0)
2. Exp4: Native simulator performance (P0)
3. Exp3: Compilation optimization (P0)
4. Exp2: Smart scheduler evaluation (P0)
5. Exp5: Algorithm template breadth (P1)
6. Exp6: Ablation study (P1)
7. Exp8: Expert baseline gap (P1)
8. Exp9: Stress test (P2)
9. Exp7: Hardware-aware compilation (P2)

Usage:
    python -m experiments.paper.run_all
    python -m experiments.paper.run_all --skip-slow  # skip Exp4, Exp9
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXPERIMENTS = [
    ("exp1_cross_platform", "Cross-Platform Consistency"),
    ("exp4_native_sim", "Native Simulator Performance"),
    ("exp3_compilation", "Compilation Optimization"),
    ("exp2_scheduler", "Smart Scheduler"),
    ("exp5_algorithms", "Algorithm Template Breadth"),
    ("exp6_ablation", "Ablation Study"),
    ("exp8_expert_gap", "Expert Baseline Gap"),
    ("exp9_stress", "Stress Test"),
    ("exp7_hardware_aware", "Hardware-Aware Compilation"),
]

SLOW_EXPERIMENTS = {"exp4_native_sim", "exp9_stress"}


def main():
    parser = argparse.ArgumentParser(description="Run all QuoNic paper experiments")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow experiments (exp4, exp9)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only the specified experiment (e.g., exp1_cross_platform)")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [(m, d) for m, d in EXPERIMENTS if m == args.only]
        if not experiments:
            print(f"Unknown experiment: {args.only}")
            print(f"Available: {', '.join(m for m, _ in EXPERIMENTS)}")
            return

    total_start = time.perf_counter()
    results_summary = []

    print(f"{'='*70}")
    print("QuoNic Paper Experiments")
    print(f"{'='*70}")
    print(f"Running {len(experiments)} experiments...")
    if args.skip_slow:
        print(f"Skipping slow: {SLOW_EXPERIMENTS}")
    print()

    for module_name, description in experiments:
        if args.skip_slow and module_name in SLOW_EXPERIMENTS:
            print(f"SKIP: {description} ({module_name})")
            results_summary.append({
                "experiment": module_name,
                "description": description,
                "status": "skipped",
                "time": 0,
            })
            continue

        print(f"\n{'='*70}")
        print(f"Running: {description}")
        print(f"Module: experiments.paper.{module_name}")
        print(f"{'='*70}")

        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(f"experiments.paper.{module_name}")
            mod.main()
            elapsed = time.perf_counter() - t0
            results_summary.append({
                "experiment": module_name,
                "description": description,
                "status": "ok",
                "time": round(elapsed, 1),
            })
            print(f"\n[OK] {description} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results_summary.append({
                "experiment": module_name,
                "description": description,
                "status": "error",
                "error": str(e)[:200],
                "time": round(elapsed, 1),
            })
            print(f"\n[FAIL] {description}: {e}")

    total_elapsed = time.perf_counter() - total_start

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<30} {'Status':>8} {'Time':>10}")
    print("-" * 50)
    for r in results_summary:
        status = r["status"].upper()
        print(f"{r['experiment']:<30} {status:>8} {r['time']:>9.1f}s")
    print("-" * 50)
    print(f"{'Total':<30} {'':>8} {total_elapsed:>9.1f}s")

    # Save summary
    output = {
        "run_all_summary": True,
        "total_time": round(total_elapsed, 1),
        "experiments": results_summary,
    }
    out_path = RESULTS_DIR / "run_all_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
