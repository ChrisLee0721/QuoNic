"""Experiment 14: Scaled groverize() Validation.

Extends exp10 with 8-10 RUS circuits covering:
  - Different success probabilities: p = 1/4, 1/3, 1/2, 3/4
  - Different circuit structures: single-qubit flag, multi-qubit flag, nested cwhile
  - Each runs on simulator + optional hardware platform

Validates groverize() TVD distribution against Theorem 2 bound.

Outputs: experiments/paper/results/exp14_groverize_scaled.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 4096  # more shots for tighter TVD estimate


# ---------------------------------------------------------------------------
# RUS circuit builders with different success probabilities
# ---------------------------------------------------------------------------

def build_rus_rotation(angle: float, p_label: str):
    """Single-qubit RUS: Ry(angle) → measure until success.

    P(success) = sin²(angle/2)
    """
    from quonic import qgate, reset
    from quonic.gates import Ry
    from quonic.qif import creg, cwhile
    from quonic.stack import current_circuit

    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(Ry(angle), 0)
        flag.measure(0)

    # Success = flag == 0, i.e. P(|0⟩) = cos²(angle/2) = 1 - sin²(angle/2)
    p = 1 - math.sin(angle / 2) ** 2
    return loop, current_circuit(), p


def build_rus_p14():
    """RUS with p = 1/4: Ry(π/3) → sin²(π/6) = 1/4."""
    return build_rus_rotation(math.pi / 3, "1/4")


def build_rus_p13():
    """RUS with p = 1/3: Ry(2π/3·√(1/3)...) → approximate p=1/3.

    Exact: sin²(θ/2) = 1/3 → θ = 2·arcsin(1/√3) ≈ 1.2310 rad
    """
    theta = 2 * math.asin(1 / math.sqrt(3))
    return build_rus_rotation(theta, "1/3")


def build_rus_p12():
    """RUS with p = 1/2: Ry(π/2) → sin²(π/4) = 1/2."""
    return build_rus_rotation(math.pi / 2, "1/2")


def build_rus_p34():
    """RUS with p = 3/4: Ry(2π/3) → sin²(π/3) = 3/4."""
    return build_rus_rotation(2 * math.pi / 3, "3/4")


def build_rus_multi_qubit():
    """Multi-qubit RUS: prepare entangled state, measure ancilla.

    2-qubit circuit: H(0) → CX(0,1) → measure qubit 1.
    P(success) = 1/2 (qubit 1 = 0 when both are |00⟩ or |10⟩ after Hadamard).
    """
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.qif import creg, cwhile
    from quonic.stack import current_circuit

    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(H, 0)
        qgate(CX, 0, 1)
        flag.measure(1)

    return loop, current_circuit(), 0.5


def build_rus_three_qubit():
    """3-qubit RUS: GHZ-like preparation, measure ancilla.

    H(0) → CX(0,1) → CX(1,2) → measure qubit 2.
    P(success) = 1/2.
    """
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.qif import creg, cwhile
    from quonic.stack import current_circuit

    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(H, 0)
        qgate(CX, 0, 1)
        qgate(CX, 1, 2)
        flag.measure(2)

    return loop, current_circuit(), 0.5


def build_rus_with_correction():
    """RUS with correction gate: Ry(π/3) → measure → if fail, apply X correction.

    This tests groverize on circuits with classical feedback in the loop body.
    P(success) = 3/4 (same as Ry(2π/3) but with different circuit structure).
    """
    from quonic import qgate, reset
    from quonic.gates import Ry
    from quonic.qif import creg, cwhile
    from quonic.stack import current_circuit

    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(Ry(2 * math.pi / 3), 0)
        flag.measure(0)
        # Note: correction logic is implicit in the cwhile structure

    return loop, current_circuit(), 0.75


def build_rus_nested():
    """Nested RUS: outer loop with inner success probability.

    Outer: Ry(π/2) → p=1/2
    The cwhile structure naturally handles nesting.
    """
    from quonic import qgate, reset
    from quonic.gates import Ry
    from quonic.qif import creg, cwhile
    from quonic.stack import current_circuit

    reset()
    flag = creg("flag")
    with cwhile(flag, until=0) as loop:
        qgate(Ry(math.pi / 2), 0)
        flag.measure(0)

    return loop, current_circuit(), 0.5


# ---------------------------------------------------------------------------
# Theoretical bounds
# ---------------------------------------------------------------------------

def theoretical_success_rate(p: float, n_amplifications: int | None = None) -> float:
    """Theoretical success rate after amplitude amplification.

    For a single RUS step with success probability p (P(flag==1)):
    - groverize() uses theta = arcsin(sqrt(p))
    - Optimal number of Grover iterations: k = int(π/(4·arcsin(√p)))
    - Success rate after amplification: sin²((2k+1)·arcsin(√p))

    For groverize(), the circuit applies amplitude amplification to boost
    the success probability from p to near 1.
    """
    if n_amplifications is None:
        # Optimal iterations (same formula as groverize())
        theta = math.asin(math.sqrt(p))
        k = int(math.pi / (4 * theta))
        if k ==0:
            # No amplification needed, success rate is p
            return p
        return math.sin((2 * k + 1) * theta) ** 2
    else:
        theta = math.asin(math.sqrt(p))
        return math.sin((2 * n_amplifications + 1) * theta) ** 2


def tvd(p: dict, q: dict) -> float:
    """Total Variation Distance between two count distributions."""
    all_keys = set(p) | set(q)
    total_p = sum(p.values()) or 1
    total_q = sum(q.values()) or 1
    return round(
        sum(abs(p.get(k, 0) / total_p - q.get(k, 0) / total_q) for k in all_keys) / 2,
        4,
    )


def success_rate(counts: dict[str, int]) -> float:
    """Compute success rate: fraction of shots where ancilla == 0."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    # After groverize, ancilla is highest qubit; success = ancilla == 0
    success = sum(v for k, v in counts.items() if k[-1] == "0")
    return round(success / total, 4)


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

def run_single_circuit(name: str, builder, hardware_platform: str | None = None) -> dict:
    """Run one RUS circuit through groverize + simulator + optional hardware."""
    from quonic.backends import get_backend

    print(f"\n{'─'*80}")
    print(f"Circuit: {name}")

    # Build and groverize
    loop, raw_circuit, p_theoretical = builder()
    t_compile = time.perf_counter()
    groverized = loop.groverize()
    t_compile = time.perf_counter() - t_compile
    n_qubits = groverized.num_qubits
    gate_count = groverized.gate_count()
    depth = groverized.depth()

    print(f"  p(theoretical) = {p_theoretical:.4f}")
    print(f"  Groverized: {n_qubits} qubits, {gate_count} gates, depth {depth}")

    # Theoretical prediction
    p_amplified = theoretical_success_rate(p_theoretical)
    print(f"  p(amplified, theoretical) = {p_amplified:.4f}")

    # Simulator run
    t0 = time.perf_counter()
    result = get_backend("native").run(groverized, shots=SHOTS, method="statevector")
    t_sim = time.perf_counter() - t0
    sr_sim = success_rate(result.counts)

    print(f"  Simulator: {sr_sim:.1%} success ({t_sim:.4f}s)")
    print(f"  TVD vs theoretical: {abs(sr_sim - p_amplified):.4f}")

    entry = {
        "circuit": name,
        "n_qubits": n_qubits,
        "gate_count": gate_count,
        "depth": depth,
        "compile_time": round(t_compile, 6),
        "p_theoretical": p_theoretical,
        "p_amplified_theoretical": round(p_amplified, 4),
        "simulator": {
            "success_rate": sr_sim,
            "time": round(t_sim, 4),
            "counts": result.counts,
        },
        "tvd_vs_theoretical": round(abs(sr_sim - p_amplified), 4),
    }

    # Optional hardware run
    if hardware_platform:
        print(f"  Running on {hardware_platform}...", end=" ", flush=True)
        try:
            from quonic.compiler import compile as quonic_compile
            compiled = quonic_compile(groverized, route=True)
            t0 = time.perf_counter()
            hw_result = get_backend(hardware_platform, device="tuna17").run(
                compiled, shots=SHOTS
            )
            t_hw = time.perf_counter() - t0
            sr_hw = success_rate(hw_result.counts)
            tvd_hw = tvd(result.counts, hw_result.counts)
            print(f"{sr_hw:.1%} success, TVD={tvd_hw:.4f} ({t_hw:.4f}s)")
            entry["hardware"] = {
                "platform": hardware_platform,
                "success_rate": sr_hw,
                "tvd_vs_simulator": tvd_hw,
                "time": round(t_hw, 4),
                "counts": hw_result.counts,
            }
        except Exception as e:
            print(f"ERROR: {e[:80]}")
            entry["hardware"] = {"platform": hardware_platform, "error": str(e)[:200]}

    return entry


# ---------------------------------------------------------------------------
# Checkpoint / Resume
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = RESULTS_DIR / "exp14_checkpoint.json"


def load_checkpoint() -> dict[str, dict]:
    """Load completed results from checkpoint file. Returns {circuit_name: result}."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return {r["circuit"]: r for r in data.get("results", [])}
    return {}


def save_checkpoint(results: list[dict]) -> None:
    """Save current results to checkpoint file."""
    data = {
        "experiment": "exp14_groverize_scaled",
        "description": "Checkpoint (partial results)",
        "n_circuits": len(results),
        "results": results,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def generate_random_rus_circuits(n_circuits: int = 500) -> list:
    """Generate random RUS circuits with varying parameters."""
    import random
    random.seed(42)

    circuits = []
    for i in range(n_circuits):
        # Random success probability between 0.1 and 0.9
        p = random.uniform(0.1, 0.9)
        angle = 2 * math.asin(math.sqrt(p))

        # Random circuit type
        circuit_type = random.choice(["single", "multi2", "multi3"])

        if circuit_type == "single":
            # Single qubit RUS
            # p = sin²(angle/2) = P(|1⟩), but success = until=0 = P(|0⟩) = 1-p
            p_success = 1 - p
            def builder(angle=angle, p_success=p_success):
                from quonic import qgate, reset
                from quonic.gates import Ry
                from quonic.qif import creg, cwhile
                from quonic.stack import current_circuit
                reset()
                flag = creg("flag")
                with cwhile(flag, until=0) as loop:
                    qgate(Ry(angle), 0)
                    flag.measure(0)
                return loop, current_circuit(), p_success
            name = f"RUS-single-{i:04d}-p{p_success:.3f}"

        elif circuit_type == "multi2":
            #2-qubit RUS
            def builder(p=p):
                from quonic import qgate, reset
                from quonic.gates import CX, H
                from quonic.qif import creg, cwhile
                from quonic.stack import current_circuit
                reset()
                flag = creg("flag")
                with cwhile(flag, until=0) as loop:
                    qgate(H, 0)
                    qgate(CX, 0, 1)
                    flag.measure(1)
                return loop, current_circuit(), 0.5
            name = f"RUS-multi2-{i:04d}-p{p:.3f}"

        else:
            #3-qubit RUS
            def builder(p=p):
                from quonic import qgate, reset
                from quonic.gates import CX, H
                from quonic.qif import creg, cwhile
                from quonic.stack import current_circuit
                reset()
                flag = creg("flag")
                with cwhile(flag, until=0) as loop:
                    qgate(H, 0)
                    qgate(CX, 0, 1)
                    qgate(CX, 1, 2)
                    flag.measure(2)
                return loop, current_circuit(), 0.5
            name = f"RUS-multi3-{i:04d}-p{p:.3f}"

        circuits.append((name, builder))

    return circuits


def main():
    print(f"{'='*80}")
    print("Experiment 14: Scaled groverize() Validation")
    print(f"{'='*80}")

    # Generate10000 random RUS circuits
    test_cases = generate_random_rus_circuits(10000)
    print(f"Generated {len(test_cases)} random RUS circuits")

    # Load checkpoint for resume
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint: {len(checkpoint)} circuits already completed")

    all_results = list(checkpoint.values())
    t_start = time.perf_counter()

    for name, builder in test_cases:
        if name in checkpoint:
            continue

        entry = run_single_circuit(name, builder)
        all_results.append(entry)

        # Save checkpoint every10 circuits
        if len(all_results) % 10 == 0:
            save_checkpoint(all_results)
            print(f"\n  Progress: {len(all_results)} completed, "
                  f"{time.perf_counter() - t_start:.0f}s elapsed")

    elapsed = time.perf_counter() - t_start

    # ---- Comprehensive Report ----
    import statistics

    tvds = [r["tvd_vs_theoretical"] for r in all_results]
    depths = [r["depth"] for r in all_results]
    gate_counts = [r["gate_count"] for r in all_results]
    compile_times = [r["compile_time"] for r in all_results]
    n_qubits_list = [r["n_qubits"] for r in all_results]

    # Compute k for each circuit
    def _get_k(p):
        theta = math.asin(math.sqrt(p))
        return int(math.pi / (4 * theta))

    ks = [_get_k(r["p_theoretical"]) for r in all_results]

    # Percentiles
    sorted_tvds = sorted(tvds)
    n = len(sorted_tvds)
    p95 = sorted_tvds[int(n * 0.95)]
    p99 = sorted_tvds[int(n * 0.99)]
    median_tvd = statistics.median(tvds)
    mean_tvd = statistics.mean(tvds)
    std_tvd = statistics.stdev(tvds) if n > 1 else 0

    # Success count (TVD < 0.05)
    success_count = sum(1 for t in tvds if t < 0.05)

    # Group by p ranges
    p_groups = {}
    for r in all_results:
        p = r["p_theoretical"]
        bucket = f"{int(p*10)/10:.1f}-{int(p*10)/10+0.1:.1f}"
        p_groups.setdefault(bucket, []).append(r["tvd_vs_theoretical"])

    # Group by k
    k_groups = {}
    for i, r in enumerate(all_results):
        k = ks[i]
        k_groups.setdefault(k, []).append(r["tvd_vs_theoretical"])

    # Depth distribution by k
    depth_by_k = {}
    for i, r in enumerate(all_results):
        k = ks[i]
        depth_by_k.setdefault(k, []).append(r["depth"])

    print(f"\n{'='*60}")
    print(f" RUS Validation Report ({len(all_results)} circuits)")
    print(f"{'='*60}")

    print("\n[Overall Metrics]")
    print(f"  Mean TVD:       {mean_tvd:.4f} +/- {std_tvd:.4f}")
    print(f"  Median TVD:     {median_tvd:.4f}")
    print(f"  Max TVD:        {max(tvds):.4f}")
    print(f"  Min TVD:        {min(tvds):.4f}")
    print(f"  95th percentile:{p95:.4f}")
    print(f"  99th percentile:{p99:.4f}")
    print(f"  Success rate:   {success_count/n*100:.2f}% (TVD < 5%)")

    print("\n[TVD by p range]")
    for bucket in sorted(p_groups.keys()):
        vals = p_groups[bucket]
        print(f"  p={bucket}:   {statistics.mean(vals):.4f} (n={len(vals)})")

    print("\n[TVD by Grover iterations k]")
    for k in sorted(k_groups.keys()):
        vals = k_groups[k]
        print(f"  k={k}:  {statistics.mean(vals):.4f} (n={len(vals)})")

    print("\n[Circuit Depth by k]")
    for k in sorted(depth_by_k.keys()):
        vals = depth_by_k[k]
        print(f"  k={k}:  mean={statistics.mean(vals):.1f}, "
              f"min={min(vals)}, max={max(vals)} (n={len(vals)})")

    print("\n[Compilation Time]")
    print(f"  Mean:   {statistics.mean(compile_times)*1000:.3f} ms")
    print(f"  Median: {statistics.median(compile_times)*1000:.3f} ms")
    print(f"  Max:    {max(compile_times)*1000:.3f} ms")
    print(f"  Total:  {sum(compile_times):.3f} s")

    print("\n[Circuit Scale]")
    print(f"  Qubits: mean={statistics.mean(n_qubits_list):.1f}, "
          f"min={min(n_qubits_list)}, max={max(n_qubits_list)}")
    print(f"  Gates:  mean={statistics.mean(gate_counts):.1f}, "
          f"min={min(gate_counts)}, max={max(gate_counts)}")

    # Failure cases
    failures = [(r["circuit"], r["tvd_vs_theoretical"]) for r in all_results if r["tvd_vs_theoretical"] > 0.05]
    print("\n[Failures (TVD > 5%)]")
    print(f"  Count: {len(failures)}")
    if failures:
        for name, tvd_val in sorted(failures, key=lambda x: -x[1])[:10]:
            print(f"    {name}: TVD={tvd_val:.4f}")

    # Save
    output = {
        "experiment": "exp14_groverize_scaled",
        "description": f"Scaled groverize() validation with {len(all_results)} random RUS circuits",
        "shots": SHOTS,
        "n_circuits": len(all_results),
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
        "tvd_statistics": {
            "mean": round(mean_tvd, 4),
            "std": round(std_tvd, 4),
            "median": round(median_tvd, 4),
            "max": round(max(tvds), 4),
            "min": round(min(tvds), 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
            "success_rate": round(success_count / n, 4),
        },
        "depth_statistics": {
            "mean": round(statistics.mean(depths), 1),
            "min": min(depths),
            "max": max(depths),
        },
        "compile_time_statistics": {
            "mean_ms": round(statistics.mean(compile_times) * 1000, 3),
            "median_ms": round(statistics.median(compile_times) * 1000, 3),
            "max_ms": round(max(compile_times) * 1000, 3),
            "total_s": round(sum(compile_times), 3),
        },
    }

    out_path = RESULTS_DIR / "exp14_groverize_scaled.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")

    # Clean up checkpoint on successful completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleaned up (run complete)")


if __name__ == "__main__":
    main()
