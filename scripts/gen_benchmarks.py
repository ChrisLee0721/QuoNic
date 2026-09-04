"""Generate comprehensive benchmarks.json with parallel execution.

Runs representative circuits across all (n, class, backend/method) combinations
using 4-way parallelism. Updates scheduler's benchmarks.json.

Usage:
    python scripts/gen_benchmarks.py
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BENCHMARKS_PATH = Path(__file__).parent.parent / "src" / "quonic" / "scheduler" / "data" / "benchmarks.json"
SHOTS = 1024
MAX_WORKERS = 2  # Reduced to avoid OOM with multiple large-circuit backends
TIMEOUT = 120  # seconds per run

# Backends to benchmark and their supported methods
BACKEND_METHODS = {
    "native": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "qiskit": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "cupy": ["statevector", "density_matrix"],
    "cirq": ["statevector"],
    "qpanda": ["statevector", "density_matrix"],
}


# ── Circuit builders ──────────────────────────────────────────────────────

def build_ghz(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit
    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def build_qaoa(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
    for _ in range(2):
        for i in range(0, n - 1, 2):
            qgate(CX, i, i + 1)
            qgate(Rz(0.5), i + 1)
            qgate(CX, i, i + 1)
        for i in range(n):
            qgate(H, i)
            qgate(Rz(0.3), i)
            qgate(H, i)
    return current_circuit()


def build_qft(n: int):
    import math
    from quonic import qgate, reset
    from quonic.gates import CP, H
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qgate(CP(angle), j, i)
    return current_circuit()


def build_random_clifford(n: int, depth: int, seed: int = 42):
    import random as rng_mod
    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, X, Y, Z
    from quonic.stack import current_circuit
    rng = rng_mod.Random(seed)
    single = [H, X, Y, Z]
    double = [CX, CZ]
    reset()
    for _ in range(depth):
        for q in range(n):
            if rng.random() < 0.7:
                qgate(rng.choice(single), q)
        pairs = list(zip(range(0, n - 1, 2), range(1, n, 2)))
        rng.shuffle(pairs)
        for ctrl, tgt in pairs[: max(1, len(pairs) // 2)]:
            qgate(rng.choice(double), ctrl, tgt)
    return current_circuit()


def build_random_non_clifford(n: int, depth: int, seed: int = 42):
    import random as rng_mod
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rx, Ry, Rz
    from quonic.stack import current_circuit
    rng = rng_mod.Random(seed)
    single = [H, Rx, Ry, Rz]
    reset()
    for _ in range(depth):
        for q in range(n):
            if rng.random() < 0.7:
                gate = rng.choice(single)
                if gate in (Rx, Ry, Rz):
                    qgate(gate(rng.uniform(0, 6.28)), q)
                else:
                    qgate(gate, q)
        pairs = list(zip(range(0, n - 1, 2), range(1, n, 2)))
        rng.shuffle(pairs)
        for ctrl, tgt in pairs[: max(1, len(pairs) // 2)]:
            qgate(CX, ctrl, tgt)
    return current_circuit()


# ── Job definitions ───────────────────────────────────────────────────────

def make_jobs():
    """Generate all benchmark jobs: (n, circuit_class, gate_bucket, depth_bucket, backend, method, builder)."""
    jobs = []

    circuit_types = {
        "clifford": [("ghz", build_ghz), ("random_clifford", lambda n: build_random_clifford(n, n * 2))],
        "low_tw": [("qaoa", build_qaoa), ("random_nc", lambda n: build_random_non_clifford(n, n))],
        "general": [("qft", build_qft)],
    }

    n_values = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    for n in n_values:
        for cls, builders in circuit_types.items():
            for name, builder in builders:
                circuit = builder(n)
                gate_count = circuit.gate_count()
                depth = circuit.depth()

                gc_bucket = "small" if gate_count < 50 else ("medium" if gate_count < 200 else "large")
                d_bucket = "shallow" if depth < 20 else ("medium" if depth < 50 else "deep")

                for backend, methods in BACKEND_METHODS.items():
                    for method in methods:
                        # Skip density_matrix for n > 14 (OOM)
                        if method == "density_matrix" and n > 14:
                            continue
                        # Skip stabilizer for non-Clifford circuits
                        if method == "stabilizer" and cls != "clifford":
                            continue
                        # Skip native/statevector for n >= 26 (OOM: 2^n * 16 bytes)
                        if backend == "native" and method == "statevector" and n >= 26:
                            continue
                        # Skip native/density_matrix for n > 12 (OOM: 2^2n)
                        if backend == "native" and method == "density_matrix" and n > 12:
                            continue
                        jobs.append({
                            "n": n,
                            "class": cls,
                            "gate_bucket": gc_bucket,
                            "depth_bucket": d_bucket,
                            "gate_count": gate_count,
                            "depth": depth,
                            "backend": backend,
                            "method": method,
                            "circuit": circuit,
                        })

    return jobs


# ── Worker function ───────────────────────────────────────────────────────

def run_single(job: dict) -> dict:
    """Run a single benchmark. Called in a subprocess."""
    import threading
    from quonic.backends import get_backend

    backend = job["backend"]
    method = job["method"]
    circuit = job["circuit"]

    result_box = {}

    def _run():
        try:
            be = get_backend(backend)
            result = be.run(circuit, shots=SHOTS, method=method)
            result_box["time"] = result
        except Exception as e:
            result_box["error"] = str(e)[:100]

    t0 = time.perf_counter()
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=TIMEOUT)
    elapsed = time.perf_counter() - t0

    if thread.is_alive():
        return {**job, "time": round(elapsed, 4), "error": f"Timeout ({TIMEOUT}s)"}
    if "error" in result_box:
        return {**job, "time": round(elapsed, 4), "error": result_box["error"]}

    return {**job, "time": round(elapsed, 4), "error": None}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Generating benchmark jobs (all backends)...")
    jobs = make_jobs()

    # Remove circuit object (not serializable for ProcessPoolExecutor)
    serializable_jobs = []
    for j in jobs:
        sj = {k: v for k, v in j.items() if k != "circuit"}
        serializable_jobs.append(sj)

    print(f"Total jobs: {len(serializable_jobs)}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Backends: {', '.join(BACKEND_METHODS.keys())}")
    print(f"Estimated time: {len(serializable_jobs) * 5 / MAX_WORKERS / 60:.0f}-{len(serializable_jobs) * 30 / MAX_WORKERS / 60:.0f} min")
    print()

    # Run with parallelism
    results = []
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for sj in serializable_jobs:
            future = pool.submit(_run_job_in_worker, sj)
            futures[future] = sj

        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                r = future.result()
                results.append(r)
                status = "OK" if r["error"] is None else f"ERR: {r['error'][:30]}"
                bm_key = f"{r['backend']}/{r['method']}"
                print(f"  [{done}/{len(serializable_jobs)}] n={r['n']:2d} {r['class']:<10} "
                      f"{bm_key:<35} {r['time']:.3f}s {status}")
            except Exception as e:
                sj = futures[future]
                bm_key = f"{sj['backend']}/{sj['method']}"
                print(f"  [{done}/{len(serializable_jobs)}] n={sj['n']:2d} {sj['class']:<10} "
                      f"{bm_key:<35} EXCEPTION: {e}")

    total_time = time.perf_counter() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Build decision table: key -> {timings: {backend/method: time}, method: fastest_backend/method}
    decisions = {}
    for r in results:
        if r["error"] is not None:
            continue
        key = f"{r['n']}|{r['class']}|{r['gate_bucket']}|{r['depth_bucket']}"
        if key not in decisions:
            decisions[key] = {"timings": {}}
        bm_key = f"{r['backend']}/{r['method']}"
        decisions[key]["timings"][bm_key] = round(r["time"], 4)

    # Pick fastest for each key
    for key, val in decisions.items():
        timings = val["timings"]
        if timings:
            val["method"] = min(timings, key=timings.get)

    # Build performance table
    perf_entries = []
    for r in results:
        if r["error"] is not None:
            continue
        bm_key = f"{r['backend']}/{r['method']}"
        perf_entries.append({
            "n": r["n"],
            "class": r["class"],
            "gate_count": r["gate_count"],
            "depth": r["depth"],
            "timings": {bm_key: round(r["time"], 4)},
        })

    # Merge timings for same (n, class, gate_count, depth)
    merged = {}
    for entry in perf_entries:
        key = (entry["n"], entry["class"], entry["gate_count"], entry["depth"])
        if key not in merged:
            merged[key] = entry
        else:
            merged[key]["timings"].update(entry["timings"])

    # Update benchmarks.json
    with open(BENCHMARKS_PATH) as f:
        benchmarks = json.load(f)

    benchmarks["performance"] = list(merged.values())
    benchmarks["decision"] = decisions
    benchmarks["meta"]["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    benchmarks["meta"]["source"] = "gen_benchmarks.py (all-backends parallel)"
    benchmarks["meta"]["total_jobs"] = len(serializable_jobs)
    benchmarks["meta"]["successful"] = sum(1 for r in results if r["error"] is None)
    benchmarks["meta"]["backends"] = list(BACKEND_METHODS.keys())

    with open(BENCHMARKS_PATH, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nUpdated {BENCHMARKS_PATH}")
    print(f"Decision entries: {len(decisions)}")
    print(f"Performance entries: {len(merged)}")


def _run_job_in_worker(sj: dict) -> dict:
    """Rebuild circuit and run benchmark in worker process."""
    cls = sj["class"]
    n = sj["n"]
    gc = sj["gate_count"]

    if cls == "clifford":
        if gc < 50:
            circuit = build_ghz(n)
        else:
            circuit = build_random_clifford(n, n * 2)
    elif cls == "low_tw":
        if gc < 200:
            circuit = build_qaoa(n)
        else:
            circuit = build_random_non_clifford(n, n)
    else:  # general
        circuit = build_qft(n)

    job = {**sj, "circuit": circuit}
    return run_single(job)


if __name__ == "__main__":
    main()
