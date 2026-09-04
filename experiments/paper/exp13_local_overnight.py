"""Experiment 13 Local: Infinite overnight run.

Keeps generating random circuit batches until Ctrl+C.
Each batch: 9 families x 4 qubit counts x shallow/medium = ~72 circuits.
Runs only ≤16 qubits to avoid density_matrix bottleneck.

Outputs: experiments/paper/results/exp13_local_overnight.json
"""

from __future__ import annotations

import json
import math
import random
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024
MAX_QUBITS = 16

ALL_BACKEND_METHODS = {
    "native": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "qiskit": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "cirq": ["statevector"],
    "qpanda": ["statevector", "density_matrix"],
}

CHECKPOINT_PATH = RESULTS_DIR / "exp13_local_checkpoint.json"

# Graceful shutdown
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    print("\n[!] Ctrl+C received, finishing current circuit then saving...")
    _shutdown = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def build_ghz(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H
    from quonic.stack import current_circuit
    reset()
    qgate(H, 0)
    for i in range(n - 1):
        qgate(CX, i, i + 1)
    return current_circuit()


def build_linear_cluster(n: int):
    from quonic import qgate, reset
    from quonic.gates import CZ, H
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
    for i in range(n - 1):
        qgate(CZ, i, i + 1)
    return current_circuit()


def build_qft(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            qgate(Rz(angle / 2), j)
            qgate(CX, j, i)
            qgate(Rz(-angle / 2), i)
            qgate(CX, j, i)
    return current_circuit()


def build_grover(n: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, X
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
    for i in range(n):
        qgate(X, i)
    qgate(H, n - 1)
    for i in range(n - 1):
        qgate(CX, i, n - 1)
    qgate(H, n - 1)
    for i in range(n):
        qgate(X, i)
    for i in range(n):
        qgate(H, i)
    for i in range(n):
        qgate(X, i)
    qgate(H, n - 1)
    for i in range(n - 1):
        qgate(CX, i, n - 1)
    qgate(H, n - 1)
    for i in range(n):
        qgate(X, i)
    for i in range(n):
        qgate(H, i)
    return current_circuit()


def build_vqe(n: int, depth: int, seed: int):
    from quonic import qgate, reset
    from quonic.gates import CX, Ry
    from quonic.stack import current_circuit
    rng = random.Random(seed)
    reset()
    for d in range(depth):
        for i in range(n):
            qgate(Ry(rng.random() * 2 * math.pi), i)
        for i in range(n - 1):
            qgate(CX, i, i + 1)
    return current_circuit()


def build_qpe(n_estimate: int, n_target: int = 1):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit
    n = n_estimate + n_target
    reset()
    for i in range(n_estimate):
        qgate(H, i)
    for i in range(n_estimate):
        angle = math.pi / (2 ** i)
        qgate(Rz(angle / 2), n_estimate)
        qgate(CX, i, n_estimate)
        qgate(Rz(-angle / 2), n_estimate)
        qgate(CX, i, n_estimate)
    return current_circuit()


def build_qaoa(n: int, depth: int, seed: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit
    rng = random.Random(seed)
    reset()
    for i in range(n):
        qgate(H, i)
    for _ in range(depth):
        for i in range(0, n - 1, 2):
            qgate(CX, i, i + 1)
            qgate(Rz(rng.random() * 2), i + 1)
            qgate(CX, i, i + 1)
        for i in range(n):
            qgate(H, i)
            qgate(Rz(rng.random()), i)
            qgate(H, i)
    return current_circuit()


def build_random_clifford(n: int, depth: int, seed: int):
    from quonic import qgate, reset
    from quonic.gates import CX, CZ, H, Z
    from quonic.stack import current_circuit
    rng = random.Random(seed)
    reset()
    clifford_1q = [H, Z]
    clifford_2q = [CX, CZ]
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(clifford_1q)
            qgate(gate, q)
        for i in range(n - 1):
            gate = rng.choice(clifford_2q)
            qgate(gate, i, i + 1)
    return current_circuit()


def build_random_non_clifford(n: int, depth: int, seed: int):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rx, Ry, Rz
    from quonic.stack import current_circuit
    rng = random.Random(seed)
    reset()
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice([H, Rx, Ry, Rz])
            if gate in (Rx, Ry, Rz):
                qgate(gate(rng.uniform(0, 6.28)), q)
            else:
                qgate(gate, q)
        for i in range(n - 1):
            qgate(CX, i, i + 1)
    return current_circuit()


# ---------------------------------------------------------------------------
# Generate one batch of circuits (varied seeds)
# ---------------------------------------------------------------------------

def generate_batch(batch_id: int) -> list[tuple[str, object, dict]]:
    """Generate one batch of ~72 circuits with unique seeds."""
    circuits = []
    rng = random.Random(batch_id * 1000 + 42)

    ns = [4, 6, 8, 10, 12, 16]
    suffix = f"-b{batch_id}"

    # Structured Clifford
    for n in [4, 8, 12, 16]:
        circuits.append((f"GHZ-{n}{suffix}", build_ghz(n),
                         {"class": "clifford", "type": "structured", "family": "GHZ", "batch": batch_id}))
    for n in [4, 8, 12, 16]:
        circuits.append((f"LinCluster-{n}{suffix}", build_linear_cluster(n),
                         {"class": "clifford", "type": "structured", "family": "LinearCluster", "batch": batch_id}))

    # Structured General
    for n in [4, 8, 12, 16]:
        circuits.append((f"QFT-{n}{suffix}", build_qft(n),
                         {"class": "general", "type": "structured", "family": "QFT", "batch": batch_id}))
    for n in ns:
        circuits.append((f"Grover-{n}{suffix}", build_grover(n),
                         {"class": "clifford", "type": "structured", "family": "Grover", "batch": batch_id}))

    # Structured Low treewidth
    for n in ns:
        circuits.append((f"QPE-{n}{suffix}", build_qpe(n),
                         {"class": "low_tw", "type": "structured", "family": "QPE", "batch": batch_id}))

    # Variational (randomized params per batch)
    for n in ns:
        seed_vqe = rng.randint(0, 99999)
        depth_vqe = rng.choice([2, 3, 4])
        circuits.append((f"VQE-{n}{suffix}", build_vqe(n, depth_vqe, seed_vqe),
                         {"class": "low_tw", "type": "variational", "family": "VQE", "batch": batch_id}))
    for n in ns:
        seed_qaoa = rng.randint(0, 99999)
        depth_qaoa = rng.choice([1, 2, 3])
        circuits.append((f"QAOA-{n}{suffix}", build_qaoa(n, depth_qaoa, seed_qaoa),
                         {"class": "low_tw", "type": "variational", "family": "QAOA", "batch": batch_id}))

    # Random Clifford (shallow + medium)
    for n in [8, 12, 16]:
        for depth_label, depth_factor in [("shallow", 2), ("medium", 5)]:
            depth = max(n, n * depth_factor // 5)
            seed = rng.randint(0, 99999)
            circuits.append(
                (f"RandCliff-{n}-{depth_label}{suffix}", build_random_clifford(n, depth, seed),
                 {"class": "clifford", "type": "random", "family": "RandomClifford",
                  "depth_bucket": depth_label, "batch": batch_id})
            )

    # Random NonClifford (shallow + medium)
    for n in [8, 12, 16]:
        for depth_label, depth_factor in [("shallow", 2), ("medium", 5)]:
            depth = max(n, n * depth_factor // 5)
            seed = rng.randint(0, 99999)
            circuits.append(
                (f"RandNonCliff-{n}-{depth_label}{suffix}", build_random_non_clifford(n, depth, seed),
                 {"class": "low_tw", "type": "random", "family": "RandomNonClifford",
                  "depth_bucket": depth_label, "batch": batch_id})
            )

    return circuits


# ---------------------------------------------------------------------------
# Run one circuit
# ---------------------------------------------------------------------------

def _run_single(backend: str, method: str, circuit, shots: int) -> dict:
    from quonic.backends import get_backend
    try:
        t0 = time.perf_counter()
        r = get_backend(backend).run(circuit, shots=shots, method=method)
        t = time.perf_counter() - t0
        return {"time": round(t, 4), "outcomes": len(r.counts)}
    except Exception as e:
        return {"time": 0, "outcomes": 0, "error": str(e)[:80]}


def run_circuit_benchmark(circuit, name: str, meta: dict) -> dict:
    from quonic.backends import get_backend
    from quonic.scheduler import schedule
    from quonic.scheduler.capabilities import decision_class, eligible_methods
    from quonic.scheduler.features import circuit_features

    features = circuit_features(circuit)
    eligible = eligible_methods(features["gate_types"], noise=False)
    dclass = decision_class(features)
    rec = schedule(circuit)

    t0 = time.perf_counter()
    get_backend(rec.backend).run(circuit, shots=SHOTS, method=rec.method)
    time_rec = time.perf_counter() - t0

    work = []
    for backend, methods in ALL_BACKEND_METHODS.items():
        for method in methods:
            if method == "density_matrix" and circuit.num_qubits > 14:
                continue
            if method == "stabilizer" and not features["is_clifford"]:
                continue
            if method not in eligible and method != "density_matrix":
                continue
            work.append((backend, method))

    all_timings = {}
    with ThreadPoolExecutor(max_workers=min(len(work), 8)) as pool:
        futures = {
            pool.submit(_run_single, b, m, circuit, SHOTS): f"{b}/{m}"
            for b, m in work
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result(timeout=60)
                all_timings[key] = result
                status = f"{result['time']:.2f}s" if result['time'] > 0 else f"ERR:{result.get('error','?')[:30]}"
                print(f"    {key}: {status}", flush=True)
            except Exception as e:
                all_timings[key] = {"time": 0, "outcomes": 0, "error": str(e)[:80]}
                print(f"    {key}: ERR:{str(e)[:30]}", flush=True)

    valid = {k: v for k, v in all_timings.items() if v["time"] > 0 and "error" not in v}
    fastest_key = min(valid, key=lambda k: valid[k]["time"]) if valid else "none"
    fastest_time = valid.get(fastest_key, {}).get("time", 0)
    rec_key = f"{rec.backend}/{rec.method}"
    rec_time = all_timings.get(rec_key, {}).get("time", time_rec)

    return {
        "circuit": name,
        "n_qubits": circuit.num_qubits,
        "gate_count": circuit.gate_count(),
        "depth": circuit.depth(),
        "features": {
            "n": features["n"],
            "depth": features["depth"],
            "gate_count": features["gate_count"],
            "is_clifford": features["is_clifford"],
            "treewidth_ub": features["treewidth_ub"],
            "entanglement": features["entanglement"],
        },
        "meta": meta,
        "decision_class": dclass,
        "recommendation": {"backend": rec.backend, "method": rec.method},
        "recommended_time": round(time_rec, 4),
        "all_timings": all_timings,
        "fastest": {"key": fastest_key, "time": fastest_time},
        "scheduler_is_fastest": rec_key == fastest_key,
        "overhead_vs_fastest": round(rec_time / fastest_time, 2) if fastest_time > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict[str, dict]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return {r["circuit"]: r for r in data.get("results", [])}
    return {}


def save_checkpoint(results: list[dict]) -> None:
    data = {
        "experiment": "exp13_local_overnight",
        "description": f"Checkpoint: {len(results)} circuits",
        "n_circuits": len(results),
        "results": results,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_statistics(results: list[dict]) -> dict:
    overheads = [r["overhead_vs_fastest"] for r in results if r["overhead_vs_fastest"] > 0]
    if not overheads:
        return {}
    overheads_sorted = sorted(overheads)
    n = len(overheads_sorted)
    return {
        "n_circuits": n,
        "scheduler_picked_fastest": sum(1 for r in results if r["scheduler_is_fastest"]),
        "accuracy_pct": round(100 * sum(1 for r in results if r["scheduler_is_fastest"]) / n, 1),
        "mean": round(sum(overheads) / n, 2),
        "median": round(overheads_sorted[n // 2], 2),
        "p95": round(overheads_sorted[int(n * 0.95)], 2),
        "worst_case": round(max(overheads), 2),
        "geometric_mean": round(math.exp(sum(math.log(x) for x in overheads) / n), 2),
    }


def compute_class_breakdown(results: list[dict]) -> dict:
    by_class = {}
    for r in results:
        cls = r["decision_class"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(r)
    breakdown = {}
    for cls, items in by_class.items():
        overheads = [r["overhead_vs_fastest"] for r in items if r["overhead_vs_fastest"] > 0]
        if not overheads:
            continue
        n = len(overheads)
        breakdown[cls] = {
            "n_circuits": n,
            "accuracy_pct": round(100 * sum(1 for r in items if r["scheduler_is_fastest"]) / n, 1),
            "mean_overhead": round(sum(overheads) / n, 2),
            "worst_case": round(max(overheads), 2),
        }
    return breakdown


def compute_scaling_curves(results: list[dict]) -> dict:
    curves = {}
    for r in results:
        family = r["meta"].get("family", "unknown")
        n = r["n_qubits"]
        if family not in curves:
            curves[family] = []
        curves[family].append({
            "n": n,
            "overhead": r["overhead_vs_fastest"],
            "scheduler": f"{r['recommendation']['backend']}/{r['recommendation']['method']}",
            "fastest": r["fastest"]["key"],
            "class": r["decision_class"],
        })
    for family in curves:
        curves[family].sort(key=lambda x: x["n"])
    return curves


# ---------------------------------------------------------------------------
# Main: infinite loop
# ---------------------------------------------------------------------------

def main():
    global _shutdown

    print("=" * 70)
    print("Experiment 13 Local: Infinite Overnight Run")
    print("Press Ctrl+C to stop and save")
    print("=" * 70)

    checkpoint = load_checkpoint()
    all_results = list(checkpoint.values())
    done_names = {r["circuit"] for r in all_results}

    if checkpoint:
        print(f"Resuming from checkpoint: {len(checkpoint)} already done")

    t_start = time.perf_counter()
    batch_id = 0
    total_circuits = 0

    while not _shutdown:
        batch = generate_batch(batch_id)
        new_circuits = [(name, circ, meta) for name, circ, meta in batch if name not in done_names]

        if not new_circuits:
            batch_id += 1
            continue

        print(f"\n{'='*60}")
        print(f"Batch {batch_id}: {len(new_circuits)} new circuits "
              f"(total done: {len(all_results)})")
        print(f"{'='*60}")

        # Run4 circuits in parallel
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = {}
            for name, circuit, meta in new_circuits:
                if _shutdown:
                    break
                total_circuits += 1
                futures[pool.submit(run_circuit_benchmark, circuit, name, meta)] = (
                    name, circuit.num_qubits, circuit.gate_count()
                )

            for future in as_completed(futures):
                if _shutdown:
                    break
                name, nq, ng = futures[future]
                try:
                    entry = future.result()
                    all_results.append(entry)
                    done_names.add(name)

                    rec = entry["recommendation"]
                    fastest = entry["fastest"]
                    overhead = entry["overhead_vs_fastest"]
                    is_fastest = entry["scheduler_is_fastest"]
                    print(f"\n  [{len(all_results)}] {name} ({nq}q, {ng} gates) | "
                          f"{rec['backend']}/{rec['method']} "
                          f"vs {fastest['key']} | {overhead}x "
                          f"{'BEST' if is_fastest else ''}")

                    # Checkpoint every10 circuits
                    if len(all_results) % 10 == 0:
                        save_checkpoint(all_results)
                        elapsed = time.perf_counter() - t_start
                        print(f"\n  --- Checkpoint: {len(all_results)} circuits, "
                              f"{elapsed/60:.1f} min ---")

                except Exception as e:
                    print(f"\n  {name} | ERROR: {e}")
                    all_results.append({
                        "circuit": name, "n_qubits": nq,
                        "error": str(e)[:200],
                    })
                    done_names.add(name)

        batch_id += 1

    # --- Save final results ---
    elapsed = time.perf_counter() - t_start
    success = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]

    stats = compute_statistics(success)
    curves = compute_scaling_curves(success)
    class_breakdown = compute_class_breakdown(success)

    print(f"\n{'='*70}")
    print(f" FINAL RESULTS ({len(success)} success, {len(failed)} failed)")
    print(f"{'='*70}")

    print(f"\n[Scheduler Accuracy]")
    print(f"  Circuits:      {stats.get('n_circuits', 0)}")
    print(f"  Accuracy:      {stats.get('accuracy_pct', 0)}% (picked fastest)")
    print(f"  Mean overhead: {stats.get('mean', 0)}x")
    print(f"  Median:        {stats.get('median', 0)}x")
    print(f"  P95:           {stats.get('p95', 0)}x")
    print(f"  Worst case:    {stats.get('worst_case', 0)}x")
    print(f"  Geometric mean:{stats.get('geometric_mean', 0)}x")

    print(f"\n[By Decision Class]")
    for cls, info in sorted(class_breakdown.items()):
        print(f"  {cls:12s}: {info['n_circuits']} circuits, "
              f"{info['accuracy_pct']}% accuracy, "
              f"{info['mean_overhead']}x mean, "
              f"{info['worst_case']}x worst")

    print(f"\n[Scaling Curves]")
    for family in sorted(curves):
        points = curves[family]
        print(f"  {family}:")
        for p in points:
            print(f"    n={p['n']:2d}: overhead={p['overhead']}x "
                  f"({p['scheduler']} vs {p['fastest']})")

    # Save
    output = {
        "experiment": "exp13_local_overnight",
        "description": f"Local infinite run: {len(success)} circuits, max {MAX_QUBITS} qubits",
        "shots": SHOTS,
        "max_qubits": MAX_QUBITS,
        "n_circuits": len(success),
        "n_failed": len(failed),
        "n_batches": batch_id + 1,
        "elapsed_seconds": round(elapsed, 1),
        "results": success,
        "scaling_curves": curves,
        "statistics": stats,
        "class_breakdown": class_breakdown,
    }

    out_path = RESULTS_DIR / "exp13_local_overnight.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")

    # Clean checkpoint on clean exit
    if CHECKPOINT_PATH.exists() and not failed:
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleaned up")


if __name__ == "__main__":
    main()
