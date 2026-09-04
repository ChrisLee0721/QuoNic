"""Experiment 13: Comprehensive Benchmark Suite + Scheduler Scaling.

Merged experiment covering:
- Experiment A: Scheduler scaling curves (Clifford/General/Low_tw swept by qubit count)
- Experiment B: Statistical distribution of scheduler accuracy over 50-100 circuits

Generates circuits across the parameter space:
  Qubit count: 4, 8, 12, 16, 20, 24, 28, 32
  Circuit class: clifford, general, low_tw
  Circuit type: structured (GHZ, QFT, QPE, Grover) + random + variational (VQE, QAOA)
  Depth: shallow / medium / deep

Outputs:
  experiments/paper/results/exp13_benchmark_suite.json
  experiments/paper/results/exp13_scaling_curves.json
"""

from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024
THREAD_TIMEOUT = 120  # seconds per backend-method run

ALL_BACKEND_METHODS = {
    "native": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "qiskit": ["statevector", "stabilizer", "matrix_product_state", "density_matrix"],
    "cirq": ["statevector"],
    "qpanda": ["statevector", "density_matrix"],
}


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


def build_vqe(n: int, depth: int = 3):
    from quonic import qgate, reset
    from quonic.gates import CX, Ry
    from quonic.stack import current_circuit
    rng = random.Random(123)
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
    n_estimate + n_target
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


def build_qaoa(n: int, depth: int = 2):
    from quonic import qgate, reset
    from quonic.gates import CX, H, Rz
    from quonic.stack import current_circuit
    reset()
    for i in range(n):
        qgate(H, i)
    for _ in range(depth):
        for i in range(0, n - 1, 2):
            qgate(CX, i, i + 1)
            qgate(Rz(0.5), i + 1)
            qgate(CX, i, i + 1)
        for i in range(n):
            qgate(H, i)
            qgate(Rz(0.3), i)
            qgate(H, i)
    return current_circuit()


def build_random_clifford(n: int, depth: int, seed: int = 42):
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


def build_random_non_clifford(n: int, depth: int, seed: int = 42):
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
# Circuit generation: cover parameter space
# ---------------------------------------------------------------------------

def generate_circuits() -> list[tuple[str, object, dict]]:
    """Generate circuits covering the parameter space.

    Returns list of (name, circuit, metadata) tuples.
    Metadata includes: class, type, n, depth_bucket for analysis.
    """
    circuits = []
    rng = random.Random(99)

    # Qubit counts to sweep
    ns = [4, 8, 12, 16, 20, 24, 28, 32]

    # --- Structured circuits (Clifford) ---
    for n in ns:
        circuits.append((f"GHZ-{n}", build_ghz(n),
                         {"class": "clifford", "type": "structured", "family": "GHZ"}))
    for n in ns:
        circuits.append((f"LinearCluster-{n}", build_linear_cluster(n),
                         {"class": "clifford", "type": "structured", "family": "LinearCluster"}))

    # --- Structured circuits (General / high treewidth) ---
    for n in [4, 8, 12, 16, 20, 24]:
        circuits.append((f"QFT-{n}", build_qft(n),
                         {"class": "general", "type": "structured", "family": "QFT"}))
    for n in [4, 6, 8, 10, 12, 16]:
        circuits.append((f"Grover-{n}", build_grover(n),
                         {"class": "clifford", "type": "structured", "family": "Grover"}))

    # --- Structured circuits (Low treewidth) ---
    for n in [4, 6, 8, 10, 12, 16, 20]:
        circuits.append((f"QPE-{n}", build_qpe(n),
                         {"class": "low_tw", "type": "structured", "family": "QPE"}))

    # --- Variational circuits (Low treewidth) ---
    for n in [4, 6, 8, 10, 12, 16, 20]:
        circuits.append((f"VQE-{n}", build_vqe(n, depth=3),
                         {"class": "low_tw", "type": "variational", "family": "VQE"}))
    for n in [4, 6, 8, 10, 12, 16, 20]:
        circuits.append((f"QAOA-{n}", build_qaoa(n, depth=2),
                         {"class": "low_tw", "type": "variational", "family": "QAOA"}))

    # --- Random circuits (vary depth for depth bucket coverage) ---
    for n in [8, 12, 16, 20, 24]:
        for depth_label, depth_factor in [("shallow", 2), ("medium", 5), ("deep", 10)]:
            depth = max(n, n * depth_factor // 5)
            seed = rng.randint(0, 10000)
            circuits.append(
                (f"RandCliff-{n}-{depth_label}", build_random_clifford(n, depth, seed),
                 {"class": "clifford", "type": "random", "family": "RandomClifford",
                  "depth_bucket": depth_label})
            )

    for n in [8, 12, 16, 20]:
        for depth_label, depth_factor in [("shallow", 2), ("medium", 5), ("deep", 10)]:
            depth = max(n, n * depth_factor // 5)
            seed = rng.randint(0, 10000)
            circuits.append(
                (f"RandNonCliff-{n}-{depth_label}", build_random_non_clifford(n, depth, seed),
                 {"class": "low_tw", "type": "random", "family": "RandomNonClifford",
                  "depth_bucket": depth_label})
            )

    return circuits


# ---------------------------------------------------------------------------
# Run one circuit against all backend-method pairs
# ---------------------------------------------------------------------------

def _run_single(backend: str, method: str, circuit, shots: int) -> dict:
    """Run one backend-method pair, return timing or error."""
    from quonic.backends import get_backend
    try:
        t0 = time.perf_counter()
        r = get_backend(backend).run(circuit, shots=shots, method=method)
        t = time.perf_counter() - t0
        return {"time": round(t, 4), "outcomes": len(r.counts)}
    except Exception as e:
        return {"time": 0, "outcomes": 0, "error": str(e)[:80]}


def run_circuit_benchmark(circuit, name: str, meta: dict) -> dict:
    """Run one circuit against all eligible backend-method pairs (parallel)."""
    from quonic.backends import get_backend
    from quonic.scheduler import schedule
    from quonic.scheduler.capabilities import decision_class, eligible_methods
    from quonic.scheduler.features import circuit_features

    features = circuit_features(circuit)
    eligible = eligible_methods(features["gate_types"], noise=False)
    dclass = decision_class(features)
    rec = schedule(circuit)

    # Run scheduler recommendation
    t0 = time.perf_counter()
    get_backend(rec.backend).run(circuit, shots=SHOTS, method=rec.method)
    time_rec = time.perf_counter() - t0

    # Build work items
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

    # Run in parallel
    all_timings = {}
    # Run all backends (no timeout - let them complete)
    with ThreadPoolExecutor(max_workers=min(len(work), 8)) as pool:
        futures = {
            pool.submit(_run_single, b, m, circuit, SHOTS): f"{b}/{m}"
            for b, m in work
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                all_timings[key] = future.result(timeout=5)
            except Exception as e:
                all_timings[key] = {"time": 0, "outcomes": 0, "error": str(e)[:80]}

    # Analysis
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
# Analysis: scaling curves + statistical distribution
# ---------------------------------------------------------------------------

def compute_scaling_curves(results: list[dict]) -> dict:
    """Extract scaling curves for Experiment A analysis.

    Groups by family and plots overhead vs n.
    """
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

    # Sort each curve by n
    for family in curves:
        curves[family].sort(key=lambda x: x["n"])

    return curves


def compute_statistics(results: list[dict]) -> dict:
    """Compute statistical distribution of scheduler overhead."""
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
    """Per-class statistics."""
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


# ---------------------------------------------------------------------------
# Checkpoint / Resume
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = RESULTS_DIR / "exp13_checkpoint.json"


def load_checkpoint() -> dict[str, dict]:
    """Load completed results from checkpoint file. Returns {circuit_name: result}."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return {r["circuit"]: r for r in data.get("results", [])}
    return {}


def save_checkpoint(results: list[dict]) -> None:
    """Save current results to checkpoint file (append-safe)."""
    data = {
        "experiment": "exp13_benchmark_suite",
        "description": "Checkpoint (partial results)",
        "n_circuits": len(results),
        "results": results,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _worker_run_batch(batch: list[tuple[str, dict]], shots: int) -> list[dict]:
    """Worker process: run a batch of circuits (each identified by name + meta).

    Circuits are regenerated inside the worker to avoid pickling Circuit objects.
    """
    # Re-import quonic in worker process

    # Map names to builder functions
    builders = {
        "GHZ": lambda n: build_ghz(n),
        "LinearCluster": lambda n: build_linear_cluster(n),
        "QFT": lambda n: build_qft(n),
        "Grover": lambda n: build_grover(n),
        "QPE": lambda n: build_qpe(n),
        "VQE": lambda n: build_vqe(n, depth=3),
        "QAOA": lambda n: build_qaoa(n, depth=2),
        "RandCliff": lambda n, d, s: build_random_clifford(n, d, s),
        "RandNonCliff": lambda n, d, s: build_random_non_clifford(n, d, s),
    }

    results = []
    for name, meta in batch:
        # Parse circuit name to reconstruct
        parts = name.split("-")
        family = parts[0]
        n = int(parts[1])

        if family in ("RandCliff", "RandNonCliff"):
            depth_label = parts[2]
            depth_map = {"shallow": 2, "medium": 5, "deep": 10}
            depth = max(n, n * depth_map[depth_label] // 5)
            circuit = builders[family](n, depth, 42)
        else:
            circuit = builders[family](n)

        result = run_circuit_benchmark(circuit, name, meta)
        results.append(result)

        rec = f"{result['recommendation']['backend']}/{result['recommendation']['method']}"
        fast = result["fastest"]["key"]
        oh = result["overhead_vs_fastest"]
        match = "YES" if result["scheduler_is_fastest"] else "NO"
        print(f"  Done: {name}: Scheduler={rec} | Fastest={fast} | Match={match} | Overhead={oh:.2f}x",
              flush=True)

    return results


def generate_bonus_circuits(n_bonus: int = 120) -> list[tuple[str, dict]]:
    """Generate bonus random circuits with different seeds to expand statistics."""
    rng = random.Random(2024)
    bonus = []
    ns = [4, 8, 12, 16, 20, 24, 28]
    depth_labels = ["shallow", "medium", "deep"]
    for i in range(n_bonus):
        n = rng.choice(ns)
        seed = rng.randint(0, 99999)
        depth_label = rng.choice(depth_labels)
        depth_map = {"shallow": 2, "medium": 5, "deep": 10}
        max(n, n * depth_map[depth_label] // 5)
        if rng.random() < 0.5:
            name = f"Bonus-RandCliff-{n}-{depth_label}-{seed}"
            meta = {"class": "clifford", "type": "random", "family": "RandomClifford",
                    "depth_bucket": depth_label, "seed": seed, "bonus": True}
        else:
            name = f"Bonus-RandNonCliff-{n}-{depth_label}-{seed}"
            meta = {"class": "low_tw", "type": "random", "family": "RandomNonClifford",
                    "depth_bucket": depth_label, "seed": seed, "bonus": True}
        bonus.append((name, meta))
    return bonus


def generate_template_compile_tasks() -> list[tuple[str, dict]]:
    """Generate tasks to test algorithm template compilation.

    Each task: build a template circuit + compile it, measuring compile time.
    """
    tasks = []
    templates = [
        ("Grover-4", lambda: grover("1100", 4, shots=1)),  # noqa: F821
        ("Grover-8", lambda: grover("11001100", 8, shots=1)),  # noqa: F821
        ("QFT-4", lambda: qft(4, shots=1)),  # noqa: F821
        ("QFT-8", lambda: qft(8, shots=1)),  # noqa: F821
        ("QFT-12", lambda: qft(12, shots=1)),  # noqa: F821
        ("QPE-pi-4", lambda: qpe(3.14159, n_precision=4, shots=1)),  # noqa: F821
        ("QPE-pi-8", lambda: qpe(3.14159, n_precision=8, shots=1)),  # noqa: F821
        ("Teleportation", lambda: teleportation(shots=1)),  # noqa: F821
        ("BB84-4", lambda: bb84(n_bits=4, shots=1)),  # noqa: F821
        ("BernsteinVazirani-4", lambda: bernstein_vazirani("1011", shots=1)),  # noqa: F821
        ("DeutschJozsa-3", lambda: deutsch_jozsa(n_qubits=3, shots=1)),  # noqa: F821
        ("BitFlip-3", lambda: bit_flip_code(shots=1)),  # noqa: F821
        ("PhaseFlip-3", lambda: phase_flip_code(shots=1)),  # noqa: F821
    ]
    for name, builder in templates:
        tasks.append((f"Template-{name}", {"type": "template_compile", "template": name}))
    return tasks


def _worker_run_one(item: tuple[str, dict], shots: int) -> dict | None:
    """Worker process: run a single circuit. Returns result or None on skip."""
    name, meta = item

    # Template compilation test
    if meta.get("type") == "template_compile":
        return _run_template_compile(name, meta)

    # Re-import quonic in worker process
    builders = {
        "GHZ": lambda n: build_ghz(n),
        "LinearCluster": lambda n: build_linear_cluster(n),
        "QFT": lambda n: build_qft(n),
        "Grover": lambda n: build_grover(n),
        "QPE": lambda n: build_qpe(n),
        "VQE": lambda n: build_vqe(n, depth=3),
        "QAOA": lambda n: build_qaoa(n, depth=2),
        "RandCliff": lambda n, d, s: build_random_clifford(n, d, s),
        "RandNonCliff": lambda n, d, s: build_random_non_clifford(n, d, s),
        "Bonus-RandCliff": lambda n, d, s: build_random_clifford(n, d, s),
        "Bonus-RandNonCliff": lambda n, d, s: build_random_non_clifford(n, d, s),
    }

    parts = name.split("-")
    # Handle "Bonus-RandCliff" / "Bonus-RandNonCliff" prefix
    if parts[0] == "Bonus" and len(parts) >= 3:
        family = f"{parts[0]}-{parts[1]}"
        n = int(parts[2])
        depth_label = parts[3] if len(parts) > 3 else "medium"
        seed = int(parts[4]) if len(parts) > 4 else 42
    else:
        family = parts[0]
        n = int(parts[1])
        depth_label = parts[2] if len(parts) > 2 else "medium"
        seed = int(parts[3]) if len(parts) > 3 else 42

    if family in ("RandCliff", "RandNonCliff", "Bonus-RandCliff", "Bonus-RandNonCliff"):
        depth_map = {"shallow": 2, "medium": 5, "deep": 10}
        depth = max(n, n * depth_map.get(depth_label, 5) // 5)
        circuit = builders[family](n, depth, seed)
    else:
        circuit = builders[family](n)

    result = run_circuit_benchmark(circuit, name, meta)

    rec = f"{result['recommendation']['backend']}/{result['recommendation']['method']}"
    fast = result["fastest"]["key"]
    oh = result["overhead_vs_fastest"]
    match = "YES" if result["scheduler_is_fastest"] else "NO"
    print(f"  Done: {name}: {rec} vs {fast} | {match} | {oh:.2f}x", flush=True)

    return result


def _run_template_compile(name: str, meta: dict) -> dict:
    """Test algorithm template compilation: build circuit + compile + measure time."""
    from quonic.algorithms import (
        bb84,
        bernstein_vazirani,
        bit_flip_code,
        deutsch_jozsa,
        grover,
        phase_flip_code,
        qft,
        qpe,
        teleportation,
    )
    from quonic.compiler import compile as quonic_compile

    template_name = meta["template"]

    builders = {
        "Grover-4": lambda: grover("1100", 4, shots=1),
        "Grover-8": lambda: grover("11001100", 8, shots=1),
        "QFT-4": lambda: qft(4, shots=1),
        "QFT-8": lambda: qft(8, shots=1),
        "QFT-12": lambda: qft(12, shots=1),
        "QPE-pi-4": lambda: qpe(3.14159, n_precision=4, shots=1),
        "QPE-pi-8": lambda: qpe(3.14159, n_precision=8, shots=1),
        "Teleportation": lambda: teleportation(shots=1),
        "BB84-4": lambda: bb84(n_bits=4, shots=1),
        "BernsteinVazirani-4": lambda: bernstein_vazirani("1011", shots=1),
        "DeutschJozsa-3": lambda: deutsch_jozsa(n_qubits=3, shots=1),
        "BitFlip-3": lambda: bit_flip_code(shots=1),
        "PhaseFlip-3": lambda: phase_flip_code(shots=1),
    }

    try:
        # Build template (this exercises the template API)
        t0 = time.perf_counter()
        result = builders[template_name]()
        t_build = time.perf_counter() - t0

        # Get the circuit from the result for compilation
        circuit = getattr(result, 'circuit', None)
        if circuit is None:
            # Some templates return Result directly, need to get circuit from IR
            from quonic.ir import Circuit
            circuit = Circuit()  # fallback

        # Compile (this exercises the compiler pipeline)
        t0 = time.perf_counter()
        quonic_compile(circuit, route=True)
        t_compile = time.perf_counter() - t0

        print(f"  Template: {name} build={t_build:.4f}s compile={t_compile:.4f}s", flush=True)

        return {
            "circuit": name,
            "type": "template_compile",
            "template": template_name,
            "build_time": round(t_build, 4),
            "compile_time": round(t_compile, 4),
            "n_qubits": circuit.num_qubits if hasattr(circuit, 'num_qubits') else 0,
            "gate_count": circuit.gate_count() if hasattr(circuit, 'gate_count') else 0,
        }
    except Exception as e:
        print(f"  Template {name} ERROR: {e}", flush=True)
        return {
            "circuit": name,
            "type": "template_compile",
            "template": template_name,
            "error": str(e)[:200],
        }


# Global semaphore for large circuit throttling (set in main)
_large_sem = None


def _run_with_throttle(item):
    """Module-level function for ProcessPoolExecutor (must be picklable)."""
    name, meta = item
    # Extract qubit count from name: find first numeric part
    n = 0
    for part in name.split("-"):
        try:
            n = int(part)
            break
        except ValueError:
            continue
    is_large = n >= 26

    if is_large and _large_sem is not None:
        _large_sem.acquire()
    try:
        return _worker_run_one(item, SHOTS)
    finally:
        if is_large and _large_sem is not None:
            _large_sem.release()


def main():
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    global _large_sem

    # Config
    MAX_RUNTIME_HOURS = 2  # stop after this many hours
    MAX_RUNTIME = MAX_RUNTIME_HOURS * 3600

    print(f"{'='*90}")
    print("Experiment 13: Comprehensive Benchmark Suite + Scheduler Scaling")
    print(f"{'='*90}")
    print(f"Max runtime: {MAX_RUNTIME_HOURS} hours")

    # Generate circuits
    print("\nGenerating circuits...")
    circuits = generate_circuits()
    print(f"Generated {len(circuits)} planned circuits")

    # Generate bonus circuits
    bonus_circuits = generate_bonus_circuits(120)
    print(f"Generated {len(bonus_circuits)} bonus random circuits")

    # Generate template compilation tasks
    template_tasks = generate_template_compile_tasks()
    print(f"Generated {len(template_tasks)} template compilation tasks")

    # Load checkpoint for resume
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint: {len(checkpoint)} circuits already completed")

    # Filter out already-completed circuits
    planned_pending = [(name, meta) for name, _, meta in circuits if name not in checkpoint]
    bonus_pending = [(name, meta) for name, meta in bonus_circuits if name not in checkpoint]
    template_pending = [(name, meta) for name, meta in template_tasks if name not in checkpoint]
    all_pending = planned_pending + bonus_pending + template_pending
    print(f"Total pending: {len(all_pending)} ({len(planned_pending)} planned + "
          f"{len(bonus_pending)} bonus + {len(template_pending)} templates)")

    # Use half of CPUs to avoid OOM (each circuit runs on multiple backends)
    n_cpus = os.cpu_count() or 4
    n_workers = min(n_cpus // 2, len(all_pending), 16)  # max 16 workers
    print(f"Using {n_workers} worker processes ({n_cpus} CPUs)")

    # Run with continuous task generation
    all_results = list(checkpoint.values())
    mp.Lock()
    _large_sem = mp.Semaphore(1)  # Only 1 large circuit at a time to avoid OOM
    completed_count = 0
    bonus_gen_counter = 0
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Submit initial batch
        futures = {
            pool.submit(_run_with_throttle, item): item[0]
            for item in all_pending
        }

        while futures:
            # Check time limit
            elapsed = time.perf_counter() - t_start
            if elapsed >= MAX_RUNTIME:
                print(f"\nTime limit reached ({MAX_RUNTIME_HOURS}h). Cancelling remaining tasks...")
                for f in futures:
                    f.cancel()
                break

            # Wait for next completion
            done = set()
            for future in futures:
                if future.done():
                    done.add(future)

            if not done:
                # No futures done yet, wait a bit
                import time as _time
                _time.sleep(0.1)
                continue

            for future in done:
                name = futures.pop(future)
                try:
                    result = future.result(timeout=1)
                    if result is not None:
                        all_results.append(result)
                        completed_count += 1
                        if completed_count % 5 == 0:
                            save_checkpoint(all_results)
                            print(f"\n  Progress: {completed_count} completed, "
                                  f"{len(futures)} in flight, "
                                  f"{elapsed:.0f}s elapsed", flush=True)
                except Exception as e:
                    print(f"\n  ERROR {name}: {e}", flush=True)

            # Auto-generate more tasks if queue is running low (with memory check)
            if len(futures) < n_workers:
                # Check available memory before submitting
                mem = psutil.virtual_memory()
                if mem.percent > 85:
                    print(f"\n  Memory high ({mem.percent:.0f}%), skipping task generation", flush=True)
                else:
                    n_new = min(n_workers - len(futures), 8)  # max 8 at a time
                    bonus_gen_counter += 1
                    new_tasks = generate_bonus_circuits(n_new)
                    print(f"\n  Auto-generating {len(new_tasks)} more circuits "
                          f"(batch #{bonus_gen_counter})", flush=True)
                    for item in new_tasks:
                        f = pool.submit(_run_with_throttle, item)
                        futures[f] = item[0]

    # Final checkpoint save
    save_checkpoint(all_results)
    elapsed = time.perf_counter() - t_start

    # Analysis
    print(f"\n{'='*90}")
    print("ANALYSIS")
    print(f"{'='*90}")

    stats = compute_statistics(all_results)
    print(f"\nOverall Statistics ({stats['n_circuits']} circuits):")
    print(f"  Accuracy: {stats['accuracy_pct']}% ({stats['scheduler_picked_fastest']}/{stats['n_circuits']})")
    print(f"  Mean overhead: {stats['mean']}x")
    print(f"  Median: {stats['median']}x")
    print(f"  P95: {stats['p95']}x")
    print(f"  Worst-case: {stats['worst_case']}x")
    print(f"  Geometric mean: {stats['geometric_mean']}x")

    breakdown = compute_class_breakdown(all_results)
    print("\nPer-class breakdown:")
    for cls, b in breakdown.items():
        print(f"  {cls}: {b['accuracy_pct']}% accuracy, "
              f"mean {b['mean_overhead']}x, worst {b['worst_case']}x ({b['n_circuits']} circuits)")

    # Scaling curves
    curves = compute_scaling_curves(all_results)
    print("\nScaling curves (per family):")
    for family, points in curves.items():
        print(f"  {family}: {len(points)} points, n range [{points[0]['n']}..{points[-1]['n']}]")

    # Save results
    output_full = {
        "experiment": "exp13_benchmark_suite",
        "description": "Comprehensive benchmark suite + scheduler scaling",
        "n_circuits": len(all_results),
        "shots": SHOTS,
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
        "statistics": stats,
        "class_breakdown": breakdown,
    }

    output_scaling = {
        "experiment": "exp13_scaling_curves",
        "description": "Scheduler scaling curves by circuit family",
        "curves": curves,
    }

    path_full = RESULTS_DIR / "exp13_benchmark_suite.json"
    path_scaling = RESULTS_DIR / "exp13_scaling_curves.json"

    with open(path_full, "w") as f:
        json.dump(output_full, f, indent=2, default=str)
    with open(path_scaling, "w") as f:
        json.dump(output_scaling, f, indent=2, default=str)

    print(f"\nResults saved to {path_full}")
    print(f"Scaling curves saved to {path_scaling}")
    print(f"Total time: {elapsed:.1f}s")

    # Clean up checkpoint on successful completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleaned up (run complete)")


if __name__ == "__main__":
    main()
