"""Experiment 15: Scaled Real Workload Validation.

20 quantum algorithms with100 random variants each (where applicable).
Measures algorithm-specific correctness metrics.

Outputs: experiments/paper/results/exp15_real_workloads.json
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHOTS = 1024
CHECKPOINT_PATH = RESULTS_DIR / "exp15_checkpoint.json"


# ---------------------------------------------------------------------------
# Random spec generators (100 variants each)
# ---------------------------------------------------------------------------

def gen_grover(n=100):
    random.seed(1001)
    specs = []
    for i in range(n):
        nq = random.randint(4, 12)
        target = "".join(random.choice("01") for _ in range(nq))
        specs.append({"name": f"Grover-{nq}q-{i:03d}", "type": "grover",
                       "n_qubits": nq, "target": target})
    return specs


def gen_qpe(n=100):
    random.seed(1002)
    specs = []
    for i in range(n):
        np_ = random.randint(4, 10)
        theta = random.uniform(0.01, 0.99)
        specs.append({"name": f"QPE-{np_}p-{i:03d}", "type": "qpe",
                       "n_precision": np_, "theta": theta, "n_qubits": np_ + 1})
    return specs


def gen_qft(n=100):
    random.seed(1003)
    return [{"name": f"QFT-{random.randint(4,16)}q-{i:03d}", "type": "qft",
             "n_qubits": random.randint(4, 16)} for i in range(n)]


def gen_qcounting(n=100):
    random.seed(1004)
    specs = []
    for i in range(n):
        nq = random.randint(4, 6)
        target = "".join(random.choice("01") for _ in range(nq))
        specs.append({"name": f"QCount-{nq}q-{i:03d}", "type": "qcounting",
                       "n_qubits": nq, "target": target})
    return specs


def gen_bv(n=100):
    random.seed(1005)
    specs = []
    for i in range(n):
        nq = random.randint(4, 12)
        secret = "".join(random.choice("01") for _ in range(nq))
        specs.append({"name": f"BV-{nq}q-{i:03d}", "type": "bv",
                       "n_qubits": nq, "secret": secret})
    return specs


def gen_dj(n=100):
    random.seed(1006)
    specs = []
    for i in range(n):
        nq = random.randint(3, 10)
        kind = random.choice(["balanced", "constant"])
        specs.append({"name": f"DJ-{nq}q-{kind[:4]}-{i:03d}", "type": "dj",
                       "n_qubits": nq, "kind": kind})
    return specs


def gen_simon(n=100):
    random.seed(1007)
    specs = []
    for i in range(n):
        nq = random.randint(3, 8)
        secret = "".join(random.choice("01") for _ in range(nq))
        specs.append({"name": f"Simon-{nq}q-{i:03d}", "type": "simon",
                       "n_qubits": nq, "secret": secret})
    return specs


def gen_qwalk(n=100):
    random.seed(1008)
    specs = []
    for i in range(n):
        npos = random.randint(4, 12)
        steps = random.randint(3, 15)
        specs.append({"name": f"QWalk-{npos}p-{steps}s-{i:03d}", "type": "qwalk",
                       "n_positions": npos, "steps": steps})
    return specs


def gen_trotter(n=100):
    random.seed(1009)
    specs = []
    for i in range(n):
        nq = random.randint(2, 6)
        steps = random.randint(1, 10)
        t = random.uniform(0.1, 2.0)
        # Random Ising-like Hamiltonian coefficients
        hz = [random.uniform(-1, 1) for _ in range(nq)]
        jzz = [random.uniform(-1, 1) for _ in range(max(nq - 1, 0))]
        specs.append({"name": f"Trotter-{nq}q-{steps}s-{i:03d}", "type": "trotter",
                       "n_qubits": nq, "steps": steps, "time": t,
                       "hz": hz, "jzz": jzz})
    return specs


def gen_qaoa_maxcut(n=100):
    random.seed(1010)
    specs = []
    for i in range(n):
        nq = random.randint(4, 10)
        p = random.randint(1, 3)
        n_edges = random.randint(nq, nq * 2)
        edges = set()
        while len(edges) < n_edges:
            u, v = random.randint(0, nq - 1), random.randint(0, nq - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        specs.append({"name": f"QAOA-MC-{nq}q-p{p}-{i:03d}", "type": "qaoa_maxcut",
                       "n_qubits": nq, "p": p, "edges": list(edges)})
    return specs


def gen_qaoa_knapsack(n=100):
    random.seed(1011)
    specs = []
    for i in range(n):
        n_items = random.randint(3, 8)
        weights = [random.randint(1, 10) for _ in range(n_items)]
        values = [random.randint(1, 10) for _ in range(n_items)]
        capacity = random.randint(sum(weights) // 3, sum(weights) * 2 // 3)
        specs.append({"name": f"QAOA-KS-{n_items}i-{i:03d}", "type": "qaoa_knapsack",
                       "n_items": n_items, "weights": weights, "values": values,
                       "capacity": capacity, "n_qubits": n_items})
    return specs


def gen_amp_est(n=100):
    random.seed(1012)
    specs = []
    for i in range(n):
        nq = random.randint(2, 6)
        np_ = random.randint(3, 8)
        specs.append({"name": f"AmpEst-{nq}q-{np_}p-{i:03d}", "type": "amp_est",
                       "n_qubits": nq, "n_precision": np_})
    return specs


def gen_clustering(n=100):
    random.seed(1013)
    specs = []
    for i in range(n):
        n_points = random.randint(4, 10)
        k = random.randint(2, min(4, n_points))
        points = [[random.uniform(0, 1), random.uniform(0, 1)] for _ in range(n_points)]
        centroids = [[random.uniform(0, 1), random.uniform(0, 1)] for _ in range(k)]
        specs.append({"name": f"QClust-{n_points}p-{k}c-{i:03d}", "type": "clustering",
                       "points": points, "centroids": centroids, "n_qubits": n_points})
    return specs


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def run_grover(spec):
    from quonic.algorithms import grover, mark_state
    oracle = mark_state(spec["target"])
    t0 = time.perf_counter()
    result = grover(oracle, spec["n_qubits"], shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    tc = result.counts.get(spec["target"], 0)
    total = sum(result.counts.values())
    return {"name": spec["name"], "type": "grover", "n_qubits": spec["n_qubits"],
            "time": round(t, 6), "success_rate": round(tc / total, 4) if total else 0}


def run_qpe(spec):
    from quonic.algorithms import qpe
    t0 = time.perf_counter()
    result = qpe(spec["theta"], n_precision=spec["n_precision"], shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    # Decode estimated phase from top outcome
    top = max(result.counts, key=result.counts.get) if result.counts else "0" * spec["n_precision"]
    j = int(top[:spec["n_precision"]], 2)
    theta_est = j / (2 ** spec["n_precision"])
    phase_error = abs(theta_est - spec["theta"])
    return {"name": spec["name"], "type": "qpe", "n_qubits": spec["n_qubits"],
            "time": round(t, 6), "phase_error": round(phase_error, 6),
            "theta_true": spec["theta"], "theta_est": round(theta_est, 6)}


def run_qft(spec):
    from quonic.algorithms import qft
    t0 = time.perf_counter()
    result = qft(spec["n_qubits"], shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    n_states = 2 ** spec["n_qubits"]
    expected = 1.0 / n_states
    total = sum(result.counts.values())
    chi2 = sum(
        (result.counts.get(format(k, f"0{spec['n_qubits']}b"), 0) / total - expected) ** 2
        for k in range(n_states)
    ) / expected
    return {"name": spec["name"], "type": "qft", "n_qubits": spec["n_qubits"],
            "time": round(t, 6), "uniformity_chi2": round(chi2, 4)}


def run_qcounting(spec):
    from quonic.algorithms import quantum_counting
    t0 = time.perf_counter()
    result = quantum_counting(spec["target"], spec["n_qubits"], shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    # True count is1 (single target state)
    estimated = result.value
    error = abs(estimated - 1)
    return {"name": spec["name"], "type": "qcounting", "n_qubits": spec["n_qubits"],
            "time": round(t, 6), "count_estimate": round(estimated, 4),
            "count_error": round(error, 4)}


def run_bv(spec):
    """Bernstein-Vazirani: oracle f(x) = s·x mod2."""
    from quonic.algorithms import bernstein_vazirani
    from quonic.ir import GateOperation
    secret = spec["secret"]
    nq = spec["n_qubits"]

    def bv_oracle(circuit, n):
        for i, bit in enumerate(reversed(secret)):
            if bit == "1":
                circuit.add(GateOperation("cx", (i, n)))

    t0 = time.perf_counter()
    result = bernstein_vazirani(nq, oracle=bv_oracle, shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    found = result.metadata.get("secret", "") if result.metadata else ""
    success = 1 if found == secret else 0
    return {"name": spec["name"], "type": "bv", "n_qubits": nq,
            "time": round(t, 6), "success": success, "secret": secret}


def run_dj(spec):
    """Deutsch-Jozsa: balanced or constant oracle."""
    from quonic.algorithms import deutsch_jozsa
    from quonic.ir import GateOperation
    nq = spec["n_qubits"]
    kind = spec["kind"]

    if kind == "constant":
        def dj_oracle(circuit, n):
            pass  # f(x) = 0 for all x
    else:
        def dj_oracle(circuit, n):
            circuit.add(GateOperation("cx", (0, n)))

    t0 = time.perf_counter()
    result = deutsch_jozsa(nq, oracle=dj_oracle, shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    # DJ: metadata has 'is_balanced'
    is_balanced = result.metadata.get("is_balanced", False) if result.metadata else False
    detected = "balanced" if is_balanced else "constant"
    success = 1 if detected == kind else 0
    return {"name": spec["name"], "type": "dj", "n_qubits": nq,
            "time": round(t, 6), "success": success, "kind": kind}


def run_simon(spec):
    """Simon's algorithm: find hidden period s."""
    from quonic.algorithms import simon
    from quonic.ir import GateOperation
    nq = spec["n_qubits"]
    secret = spec["secret"]

    def simon_oracle(circuit, n):
        # f(x) = f(x⊕s): copy input to output, then XOR with secret
        for i in range(n):
            circuit.add(GateOperation("cx", (i, n + i)))
        for i, bit in enumerate(reversed(secret)):
            if bit == "1":
                circuit.add(GateOperation("cx", (i, n + i)))

    t0 = time.perf_counter()
    result = simon(nq, oracle=simon_oracle, shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    found_secret = result.metadata.get("secret", "") if result.metadata else ""
    success = 1 if found_secret == secret else 0
    return {"name": spec["name"], "type": "simon", "n_qubits": nq,
            "time": round(t, 6), "success": success, "secret": secret}


def run_qwalk(spec):
    from quonic.algorithms import quantum_walk
    t0 = time.perf_counter()
    result = quantum_walk(spec["n_positions"], steps=spec["steps"], shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    # Entropy of distribution
    total = sum(result.counts.values())
    probs = [v / total for v in result.counts.values() if v > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(spec["n_positions"])
    return {"name": spec["name"], "type": "qwalk", "n_qubits": spec["n_positions"],
            "time": round(t, 6), "entropy": round(entropy, 4),
            "max_entropy": round(max_entropy, 4)}


def run_trotter(spec):
    from quonic.algorithms import trotter
    nq = spec["n_qubits"]
    # Build Hamiltonian as list of (coeff, pauli_string)
    ham = []
    for i, h in enumerate(spec["hz"]):
        pauli = "I" * i + "Z" + "I" * (nq - i - 1)
        ham.append((h, pauli))
    for i, j in enumerate(spec["jzz"]):
        pauli = "I" * i + "ZZ" + "I" * (nq - i - 2)
        ham.append((j, pauli))
    t0 = time.perf_counter()
    result = trotter(ham, time=spec["time"], steps=spec["steps"],
                     n_qubits=nq, shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    return {"name": spec["name"], "type": "trotter", "n_qubits": nq,
            "time": round(t, 6)}


def run_qaoa_maxcut(spec):
    from quonic.algorithms import qaoa_maxcut
    t0 = time.perf_counter()
    result = qaoa_maxcut(spec["edges"], spec["n_qubits"], p=spec["p"], maxiter=100)
    t = time.perf_counter() - t0
    return {"name": spec["name"], "type": "qaoa_maxcut", "n_qubits": spec["n_qubits"],
            "time": round(t, 6), "p": spec["p"],
            "cut_value": result.value if hasattr(result, "value") else None}


def run_qaoa_knapsack(spec):
    from quonic.algorithms import qaoa_knapsack
    t0 = time.perf_counter()
    result = qaoa_knapsack(spec["weights"], spec["values"], spec["capacity"],
                           p=2, maxiter=100)
    t = time.perf_counter() - t0
    return {"name": spec["name"], "type": "qaoa_knapsack", "n_qubits": spec["n_qubits"],
            "time": round(t, 6),
            "value": result.value if hasattr(result, "value") else None}


def run_amp_est(spec):
    from quonic.algorithms import amplitude_estimation
    t0 = time.perf_counter()
    result = amplitude_estimation(spec["n_qubits"], n_precision=spec["n_precision"],
                                  shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    est = result.value if result.value is not None else 0
    return {"name": spec["name"], "type": "amp_est", "n_qubits": spec["n_qubits"],
            "time": round(t, 6), "estimate": round(est, 6)}


def run_clustering(spec):
    from quonic.algorithms import quantum_clustering
    t0 = time.perf_counter()
    result = quantum_clustering(spec["points"], spec["centroids"],
                                max_iter=20, shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    return {"name": spec["name"], "type": "clustering", "n_qubits": spec["n_qubits"],
            "time": round(t, 6)}


# Single-run algorithms
def run_vqe():
    from quonic.algorithms import molecule_vqe
    t0 = time.perf_counter()
    result = molecule_vqe(maxiter=200)
    t = time.perf_counter() - t0
    exact = -1.8573
    energy_err = abs(result.value - exact)
    return {"name": "VQE-H2", "type": "vqe", "n_qubits": 2, "time": round(t, 6),
            "energy": round(result.value, 6), "energy_error": round(energy_err, 6),
            "converged": energy_err < 0.01}


def run_hhl():
    from quonic.algorithms import hhl
    # Random2x2 diagonal matrix
    random.seed(9999)
    a = random.uniform(0.5, 2.0)
    b = random.uniform(0.5, 2.0)
    matrix = [[a, 0], [0, b]]
    vector = [1.0, 0.0]
    t0 = time.perf_counter()
    result = hhl(matrix, vector, n_clock=4, shots=SHOTS, backend="native")
    t = time.perf_counter() - t0
    return {"name": "HHL-2x2", "type": "hhl", "n_qubits": 6, "time": round(t, 6),
            "matrix": matrix}


def run_qaoa_tsp():
    from quonic.algorithms import qaoa_tsp
    random.seed(9998)
    n_cities = 4
    distances = {}
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            d = round(random.uniform(1, 10), 2)
            distances[(i, j)] = d
    t0 = time.perf_counter()
    result = qaoa_tsp(distances, n_cities, p=2, maxiter=100)
    t = time.perf_counter() - t0
    return {"name": "QAOA-TSP-4", "type": "qaoa_tsp", "n_qubits": n_cities ** 2,
            "time": round(t, 6),
            "tour_length": result.value if hasattr(result, "value") else None}


def run_qnn():
    from quonic.algorithms import qnn
    t0 = time.perf_counter()
    result = qnn(n_qubits=4, depth=3)
    t = time.perf_counter() - t0
    return {"name": "QNN-4q-3d", "type": "qnn", "n_qubits": 4, "time": round(t, 6)}


def run_qgan():
    from quonic.algorithms import qgan
    t0 = time.perf_counter()
    result = qgan(n_steps=100)
    t = time.perf_counter() - t0
    return {"name": "QGAN-100s", "type": "qgan", "n_qubits": 4, "time": round(t, 6)}


def run_qsvm():
    from quonic.algorithms import qsvm
    t0 = time.perf_counter()
    result = qsvm()
    t = time.perf_counter() - t0
    return {"name": "QSVM", "type": "qsvm", "n_qubits": 2, "time": round(t, 6),
            "accuracy": result.value if hasattr(result, "value") else None}


def run_ham_sim():
    from quonic.algorithms import hamiltonian_simulation
    t0 = time.perf_counter()
    result = hamiltonian_simulation()
    t = time.perf_counter() - t0
    return {"name": "HamSim", "type": "ham_sim", "n_qubits": 2, "time": round(t, 6)}


# Runner dispatch
RUNNERS = {
    "grover": run_grover,
    "qpe": run_qpe,
    "qft": run_qft,
    "qcounting": run_qcounting,
    "bv": run_bv,
    "dj": run_dj,
    "simon": run_simon,
    "qwalk": run_qwalk,
    "trotter": run_trotter,
    "qaoa_maxcut": run_qaoa_maxcut,
    "qaoa_knapsack": run_qaoa_knapsack,
    "amp_est": run_amp_est,
    "clustering": run_clustering,
}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return {r["name"]: r for r in data.get("results", [])}
    return {}


def save_checkpoint(results):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{'='*80}")
    print("Experiment 15: Scaled Real Workload Validation (20 algorithms)")
    print(f"{'='*80}")

    # Generate all specs
    all_specs = []
    all_specs.extend(gen_grover())
    all_specs.extend(gen_qpe())
    all_specs.extend(gen_qft())
    all_specs.extend(gen_qcounting())
    all_specs.extend(gen_bv())
    all_specs.extend(gen_dj())
    all_specs.extend(gen_simon())
    all_specs.extend(gen_qwalk())
    all_specs.extend(gen_trotter())
    all_specs.extend(gen_qaoa_maxcut())
    all_specs.extend(gen_qaoa_knapsack())
    all_specs.extend(gen_amp_est())
    all_specs.extend(gen_clustering())
    print(f"Generated {len(all_specs)} random circuit specs")

    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resuming: {len(checkpoint)} already done")

    all_results = list(checkpoint.values())
    t_start = time.perf_counter()

    # Run randomized algorithms
    for spec in all_specs:
        if spec["name"] in checkpoint:
            continue
        runner = RUNNERS[spec["type"]]
        try:
            entry = runner(spec)
            all_results.append(entry)
        except Exception as e:
            all_results.append({"name": spec["name"], "type": spec["type"],
                                "n_qubits": spec.get("n_qubits", 0),
                                "error": str(e)[:200]})
        if len(all_results) % 50 == 0:
            save_checkpoint(all_results)
            print(f"  Progress: {len(all_results)}, {time.perf_counter()-t_start:.0f}s")

    # Run single-run algorithms
    singles = [
        ("VQE-H2", run_vqe),
        ("HHL-2x2", run_hhl),
        ("QAOA-TSP-4", run_qaoa_tsp),
        ("QNN-4q-3d", run_qnn),
        ("QGAN-100s", run_qgan),
        ("QSVM", run_qsvm),
        ("HamSim", run_ham_sim),
    ]
    for label, runner in singles:
        if label in checkpoint:
            continue
        print(f"  Running {label}...")
        try:
            entry = runner()
            all_results.append(entry)
            print(f"    Done: {entry['time']:.4f}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results.append({"name": label, "type": label.lower(), "error": str(e)[:200]})

    elapsed = time.perf_counter() - t_start

    # ---- Report ----
    by_type = {}
    for r in all_results:
        by_type.setdefault(r.get("type", "?"), []).append(r)

    print(f"\n{'='*60}")
    print(f" Real Workload Report ({len(all_results)} circuits)")
    print(f"{'='*60}")

    errors = [r for r in all_results if "error" in r]
    print(f"\nTotal: {len(all_results)} circuits, {len(errors)} errors, {elapsed:.1f}s")

    # Per-algorithm summary
    algo_metrics = {
        "grover": ("success_rate", "Success Rate"),
        "qpe": ("phase_error", "Phase Error"),
        "qft": ("uniformity_chi2", "Uniformity chi2"),
        "qcounting": ("count_error", "Count Error"),
        "bv": ("success", "Success Rate"),
        "dj": ("success", "Success Rate"),
        "simon": ("success", "Success Rate"),
        "qwalk": ("entropy", "Entropy"),
        "trotter": (None, None),
        "qaoa_maxcut": ("cut_value", "Cut Value"),
        "qaoa_knapsack": ("value", "Knapsack Value"),
        "amp_est": ("estimate", "Estimate"),
        "clustering": (None, None),
        "vqe": ("error", "Energy Error"),
        "hhl": (None, None),
        "qaoa_tsp": (None, None),
        "qnn": (None, None),
        "qgan": (None, None),
        "qsvm": (None, None),
        "ham_sim": (None, None),
    }

    print(f"\n{'Algorithm':<20} {'n':>5} {'Err':>4} {'Time(ms)':>10} {'Metric':>12} {'Value':>10}")
    print("-" * 65)
    for algo_type, results in sorted(by_type.items()):
        ok = [r for r in results if "error" not in r]
        errs = len(results) - len(ok)
        times = [r["time"] for r in ok if "time" in r]
        if not times:
            print(f"{algo_type:<20} {len(results):>5} {errs:>4} {'N/A':>10}")
            continue
        mean_t = statistics.mean(times) * 1000
        metric_key, metric_name = algo_metrics.get(algo_type, (None, None))
        if metric_key and ok:
            vals = [r[metric_key] for r in ok if r.get(metric_key) is not None]
            if vals:
                mean_v = statistics.mean(vals)
                print(f"{algo_type:<20} {len(results):>5} {errs:>4} {mean_t:>10.2f} "
                      f"{metric_name:>12} {mean_v:>10.4f}")
            else:
                print(f"{algo_type:<20} {len(results):>5} {errs:>4} {mean_t:>10.2f}")
        else:
            print(f"{algo_type:<20} {len(results):>5} {errs:>4} {mean_t:>10.2f}")

    # Time by qubit count
    print(f"\n[Time by qubit count]")
    by_q = {}
    for r in all_results:
        if "error" in r or "time" not in r:
            continue
        n = r.get("n_qubits", 0)
        by_q.setdefault(n, []).append(r["time"])
    for n in sorted(by_q):
        vals = by_q[n]
        print(f"  {n:2d} qubits: mean={statistics.mean(vals)*1000:.2f}ms, n={len(vals)}")

    # Save
    output = {
        "experiment": "exp15_real_workloads",
        "description": f"Scaled real workload validation: {len(all_results)} circuits, 20 algorithms",
        "shots": SHOTS,
        "n_circuits": len(all_results),
        "n_algorithms": len(by_type),
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
        "by_type": {
            t: {"count": len(rs), "errors": sum(1 for r in rs if "error" in r),
                "mean_time_ms": round(statistics.mean(
                    [r["time"] for r in rs if "time" in r and "error" not in r]
                ) * 1000, 2) if any("time" in r and "error" not in r for r in rs) else None}
            for t, rs in by_type.items()
        },
    }
    out_path = RESULTS_DIR / "exp15_real_workloads.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleaned up")


if __name__ == "__main__":
    main()
