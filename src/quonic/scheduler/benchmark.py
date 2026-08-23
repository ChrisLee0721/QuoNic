"""Offline benchmark: measure "gate type x bit count x method" crossover points on a reference machine and generate data.

It produces two things for the scheduler to query at runtime (the moat = others can copy the code but not the measured data you have accumulated):

- **capability matrix** (capabilities, static) -- which gates each method handles / whether noise is supported
- **performance data** (performance + decision, dynamic) -- per-method timings + crossover thresholds

Usage:
    python -m quonic.scheduler.benchmark -o scheduler/data/benchmarks.json

Note: timings correlate strongly with CPU / memory / BLAS / backend versions and drift
across machines. So this script must be re-runnable for calibration; what ships with the
package is a frozen table from a "reference machine" for cold-start fallback.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .._i18n import tr
from ..backends import get_backend
from ..ir import Circuit, GateOperation
from ..noise import NoiseModel, depolarizing
from .capabilities import METHOD_CAPABILITIES

# the "challenger method" for each circuit class (the alternative to statevector)
_ALT_METHOD: dict[str, str] = {"clifford": "stabilizer", "low_tw": "matrix_product_state"}


def _timed_run(
    circuit: Circuit,
    backend: str,
    method: str,
    shots: int = 256,
    noise: NoiseModel | float | None = None,
) -> float | None:
    """Run once and return the elapsed time; return None when the method does not support this circuit.

    When the method does not support the circuit, Aer prints an error to stderr;
    this is silenced here, keeping only the timing. noise can be a NoiseModel or
    a probability value, passed to the backend (for the density_matrix noise benchmark).
    """
    be = get_backend(backend)
    t0 = time.time()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            be.run(circuit, shots=shots, method=method, noise=noise)
    except (ImportError, RuntimeError, ValueError):
        return None
    return time.time() - t0


# ---------------------------------------------------------------------------
# Circuit families
# ---------------------------------------------------------------------------

def _ghz(n: int) -> Circuit:
    """GHZ: a pure basic Clifford chain (H + CX), treewidth=1."""
    c = Circuit()
    c.add(GateOperation("h", (0,)))
    for i in range(n - 1):
        c.add(GateOperation("cx", (i, i + 1)))
    return c


def _chain_rotation(n: int) -> Circuit:
    """Low-treewidth chain with rotations: non-Clifford (rz) + CX chain, treewidth=1."""
    c = Circuit()
    c.add(GateOperation("rz", (0,), (0.3,)))
    for i in range(n - 1):
        c.add(GateOperation("cx", (i, i + 1)))
    return c


def _qaoa(n: int, p: int = 1) -> Circuit:
    """QAOA warm-up: one layer of ry/rz rotations + a CX chain, non-Clifford and low treewidth."""
    c = Circuit()
    for q in range(n):
        c.add(GateOperation("ry", (q,), (0.5,)))
    for i in range(n - 1):
        c.add(GateOperation("cx", (i, i + 1)))
    for q in range(n):
        c.add(GateOperation("rz", (q,), (0.4,)))
    return c


def _grover(n: int) -> Circuit:
    """One Grover iteration: oracle + diffusion (ccx is non-Clifford, higher treewidth)."""
    c = Circuit()
    for q in range(n):
        c.add(GateOperation("h", (q,)))
    # oracle: flip the phase of the target state |1...1> (built with ccx, non-Clifford)
    c.add(GateOperation("x", (0,)))
    c.add(GateOperation("ccx", (0, 1, n - 1)))
    c.add(GateOperation("x", (0,)))
    # diffusion operator: full H layer + full X layer + mcz + full X layer + full H layer
    for q in range(n):
        c.add(GateOperation("h", (q,)))
        c.add(GateOperation("x", (q,)))
    c.add(GateOperation("mcz", tuple(range(n))))
    for q in range(n):
        c.add(GateOperation("x", (q,)))
        c.add(GateOperation("h", (q,)))
    return c


def _qft(n: int) -> Circuit:
    """Quantum Fourier transform: fully-connected controlled phases (cp), non-Clifford with treewidth n-1.

    The standard QFT H + controlled-rotation structure (the trailing bit-order
    reversal is omitted, which does not affect treewidth/timing characteristics).
    """
    c = Circuit()
    for i in range(n):
        c.add(GateOperation("h", (i,)))
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            c.add(GateOperation("cp", (j, i), (angle,)))
    return c


def representative_circuits() -> list[tuple[str, Circuit]]:
    """Representative circuits for demos / quick benchmarks."""
    return [
        ("ghz24", _ghz(24)),
        ("qaoa24", _qaoa(24)),
        ("grover8", _grover(8)),
    ]


# ---------------------------------------------------------------------------
# Grid benchmark
# ---------------------------------------------------------------------------

def benchmark_methods(
    circuit: Circuit,
    methods: Iterable[str],
    backend: str = "qiskit",
    shots: int = 256,
    repeats: int = 3,
) -> dict[str, float]:
    """Measure each method's timing and take the minimum over repeats (to suppress single-run timing jitter)."""
    timings: dict[str, float] = {}
    for m in methods:
        samples: list[float] = []
        for _ in range(repeats):
            t = _timed_run(circuit, backend, m, shots)
            if t is not None:
                samples.append(t)
        if samples:
            timings[m] = round(min(samples), 4)
    return timings


def benchmark_grid(
    n_values: Iterable[int] = (8, 12, 16, 20, 24),
    backend: str = "qiskit",
    shots: int = 256,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """For the clifford / low_tw classes, measure each method's timing per n."""
    _timed_run(_ghz(4), backend, "statevector", shots)  # warm-up: trigger backend import/compile
    performance: list[dict[str, Any]] = []
    for n in n_values:
        c = _ghz(n)
        performance.append({
            "n": n,
            "class": "clifford",
            "timings": benchmark_methods(
                c, ("statevector", "stabilizer", "matrix_product_state"), backend, shots, repeats
            ),
        })
        c = _chain_rotation(n)
        performance.append({
            "n": n,
            "class": "low_tw",
            "timings": benchmark_methods(
                c, ("statevector", "matrix_product_state"), backend, shots, repeats
            ),
        })
    return performance


def benchmark_general(
    n_values: Iterable[int] = (8, 12, 16),
    backend: str = "qiskit",
    shots: int = 256,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Measure the statevector timing of high-treewidth non-Clifford circuits (QFT / Grover).

    Only statevector can run these two circuit families (they include mcz /
    fully-connected cp); this function does not find crossover points but
    validates the "general -> statevector" classification and records the 2^n
    ceiling of statevector as n grows.
    """
    _timed_run(_ghz(4), backend, "statevector", shots)  # warm-up
    performance: list[dict[str, Any]] = []
    for n in n_values:
        for name, fn in (("qft", _qft), ("grover", _grover)):
            samples: list[float] = []
            for _ in range(repeats):
                t = _timed_run(fn(n), backend, "statevector", shots)
                if t is not None:
                    samples.append(t)
            if samples:
                performance.append({
                    "circuit": name,
                    "n": n,
                    "time": round(min(samples), 4),
                })
    return performance


def benchmark_noise(
    n_values: Iterable[int] = (2, 4, 6, 8, 10, 12),
    noise: float = 0.01,
    backend: str = "qiskit",
    shots: int = 256,
    repeats: int = 3,
    budget: float = 0.5,
) -> dict[str, Any]:
    """Measure the density_matrix cost curve under noise (the only noise-capable method, 4^n resources).

    Returns {"method", "noise", "budget", "performance", "infeasible_n"}:
    infeasible_n is the first bit count whose timing exceeds budget seconds;
    None if the whole grid runs within budget.
    """
    _timed_run(_ghz(2), backend, "density_matrix", shots, noise=depolarizing(noise))  # warm-up
    performance: list[dict[str, Any]] = []
    infeasible_n: int | None = None
    for n in n_values:
        samples: list[float] = []
        for _ in range(repeats):
            t = _timed_run(_ghz(n), backend, "density_matrix", shots, noise=depolarizing(noise))
            if t is not None:
                samples.append(t)
        if not samples:
            continue
        tmin = round(min(samples), 4)
        performance.append({"n": n, "time": tmin})
        if infeasible_n is None and tmin > budget:
            infeasible_n = n
    return {
        "method": "density_matrix",
        "noise": noise,
        "budget": budget,
        "performance": performance,
        "infeasible_n": infeasible_n,
    }


def derive_decision(
    performance: list[dict[str, Any]], margin: float = 0.2
) -> dict[str, dict[str, Any]]:
    """Derive the crossover point for each class from measured data.

    The crossover point is the smallest n where the alternative method is first
    "clearly" faster than statevector. The alternative must be faster by at least
    margin (default 20%) to keep microsecond-scale timing jitter from wobbling
    the crossover point at small n -- at small n both methods are on the order of
    milliseconds, and a 1% difference is noise that must not change routing.
    """
    decision: dict[str, dict[str, Any]] = {}
    for cls, alt in _ALT_METHOD.items():
        rows = [r for r in performance if r["class"] == cls]
        above: int | None = None
        for r in sorted(rows, key=lambda x: x["n"]):
            sv = r["timings"].get("statevector")
            at = r["timings"].get(alt)
            if sv is not None and at is not None and at < sv * (1 - margin):
                above = r["n"]
                break
        if above is not None:
            decision[cls] = {"method": alt, "above_n": above}
    return decision


def _meta(backend: str, shots: int) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import platform

        info["machine"] = platform.platform()
    except (ImportError, RuntimeError, ValueError):
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except (ImportError, RuntimeError, ValueError):
        pass
    try:
        import qiskit_aer

        info["qiskit_aer"] = qiskit_aer.__version__
    except (ImportError, RuntimeError, ValueError):
        pass
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "shots": shots,
        **info,
    }


# ---------------------------------------------------------------------------
# GPU backend benchmark
# ---------------------------------------------------------------------------

def benchmark_gpu_backends(
    n_values: Iterable[int] = (8, 12, 16, 20, 24),
    backends: Iterable[str] = ("qulacs", "tensorcircuit", "cupy"),
    shots: int = 256,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Measure GPU backend timings across circuit families and qubit counts.

    Returns a list of dicts: {"n", "class", "backend", "time"}.
    """
    backends = list(backends)
    # warm-up: trigger import/compile for each backend
    for b in backends:
        _timed_run(_ghz(4), b, "gpu", shots)

    performance: list[dict[str, Any]] = []
    for n in n_values:
        for cls, fn in [("clifford", _ghz), ("low_tw", _chain_rotation), ("general", _grover)]:
            for b in backends:
                samples: list[float] = []
                for _ in range(repeats):
                    t = _timed_run(fn(n), b, "gpu", shots)
                    if t is not None:
                        samples.append(t)
                if samples:
                    performance.append({
                        "n": n,
                        "class": cls,
                        "backend": b,
                        "time": round(min(samples), 4),
                    })
    return performance


def derive_gpu_decision(
    performance: list[dict[str, Any]], margin: float = 0.2
) -> dict[str, dict[str, Any]]:
    """Derive the best GPU backend for each circuit class from measured data.

    For each class, find the fastest backend across all n values.
    When no data exists, returns empty dict (caller falls back to hardcoded defaults).
    """
    decision: dict[str, dict[str, Any]] = {}
    classes = {r["class"] for r in performance}
    for cls in classes:
        rows = [r for r in performance if r["class"] == cls]
        if not rows:
            continue
        # Find the backend with the best (lowest) average time
        backend_times: dict[str, list[float]] = {}
        for r in rows:
            backend_times.setdefault(r["backend"], []).append(r["time"])
        best_backend = min(backend_times, key=lambda b: sum(backend_times[b]) / len(backend_times[b]))
        decision[cls] = {"backend": best_backend}
    return decision


def build_gpu_benchmark_data(
    n_values: Iterable[int] = (8, 12, 16, 20, 24),
    backends: Iterable[str] = ("qulacs", "tensorcircuit", "cupy"),
    shots: int = 256,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run the full GPU benchmark and return structured data.

    Produces:
        meta        -- machine/version information
        performance -- per-backend timings per circuit class per n
        decision    -- best backend per circuit class
    """
    backends = list(backends)
    performance = benchmark_gpu_backends(n_values, backends=backends, shots=shots, repeats=repeats)
    return {
        "meta": {"backends": backends, "shots": shots, **_meta("gpu", shots)},
        "performance": performance,
        "decision": derive_gpu_decision(performance),
    }


def build_benchmark_data(
    n_values: Iterable[int] = (8, 12, 16, 20, 24),
    noise_n: Iterable[int] = (2, 4, 6, 8, 10, 12),
    general_n: Iterable[int] = (8, 12, 16),
    backend: str = "qiskit",
    shots: int = 256,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run the full benchmark and return structured data.

    Produces:
        meta          -- machine/version information
        capabilities  -- static capability matrix
        performance   -- clifford/low_tw crossover grid
        general       -- QFT/Grover statevector validation points (records the 2^n ceiling)
        noise         -- density_matrix + noise cost curve (records the 4^n cost)
        decision      -- crossover thresholds derived from performance
    """
    performance = benchmark_grid(n_values, backend=backend, shots=shots, repeats=repeats)
    return {
        "meta": _meta(backend, shots),
        "capabilities": METHOD_CAPABILITIES,
        "performance": performance,
        "general": benchmark_general(general_n, backend=backend, shots=shots, repeats=repeats),
        "noise": benchmark_noise(noise_n, backend=backend, shots=shots, repeats=repeats),
        "decision": derive_decision(performance),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline benchmark: generate scheduling data (capability matrix + performance)")
    parser.add_argument("-o", "--output", default="scheduler/data/benchmarks.json")
    parser.add_argument("--backend", default="qiskit")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--n", default="8,12,16,20,24", help="comma-separated grid of bit counts")
    parser.add_argument("--noise-n", default="2,4,6,8,10,12",
                        help="bit-count grid for the noise benchmark (density_matrix, 4^n cost)")
    parser.add_argument("--general-n", default="8,12,16",
                        help="bit-count grid for the QFT/Grover validation points")
    args = parser.parse_args(argv)

    n_values = [int(x) for x in args.n.split(",")]
    noise_n = [int(x) for x in args.noise_n.split(",")]
    general_n = [int(x) for x in args.general_n.split(",")]
    data = build_benchmark_data(
        n_values, noise_n=noise_n, general_n=general_n,
        backend=args.backend, shots=args.shots,
    )

    print(tr("bench.capabilities"))
    for m, cap in data["capabilities"].items():
        print(f"  {m:24s} noise={cap['noise']!s:5s} gates={cap['gates']}")
    print(tr("bench.performance"))
    for r in data["performance"]:
        t = ", ".join(f"{m}={s}s" for m, s in r["timings"].items())
        print(f"  n={r['n']:>3d} {r['class']:8s} {t}")
    print(tr("bench.decision"))
    for cls, d in data["decision"].items():
        print(f"  {cls:8s} -> {d['method']} (n >= {d['above_n']})")

    print(tr("bench.general"))
    for r in data["general"]:
        print(f"  {r['circuit']:6s} n={r['n']:>3d}  statevector={r['time']}s")

    noise = data["noise"]
    print(tr("bench.noise"))
    for r in noise["performance"]:
        print(f"  n={r['n']:>3d}  density_matrix={r['time']}s")
    print(tr("bench.infeasible", budget=noise["budget"], infeasible_n=noise["infeasible_n"]))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(tr("bench.written", output=args.output))


if __name__ == "__main__":
    main()
