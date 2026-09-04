"""qshow — run the current circuit and display the result in the terminal / Jupyter.

Two usages:
    qshow(shots=1024)                     # run the current circuit and display
    qshow(result)                          # directly visualize a Result (algorithm output, etc.)

After running the current circuit, the circuit is automatically cleared (each qshow is a complete program);
call reset() to clear it manually.

Passing cache=LocalCacheRegistry(...) enables the local scheduling cache: the first run records
"circuit features -> backend", and later runs with tweaked circuits hit the cache directly, avoiding repeated decisions.

backend only selects the engine (qiskit/cirq/pennylane/native/qi); the specific real-hardware chip
goes through the device parameter (only effective when backend="qi", with values tuna9/tuna17/qx).

Passing groverize=True compiles cwhile loops into static Grover circuits before execution,
enabling dynamic algorithms to run on backends without mid-circuit measurement support.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

from ._i18n import tr
from .backends import get_backend, get_backend_for_method
from .compiler import decompose, route_swaps
from .ir import Circuit
from .noise import NoiseModel, resolve_noise
from .result import Result
from .scheduler import (
    LocalCacheRegistry,
    circuit_features,
    load_noise_cost,
    recommend_method,
    schedule,
)
from .stack import current_circuit, reset
from .topology import CouplingMap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def qshow(
    result: Result | None = None,
    backend: str = "auto",
    shots: int = 1024,
    noise: NoiseModel | float | None = None,
    report: bool = False,
    cache: LocalCacheRegistry | None = None,
    coupling_map: CouplingMap | None = None,
    device: str | None = None,
    requires_grad: bool = False,
    groverize: bool = False,
) -> Result | None:
    if result is not None:
        if not isinstance(result, Result):
            raise TypeError(tr("err.qshow_arg"))
        _print_result(result, backend_name=None)
        return result

    circuit = current_circuit()
    circuit.requires_grad = requires_grad
    if circuit.is_empty():
        print(tr("show.empty_circuit"))
        return None

    # groverize: compile cwhile loops into static Grover circuits
    if groverize:
        from .compiler import groverize as _groverize
        from .ir import ClassicalWhileOperation
        cwhile_ops = [op for op in circuit.ops if isinstance(op, ClassicalWhileOperation)]
        if cwhile_ops:
            circuit = _groverize(cwhile_ops[0])
        # if no cwhile ops, just run the circuit as-is

    # target topology: first gate decomposition (high-level gates → basic gates), then SWAP routing onto the coupling map
    if coupling_map is not None:
        circuit = route_swaps(decompose(circuit), coupling_map)

    if report:
        _print_circuit_report(circuit)

    # scheduling: resolve the circuit to pick method; when backend is "auto",
    # use the full scheduler chain (cache -> profiles -> table -> rules)
    noise_enabled = resolve_noise(noise).enabled
    _profiles = None
    if noise_enabled:
        _warn_noise_cost(circuit.num_qubits)
    if backend == "auto":
        from .scheduler.profiles import default_profiles as _dp
        _profiles = _dp()
        rec = schedule(circuit, cache=cache, profiles=_profiles, noise=noise_enabled)
        be_name = rec.backend
        method = rec.method
    else:
        be_name = backend
        rec = recommend_method(circuit_features(circuit), noise=noise_enabled)
        method = rec.method

    # noise is handled by each backend itself (qiskit/native use density_matrix, cirq/pennylane
    # use channels), no method downgrade; without noise, match method capability and downgrade to native.
    if noise_enabled:
        be = get_backend(be_name, device=device)
    else:
        be = get_backend_for_method(be_name, method, device=device)
    t0 = time.time()
    try:
        result = be.run(circuit, shots=shots, noise=noise, method=method)
    except ImportError:
        from .backends import _detect_available
        from .scheduler.registry import Recommendation as _Rec
        fallback = _detect_available()
        be = get_backend(fallback)
        rec = _Rec(backend=fallback, method="statevector")
        be_name = fallback
        method = "statevector"
        result = be.run(circuit, shots=shots, noise=noise, method=method)
    elapsed = time.time() - t0

    if cache is not None:
        cache.report_result(circuit_features(circuit), f"{be.name}:{method}", elapsed, None)

    # Update profile-based scheduler with real timing data
    try:
        from .backends import _background_explore, _pick_alternative
        from .scheduler.registry import Recommendation
        if _profiles is not None:
            feats = circuit_features(circuit)
            _profiles.report_result(feats, f"{be.name}/{method}", elapsed, None)
            _profiles._save()  # persist immediately

            # 5% exploration in background
            import random as _random
            import threading as _threading
            if _random.random() < 0.05:
                _rec = Recommendation(be.name, method)
                explore = _pick_alternative(_rec, _profiles, feats)
                if explore is not None:
                    _threading.Thread(
                        target=_background_explore,
                        args=(circuit, explore.backend, explore.method, _profiles, feats),
                        daemon=True,
                    ).start()
    except Exception:
        pass

    _print_result(result, backend_name=be.name)
    reset()
    return result


def _warn_noise_cost(n: int) -> None:
    """With noise, warn about the 4^n cost of density_matrix based on measured data (silent without measured data)."""
    cost = load_noise_cost()
    infeasible = cost.get("infeasible_n")
    if infeasible is not None and n >= infeasible:
        print(tr("show.noise_cost", infeasible=infeasible, n=n))


def _print_circuit_report(circuit: Circuit) -> None:
    print(tr("show.circuit_resources"))
    print(tr("show.gate_count", n=circuit.gate_count()))
    print(tr("show.depth", n=circuit.depth()))
    print(tr("show.qubit_count", n=circuit.num_qubits))


def _print_result(result: Result, backend_name: str | None = None) -> None:
    if result.kind == "counts":
        _print_counts(result, backend_name)
    elif result.kind == "value":
        _print_value(result)
    else:
        raise ValueError(tr("err.unknown_result_kind", kind=result.kind))


def _print_counts(result: Result, backend_name: str | None) -> None:
    header = tr("show.backend_header", name=backend_name) if backend_name else ""
    print(f"{header}{tr('show.shots', shots=result.shots)}")
    print(tr("show.result"))
    counts = result.counts or {}
    total = sum(counts.values()) or 1
    for bitstring in sorted(counts):
        n = counts[bitstring]
        bar = "#" * round(40 * n / total)
        print(f"  |{bitstring}>  {n:>6d}  ({n / total:6.1%})  {bar}")


def _print_value(result: Result) -> None:
    print(tr("show.result"))
    print(f"  {result.value}")
    for key, val in result.metadata.items():
        print(f"  {key} = {val}")


# ---------------------------------------------------------------------------
# qshow_all — run the current circuit on multiple backends in parallel
# ---------------------------------------------------------------------------


def qshow_all(
    backends: list[str],
    shots: int = 1024,
    noise: NoiseModel | float | None = None,
    print_results: bool = True,
) -> dict[str, Result]:
    """Run the current circuit on multiple backends in parallel.

    Each backend runs in a separate process (independent global state).
    Returns a dict mapping backend name to Result.

    Example::

        qgate(H, 0)
        qgate(CX, 0, 1)
        results = qshow_all(['qiskit', 'cirq', 'qulacs'])
        # {'qiskit': Result(...), 'cirq': Result(...), 'qulacs': Result(...)}
    """
    circuit = current_circuit()
    if circuit.is_empty():
        print(tr("show.empty_circuit"))
        return {}

    # Warn about GPU backends in multiprocessing
    gpu_backends = {"qulacs", "tensorcircuit", "cudaq", "mindquantum", "qpanda", "cupy"}
    requested_gpu = [b for b in backends if b in gpu_backends]
    if requested_gpu and len(backends) > 1:
        import warnings
        warnings.warn(
            f"GPU backends {requested_gpu} used with qshow_all (multiprocessing). "
            f"CUDA contexts cannot be shared across processes; GPU backends may "
            f"fail or fall back to CPU in subprocesses.",
            stacklevel=2,
        )

    # Capture circuit ops for serialization to subprocesses
    ops = list(circuit.ops)
    n = circuit.num_qubits

    # _run_one must be at module level for ProcessPoolExecutor pickle
    args = [(backend, ops, n, shots, noise) for backend in backends]

    results: dict[str, Result] = {}
    if len(backends) == 1:
        results[backends[0]] = _run_one_in_subprocess(args[0])
    else:
        with ProcessPoolExecutor(max_workers=len(backends)) as pool:
            futures = {pool.submit(_run_one_in_subprocess, a): a[0] for a in args}
            for future, be_name in futures.items():
                results[be_name] = future.result()

    if print_results:
        for be_name, result in results.items():
            _print_result(result, backend_name=be_name)

    return results


def _run_one_in_subprocess(args: tuple) -> Result:
    """Module-level function for ProcessPoolExecutor (must be picklable)."""
    backend_name, ops, n, shots, noise = args
    from .ir import Circuit as _Circuit

    c = _Circuit()
    c.num_qubits = n
    c.ops = list(ops)
    be = get_backend(backend_name)
    return be.run(c, shots=shots, noise=noise)


# ---------------------------------------------------------------------------
# run_circuits — run multiple different circuits in parallel
# ---------------------------------------------------------------------------


def run_circuits(
    builders: list[Callable[[], None]],
    backend: str = "auto",
    shots: int = 1024,
    noise: NoiseModel | float | None = None,
    print_results: bool = True,
) -> dict[int, Result]:
    """Run multiple different circuits in parallel.

    Each builder is a function that calls qgate() to build a circuit.
    Each builder runs in a separate process with its own global state.

    Returns a dict mapping index (0, 1, 2, ...) to Result.

    Example::

        def bell():
            qgate(H, 0)
            qgate(CX, 0, 1)

        def flip():
            qgate(X, 0)

        results = run_circuits([bell, flip], backend='qiskit')
        # {0: Result(bell), 1: Result(flip)}
    """
    if not builders:
        return {}

    # Capture each circuit by running the builder in the main process
    # and recording the ops, then replay in subprocesses.
    captured: list[tuple] = []
    for builder in builders:
        reset()
        builder()
        circ = current_circuit()
        captured.append((list(circ.ops), circ.num_qubits))
        reset()

    if len(builders) == 1:
        ops, n = captured[0]
        result = _run_circuit_subprocess((backend, ops, n, shots, noise))
        if print_results:
            _print_result(result, backend_name=backend)
        return {0: result}

    args_list = [(backend, ops, n, shots, noise) for ops, n in captured]
    results: dict[int, Result] = {}

    with ProcessPoolExecutor(max_workers=len(builders)) as pool:
        futures = {pool.submit(_run_circuit_subprocess, a): i for i, a in enumerate(args_list)}
        for future, idx in futures.items():
            results[idx] = future.result()

    if print_results:
        for idx, result in results.items():
            _print_result(result, backend_name=f"{backend} #{idx}")

    return results


def _run_circuit_subprocess(args: tuple) -> Result:
    """Module-level function for ProcessPoolExecutor (must be picklable)."""
    backend_name, ops, n, shots, noise = args
    from .ir import Circuit as _Circuit

    c = _Circuit()
    c.num_qubits = n
    c.ops = list(ops)
    be = get_backend(backend_name)
    return be.run(c, shots=shots, noise=noise)
