"""Experiment 3: Two-Level Compilation Pipeline.

Compares:
1. Direct QuoNic compile
2. Direct Qiskit L2 compile
3. Direct t|ket> compile
4. QuoNic → Qiskit L2 (two-level)
5. QuoNic → t|ket> (two-level)

Measures: gate count reduction (%) and compilation time (ms).
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from quonic.ir import Circuit, GateOperation
from quonic.compiler import compile as quonic_compile
from quonic.gates import H, CX

# --- Circuit builders ---

def build_qft(n: int) -> Circuit:
    """Build QFT circuit with high-level CP gates."""
    import math
    c = Circuit()
    c.allocate(n)
    for i in range(n):
        c.add(GateOperation("h", (i,)))
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            c.add(GateOperation("cp", (j, i), (angle,)))
    for i in range(n // 2):
        c.add(GateOperation("swap", (i, n - 1 - i)))
    return c


def build_grover(n: int) -> Circuit:
    """Build Grover with MCX/MCZ high-level gates."""
    c = Circuit()
    c.allocate(n)
    for i in range(n):
        c.add(GateOperation("h", (i,)))
    # Oracle: multi-controlled X
    c.add(GateOperation("mcx", tuple(range(n)), ()))
    # Diffusion
    for i in range(n):
        c.add(GateOperation("h", (i,)))
    for i in range(n):
        c.add(GateOperation("x", (i,)))
    c.add(GateOperation("mcz", tuple(range(n)), ()))
    for i in range(n):
        c.add(GateOperation("x", (i,)))
    for i in range(n):
        c.add(GateOperation("h", (i,)))
    return c


def build_qpe(n_estimate: int, n_target: int = 1) -> Circuit:
    """Build QPE with controlled rotation gates."""
    import math
    n = n_estimate + n_target
    c = Circuit()
    c.allocate(n)
    # Hadamard on estimation register
    for i in range(n_estimate):
        c.add(GateOperation("h", (i,)))
    # Controlled rotations
    for i in range(n_estimate):
        angle = math.pi / (2 ** i)
        c.add(GateOperation("cp", (i, n_estimate), (angle,)))
    # Inverse QFT on estimation register
    for i in range(n_estimate // 2):
        c.add(GateOperation("swap", (i, n_estimate - 1 - i)))
    for i in range(n_estimate):
        c.add(GateOperation("h", (i,)))
        for j in range(i):
            angle = -math.pi / (2 ** (i - j))
            c.add(GateOperation("cp", (j, i), (angle,)))
    return c


def build_vqe(n: int, depth: int = 3) -> Circuit:
    """Build VQE ansatz with CCX entangling gates."""
    import math
    import random
    random.seed(123)
    c = Circuit()
    c.allocate(n)
    for d in range(depth):
        # Rotation layer
        for i in range(n):
            angle = random.random() * 2 * math.pi
            c.add(GateOperation("ry", (i,), (angle,)))
        # Entangling layer with CCX
        for i in range(0, n - 2, 2):
            c.add(GateOperation("ccx", (i, i + 1, i + 2)))
        # CX layer
        for i in range(n - 1):
            c.add(GateOperation("cx", (i, i + 1)))
    return c


# --- QuoNic compilation ---

def quonic_gate_count(circuit: Circuit) -> int:
    """Count gates after QuoNic compilation."""
    compiled = quonic_compile(circuit)
    return len(compiled)


def quonic_compile_time(circuit: Circuit) -> float:
    """Measure QuoNic compilation time in ms."""
    t0 = time.perf_counter()
    quonic_compile(circuit)
    return (time.perf_counter() - t0) * 1000


# --- Qiskit compilation ---

def quonic_to_qiskit(circuit: Circuit):
    """Convert QuoNic IR to Qiskit QuantumCircuit."""
    from qiskit import QuantumCircuit
    n = circuit.num_qubits
    qc = QuantumCircuit(n)
    for op in circuit:
        name = op.name.lower()
        qubits = op.qubits
        params = op.params or ()
        if name == "h":
            qc.h(qubits[0])
        elif name == "x":
            qc.x(qubits[0])
        elif name == "y":
            qc.y(qubits[0])
        elif name == "z":
            qc.z(qubits[0])
        elif name == "s":
            qc.s(qubits[0])
        elif name == "t":
            qc.t(qubits[0])
        elif name == "rx":
            qc.rx(params[0], qubits[0])
        elif name == "ry":
            qc.ry(params[0], qubits[0])
        elif name == "rz":
            qc.rz(params[0], qubits[0])
        elif name == "cx":
            qc.cx(qubits[0], qubits[1])
        elif name == "cz":
            qc.cz(qubits[0], qubits[1])
        elif name == "cp":
            qc.cp(params[0], qubits[0], qubits[1])
        elif name == "ccx":
            qc.ccx(qubits[0], qubits[1], qubits[2])
        elif name == "mcx":
            qc.mcx(list(qubits[:-1]), qubits[-1])
        elif name == "mcz":
            qc.h(qubits[-1])
            qc.mcx(list(qubits[:-1]), qubits[-1])
            qc.h(qubits[-1])
        elif name == "swap":
            qc.swap(qubits[0], qubits[1])
        elif name == "p":
            qc.p(params[0], qubits[0])
        else:
            # Skip unknown gates
            pass
    return qc


def qiskit_gate_count(qc) -> int:
    """Count gates in Qiskit circuit."""
    return qc.size()


def qiskit_compile(qc, opt_level: int = 2):
    """Compile with Qiskit transpiler."""
    from qiskit import transpile
    return transpile(qc, optimization_level=opt_level)


def qiskit_compile_time(qc, opt_level: int = 2) -> float:
    """Measure Qiskit compilation time in ms."""
    from qiskit import transpile
    t0 = time.perf_counter()
    transpile(qc, optimization_level=opt_level)
    return (time.perf_counter() - t0) * 1000


# --- t|ket> compilation ---

def quonic_to_tket(circuit: Circuit):
    """Convert QuoNic IR to t|ket> Circuit."""
    from pytket import Circuit as TketCircuit
    from pytket import OpType
    n = circuit.num_qubits
    tc = TketCircuit(n)
    for op in circuit:
        name = op.name.lower()
        qubits = op.qubits
        params = op.params or ()
        if name == "h":
            tc.H(qubits[0])
        elif name == "x":
            tc.X(qubits[0])
        elif name == "y":
            tc.Y(qubits[0])
        elif name == "z":
            tc.Z(qubits[0])
        elif name == "s":
            tc.S(qubits[0])
        elif name == "t":
            tc.T(qubits[0])
        elif name == "rx":
            tc.Rx(params[0], qubits[0])
        elif name == "ry":
            tc.Ry(params[0], qubits[0])
        elif name == "rz":
            tc.Rz(params[0], qubits[0])
        elif name == "cx":
            tc.CX(qubits[0], qubits[1])
        elif name == "cz":
            tc.CZ(qubits[0], qubits[1])
        elif name == "cp":
            # t|ket> doesn't have CP directly, decompose
            tc.Rz(params[0] / 2, qubits[0])
            tc.CX(qubits[0], qubits[1])
            tc.Rz(-params[0] / 2, qubits[1])
            tc.CX(qubits[0], qubits[1])
            tc.Rz(params[0] / 2, qubits[1])
        elif name == "ccx":
            tc.CCX(qubits[0], qubits[1], qubits[2])
        elif name == "swap":
            tc.SWAP(qubits[0], qubits[1])
        elif name == "p":
            tc.Rz(params[0], qubits[0])
        else:
            # Skip unknown gates
            pass
    return tc


def tket_gate_count(tc) -> int:
    """Count gates in t|ket> circuit."""
    return tc.n_gates


def tket_compile(tc):
    """Compile with t|ket> FullPeepholeOptimise."""
    from pytket.passes import FullPeepholeOptimise
    tc_copy = tc.copy()
    FullPeepholeOptimise().apply(tc_copy)
    return tc_copy


def tket_compile_time(tc) -> float:
    """Measure t|ket> compilation time in ms."""
    from pytket.passes import FullPeepholeOptimise
    tc_copy = tc.copy()
    t0 = time.perf_counter()
    FullPeepholeOptimise().apply(tc_copy)
    return (time.perf_counter() - t0) * 1000


# --- Main experiment ---

def run_experiment():
    circuits = {
        "QFT-8": build_qft(8),
        "Grover-4": build_grover(4),
        "QPE-4": build_qpe(4),
        "VQE-6": build_vqe(6, depth=3),
    }

    results = []

    for name, circ in circuits.items():
        print(f"\n{'='*60}")
        print(f"Circuit: {name}")
        print(f"{'='*60}")

        # Original gate count
        orig_gates = len(circ)
        print(f"  Original gates: {orig_gates}")

        # --- QuoNic direct ---
        q_compiled = quonic_compile(circ)
        q_gates = len(q_compiled)
        q_time = quonic_compile_time(circ)
        q_reduction = (1 - q_gates / orig_gates) * 100 if orig_gates > 0 else 0
        print(f"  QuoNic: {q_gates} gates ({q_reduction:.1f}% reduction), {q_time:.2f} ms")

        # --- Qiskit direct L2 ---
        qiskit_circ = quonic_to_qiskit(circ)
        qiskit_orig = qiskit_gate_count(qiskit_circ)
        qiskit_compiled = qiskit_compile(qiskit_circ, opt_level=2)
        qiskit_gates = qiskit_gate_count(qiskit_compiled)
        qiskit_time = qiskit_compile_time(qiskit_circ, opt_level=2)
        qiskit_reduction = (1 - qiskit_gates / qiskit_orig) * 100 if qiskit_orig > 0 else 0
        print(f"  Qiskit L2: {qiskit_gates} gates ({qiskit_reduction:.1f}% reduction), {qiskit_time:.2f} ms")

        # --- t|ket> direct ---
        tket_circ = quonic_to_tket(circ)
        tket_orig = tket_gate_count(tket_circ)
        tket_compiled = tket_compile(tket_circ)
        tket_gates = tket_gate_count(tket_compiled)
        tket_time = tket_compile_time(tket_circ)
        tket_reduction = (1 - tket_gates / tket_orig) * 100 if tket_orig > 0 else 0
        print(f"  t|ket>: {tket_gates} gates ({tket_reduction:.1f}% reduction), {tket_time:.2f} ms")

        # --- Two-level: QuoNic → Qiskit L2 ---
        q_then_qiskit_circ = quonic_to_qiskit(q_compiled)
        q_then_qiskit_orig = qiskit_gate_count(q_then_qiskit_circ)
        q_then_qiskit_compiled = qiskit_compile(q_then_qiskit_circ, opt_level=2)
        q_then_qiskit_gates = qiskit_gate_count(q_then_qiskit_compiled)
        q_then_qiskit_time = q_time + qiskit_compile_time(q_then_qiskit_circ, opt_level=2)
        q_then_qiskit_reduction = (1 - q_then_qiskit_gates / qiskit_orig) * 100 if qiskit_orig > 0 else 0
        print(f"  QuoNic→Qiskit L2: {q_then_qiskit_gates} gates ({q_then_qiskit_reduction:.1f}% reduction), {q_then_qiskit_time:.2f} ms")

        # --- Two-level: QuoNic → t|ket> ---
        q_then_tket_circ = quonic_to_tket(q_compiled)
        q_then_tket_orig = tket_gate_count(q_then_tket_circ)
        q_then_tket_compiled = tket_compile(q_then_tket_circ)
        q_then_tket_gates = tket_gate_count(q_then_tket_compiled)
        q_then_tket_time = q_time + tket_compile_time(q_then_tket_circ)
        q_then_tket_reduction = (1 - q_then_tket_gates / tket_orig) * 100 if tket_orig > 0 else 0
        print(f"  QuoNic→t|ket>: {q_then_tket_gates} gates ({q_then_tket_reduction:.1f}% reduction), {q_then_tket_time:.2f} ms")

        # Overhead
        qiskit_overhead = q_then_qiskit_reduction - qiskit_reduction
        tket_overhead = q_then_tket_reduction - tket_reduction
        print(f"  Two-level overhead (Qiskit): {qiskit_overhead:+.1f}%")
        print(f"  Two-level overhead (t|ket>): {tket_overhead:+.1f}%")

        results.append({
            "circuit": name,
            "orig_gates": orig_gates,
            "quonic_gates": q_gates,
            "quonic_reduction": round(q_reduction, 1),
            "quonic_time_ms": round(q_time, 2),
            "qiskit_gates": qiskit_gates,
            "qiskit_reduction": round(qiskit_reduction, 1),
            "qiskit_time_ms": round(qiskit_time, 2),
            "tket_gates": tket_gates,
            "tket_reduction": round(tket_reduction, 1),
            "tket_time_ms": round(tket_time, 2),
            "two_level_qiskit_gates": q_then_qiskit_gates,
            "two_level_qiskit_reduction": round(q_then_qiskit_reduction, 1),
            "two_level_qiskit_time_ms": round(q_then_qiskit_time, 2),
            "two_level_tket_gates": q_then_tket_gates,
            "two_level_tket_reduction": round(q_then_tket_reduction, 1),
            "two_level_tket_time_ms": round(q_then_tket_time, 2),
            "overhead_qiskit": round(qiskit_overhead, 1),
            "overhead_tket": round(tket_overhead, 1),
        })

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp3_two_level_compilation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_experiment()
