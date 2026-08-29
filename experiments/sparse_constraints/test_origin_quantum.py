"""Test: Origin Quantum (pyqpanda3) vs Qiskit-Aer Grover search benchmark."""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def build_mcz_pq(qubits):
    """Build multi-controlled Z on pyqpanda3 (no ancilla)."""
    import pyqpanda3 as pq
    qc = pq.core.QCircuit()
    qc << pq.core.H(qubits[-1])
    x_circ = pq.core.QCircuit()
    x_circ << pq.core.X(qubits[-1])
    mcx = x_circ.control(qubits[:-1])
    qc << mcx
    qc << pq.core.H(qubits[-1])
    return qc


def build_mcz_qiskit(n):
    """Build multi-controlled Z on qiskit (no ancilla)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    return qc


def build_oracle_pq(n_data, valid_states):
    """Build phase oracle on pyqpanda3."""
    import pyqpanda3 as pq
    data = [pq.core.Qubit(i) for i in range(n_data)]
    qc = pq.core.QCircuit()

    for state in valid_states:
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                qc << pq.core.X(data[i])
        qc << build_mcz_pq(data)
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                qc << pq.core.X(data[i])

    return qc


def build_oracle_qiskit(n_data, valid_states):
    """Build phase oracle on qiskit."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_data)

    for state in valid_states:
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                qc.x(i)
        qc.compose(build_mcz_qiskit(n_data), inplace=True)
        for i, bit in enumerate(reversed(state)):
            if bit == '0':
                qc.x(i)

    return qc


def build_diffuser_pq(n_data):
    """Build diffuser on pyqpanda3."""
    import pyqpanda3 as pq
    data = [pq.core.Qubit(i) for i in range(n_data)]
    qc = pq.core.QCircuit()
    for q in data:
        qc << pq.core.H(q)
        qc << pq.core.X(q)
    qc << build_mcz_pq(data)
    for q in data:
        qc << pq.core.X(q)
        qc << pq.core.H(q)
    return qc


def build_diffuser_qiskit(n_data):
    """Build diffuser on qiskit."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_data)
    qc.h(range(n_data))
    qc.x(range(n_data))
    qc.compose(build_mcz_qiskit(n_data), inplace=True)
    qc.x(range(n_data))
    qc.h(range(n_data))
    return qc


def run_pq(oracle, diffuser, n_data, n_shots, n_iterations):
    """Run Grover on pyqpanda3 CPUQVM."""
    import pyqpanda3 as pq
    data = [pq.core.Qubit(i) for i in range(n_data)]
    prog = pq.core.QProg()
    for q in data:
        prog << pq.core.H(q)
    for _ in range(n_iterations):
        prog << oracle
        prog << diffuser
    for i in range(n_data):
        prog << pq.core.measure(data[i], i)
    qvm = pq.core.CPUQVM()
    t0 = time.time()
    qvm.run(prog, n_shots)
    counts = qvm.result().get_counts()
    return counts, time.time() - t0


def run_qiskit(oracle, diffuser, n_data, n_shots, n_iterations):
    """Run Grover on qiskit-aer."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    qc = QuantumCircuit(n_data, n_data)
    qc.h(range(n_data))
    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_data), range(n_data))
    sim = AerSimulator()
    t0 = time.time()
    counts = sim.run(qc, shots=n_shots).result().get_counts()
    return counts, time.time() - t0


def main():
    print("=" * 70)
    print("Origin Quantum (pyqpanda3) vs Qiskit-Aer: Grover Search")
    print("=" * 70)

    tests = [
        (4, ["0101", "1010"]),
        (6, ["010101", "101010"]),
        (8, ["01010101", "10101010"]),
        (10, ["0101010101", "1010101010"]),
        (12, ["010101010101", "101010101010"]),
        (14, ["01010101010101", "10101010101010"]),
        (16, ["0101010101010101", "1010101010101010"]),
        (18, ["010101010101010101", "101010101010101010"]),
        (20, ["01010101010101010101", "10101010101010101010"]),
    ]

    n_shots = 1000

    print(f"\n  {'Qubits':>6s} {'Iter':>4s} │ {'pyqpanda3':>12s} {'Hits':>6s} │ {'qiskit-aer':>12s} {'Hits':>6s} │ {'Winner':>10s}")
    print(f"  {'-'*6} {'-'*4} │ {'-'*12} {'-'*6} │ {'-'*12} {'-'*6} │ {'-'*10}")

    for n_data, valid_states in tests:
        N = 2 ** n_data
        n_iter = max(1, int(math.pi / 4 * math.sqrt(N / len(valid_states))))

        t_pq = t_qs = None
        hits_pq = hits_qs = 0

        # pyqpanda3
        try:
            oracle_pq = build_oracle_pq(n_data, valid_states)
            diffuser_pq = build_diffuser_pq(n_data)
            counts_pq, t_pq = run_pq(oracle_pq, diffuser_pq, n_data, n_shots, n_iter)
            hits_pq = sum(counts_pq.get(s, 0) for s in valid_states)
        except Exception as e:
            t_pq = None

        # qiskit-aer
        try:
            oracle_qs = build_oracle_qiskit(n_data, valid_states)
            diffuser_qs = build_diffuser_qiskit(n_data)
            counts_qs, t_qs = run_qiskit(oracle_qs, diffuser_qs, n_data, n_shots, n_iter)
            hits_qs = sum(counts_qs.get(s, 0) for s in valid_states)
        except Exception as e:
            t_qs = None

        # Format output
        pq_str = f"{t_pq:.3f}s" if t_pq else "FAILED"
        qs_str = f"{t_qs:.3f}s" if t_qs else "FAILED"
        pq_hits = f"{100*hits_pq/n_shots:.0f}%" if t_pq else "-"
        qs_hits = f"{100*hits_qs/n_shots:.0f}%" if t_qs else "-"

        winner = "-"
        if t_pq and t_qs:
            winner = "pyqpanda3" if t_pq < t_qs else "qiskit-aer"
            ratio = max(t_pq, t_qs) / min(t_pq, t_qs)
            winner += f" {ratio:.1f}x"

        print(f"  {n_data:6d} {n_iter:4d} │ {pq_str:>12s} {pq_hits:>6s} │ {qs_str:>12s} {qs_hits:>6s} │ {winner:>10s}")


if __name__ == "__main__":
    main()
