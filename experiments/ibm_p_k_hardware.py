"""
Two-layer paradigm on IBM hardware:
Measure P(k̂) distribution via quantum counting, then weighted success rate.
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler


def build_oracle(n_search, marked_states):
    qc = QuantumCircuit(n_search)
    for state in marked_states:
        bits = format(state, f'0{n_search}b')
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)
        if n_search == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_search - 1)
            qc.mcx(list(range(n_search - 1)), n_search - 1)
            qc.h(n_search - 1)
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)
    return qc


def build_diffusion(n_search):
    qc = QuantumCircuit(n_search)
    qc.h(range(n_search))
    qc.x(range(n_search))
    if n_search == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_search - 1)
        qc.mcx(list(range(n_search - 1)), n_search - 1)
        qc.h(n_search - 1)
    qc.x(range(n_search))
    qc.h(range(n_search))
    return qc


def build_grover_iteration(n_search, marked_states):
    qc = QuantumCircuit(n_search)
    qc.compose(build_oracle(n_search, marked_states), inplace=True)
    qc.compose(build_diffusion(n_search), inplace=True)
    return qc


def build_quantum_counting(n_count, n_search, marked_states):
    n_total = n_count + n_search
    qc = QuantumCircuit(n_total, n_count)
    qc.h(range(n_count))
    qc.h(range(n_count, n_total))

    grover_op = build_grover_iteration(n_search, marked_states)
    for i in range(n_count):
        power = 2 ** i
        for _ in range(power):
            qc.compose(grover_op.control(), inplace=True,
                       qubits=[i] + list(range(n_count, n_total)))

    # Inverse QFT
    for j in range(n_count):
        for m in range(j):
            qc.cp(-np.pi / 2**(j - m), m, j)
        qc.h(j)

    qc.measure(range(n_count), range(n_count))
    return qc


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"experiments/ibm_p_k_{ts}.json"

    print("=== IBM Hardware: P(k_hat) via Quantum Counting ===")
    print(f"Time: {datetime.now().isoformat()}")

    # Connect to IBM
    service = QiskitRuntimeService()
    backend = service.backend('ibm_kingston')
    print(f"Backend: {backend.name}")
    print(f"Qubits: {backend.num_qubits}")

    # Parameters
    n_search = 2
    n_count = 2
    N = 2**n_search
    M_true = 2
    marked_states = [1, 3]
    R = 2000  # shots for Layer 1
    shots_per_k = 2000  # shots for Layer 2

    print(f"\nN={N}, M_true={M_true}, marked={marked_states}")
    print(f"R={R}, shots_per_k={shots_per_k}")

    # Build counting circuit
    qc_count = build_quantum_counting(n_count, n_search, marked_states)
    print(f"\nLayer 1 circuit depth: {qc_count.depth()}")

    # Transpile for hardware
    qc_count_t = transpile(qc_count, backend, optimization_level=1,
                           initial_layout=list(range(n_total := n_count + n_search)))
    print(f"Transpiled depth: {qc_count_t.depth()}")

    # Submit Layer 1
    print(f"\n--- Layer 1: Submitting quantum counting ({R} shots) ---")
    sampler = Sampler(backend)
    job = sampler.run([qc_count_t], shots=R)
    print(f"Job ID: {job.job_id()}")

    # Wait for result
    print("Waiting for Layer 1...")
    result = job.result()
    counts = result[0].data.c.get_counts()
    print(f"Raw counts: {counts}")

    # Convert to k̂
    k_hat_counts = {}
    for bitstring, count in counts.items():
        phase_int = int(bitstring, 2)
        phase = phase_int / (2**n_count)
        k_hat = round(N * np.sin(np.pi * phase) ** 2)
        k_hat = max(0, min(N, k_hat))
        k_hat_counts[k_hat] = k_hat_counts.get(k_hat, 0) + count

    P_k = {k: v / R for k, v in sorted(k_hat_counts.items())}
    print(f"\nP(k_hat) distribution:")
    for k, p in P_k.items():
        print(f"  k_hat={k}: {p:.3f} ({k_hat_counts[k]} counts)")

    # Layer 2: Grover for each k̂
    print(f"\n--- Layer 2: Grover for each k_hat ---")
    F_layer2 = {}

    for k_hat in P_k.keys():
        if k_hat == 0:
            F_layer2[k_hat] = 0.0
            continue

        # Build Grover circuit
        qc_grover = QuantumCircuit(n_search, n_search)
        qc_grover.h(range(n_search))
        n_iter = max(1, round(k_hat * np.pi / (4 * np.arcsin(np.sqrt(M_true / N)))))
        for _ in range(n_iter):
            qc_grover.compose(build_grover_iteration(n_search, marked_states), inplace=True)
        qc_grover.measure(range(n_search), range(n_search))

        # Transpile and submit
        qc_grover_t = transpile(qc_grover, backend, optimization_level=1,
                                initial_layout=list(range(n_search)))
        job = sampler.run([qc_grover_t], shots=shots_per_k)
        print(f"k_hat={k_hat} ({n_iter} iter): job {job.job_id()}...")

        result = job.result()
        counts_g = result[0].data.c.get_counts()

        success = sum(c for s, c in counts_g.items() if int(s, 2) in marked_states)
        F_layer2[k_hat] = success / shots_per_k
        print(f"  F = {F_layer2[k_hat]:.3f}")

    # Results
    print(f"\n=== RESULTS ===")
    F_weighted = sum(P_k[k] * F_layer2[k] for k in P_k)
    k_mle = max(P_k, key=P_k.get)
    F_point = F_layer2[k_mle]

    print(f"k_hat_MLE = {k_mle}")
    print(f"F_point_estimate = {F_point:.4f}")
    print(f"F_weighted = {F_weighted:.4f}")
    print(f"Difference = {F_point - F_weighted:.4f} ({(F_point-F_weighted)/F_point*100:.1f}%)")

    # Save
    output = {
        'timestamp': ts,
        'backend': backend.name,
        'params': {'N': N, 'M_true': M_true, 'R': R, 'shots_per_k': shots_per_k},
        'P_k': P_k,
        'F_layer2': F_layer2,
        'F_weighted': F_weighted,
        'F_point': F_point,
        'k_mle': k_mle,
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
