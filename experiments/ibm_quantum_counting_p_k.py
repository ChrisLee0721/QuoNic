"""
Two-layer paradigm: measure P(k_hat) distribution via quantum counting,
then compute weighted success rate vs point estimate.

Search space: N=4 items (2 qubits), M=2 marked items
Layer 1: Quantum counting (2 counting qubits + 2 search qubits)
Layer 2: Grover with estimated k_hat
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler


def build_oracle(n_search, marked_states):
    """Oracle that flips marked states."""
    qc = QuantumCircuit(n_search)
    for state in marked_states:
        # Flip phase of |state>
        bits = format(state, f'0{n_search}b')
        # Apply X to qubits that are 0 in the state
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)
        # Multi-controlled Z
        if n_search == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_search - 1)
            qc.mcx(list(range(n_search - 1)), n_search - 1)
            qc.h(n_search - 1)
        # Undo X gates
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(i)
    return qc


def build_diffusion(n_search):
    """Grover diffusion operator."""
    qc = QuantumCircuit(n_search)
    qc.h(range(n_search))
    qc.x(range(n_search))
    # Multi-controlled Z
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
    """One Grover iteration: Oracle + Diffusion."""
    qc = QuantumCircuit(n_search)
    qc.compose(build_oracle(n_search, marked_states), inplace=True)
    qc.compose(build_diffusion(n_search), inplace=True)
    return qc


def build_quantum_counting(n_count, n_search, marked_states, theta_k):
    """
    Quantum counting via QPE on Grover operator.
    n_count: counting qubits
    n_search: search qubits
    theta_k: rotation angle = 2*arcsin(sqrt(k/N))
    """
    n_total = n_count + n_search
    qc = QuantumCircuit(n_total, n_count)

    # Initialize counting qubits in superposition
    qc.h(range(n_count))

    # Initialize search qubits in superposition
    qc.h(range(n_count, n_total))

    # Controlled Grover iterations
    grover_op = build_grover_iteration(n_search, marked_states)
    for i in range(n_count):
        # Apply 2^i controlled Grover iterations
        power = 2 ** i
        for _ in range(power):
            # Controlled version of Grover iteration
            qc.compose(grover_op.control(), inplace=True,
                       qubits=[i] + list(range(n_count, n_total)))

    # Inverse QFT on counting qubits
    qc.append(_iqft(n_count), range(n_count))

    # Measure counting qubits
    qc.measure(range(n_count), range(n_count))

    return qc


def _iqft(n):
    """Inverse QFT."""
    qc = QuantumCircuit(n)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / 2**(j - m), m, j)
        qc.h(j)
    return qc.to_instruction()


def run_experiment():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"experiments/quantum_counting_{ts}.json"

    print("=== Two-Layer Paradigm: P(k_hat) Measurement ===")
    print(f"Time: {datetime.now().isoformat()}")

    # Parameters
    n_search = 2  # 4 items
    n_count = 2   # 2 counting qubits
    N = 2**n_search  # 4 items
    M_true = 2    # True number of marked items
    marked_states = [1, 3]  # |01⟩ and |11⟩ are marked

    # True success probability
    theta = np.arcsin(np.sqrt(M_true / N))
    print(f"\nSearch space: N={N}, M_true={M_true}")
    print(f"True theta = {theta:.4f}")
    print(f"Marked states: {marked_states}")

    # Layer 1: Quantum counting - run R times
    R = 500
    print(f"\n--- Layer 1: Quantum counting (R={R} runs) ---")

    # Build counting circuit
    qc_count = build_quantum_counting(n_count, n_search, marked_states, theta)
    print(f"Circuit depth: {qc_count.depth()}")
    print(f"Circuit size: {qc_count.size()}")

    # Run on simulator (for speed)
    sim = AerSimulator()
    qc_count_t = transpile(qc_count, sim, optimization_level=1)
    job = sim.run(qc_count_t, shots=R)
    result = job.result()
    counts = result.get_counts()

    print(f"Raw counts: {counts}")

    # Convert counts to k_hat estimates
    # Quantum counting measures phases, which map to k_hat
    # For simplicity: read the binary count as phase estimate
    k_hat_counts = {}
    for bitstring, count in counts.items():
        # bitstring is in format |q1 q0⟩ (counting qubits)
        phase_int = int(bitstring, 2)
        # Phase → k_hat mapping: k_hat = round(N * sin²(π * phase_int / 2^n_count))
        phase = phase_int / (2**n_count)
        k_hat = round(N * np.sin(np.pi * phase) ** 2)
        k_hat = max(0, min(N, k_hat))
        k_hat_counts[k_hat] = k_hat_counts.get(k_hat, 0) + count

    # Normalize to get P(k_hat)
    P_k = {k: v / R for k, v in sorted(k_hat_counts.items())}
    print(f"\nP(k_hat) distribution:")
    for k, p in P_k.items():
        print(f"  k_hat={k}: {p:.3f} ({k_hat_counts[k]} counts)")

    # Layer 2: Grover for each k_hat
    print(f"\n--- Layer 2: Grover success rate for each k_hat ---")
    F_layer2 = {}
    shots_per_k = 1000

    for k_hat in P_k.keys():
        if k_hat == 0:
            F_layer2[k_hat] = 0.0
            continue

        # Build Grover circuit with k_hat iterations
        qc_grover = QuantumCircuit(n_search, n_search)
        qc_grover.h(range(n_search))
        n_iterations = max(1, round(k_hat * np.pi / (4 * np.arcsin(np.sqrt(M_true / N)))))
        for _ in range(n_iterations):
            qc_grover.compose(build_grover_iteration(n_search, marked_states), inplace=True)
        qc_grover.measure(range(n_search), range(n_search))

        # Run
        qc_grover_t = transpile(qc_grover, sim, optimization_level=1)
        job = sim.run(qc_grover_t, shots=shots_per_k)
        result = job.result()
        counts_g = result.get_counts()

        # Success = measure a marked state
        success_count = 0
        for bitstring, count in counts_g.items():
            state = int(bitstring, 2)
            if state in marked_states:
                success_count += count

        F_layer2[k_hat] = success_count / shots_per_k
        print(f"  k_hat={k_hat} ({n_iterations} iterations): F={F_layer2[k_hat]:.3f}")

    # Compute weighted success rate
    print(f"\n--- Results ---")
    F_weighted = sum(P_k[k] * F_layer2[k] for k in P_k)
    k_mle = max(P_k, key=P_k.get)
    F_point = F_layer2[k_mle]

    print(f"k_hat_MLE = {k_mle} (most likely)")
    print(f"F_point_estimate = {F_point:.3f}")
    print(f"F_weighted = {F_weighted:.3f}")
    print(f"Difference = {F_point - F_weighted:.3f} ({(F_point - F_weighted)/F_point*100:.1f}%)")

    # Save results
    output = {
        'timestamp': ts,
        'params': {
            'n_search': n_search, 'n_count': n_count,
            'N': N, 'M_true': M_true, 'marked_states': marked_states,
            'R': R, 'shots_per_k': shots_per_k,
        },
        'P_k': P_k,
        'F_layer2': F_layer2,
        'F_weighted': F_weighted,
        'F_point': F_point,
        'k_mle': k_mle,
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")

    return output


if __name__ == '__main__':
    run_experiment()
