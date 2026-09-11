"""
Two-layer paradigm with SOFT decoding:
Instead of hard MLE for k̂, compute posterior P(k̂|measurement) and weight.
"""

import numpy as np
import json
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


def soft_decode(counts, n_count, N, sigma=None):
    """
    Soft decoding: compute P(k̂|measurement) using Gaussian likelihood.

    For each measured phase φ̂, compute likelihood of each k̂:
    P(φ̂|k̂) = exp(-(φ̂ - φ_k̂)² / (2σ²))

    where φ_k̂ = 2*arcsin(sqrt(k̂/N)) is the theoretical phase for k̂ marked items.
    """
    if sigma is None:
        sigma = 1.0 / (2**n_count)  # Statistical uncertainty from bin width

    # Possible k̂ values (1 to N, skip 0 since it's degenerate)
    k_values = list(range(1, N + 1))

    # Theoretical phases for each k
    phi_theory = {k: 2 * np.arcsin(np.sqrt(k / N)) for k in k_values}

    # Accumulate posterior over all measurements
    posterior_accum = {k: 0.0 for k in k_values}
    total_shots = sum(counts.values())

    for bitstring, count in counts.items():
        phase_int = int(bitstring, 2)
        phi_measured = 2 * np.pi * phase_int / (2**n_count)

        # Compute likelihood for each k̂
        likelihoods = {}
        for k in k_values:
            # Gaussian likelihood (wrapped for periodicity)
            diff = phi_measured - phi_theory[k]
            # Handle periodicity: phase is mod 2π
            diff = np.angle(np.exp(1j * diff))
            likelihoods[k] = np.exp(-diff**2 / (2 * sigma**2))

        # Normalize to get posterior for this measurement
        total_likelihood = sum(likelihoods.values())
        if total_likelihood > 0:
            for k in k_values:
                posterior_accum[k] += count * likelihoods[k] / total_likelihood

    # Normalize overall
    P_k = {k: v / total_shots for k, v in posterior_accum.items()}
    return P_k


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f"experiments/ibm_p_k_soft_{ts}.json"

    print("=== IBM Hardware: P(k_hat) with SOFT Decoding ===")
    print(f"Time: {datetime.now().isoformat()}")

    # Connect to IBM
    service = QiskitRuntimeService()
    backend = service.backend('ibm_kingston')
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    # Parameters
    n_search = 2
    n_count = 2
    N = 2**n_search
    M_true = 2
    marked_states = [1, 3]
    R = 2000
    shots_per_k = 2000

    print(f"\nN={N}, M_true={M_true}, marked={marked_states}")
    print(f"R={R}, shots_per_k={shots_per_k}")

    # Build and run quantum counting
    qc_count = build_quantum_counting(n_count, n_search, marked_states)
    qc_count_t = transpile(qc_count, backend, optimization_level=1,
                           initial_layout=list(range(n_count + n_search)))

    print(f"\n--- Layer 1: Quantum counting ({R} shots) ---")
    sampler = Sampler(backend)
    job = sampler.run([qc_count_t], shots=R)
    print(f"Job: {job.job_id()}")
    result = job.result()
    counts = result[0].data.c.get_counts()
    print(f"Raw counts: {counts}")

    # Hard decoding (for comparison)
    print(f"\n--- Hard Decoding (MLE) ---")
    k_hat_counts = {}
    for bitstring, count in counts.items():
        phase_int = int(bitstring, 2)
        phase = phase_int / (2**n_count)
        k_hat = round(N * np.sin(np.pi * phase) ** 2)
        k_hat = max(0, min(N, k_hat))
        k_hat_counts[k_hat] = k_hat_counts.get(k_hat, 0) + count

    P_k_hard = {k: v / R for k, v in sorted(k_hat_counts.items())}
    print(f"P(k_hat) hard: {P_k_hard}")

    # Soft decoding
    print(f"\n--- Soft Decoding ---")
    P_k_soft = soft_decode(counts, n_count, N, sigma=0.3)
    print(f"P(k_hat) soft: { {k: round(v, 3) for k, v in P_k_soft.items()} }")

    # Layer 2: Grover for each k̂
    print(f"\n--- Layer 2: Grover success rates ---")
    F_layer2 = {}

    for k_hat in range(1, N + 1):
        # Build Grover circuit
        qc_grover = QuantumCircuit(n_search, n_search)
        qc_grover.h(range(n_search))
        n_iter = max(1, round(k_hat * np.pi / (4 * np.arcsin(np.sqrt(M_true / N)))))
        for _ in range(n_iter):
            qc_grover.compose(build_grover_iteration(n_search, marked_states), inplace=True)
        qc_grover.measure(range(n_search), range(n_search))

        qc_grover_t = transpile(qc_grover, backend, optimization_level=1,
                                initial_layout=list(range(n_search)))
        job = sampler.run([qc_grover_t], shots=shots_per_k)
        result = job.result()
        counts_g = result[0].data.c.get_counts()

        success = sum(c for s, c in counts_g.items() if int(s, 2) in marked_states)
        F_layer2[k_hat] = success / shots_per_k
        print(f"  k_hat={k_hat} ({n_iter} iter): F={F_layer2[k_hat]:.3f}")

    # Compare hard vs soft
    print(f"\n=== RESULTS ===")

    # Hard decoding result
    F_hard = sum(P_k_hard.get(k, 0) * F_layer2.get(k, 0) for k in P_k_hard)
    k_mle = max(P_k_hard, key=P_k_hard.get)
    F_point = F_layer2.get(k_mle, 0)

    # Soft decoding result
    F_soft = sum(P_k_soft.get(k, 0) * F_layer2.get(k, 0) for k in P_k_soft)

    print(f"Point estimate (k_MLE={k_mle}): F = {F_point:.4f}")
    print(f"Hard decoding weighted: F = {F_hard:.4f}")
    print(f"Soft decoding weighted: F = {F_soft:.4f}")
    print(f"\nImprovement (soft vs hard): {F_soft - F_hard:.4f} ({(F_soft-F_hard)/F_hard*100:.1f}%)")
    print(f"Gap to point estimate: {F_point - F_soft:.4f}")

    # Save
    output = {
        'timestamp': ts,
        'backend': backend.name,
        'params': {'N': N, 'M_true': M_true, 'R': R, 'shots_per_k': shots_per_k,
                   'n_count': n_count, 'sigma': 0.3},
        'P_k_hard': P_k_hard,
        'P_k_soft': P_k_soft,
        'F_layer2': F_layer2,
        'F_hard': F_hard,
        'F_soft': F_soft,
        'F_point': F_point,
        'k_mle': k_mle,
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
