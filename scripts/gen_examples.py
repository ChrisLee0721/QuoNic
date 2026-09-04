"""Generate missing algorithm examples."""

import os

EXAMPLES = {
    # --- Quantum Algorithms ---
    "amplitude_amplification": '''"""Amplitude amplification: boost probability of a target state.

Like Grover but with a custom state preparation oracle.
Output: amplified probability of the marked state.
"""

from quonic.algorithms import amplitude_amplification, mark_state

oracle_fn = mark_state("11")
result = amplitude_amplification(2, oracle_fn, shots=1024)
print(result.counts)
''',
    "amplitude_estimation": '''"""Amplitude estimation: estimate the amplitude of a marked state.

Uses precision qubits to estimate the success probability of an oracle.
Output: estimated amplitude close to the true value.
"""

from quonic.algorithms import amplitude_estimation

result = amplitude_estimation(n_qubits=2, n_precision=3, shots=1024)
print(result.counts)
''',
    "bernstein_vazirani": '''"""Bernstein-Vazirani: find the hidden bitstring s in f(x) = s·x mod 2.

One query suffices — the algorithm reads s directly from the output.
Output: all shots give the hidden string.
"""

from quonic.algorithms import bernstein_vazirani, oracle

n = 4
ora = oracle(n)(lambda i: [True, False, True, False][i])  # s = "1010"
result = bernstein_vazirani(n, ora, shots=1024)
print(result.counts)
''',
    "color_code": '''"""Color code error correction demo.

Demonstrates the color code — a topological code with transversal gates.
Output: corrected logical state.
"""

from quonic.algorithms import color_code

result = color_code(shots=100)
print(result.counts)
''',
    "deutsch_jozsa": '''"""Deutsch-Jozsa: determine if f is constant or balanced in one query.

Classical requires 2^(n-1)+1 queries; quantum needs 1.
Output: all zeros = constant, anything else = balanced.
"""

from quonic.algorithms import deutsch_jozsa, oracle

n = 3
# Balanced oracle: returns True for first half
ora = oracle(n)(lambda i: i < 2**(n-1))
result = deutsch_jozsa(n, ora, shots=100)
print(result.counts)
''',
    "discrete_log": '''"""Discrete logarithm: find x such that a^x = b mod p.

Quantum algorithm for the discrete log problem.
Output: the discrete logarithm.
"""

from quonic.algorithms import discrete_log

result = discrete_log(a=2, b=8, p=11)
print(result.counts)
''',
    "dqaoa": '''"""Dynamic QAOA (DQAOA): adaptive layer QAOA variant.

Adds layers dynamically until convergence.
Output: approximate optimal solution.
"""

from quonic.algorithms import dqaoa

result = dqaoa()
print(result.counts)
''',
    "dynamics_simulation": '''"""Quantum dynamics simulation: simulate time evolution of a quantum system.

Uses Trotterization to approximate e^{-iHt}.
Output: evolved state measurements.
"""

from quonic.algorithms import dynamics_simulation

result = dynamics_simulation(n_steps=10, shots=1024)
print(result.counts)
''',
    "e91": '''"""E91 quantum key distribution protocol.

Uses entangled pairs and Bell inequality tests for secure key exchange.
Output: shared secret key bits.
"""

from quonic.algorithms import e91

result = e91(n_rounds=100)
print(f"Key length: {len(result.counts)}")
''',
    "elliptic_curve": '''"""Elliptic curve quantum algorithm demo.

Quantum approach to elliptic curve discrete log.
Output: approximate solution.
"""

from quonic.algorithms import elliptic_curve

result = elliptic_curve()
print(result.counts)
''',
    "ft_gate": '''"""Fault-tolerant gate demo.

Demonstrates gates implemented with error detection/correction.
Output: logically encoded state.
"""

from quonic.algorithms import ft_gate

result = ft_gate(shots=100)
print(result.counts)
''',
    "hadamard_test": '''"""Hadamard test: estimate Re(<psi|U|psi>).

A primitive for many quantum algorithms (inner product estimation).
Output: probability of measuring |0> encodes the real part.
"""

from quonic import qgate
from quonic.algorithms import hadamard_test
from quonic.gates import X

def prep_psi(qc, q):
    qgate(X, q)  # |1>

def apply_u(qc, q):
    pass  # Identity

result = hadamard_test(1, prep_psi, apply_u, shots=10000)
print(result.counts)
''',
    "hamiltonian_simulation": '''"""Hamiltonian simulation via Trotterization.

Simulates e^{-iHt} for a given Hamiltonian H.
Output: evolved state measurements.
"""

from quonic.algorithms import hamiltonian_simulation

result = hamiltonian_simulation()
print(result.counts)
''',
    "hhl": '''"""HHL algorithm: quantum linear system solver.

Solves Ax = b exponentially faster than classical for sparse matrices.
Output: quantum state proportional to x = A^{-1}b.
"""

from quonic.algorithms import hhl

result = hhl()
print(result.counts)
''',
    "hsp": '''"""Hidden Subgroup Problem demo.

General framework for Simon, Shor, and other HSP-based algorithms.
Output: subgroup generators.
"""

from quonic.algorithms import hsp

result = hsp()
print(result.counts)
''',
    "jordan_wigner": '''"""Jordan-Wigner transform: 2-site Hubbard model simulation.

Maps fermionic Hamiltonian to qubit Hamiltonian.
Output: ground state energy estimate.
"""

from quonic.algorithms import jordan_wigner_2site

result = jordan_wigner_2site(t=1.0, U=2.0)
print(result.counts)
''',
    "lattice_svp": '''"""Lattice SVP (Shortest Vector Problem) quantum demo.

Quantum approach to lattice-based cryptography problems.
Output: approximate shortest vector.
"""

from quonic.algorithms import lattice_svp

result = lattice_svp()
print(result.counts)
''',
    "molecule_vqe": '''"""Molecular VQE: compute ground state energy of a molecule.

Uses variational quantum eigensolver with chemistry-inspired ansatz.
Output: ground state energy.
"""

from quonic.algorithms import molecule_vqe

result = molecule_vqe(maxiter=200)
print(f"Ground state energy: {result.value}")
''',
    "phase_flip_code": '''"""Phase flip error correction code.

Encodes logical qubit against phase errors using 3 physical qubits.
Output: corrected logical state.
"""

from quonic.algorithms import phase_flip_code

result = phase_flip_code(error_qubit=0, shots=100)
print(result.counts)
''',
    "qaoa_knapsack": '''"""QAOA for the knapsack problem.

Finds the optimal subset of items maximizing value within weight capacity.
Output: approximate optimal value.
"""

from quonic.algorithms import qaoa_knapsack

weights = [2, 3, 4]
values = [3, 4, 5]
capacity = 5
result = qaoa_knapsack(weights, values, capacity, p=1, maxiter=100)
print(f"Optimal value: {result.value}")
''',
    "qaoa_maxcut": '''"""QAOA for MaxCut: find the maximum cut of a graph.

Partitions vertices to maximize edges between partitions.
Output: approximate max cut value.
"""

from quonic.algorithms import qaoa_maxcut

edges = [(0, 1), (1, 2), (0, 2)]
result = qaoa_maxcut(edges, 3, p=1, maxiter=100)
print(f"Max cut: {result.value}")
''',
    "qaoa_mis": '''"""QAOA for Maximum Independent Set.

Finds the largest set of non-adjacent vertices.
Output: approximate MIS size.
"""

from quonic.algorithms import qaoa_mis

edges = [(0, 1), (1, 2)]
result = qaoa_mis(edges, 3, p=1, maxiter=100)
print(f"MIS size: {result.value}")
''',
    "qaoa_tsp": '''"""QAOA for the Traveling Salesman Problem.

Finds the shortest route visiting all cities exactly once.
Output: approximate tour cost.
"""

from quonic.algorithms import qaoa_tsp

distances = {
    (0, 1): 1.0, (1, 0): 1.0,
    (1, 2): 2.0, (2, 1): 2.0,
    (0, 2): 1.5, (2, 0): 1.5,
}
result = qaoa_tsp(distances, 3, p=1, maxiter=100)
print(f"Tour cost: {result.value}")
''',
    "qbm": '''"""Quantum Boltzmann Machine demo.

Quantum version of Boltzmann machine for generative modeling.
Output: learned distribution.
"""

from quonic.algorithms import qbm

result = qbm(temperature=1.0)
print(result.counts)
''',
    "qcnn": '''"""Quantum Convolutional Neural Network demo.

Quantum CNN for image classification tasks.
Output: classification accuracy.
"""

from quonic.algorithms import qcnn

result = qcnn(maxiter=50)
print(f"Accuracy: {result.value}")
''',
    "qft": '''"""Quantum Fourier Transform.

The quantum analogue of the discrete Fourier transform.
Output: QFT of the input state.
"""

from quonic.algorithms import qft

result = qft(n_qubits=3, shots=1024)
print(result.counts)
''',
    "qgan": '''"""Quantum GAN (Generative Adversarial Network) demo.

Quantum generator + classical discriminator for data generation.
Output: generated distribution.
"""

from quonic.algorithms import qgan

result = qgan(n_steps=50)
print(result.counts)
''',
    "qgnn": '''"""Quantum Graph Neural Network demo.

Quantum version of GNN for graph-structured data.
Output: node/graph embeddings.
"""

from quonic.algorithms import qgnn

result = qgnn()
print(result.counts)
''',
    "qng": '''"""Quantum Natural Gradient demo.

Uses the quantum Fisher information matrix for better optimization.
Output: optimized parameters.
"""

from quonic.algorithms import qng

result = qng(n_params=2, maxiter=50)
print(f"Final loss: {result.value}")
''',
    "qnn": '''"""Quantum Neural Network demo.

Variational quantum circuit as a neural network.
Output: trained model predictions.
"""

from quonic.algorithms import qnn

result = qnn(n_qubits=2, depth=2)
print(result.counts)
''',
    "qpca": '''"""Quantum PCA (Principal Component Analysis) demo.

Exponentially faster PCA for density matrices.
Output: principal eigenvalues.
"""

from quonic.algorithms import qpca

result = qpca()
print(result.counts)
''',
    "qrl": '''"""Quantum Reinforcement Learning demo.

Quantum agent learning in a classical environment.
Output: learned policy.
"""

from quonic.algorithms import qrl

result = qrl(n_episodes=10)
print(result.counts)
''',
    "qsp": '''"""Quantum Signal Processing demo.

Core subroutine for quantum singular value transformation.
Output: transformed signal.
"""

from quonic.algorithms import qsp

result = qsp(angle=0.785)
print(result.counts)
''',
    "qsvm": '''"""Quantum Support Vector Machine demo.

Uses quantum kernel for classification.
Output: classification accuracy.
"""

from quonic.algorithms import qsvm

result = qsvm()
print(result.counts)
''',
    "qtda": '''"""Quantum Topological Data Analysis demo.

Quantum algorithm for persistent homology.
Output: topological features.
"""

from quonic.algorithms import qtda

result = qtda()
print(result.counts)
''',
    "qtransformer": '''"""Quantum Transformer demo.

Quantum attention mechanism for sequence modeling.
Output: attention weights.
"""

from quonic.algorithms import qtransformer

result = qtransformer()
print(result.counts)
''',
    "quantum_annealing": '''"""Quantum annealing with hybrid classical-quantum solver.

Simulates quantum annealing for optimization problems.
Output: approximate ground state.
"""

from quonic.algorithms import quantum_annealing_hybrid

result = quantum_annealing_hybrid(n_spins=4, n_steps=100)
print(result.counts)
''',
    "quantum_bayesian": '''"""Quantum Bayesian inference demo.

Quantum version of Bayesian updating.
Output: posterior probabilities.
"""

from quonic.algorithms import quantum_bayesian

result = quantum_bayesian(prior_h0=0.5, likelihood_h0=0.8, likelihood_h1=0.3)
print(result.value)
''',
    "quantum_clustering": '''"""Quantum k-means clustering with SWAP test.

Quantum algorithm for unsupervised clustering using SWAP test for distance estimation.
Output: cluster assignments.
"""

from quonic.algorithms import quantum_clustering

points = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
centroids = [[0.0, 0.0], [1.0, 1.0]]
result = quantum_clustering(points, centroids)
print(result.metadata["assignments"])
''',
    "quantum_eigenvalue": '''"""Quantum eigenvalue estimation demo.

Estimates eigenvalues of a unitary operator.
Output: eigenvalue estimates.
"""

from quonic.algorithms import quantum_eigenvalue

result = quantum_eigenvalue()
print(result.counts)
''',
    "quantum_fitting": '''"""Quantum curve fitting demo.

Quantum version of regression/curve fitting.
Output: fitted parameters.
"""

from quonic.algorithms import quantum_fitting

result = quantum_fitting()
print(result.counts)
''',
    "quantum_kernel": '''"""Quantum kernel estimation.

Computes quantum kernel matrix for machine learning.
Output: kernel matrix entries.
"""

from quonic.algorithms import quantum_kernel

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
result = quantum_kernel(X, n_qubits=2, shots=10000)
print(result.counts)
''',
    "quantum_matrix_inversion": '''"""Quantum matrix inversion demo.

HHL-based matrix inversion for linear systems.
Output: solution vector.
"""

from quonic.algorithms import quantum_matrix_inversion

result = quantum_matrix_inversion()
print(result.counts)
''',
    "quantum_monte_carlo": '''"""Quantum Monte Carlo integration demo.

Quantum speedup for Monte Carlo methods.
Output: estimated integral value.
"""

from quonic.algorithms import quantum_monte_carlo

result = quantum_monte_carlo(n_qubits=2, shots=1024)
print(f"Estimated value: {result.value}")
''',
    "quantum_ode": '''"""Quantum ODE solver demo.

Quantum algorithm for solving ordinary differential equations.
Output: solution trajectory.
"""

from quonic.algorithms import quantum_ode

result = quantum_ode(shots=1024)
print(result.counts)
''',
    "quantum_pde": '''"""Quantum PDE solver demo.

Quantum algorithm for solving partial differential equations.
Output: solution field.
"""

from quonic.algorithms import quantum_pde

result = quantum_pde(shots=1024)
print(result.counts)
''',
    "quantum_walk": '''"""Quantum walk on a line.

The quantum analogue of a random walk — spreads quadratically faster.
Output: position distribution after n steps.
"""

from quonic.algorithms import quantum_walk

result = quantum_walk(n_positions=5, steps=10, shots=1024)
print(result.counts)
''',
    "rejection_sampling": '''"""Quantum rejection sampling demo.

Quantum-enhanced rejection sampling from a target distribution.
Output: samples from the target distribution.
"""

from quonic.algorithms import rejection_sampling

result = rejection_sampling(n_samples=100)
print(result.counts)
''',
    "shor_code": '''"""Shor's 9-qubit code: the first quantum error correction code.

Corrects arbitrary single-qubit errors.
Output: corrected logical state.
"""

from quonic.algorithms import shor_code

result = shor_code(error_qubit=0, shots=100)
print(result.counts)
''',
    "simon": '''"""Simon's algorithm: find the period of a 2-to-1 function.

Exponential speedup over classical; precursor to Shor's algorithm.
Output: the hidden period string.
"""

from quonic.algorithms import simon, oracle

n = 3
# f(x) = f(x XOR s) with s = "101"
ora = oracle(n)(lambda i: i ^ 5 if i < 4 else i)  # simplified
result = simon(n, ora, shots=200)
print(result.counts)
''',
    "stabilizer": '''"""Stabilizer formalism demo.

Demonstrates Clifford group simulation via stabilizer tableau.
Output: stabilizer state measurements.
"""

from quonic.algorithms import stabilizer

result = stabilizer(n_qubits=3, shots=100)
print(result.counts)
''',
    "steane_code": '''"""Steane code: [[7,1,3]] CSS code.

Corrects any single-qubit error using 7 physical qubits.
Output: corrected logical state.
"""

from quonic.algorithms import steane_code

result = steane_code(error_qubit=0, shots=100)
print(result.counts)
''',
    "superdense_coding": '''"""Superdense coding: send 2 classical bits using 1 qubit.

Alice encodes 2 bits by manipulating her half of an entangled pair.
Output: decoded message.
"""

from quonic.algorithms import superdense_coding

for msg in ["00", "01", "10", "11"]:
    result = superdense_coding(message=msg, shots=100)
    decoded = max(result.counts, key=result.counts.get)
    print(f"Sent: {msg}, Decoded: {decoded}")
''',
    "surface_code": '''"""Surface code error correction demo.

The leading candidate for fault-tolerant quantum computation.
Output: logical qubit with error correction.
"""

from quonic.algorithms import surface_code

result = surface_code(distance=3, shots=100)
print(result.counts)
''',
    "swap_test": '''"""SWAP test: estimate overlap between two quantum states.

Output: P(|0>) = (1 + |<a|b>|^2) / 2, so high P means similar states.
"""

from quonic import qgate
from quonic.algorithms import swap_test
from quonic.gates import X

def prep_a(qc, q):
    pass  # |0>

def prep_b(qc, q):
    qgate(X, q)  # |1>

result = swap_test(1, prep_a, prep_b, shots=10000)
print(result.counts)
''',
    "syndrome": '''"""Syndrome measurement demo.

Extracts error syndromes without disturbing the encoded state.
Output: syndrome bits indicating error location.
"""

from quonic.algorithms import syndrome

result = syndrome(n_data=3, shots=100)
print(result.counts)
''',
    "vqr": '''"""Variational Quantum Regressor.

Quantum model for regression tasks.
Output: predicted values.
"""

from quonic.algorithms import vqr

X = [[0.0], [0.5], [1.0], [1.5]]
y = [0.0, 0.479, 0.841, 0.997]
result = vqr(X, y, n_params=2, maxiter=100)
print(f"Final loss: {result.value}")
''',
}

def main():
    for name, content in EXAMPLES.items():
        d = os.path.join("examples", name)
        os.makedirs(d, exist_ok=True)
        fname = os.path.join(d, f"{name}.py")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  {fname}")

if __name__ == "__main__":
    main()
