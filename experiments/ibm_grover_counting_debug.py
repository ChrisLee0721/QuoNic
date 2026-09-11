"""Debug: quantum counting with correct phase extraction.

Grover operator eigenvalue: e^{i(π - 2θ)}, NOT e^{2iθ}.
Phase extraction: θ = π(0.5 - φ), not θ = πφ.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import numpy as np

# ============================================================
# Problem setup
# ============================================================
N_SEARCH_QUBITS = 3
N_STATES = 2 ** N_SEARCH_QUBITS  # 8
M_SOLUTIONS = 1
SOLUTIONS = [5]
SHOTS = 4000

theta_ideal = np.arcsin(np.sqrt(M_SOLUTIONS / N_STATES))
print(f"Search: {N_STATES} states, M={M_SOLUTIONS}, θ={theta_ideal:.4f}")
print(f"Grover eigenvalue: e^(i(π - 2θ)) = e^(i*{np.pi - 2*theta_ideal:.4f})")
print(f"Phase = 0.5 - θ/π = {0.5 - theta_ideal/np.pi:.4f}")

# ============================================================
# Oracle and Grover as Gate objects
# ============================================================
def make_oracle_gate(n_qubits, solutions):
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)
    for sol in solutions:
        bits = format(sol, f'0{n_qubits}b')[::-1]
        for i, b in enumerate(bits):
            if b == '0': qc.x(qr[i])
        qc.h(qr[-1])
        qc.mcx(list(qr[:-1]), qr[-1])
        qc.h(qr[-1])
        for i, b in enumerate(bits):
            if b == '0': qc.x(qr[i])
    return qc.to_gate(label='O')

def make_diffusion_gate(n_qubits):
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)
    qc.h(qr)
    qc.x(qr)
    qc.h(qr[-1])
    qc.mcx(list(qr[:-1]), qr[-1])
    qc.h(qr[-1])
    qc.x(qr)
    qc.h(qr)
    return qc.to_gate(label='D')

def make_grover_gate(n_qubits, solutions):
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)
    oracle = make_oracle_gate(n_qubits, solutions)
    diffusion = make_diffusion_gate(n_qubits)
    qc.append(oracle, qr)
    qc.append(diffusion, qr)
    return qc.to_gate(label='G')

grover_gate = make_grover_gate(N_SEARCH_QUBITS, SOLUTIONS)

# ============================================================
# Quantum counting circuit
# ============================================================
def build_counting_circuit(n_counting, n_search, grover_gate):
    counting_qubits = QuantumRegister(n_counting, 'count')
    search_qubits = QuantumRegister(n_search, 'search')
    cr = ClassicalRegister(n_counting, 'c')
    qc = QuantumCircuit(counting_qubits, search_qubits, cr)

    # Prepare search in uniform superposition
    qc.h(search_qubits)

    # Controlled Grover^{2^j}
    for j in range(n_counting):
        power = 2 ** j
        c_grover = grover_gate.control(1, label=f'c-G^{power}')
        for _ in range(power):
            qc.append(c_grover, [counting_qubits[j]] + list(search_qubits))

    # Inverse QFT
    if n_counting == 1:
        qc.h(counting_qubits[0])
    elif n_counting == 2:
        qc.h(counting_qubits[1])
        qc.cp(-np.pi/2, counting_qubits[0], counting_qubits[1])
        qc.h(counting_qubits[0])

    qc.measure(counting_qubits, cr)
    return qc

def parse_phase(counts, n_counting, shots):
    """Extract θ from QPE measurement, accounting for Grover eigenvalue structure."""
    top = sorted(counts.items(), key=lambda x: -x[1])[:4]
    # Measured phase φ
    phi = sum(int(bs, 2) / (2**n_counting) * cnt for bs, cnt in top) / shots
    # Grover eigenvalue: e^{i(π - 2θ)} → φ = 0.5 - θ/π → θ = π(0.5 - φ)
    theta_est = np.pi * (0.5 - phi)
    # Also try the other branch: φ = 0.5 + θ/π → θ = π(φ - 0.5)
    theta_est2 = np.pi * (phi - 0.5)
    # Pick the positive one that makes sense
    if theta_est > 0:
        theta = theta_est
    else:
        theta = abs(theta_est2)
    M_est = N_STATES * np.sin(theta) ** 2
    if M_est > 0 and M_est < N_STATES:
        k_opt = max(1, round(np.pi / (4 * theta) - 0.5))
    else:
        k_opt = 1
    return phi, theta, M_est, k_opt, top

# ============================================================
# Test on simulator
# ============================================================
print(f"\n{'='*60}")
print("SIMULATOR (noiseless)")

sim = AerSimulator()
for n_count in [1, 2]:
    qc = build_counting_circuit(n_count, N_SEARCH_QUBITS, grover_gate)
    qc_sim = transpile(qc, basis_gates=['cx', 'u3', 'u2', 'u1', 'id'])
    result = sim.run(qc_sim, shots=SHOTS).result()
    counts = result.get_counts()

    phi, theta, M_est, k_opt, top = parse_phase(counts, n_count, SHOTS)
    print(f"\n  {n_count} counting qubit(s), depth {qc.depth()}:")
    print(f"    Outcomes: {top}")
    print(f"    Phase φ = {phi:.4f}")
    print(f"    θ = {theta:.4f} (true: {theta_ideal:.4f})")
    print(f"    M ≈ {M_est:.2f} (true: {M_SOLUTIONS})")
    print(f"    k_opt = {k_opt}")

# ============================================================
# Test on hardware
# ============================================================
print(f"\n{'='*60}")
print("HARDWARE")

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token='aKHCjMbuXdGqGPXYYb6p38eYBOtun5DiDFD5kw9CBpOL',
    instance='auto'
)
backend = service.backend('ibm_marrakesh')
pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

for n_count in [1, 2]:
    qc = build_counting_circuit(n_count, N_SEARCH_QUBITS, grover_gate)
    qc_t = pm.run(qc)

    print(f"\n  {n_count} counting qubit(s), depth {qc_t.depth()}:")

    sampler = Sampler(backend)
    sampler.options.default_shots = SHOTS
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False

    job = sampler.run([qc_t])
    print(f"  Job: {job.job_id()}")
    res = job.result()
    counts = res[0].data.c.get_counts()

    phi, theta, M_est, k_opt, top = parse_phase(counts, n_count, SHOTS)
    print(f"    Outcomes: {top}")
    print(f"    Phase φ = {phi:.4f}")
    print(f"    θ = {theta:.4f} (true: {theta_ideal:.4f})")
    print(f"    M ≈ {M_est:.2f} (true: {M_SOLUTIONS})")
    print(f"    k_opt = {k_opt}")
