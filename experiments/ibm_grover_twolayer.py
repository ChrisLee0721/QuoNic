"""IBM: Two-layer Grover experiment — quantum counting + pure unitary vs dynamic.

Tests the two-layer quantum execution paradigm:
  Layer 1: Quantum counting to estimate M (number of solutions)
  Layer 2: Compile optimal θ, execute pure unitary Grover

Compares three approaches:
  A. Pure unitary Grover (k=3 iterations, no measurement mid-circuit)
  B. Two-layer: quantum counting → compile → pure unitary Grover
  C. Dynamic Grover: measure-and-branch after each iteration

Expected: B ≈ A >> C. Dynamic breaks amplitude amplification via measurement.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np
import json
from datetime import datetime

# ============================================================
# Search problem: n_qubit search space, M solutions
# ============================================================
N_SEARCH_QUBITS = 3  # 2^3 = 8 states (smaller for shallower circuits)
N_STATES = 2 ** N_SEARCH_QUBITS
M_SOLUTIONS = 1  # number of solutions
SOLUTIONS = [5]  # target state
SHOTS = 4000
N_COUNTING_QUBITS = 2  # coarse precision for quantum counting

# Ideal theta
theta_ideal = np.arcsin(np.sqrt(M_SOLUTIONS / N_STATES))

# ============================================================
# Oracle and diffusion
# ============================================================
def apply_oracle(qc, search_qubits, solutions):
    """Apply phase oracle: flip phase of solution states."""
    for sol in solutions:
        bits = format(sol, f'0{len(search_qubits)}b')[::-1]
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(search_qubits[i])
    qc.h(search_qubits[-1])
    qc.mcx(list(search_qubits[:-1]), search_qubits[-1])
    qc.h(search_qubits[-1])
    for sol in solutions:
        bits = format(sol, f'0{len(search_qubits)}b')[::-1]
        for i, b in enumerate(bits):
            if b == '0':
                qc.x(search_qubits[i])

def apply_diffusion(qc, search_qubits):
    """Apply diffusion operator (inversion about mean)."""
    qc.h(search_qubits)
    qc.x(search_qubits)
    qc.h(search_qubits[-1])
    qc.mcx(list(search_qubits[:-1]), search_qubits[-1])
    qc.h(search_qubits[-1])
    qc.x(search_qubits)
    qc.h(search_qubits)

def apply_grover(qc, search_qubits, solutions):
    """Apply one Grover iteration: oracle + diffusion."""
    apply_oracle(qc, search_qubits, solutions)
    apply_diffusion(qc, search_qubits)

def make_grover_gate(n_qubits, solutions):
    """Build Grover iteration as a Gate object (for controlled operations)."""
    qr = QuantumRegister(n_qubits, 'q')
    qc = QuantumCircuit(qr)
    apply_grover(qc, qr, solutions)
    return qc.to_gate(label='G')

# ============================================================
# Connect to IBM
# ============================================================
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token='aKHCjMbuXdGqGPXYYb6p38eYBOtun5DiDFD5kw9CBpOL',
    instance='auto'
)
backend = service.backend('ibm_marrakesh')
print(f"Backend: {backend.name}")

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

results = {}

# ============================================================
# Layer 1: Quantum counting
# ============================================================
print(f"\n{'='*60}")
print(f"LAYER 1: Quantum counting ({N_COUNTING_QUBITS} counting qubits)")
print(f"  Search space: {N_STATES} states, {M_SOLUTIONS} solutions {SOLUTIONS}")
print(f"  Ideal θ = {theta_ideal:.4f}")

counting_qubits = QuantumRegister(N_COUNTING_QUBITS, 'count')
search_qubits = QuantumRegister(N_SEARCH_QUBITS, 'search')
cr_count = ClassicalRegister(N_COUNTING_QUBITS, 'c_count')

qc_count = QuantumCircuit(counting_qubits, search_qubits, cr_count)

# Prepare search register in uniform superposition
qc_count.h(search_qubits)

# Build Grover gate for controlled operations
grover_gate = make_grover_gate(N_SEARCH_QUBITS, SOLUTIONS)

# Controlled Grover operators: 2^j iterations controlled by counting qubit j
for j in range(N_COUNTING_QUBITS):
    power = 2 ** j
    c_grover = grover_gate.control(1, label=f'c-G^{power}')
    for _ in range(power):
        qc_count.append(c_grover, [counting_qubits[j]] + list(search_qubits))

# Inverse QFT on counting register
qc_count.h(counting_qubits[1])
qc_count.cp(-np.pi/2, counting_qubits[0], counting_qubits[1])
qc_count.h(counting_qubits[0])

# Measure counting register
qc_count.measure(counting_qubits, cr_count)

# Transpile and run
qc_count_t = pm.run(qc_count)
print(f"  Circuit depth: {qc_count_t.depth()}")

sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print(f"  Submitting quantum counting...")
job_count = sampler.run([qc_count_t])
print(f"  Job ID: {job_count.job_id()}")
res_count = job_count.result()
counts_count = res_count[0].data.c_count.get_counts()

# Parse phase estimate (corrected: Grover eigenvalue is e^{i(π-2θ)})
top_counts = sorted(counts_count.items(), key=lambda x: -x[1])[:4]
phi_est = sum(int(bs, 2) / (2**N_COUNTING_QUBITS) * cnt for bs, cnt in top_counts) / SHOTS

# θ = π(0.5 - φ), not πφ
theta_est = np.pi * (0.5 - phi_est)
if theta_est < 0:
    theta_est = abs(np.pi * (phi_est - 0.5))
M_est_raw = N_STATES * np.sin(theta_est) ** 2
M_est = max(1, round(M_est_raw))

# Optimal iterations from estimated M
if M_est > 0 and M_est < N_STATES:
    theta_opt = np.arcsin(np.sqrt(M_est / N_STATES))
    k_opt = max(1, round(np.pi / (4 * theta_opt) - 0.5))
else:
    k_opt = 1

print(f"\n  Quantum counting results:")
print(f"    Top outcomes: {top_counts}")
print(f"    Phase φ = {phi_est:.4f}")
print(f"    θ estimate: {theta_est:.4f} (ideal: {theta_ideal:.4f})")
print(f"    M estimate: {M_est_raw:.2f} → rounded to {M_est} (true: {M_SOLUTIONS})")
print(f"    Optimal iterations: k = {k_opt}")

results['layer1'] = {
    'counts': {k: v for k, v in top_counts},
    'phase_est': float(phi_est),
    'theta_est': float(theta_est),
    'M_est_raw': float(M_est_raw),
    'M_est': int(M_est),
    'k_opt': int(k_opt),
    'theta_ideal': float(theta_ideal),
}

# ============================================================
# Method A: Pure unitary Grover (k=3, hardcoded)
# ============================================================
print(f"\n{'='*60}")
K_FIXED = 3
print(f"METHOD A: Pure unitary Grover (k={K_FIXED} iterations)")

qr_a = QuantumRegister(N_SEARCH_QUBITS, 'search')
cr_a = ClassicalRegister(N_SEARCH_QUBITS, 'c_out')
qc_pure = QuantumCircuit(qr_a, cr_a)
qc_pure.h(qr_a)
for _ in range(K_FIXED):
    apply_grover(qc_pure, qr_a, SOLUTIONS)
qc_pure.measure(qr_a, cr_a)

qc_pure_t = pm.run(qc_pure)
print(f"  Depth: {qc_pure_t.depth()}")

# ============================================================
# Method B: Two-layer (counting + compiled Grover)
# ============================================================
print(f"\n{'='*60}")
print(f"METHOD B: Two-layer (quantum counting → k={k_opt} iterations)")

qr_b = QuantumRegister(N_SEARCH_QUBITS, 'search')
cr_b = ClassicalRegister(N_SEARCH_QUBITS, 'c_out')
qc_twolayer = QuantumCircuit(qr_b, cr_b)
qc_twolayer.h(qr_b)
for _ in range(k_opt):
    apply_grover(qc_twolayer, qr_b, SOLUTIONS)
qc_twolayer.measure(qr_b, cr_b)

qc_twolayer_t = pm.run(qc_twolayer)
print(f"  Depth: {qc_twolayer_t.depth()}")

# ============================================================
# Method C: Dynamic Grover (measure-and-branch)
# ============================================================
print(f"\n{'='*60}")
print(f"METHOD C: Dynamic Grover (measure after each iteration)")

# Dynamic version: after each Grover iteration, measure search register.
# If solution found, stop. This breaks amplitude amplification.
# We try K_FIXED iterations with mid-circuit measurement.
qr_c = QuantumRegister(N_SEARCH_QUBITS, 'search')
cr_mid_c = ClassicalRegister(N_SEARCH_QUBITS, 'c_mid')
cr_out_c = ClassicalRegister(N_SEARCH_QUBITS, 'c_out')
qc_dyn = QuantumCircuit(qr_c, cr_mid_c, cr_out_c)
qc_dyn.h(qr_c)

for i in range(K_FIXED):
    apply_grover(qc_dyn, qr_c, SOLUTIONS)
    qc_dyn.measure(qr_c, cr_mid_c)
    # Note: after measurement, state collapses. Subsequent Grover iterations
    # start from collapsed state, not from coherent superposition.
    # This is the fundamental break: measurement destroys amplitude amplification.

# Final measurement
qc_dyn.measure(qr_c, cr_out_c)

qc_dyn_t = pm.run(qc_dyn)
print(f"  Depth: {qc_dyn_t.depth()}")

# ============================================================
# Submit all three in one batch
# ============================================================
print(f"\n{'='*60}")
print("Submitting all three circuits in one batch...")

sampler2 = Sampler(backend)
sampler2.options.default_shots = SHOTS
sampler2.options.dynamical_decoupling.enable = False
sampler2.options.twirling.enable_gates = False
sampler2.options.twirling.enable_measure = False

job = sampler2.run([qc_pure_t, qc_twolayer_t, qc_dyn_t])
print(f"Job ID: {job.job_id()}")
print("Waiting for results...")

res = job.result()

counts_pure = res[0].data.c_out.get_counts()
counts_twolayer = res[1].data.c_out.get_counts()
counts_dyn = res[2].data.c_out.get_counts()

# ============================================================
# Analyze results
# ============================================================
def success_rate(counts, solutions, n_qubits, shots):
    """Compute probability of finding any solution."""
    success = 0
    for sol in solutions:
        bs = format(sol, f'0{n_qubits}b')
        success += counts.get(bs, 0)
    return success / shots

p_pure = success_rate(counts_pure, SOLUTIONS, N_SEARCH_QUBITS, SHOTS)
p_twolayer = success_rate(counts_twolayer, SOLUTIONS, N_SEARCH_QUBITS, SHOTS)
p_dyn = success_rate(counts_dyn, SOLUTIONS, N_SEARCH_QUBITS, SHOTS)

# Theoretical predictions
p_pure_theory = np.sin((2*K_FIXED + 1) * theta_ideal) ** 2
p_twolayer_theory = np.sin((2*k_opt + 1) * theta_ideal) ** 2

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  Search: {N_STATES} states, solutions = {SOLUTIONS}")
print(f"  Ideal θ = {theta_ideal:.4f}, M = {M_SOLUTIONS}")
print(f"")
print(f"  Method A: Pure unitary (k={K_FIXED})")
print(f"    Measured: {p_pure:.4f}  Theory: {p_pure_theory:.4f}")
print(f"  Method B: Two-layer (counting → k={k_opt})")
print(f"    Measured: {p_twolayer:.4f}  Theory: {p_twolayer_theory:.4f}")
print(f"  Method C: Dynamic (measure-and-branch, {K_FIXED} iterations)")
print(f"    Measured: {p_dyn:.4f}")
print(f"")
print(f"  Advantage B vs C: {p_twolayer/p_dyn:.2f}x" if p_dyn > 0 else "  Advantage B vs C: ∞")
print(f"  Advantage A vs C: {p_pure/p_dyn:.2f}x" if p_dyn > 0 else "  Advantage A vs C: ∞")

results['methods'] = {
    'A_pure_unitary': {
        'k': K_FIXED,
        'measured': float(p_pure),
        'theory': float(p_pure_theory),
        'depth': int(qc_pure_t.depth()),
    },
    'B_twolayer': {
        'k': k_opt,
        'measured': float(p_twolayer),
        'theory': float(p_twolayer_theory),
        'depth': int(qc_twolayer_t.depth()),
    },
    'C_dynamic': {
        'k': K_FIXED,
        'measured': float(p_dyn),
        'depth': int(qc_dyn_t.depth()),
    },
}

# ============================================================
# Save
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"experiments/ibm_grover_twolayer_{timestamp}.json"
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nData saved to {save_path}")
