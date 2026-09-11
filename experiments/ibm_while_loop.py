"""IBM: test while_loop primitive.

while q0 == 1:
    H(q0)
    measure q0

This is NOT nested if_test - it's a while_loop primitive.
IBM might support this even if nested if_test is unsupported.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token='xJA97lPiGPyWT4wXKb0k8gkBGfmou5lB9hTGEBaHJD2a',
    instance='auto'
)
backend = service.backend('ibm_marrakesh')
print(f"Backend: {backend.name}")

SHOTS = 4000

# ============================================================
# while loop: while q0 == 1, apply H and measure
# ============================================================
print("\n=== Building while loop circuit ===")

qr = QuantumRegister(1, 'q')
cr = ClassicalRegister(1, 'c')
cr_out = ClassicalRegister(1, 'out')

qc = QuantumCircuit(qr, cr, cr_out)
qc.x(0)  # q0 = |1>
qc.barrier()

# while cr == 1: H(q0), measure(q0)
# Initially cr is 0, so we need to set it to 1 first
# Or: use a do-while pattern
# Let's try: measure first, then while
qc.measure(0, cr[0])  # initial measurement
with qc.while_loop((cr, 1)):
    qc.h(0)
    qc.measure(0, cr[0])

qc.barrier()
qc.measure(0, cr_out[0])

print(f"Depth: {qc.depth()}, gates: {qc.size()}")
print("Circuit built OK")

# ============================================================
# Groverize: RY rotation for equivalent distribution
# ============================================================
# For while loop with H + measure, max iterations unbounded:
# P(stop at iteration k) = (1/2)^k
# P(0) = sum_{k=1}^{inf} (1/2)^k = 1
# But in practice, hardware has finite coherence time
# Let's assume max ~10 iterations: P(0) = 1 - (1/2)^10 ≈ 0.999

# For fair comparison, use same max iterations as dynamic circuit
# Dynamic circuit will terminate naturally when measurement gives 0

print("\n=== Building Groverize (RY) circuit ===")
# Theoretical: while loop with H + measure terminates with P(0)→1
# In practice on hardware: limited by coherence time
# RY rotation: prepare |0> with high probability
theta = 2 * np.arcsin(np.sqrt(0.99))  # P(0) = 0.99

qr2 = QuantumRegister(1, 'q')
cr_out2 = ClassicalRegister(1, 'out')

qc_grov = QuantumCircuit(qr2, cr_out2)
qc_grov.x(0)
qc_grov.ry(theta, 0)
qc_grov.barrier()
qc_grov.measure(0, cr_out2[0])

print(f"Depth: {qc_grov.depth()}, gates: {qc_grov.size()}")

# ============================================================
# Submit
# ============================================================
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
qc_t = pm.run(qc)
qc_grov_t = pm.run(qc_grov)

print(f"\nDynamic transpiled: depth={qc_t.depth()}, gates={qc_t.size()}")
print(f"Groverize transpiled: depth={qc_grov_t.depth()}, gates={qc_grov_t.size()}")

sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print(f"\nSubmitting to {backend.name}...")
job_dyn = sampler.run([qc_t])
job_grov = sampler.run([qc_grov_t])
print(f"Dynamic job: {job_dyn.job_id()}")
print(f"Groverize job: {job_grov.job_id()}")
print("Waiting...")

try:
    res_dyn = job_dyn.result()
    counts_dyn = res_dyn[0].data.out.get_counts()
    p0_dyn = counts_dyn.get('0', 0) / SHOTS
    p1_dyn = counts_dyn.get('1', 0) / SHOTS
    print(f"\nDynamic:   P(0)={p0_dyn:.4f}  P(1)={p1_dyn:.4f}")
except Exception as e:
    print(f"\nDynamic FAILED: {e}")

try:
    res_grov = job_grov.result()
    counts_grov = res_grov[0].data.out.get_counts()
    p0_grov = counts_grov.get('0', 0) / SHOTS
    p1_grov = counts_grov.get('1', 0) / SHOTS
    print(f"Groverize: P(0)={p0_grov:.4f}  P(1)={p1_grov:.4f}")
except Exception as e:
    print(f"Groverize FAILED: {e}")
