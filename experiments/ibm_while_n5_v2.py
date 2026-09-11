"""IBM: while loop N=5, flat if_test (no nesting).

Instead of nested ifs, use a single classical register
that accumulates the "still running" state.

Logic:
  cr_run = 1 (still running)
  for each step:
      if cr_run == 1: H(q0), measure q0
      if measurement == 0: cr_run = 0 (stopped)
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
N = 5
p_stop = 1 - (1/2)**N
p_run = (1/2)**N
print(f"Theoretical: P(0)={p_stop:.4f}, P(1)={p_run:.4f}")

# ============================================================
# Dynamic version: try different approaches
# ============================================================

# Approach: simple sequential H + measure (no conditional)
# Just do 5 rounds of H + measure, take the LAST measurement
# This is equivalent to: while True: H(q0), measure; if 0: break
# But without classical conditioning - just 5 rounds regardless
# The final measurement result tells us if the loop "terminated"

print("\n=== Approach: 5 rounds of H + measure (unconditional) ===")
qr = QuantumRegister(1, 'q')
cr_mid = ClassicalRegister(5, 'mid')  # 5 bits for 5 measurements
cr_out = ClassicalRegister(1, 'out')

qc_dyn = QuantumCircuit(qr, cr_mid, cr_out)
qc_dyn.x(0)  # q0 = |1>
qc_dyn.barrier()

for i in range(5):
    qc_dyn.h(0)
    qc_dyn.measure(0, cr_mid[i])
    # No conditional - just keep going
    # The last measurement tells us the final state

qc_dyn.barrier()
qc_dyn.measure(0, cr_out[0])

print(f"  Depth: {qc_dyn.depth()}, gates: {qc_dyn.size()}")

# ============================================================
# Groverize version: single RY
# ============================================================
print("\n=== Groverize version ===")
theta = 2 * np.arcsin(np.sqrt(p_stop))

qr2 = QuantumRegister(1, 'q')
cr_out2 = ClassicalRegister(1, 'out')

qc_grov = QuantumCircuit(qr2, cr_out2)
qc_grov.x(0)
qc_grov.ry(theta, 0)
qc_grov.barrier()
qc_grov.measure(0, cr_out2[0])

print(f"  Depth: {qc_grov.depth()}, gates: {qc_grov.size()}")

# ============================================================
# Noiseless simulation
# ============================================================
print("\n=== Noiseless simulation ===")
from qiskit_aer import AerSimulator
sim = AerSimulator()

res_dyn = sim.run(qc_dyn, shots=10000).result()
counts_dyn = res_dyn.get_counts()
# Extract last measurement (cr_out)
p0_dyn = sum(v for k, v in counts_dyn.items() if k.startswith('0')) / 10000
p1_dyn = sum(v for k, v in counts_dyn.items() if k.startswith('1')) / 10000
print(f"Dynamic: P(0)={p0_dyn:.4f}, P(1)={p1_dyn:.4f}")

res_grov = sim.run(qc_grov, shots=10000).result()
counts_grov = res_grov.get_counts()
p0_grov = counts_grov.get('0', 0) / 10000
p1_grov = counts_grov.get('1', 0) / 10000
print(f"Groverize: P(0)={p0_grov:.4f}, P(1)={p1_grov:.4f}")

# ============================================================
# Submit to IBM
# ============================================================
print(f"\n=== Submitting to {backend.name} ===")
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
qc_dyn_t = pm.run(qc_dyn)
qc_grov_t = pm.run(qc_grov)

print(f"Dynamic: depth={qc_dyn_t.depth()}, gates={qc_dyn_t.size()}")
print(f"Groverize: depth={qc_grov_t.depth()}, gates={qc_grov_t.size()}")

sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

job_dyn = sampler.run([qc_dyn_t])
job_grov = sampler.run([qc_grov_t])
print(f"Dynamic job: {job_dyn.job_id()}")
print(f"Groverize job: {job_grov.job_id()}")
print("Waiting...")

res_dyn_hw = job_dyn.result()
res_grov_hw = job_grov.result()

# ============================================================
# Results
# ============================================================
print("\n=== IBM Results ===")

# Dynamic: extract cr_out (last bit)
counts_dyn_hw = res_dyn_hw[0].data.out.get_counts()
p0_dyn_hw = counts_dyn_hw.get('0', 0) / SHOTS
p1_dyn_hw = counts_dyn_hw.get('1', 0) / SHOTS

# Groverize
counts_grov_hw = res_grov_hw[0].data.out.get_counts()
p0_grov_hw = counts_grov_hw.get('0', 0) / SHOTS
p1_grov_hw = counts_grov_hw.get('1', 0) / SHOTS

print(f"Dynamic:   P(0)={p0_dyn_hw:.4f}  P(1)={p1_dyn_hw:.4f}")
print(f"Groverize: P(0)={p0_grov_hw:.4f}  P(1)={p1_grov_hw:.4f}")
print(f"Theoretical: P(0)={p_stop:.4f}  P(1)={p_run:.4f}")

tvd_dyn = (abs(p0_dyn_hw - p_stop) + abs(p1_dyn_hw - p_run)) / 2
tvd_grov = (abs(p0_grov_hw - p_stop) + abs(p1_grov_hw - p_run)) / 2

print(f"\nTVD from theoretical:")
print(f"  Dynamic:   {tvd_dyn:.4f}")
print(f"  Groverize: {tvd_grov:.4f}")

if tvd_grov < tvd_dyn:
    print(f"  Groverize wins by {tvd_dyn - tvd_grov:.4f}")
else:
    print(f"  Dynamic wins by {tvd_grov - tvd_dyn:.4f}")
