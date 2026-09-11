"""IBM: real while loop experiment, N=5 max iterations.

Program:
    q0 = |1>
    while q0 == 1:
        H(q0)
        measure q0

Dynamic version: nested if_test, each step = H + measure + conditional
Groverize version: single RY rotation that produces the same final distribution

After N iterations of H + measure:
  P(q0=0) = 1 - (1/2)^N = 31/32 = 96.875%
  P(q0=1) = (1/2)^N = 1/32 = 3.125%

Groverize: RY(theta) where cos²(theta/2) = 31/32
  theta = 2 * arccos(sqrt(31/32)) ≈ 0.354 rad
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
N = 5  # max iterations

# Theoretical distribution after N iterations of H + measure:
# P(0) = 1 - (1/2)^N, P(1) = (1/2)^N
p_stop = 1 - (1/2)**N  # probability of q0=0 (loop terminated)
p_run = (1/2)**N         # probability of q0=1 (still running)
print(f"\nTheoretical: P(0)={p_stop:.4f}, P(1)={p_run:.4f}")

# ============================================================
# Dynamic version: nested if_test
# ============================================================
print("\n=== Building Dynamic version ===")
# Each step: H(q0), measure(q0), if q0==1: continue

qr = QuantumRegister(1, 'q')
# N classical registers for mid-circuit measurements
cr_list = [ClassicalRegister(1, f'c{i}') for i in range(N)]
cr_out = ClassicalRegister(1, 'out')

qc_dyn = QuantumCircuit(qr, *cr_list, cr_out)
qc_dyn.x(0)  # q0 = |1>
qc_dyn.barrier()

# Step 1
qc_dyn.h(0)
qc_dyn.measure(0, cr_list[0][0])
with qc_dyn.if_test((cr_list[0], 1)):  # if q0 still |1>
    # Step 2
    qc_dyn.h(0)
    qc_dyn.measure(0, cr_list[1][0])
    with qc_dyn.if_test((cr_list[1], 1)):
        # Step 3
        qc_dyn.h(0)
        qc_dyn.measure(0, cr_list[2][0])
        with qc_dyn.if_test((cr_list[2], 1)):
            # Step 4
            qc_dyn.h(0)
            qc_dyn.measure(0, cr_list[3][0])
            with qc_dyn.if_test((cr_list[3], 1)):
                # Step 5
                qc_dyn.h(0)
                qc_dyn.measure(0, cr_list[4][0])

qc_dyn.barrier()
qc_dyn.measure(0, cr_out[0])  # final readout

print(f"  Dynamic depth: {qc_dyn.depth()}, gates: {qc_dyn.size()}")

# ============================================================
# Groverize version: single RY rotation
# ============================================================
print("\n=== Building Groverize version ===")
# The while loop's final distribution is:
# P(0) = 1 - (1/2)^5 = 31/32
# P(1) = 1/32
# Single RY rotation from |1> to this distribution:
# RY(theta)|1> = cos(theta/2)|1> + sin(theta/2)|0>
# We want sin²(theta/2) = P(0) = 31/32
# theta = 2 * arcsin(sqrt(31/32))

theta = 2 * np.arcsin(np.sqrt(p_stop))
print(f"  RY angle: {theta:.4f} rad ({np.degrees(theta):.1f} degrees)")

qr2 = QuantumRegister(1, 'q')
cr_out2 = ClassicalRegister(1, 'out')

qc_grov = QuantumCircuit(qr2, cr_out2)
qc_grov.x(0)  # q0 = |1>
qc_grov.ry(theta, 0)  # rotate to target distribution
qc_grov.barrier()
qc_grov.measure(0, cr_out2[0])

print(f"  Groverize depth: {qc_grov.depth()}, gates: {qc_grov.size()}")

# ============================================================
# Simulate both (noiseless) to verify correctness
# ============================================================
print("\n=== Noiseless simulation ===")
from qiskit_aer import AerSimulator

sim = AerSimulator()

# Dynamic: need to extract final measurement only
# Count shots where final cr_out = 0 and cr_out = 1
result_dyn = sim.run(qc_dyn, shots=10000).result()
counts_dyn_raw = result_dyn.get_counts()
# Parse: key format is "out c4 c3 c2 c1 c0" (reverse order)
counts_dyn = {'0': 0, '1': 0}
for key, val in counts_dyn_raw.items():
    # The 'out' register is the leftmost bit
    out_bit = key.split()[0] if ' ' in key else key[0]
    counts_dyn[out_bit] = counts_dyn.get(out_bit, 0) + val
print(f"Dynamic noiseless: {counts_dyn}")
p0_dyn = counts_dyn.get('0', 0) / 10000
p1_dyn = counts_dyn.get('1', 0) / 10000
print(f"  P(0)={p0_dyn:.4f}, P(1)={p1_dyn:.4f} (expected: P(0)={p_stop:.4f})")

result_grov = sim.run(qc_grov, shots=10000).result()
counts_grov = result_grov.get_counts()
print(f"Groverize noiseless: {counts_grov}")
p0_grov = counts_grov.get('0', 0) / 10000
p1_grov = counts_grov.get('1', 0) / 10000
print(f"  P(0)={p0_grov:.4f}, P(1)={p1_grov:.4f} (expected: P(0)={p_stop:.4f})")

# ============================================================
# Submit to IBM hardware
# ============================================================
print(f"\n=== Submitting to {backend.name} ===")

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
qc_dyn_t = pm.run(qc_dyn)
qc_grov_t = pm.run(qc_grov)

print(f"  Dynamic transpiled depth: {qc_dyn_t.depth()}, gates: {qc_dyn_t.size()}")
print(f"  Groverize transpiled depth: {qc_grov_t.depth()}, gates: {qc_grov_t.size()}")

sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

job_dyn = sampler.run([qc_dyn_t])
job_grov = sampler.run([qc_grov_t])

print(f"  Dynamic job: {job_dyn.job_id()}")
print(f"  Groverize job: {job_grov.job_id()}")
print("Waiting...")

res_dyn = job_dyn.result()
res_grov = job_grov.result()

# ============================================================
# Parse results
# ============================================================
print("\n=== IBM Results ===")

# Dynamic: extract final measurement
counts_dyn_hw = {}
for key, val in res_dyn[0].data.out.get_counts().items():
    counts_dyn_hw[key] = val

# Groverize
counts_grov_hw = res_grov[0].data.out.get_counts()

p0_dyn_hw = counts_dyn_hw.get('0', 0) / SHOTS
p1_dyn_hw = counts_dyn_hw.get('1', 0) / SHOTS
p0_grov_hw = counts_grov_hw.get('0', 0) / SHOTS
p1_grov_hw = counts_grov_hw.get('1', 0) / SHOTS

print(f"Dynamic:   P(0)={p0_dyn_hw:.4f}  P(1)={p1_dyn_hw:.4f}")
print(f"Groverize: P(0)={p0_grov_hw:.4f}  P(1)={p1_grov_hw:.4f}")
print(f"Theoretical: P(0)={p_stop:.4f}  P(1)={p_run:.4f}")

# TVD from theoretical
tvd_dyn = (abs(p0_dyn_hw - p_stop) + abs(p1_dyn_hw - p_run)) / 2
tvd_grov = (abs(p0_grov_hw - p_stop) + abs(p1_grov_hw - p_run)) / 2

print(f"\nTVD from theoretical:")
print(f"  Dynamic:   {tvd_dyn:.4f}")
print(f"  Groverize: {tvd_grov:.4f}")

if tvd_grov < tvd_dyn:
    print(f"\n  Groverize closer to theory by {tvd_dyn - tvd_grov:.4f} TVD")
else:
    print(f"\n  Dynamic closer to theory by {tvd_grov - tvd_dyn:.4f} TVD")
