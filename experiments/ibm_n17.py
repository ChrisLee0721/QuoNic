"""IBM: N=17 extreme scaling test.

18 qubits, 17 conditional operations.
Dynamic: 17 mid-circuit measurements
Groverize: 17 CX gates

Expected: both degrade significantly, but groverize should still be better.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token='xJA97lPiGPyWT4wXKb0k8gkBGfmou5lB9hTGEBaHJD2a',
    instance='auto'
)
backend = service.backend('ibm_marrakesh')
print(f"Backend: {backend.name}")

N = 17
n_qubits = N + 1  # 18 qubits
SHOTS = 4000

print(f"N = {N}, {n_qubits} qubits")

# --- Dynamic ---
print("Building dynamic circuit...")
qr = QuantumRegister(n_qubits, 'q')
cr_mid_list = [ClassicalRegister(1, f'c{i}') for i in range(N)]
cr_out = ClassicalRegister(n_qubits, 'c_out')
qc_dyn = QuantumCircuit(qr, *cr_mid_list, cr_out)
qc_dyn.x(0)
qc_dyn.barrier()

for i in range(N):
    qc_dyn.measure(0, cr_mid_list[i][0])
    with qc_dyn.if_test((cr_mid_list[i], 1)):
        qc_dyn.x(i + 1)
    qc_dyn.barrier()

for j in range(n_qubits):
    qc_dyn.measure(j, cr_out[j])

# --- Groverize ---
print("Building groverize circuit...")
qr2 = QuantumRegister(n_qubits, 'q')
cr_out2 = ClassicalRegister(n_qubits, 'c_out')
qc_grov = QuantumCircuit(qr2, cr_out2)
qc_grov.x(0)
qc_grov.barrier()
for i in range(N):
    qc_grov.cx(0, i + 1)
qc_grov.barrier()
for j in range(n_qubits):
    qc_grov.measure(j, cr_out2[j])

# --- Transpile ---
print("Transpiling...")
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
qc_dyn_t = pm.run(qc_dyn)
qc_grov_t = pm.run(qc_grov)

print(f"  Dynamic depth: {qc_dyn_t.depth()}, gates: {qc_dyn_t.size()}")
print(f"  Groverize depth: {qc_grov_t.depth()}, gates: {qc_grov_t.size()}")

# --- Submit ---
sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print("Submitting...")
job_dyn = sampler.run([qc_dyn_t])
job_grov = sampler.run([qc_grov_t])

print(f"  Dynamic job: {job_dyn.job_id()}")
print(f"  Groverize job: {job_grov.job_id()}")
print("Waiting...")

res_dyn = job_dyn.result()
res_grov = job_grov.result()

counts_dyn = res_dyn[0].data.c_out.get_counts()
counts_grov = res_grov[0].data.c_out.get_counts()

ideal = '1' * n_qubits
p_dyn = counts_dyn.get(ideal, 0) / SHOTS
p_grov = counts_grov.get(ideal, 0) / SHOTS

print(f"\n=== N={N} RESULTS ===")
print(f"Ideal bitstring: {ideal}")
print(f"Dynamic   P({ideal}) = {p_dyn:.4f}  ({p_dyn*100:.1f}%)")
print(f"Groverize P({ideal}) = {p_grov:.4f}  ({p_grov*100:.1f}%)")
print(f"Advantage: {(p_grov - p_dyn)*100:+.1f}%")
print(f"Dynamic error rate: {(1-p_dyn)*100:.1f}%")
print(f"Groverize error rate: {(1-p_grov)*100:.1f}%")
print(f"Error rate reduction: {(1-p_dyn)/(1-p_grov)*100 - 100:+.1f}%")

# Top 5 outcomes
print(f"\n--- Dynamic top 5 ---")
for k, v in sorted(counts_dyn.items(), key=lambda x: -x[1])[:5]:
    print(f"  {k}: {v/SHOTS:.4f}")

print(f"\n--- Groverize top 5 ---")
for k, v in sorted(counts_grov.items(), key=lambda x: -x[1])[:5]:
    print(f"  {k}: {v/SHOTS:.4f}")
