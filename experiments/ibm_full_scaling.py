"""IBM: full scaling N=1..17, batch submission.

Fill in the gaps: N=5,6,7,8,9,10,11,12,13,14,15,16
Plus re-run N=1,2,3,4,17 for consistency (same session, same calibration).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np
import json
from datetime import datetime

service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token='xJA97lPiGPyWT4wXKb0k8gkBGfmou5lB9hTGEBaHJD2a',
    instance='auto'
)
backend = service.backend('ibm_marrakesh')
print(f"Backend: {backend.name}")

N_values = list(range(1, 18))  # 1 to 17
SHOTS = 4000

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

# Build and transpile all circuits
print("Building and transpiling all circuits...")
dyn_circuits = []
grov_circuits = []
metadata = []

for N in N_values:
    n_qubits = N + 1

    # Dynamic
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
    dyn_circuits.append(pm.run(qc_dyn))

    # Groverize
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
    grov_circuits.append(pm.run(qc_grov))

    metadata.append({
        'N': N, 'n_qubits': n_qubits,
        'dyn_depth': dyn_circuits[-1].depth(),
        'dyn_gates': dyn_circuits[-1].size(),
        'grov_depth': grov_circuits[-1].depth(),
        'grov_gates': grov_circuits[-1].size(),
    })
    print(f"  N={N:2d}: dyn depth={metadata[-1]['dyn_depth']:3d}, grov depth={metadata[-1]['grov_depth']:3d}")

# Batch submit all dynamic circuits
print(f"\nSubmitting {len(dyn_circuits)} dynamic circuits...")
dyn_jobs = sampler.run(dyn_circuits)
print(f"Dynamic batch job: {dyn_jobs.job_id()}")

# Batch submit all groverize circuits
print(f"Submitting {len(grov_circuits)} groverize circuits...")
grov_jobs = sampler.run(grov_circuits)
print(f"Groverize batch job: {grov_jobs.job_id()}")

print("\nWaiting for results...")
dyn_results = dyn_jobs.result()
grov_results = grov_jobs.result()

# Extract results
print("\n=== FULL SCALING RESULTS ===")
print(f"{'N':>3s} {'Dyn P':>8s} {'Grov P':>8s} {'Abs Diff':>10s} {'Rel x':>8s} {'Dyn depth':>10s} {'Grov depth':>10s}")

data = {}
for idx, N in enumerate(N_values):
    n_qubits = N + 1
    ideal = '1' * n_qubits

    counts_dyn = dyn_results[idx].data.c_out.get_counts()
    counts_grov = grov_results[idx].data.c_out.get_counts()

    p_dyn = counts_dyn.get(ideal, 0) / SHOTS
    p_grov = counts_grov.get(ideal, 0) / SHOTS
    rel = p_grov / p_dyn if p_dyn > 0 else float('inf')

    data[N] = {
        'p_dynamic': p_dyn, 'p_groverize': p_grov,
        'abs_diff': p_grov - p_dyn, 'relative': rel,
        'dyn_depth': metadata[idx]['dyn_depth'],
        'grov_depth': metadata[idx]['grov_depth'],
        'dyn_counts': counts_dyn, 'grov_counts': counts_grov,
    }

    print(f"{N:>3d} {p_dyn:>8.4f} {p_grov:>8.4f} {(p_grov-p_dyn)*100:>+9.1f}% {rel:>7.2f}x {metadata[idx]['dyn_depth']:>10d} {metadata[idx]['grov_depth']:>10d}")

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"experiments/ibm_full_scaling_{ts}.json"
with open(path, 'w') as f:
    json.dump(data, f, indent=2, default=str)
print(f"\nSaved to {path}")

# Print summary for paper
print("\n=== FOR PAPER ===")
print(f"{'N':>3s} {'Dynamic':>10s} {'Groverize':>10s} {'Advantage':>10s} {'Relative':>10s}")
for N in N_values:
    d = data[N]
    print(f"{N:>3d} {d['p_dynamic']*100:>9.1f}% {d['p_groverize']*100:>9.1f}% {d['abs_diff']*100:>+9.1f}% {d['relative']:>9.2f}x")
