"""IBM: extend fixed layout scaling to N=18..23.

Continues from the N=1..17 experiment with the same fixed qubit layout.
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

# Same layout as before, extended
base_layout = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 35]

# Find more connected qubits from 35
coupling = backend.coupling_map
neighbors = {}
for edge in coupling.get_edges():
    a, b = edge
    neighbors.setdefault(a, []).append(b)
    neighbors.setdefault(b, []).append(a)

# Extend from the end of base_layout
layout = list(base_layout)
visited = set(layout)
while len(layout) < 24:  # need 24 for N=23
    current = layout[-1]
    found = False
    for nb in neighbors.get(current, []):
        if nb not in visited:
            layout.append(nb)
            visited.add(nb)
            found = True
            break
    if not found:
        # try extending from earlier
        for i in range(len(layout) - 2, -1, -1):
            for nb in neighbors.get(layout[i], []):
                if nb not in visited:
                    layout = layout[:i+1] + [nb] + layout[i+1:]
                    visited.add(nb)
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"Could only extend to {len(layout)} qubits")
            break

print(f"Layout: {layout}")
print(f"Chain length: {len(layout)}")

N_values = list(range(18, 24))  # 18 to 23
SHOTS = 4000

print("\nBuilding and transpiling...")
dyn_circuits = []
grov_circuits = []
metadata = []

for N in N_values:
    n_qubits = N + 1
    phys_qubits = layout[:n_qubits]

    pm = generate_preset_pass_manager(
        optimization_level=1, backend=backend,
        initial_layout=phys_qubits,
    )

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

    dyn_t = pm.run(qc_dyn)
    grov_t = pm.run(qc_grov)

    dyn_circuits.append(dyn_t)
    grov_circuits.append(grov_t)

    metadata.append({
        'N': N, 'n_qubits': n_qubits, 'phys_qubits': phys_qubits,
        'dyn_depth': dyn_t.depth(),
        'dyn_gates': dyn_t.size(),
        'grov_depth': grov_t.depth(),
        'grov_gates': grov_t.size(),
    })
    print(f"  N={N:2d}: phys={phys_qubits}, dyn depth={metadata[-1]['dyn_depth']:3d}, grov depth={metadata[-1]['grov_depth']:3d}")

sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print(f"\nSubmitting {len(dyn_circuits)} dynamic circuits...")
dyn_jobs = sampler.run(dyn_circuits)
print(f"Dynamic job: {dyn_jobs.job_id()}")

print(f"Submitting {len(grov_circuits)} groverize circuits...")
grov_jobs = sampler.run(grov_circuits)
print(f"Groverize job: {grov_jobs.job_id()}")

print("\nWaiting for results...")
dyn_results = dyn_jobs.result()
grov_results = grov_jobs.result()

# Extract
print("\n=== RESULTS (N=18..23) ===")
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
        'phys_qubits': metadata[idx]['phys_qubits'],
        'dyn_counts': counts_dyn, 'grov_counts': counts_grov,
    }
    print(f"{N:>3d} {p_dyn:>8.4f} {p_grov:>8.4f} {(p_grov-p_dyn)*100:>+9.1f}% {rel:>7.2f}x {metadata[idx]['dyn_depth']:>10d} {metadata[idx]['grov_depth']:>10d}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"experiments/ibm_scaling_extend_18_23_{ts}.json"
with open(path, 'w') as f:
    json.dump(data, f, indent=2, default=str)
print(f"\nSaved to {path}")
