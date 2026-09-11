"""IBM: extended scaling N=31,33,35,37,39,41,43,45,47,49.

Uses first 50 qubits from ibm_marrakesh.
Circuit structure: same as ibm_full_scaling_fixed_layout.py
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
    token='sXPqy4J-tYZVmoIYR908WhjT8L2Ebc0mUceEU3ntvcSx',
    instance='auto'
)
backend = service.backend('ibm_marrakesh')
print(f"Backend: {backend.name}")

N_values = [31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
SHOTS = 4000
MAX_QUBITS = 50

# Use calibrated qubits starting from qubit15 (best connected component)
layout = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 69, 70, 71, 72, 73, 74, 75]
print(f"Using qubits: {layout[:5]}...{layout[-3:]}")

# ============================================================
# Build, transpile, and collect results for each N
# ============================================================
results = {}
dyn_transpiled = []
grov_transpiled = []
valid_N = []

for N in N_values:
    n_qubits = N + 1
    if n_qubits > MAX_QUBITS:
        print(f"Skipping N={N}: needs {n_qubits} qubits, max is {MAX_QUBITS}")
        continue

    phys_qubits = layout[:n_qubits]
    print(f"\nBuilding N={N} ({n_qubits} qubits)...")

    # Pass manager with fixed layout for this N
    pm = generate_preset_pass_manager(
        optimization_level=1, backend=backend,
        initial_layout=phys_qubits,
    )

    # --- Dynamic circuit ---
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

    # --- Groverize (pure unitary) circuit ---
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
    print(f"Transpiling N={N}...")
    try:
        dyn_t = pm.run(qc_dyn)
        grov_t = pm.run(qc_grov)

        dyn_transpiled.append(dyn_t)
        grov_transpiled.append(grov_t)
        valid_N.append(N)

        results[str(N)] = {
            'dyn_depth': dyn_t.depth(),
            'grov_depth': grov_t.depth(),
            'dyn_gates': dyn_t.size(),
            'grov_gates': grov_t.size(),
            'phys_qubits': phys_qubits
        }

        print(f"  dyn_depth={dyn_t.depth()}, grov_depth={grov_t.depth()}")

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print(f"\nSuccessfully transpiled {len(valid_N)} circuits")

# ============================================================
# Run on hardware
# ============================================================
if len(valid_N) > 0:
    print("\nRunning dynamic circuits...")
    sampler = Sampler(backend)
    sampler.options.default_shots = SHOTS
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False

    dyn_jobs = sampler.run(dyn_transpiled)
    print(f"Dynamic batch job: {dyn_jobs.job_id()}")

    print("Running groverize circuits...")
    grov_jobs = sampler.run(grov_transpiled)
    print(f"Groverize batch job: {grov_jobs.job_id()}")

    # Get results
    print("\nWaiting for results...")
    dyn_results = dyn_jobs.result()
    grov_results = grov_jobs.result()

    # Process results
    for i, N in enumerate(valid_N):
        n_qubits = N + 1
        target = '1' * n_qubits

        # Dynamic counts
        dyn_counts = dyn_results[i].data.c_out.get_counts()
        p_dyn = dyn_counts.get(target, 0) / SHOTS

        # Groverize counts
        grov_counts = grov_results[i].data.c_out.get_counts()
        p_grov = grov_counts.get(target, 0) / SHOTS

        results[str(N)].update({
            'p_dynamic': p_dyn,
            'p_groverize': p_grov,
            'abs_diff': abs(p_grov - p_dyn),
            'relative': p_grov / p_dyn if p_dyn > 0 else float('inf'),
            'dyn_counts': dict(dyn_counts),
            'grov_counts': dict(grov_counts),
        })

        print(f"N={N}: Dynamic={p_dyn:.3f}, Groverize={p_grov:.3f}, "
              f"Relative={p_grov/p_dyn if p_dyn>0 else float('inf'):.1f}x")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"experiments/ibm_scaling_extended_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {filename}")
