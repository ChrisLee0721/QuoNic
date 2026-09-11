"""IBM: scaling experiment — N conditional operations.

Program: for i in range(N):
             if q0 == 1: flip q[i+1]

Dynamic version: N mid-circuit measurements + N classical feedbacks
Groverize version: N controlled-X gates (CX from q0 to each q[i+1])

As N increases, dynamic accumulates N measurement errors.
Groverize only has deeper circuit (no measurement errors).

Expected: groverize advantage grows with N.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np
import json
from datetime import datetime

# Connect
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token='xJA97lPiGPyWT4wXKb0k8gkBGfmou5lB9hTGEBaHJD2a',
    instance='auto'
)
backend = service.backend('ibm_marrakesh')
print(f"Backend: {backend.name}")

# ============================================================
# Build circuits for N = 1, 2, 3, 4
# ============================================================
N_values = [1, 2, 3, 4]
SHOTS = 4000

results = {}

for N in N_values:
    n_qubits = N + 1  # q0 (control) + q1..qN (targets)
    print(f"\n{'='*60}")
    print(f"N = {N}: {n_qubits} qubits, {N} conditional operations")

    # --- Dynamic version ---
    qr = QuantumRegister(n_qubits, 'q')
    cr_mid = ClassicalRegister(N, 'c_mid')  # N mid-circuit measurements
    cr_out = ClassicalRegister(n_qubits, 'c_out')

    qc_dyn = QuantumCircuit(qr, cr_mid, cr_out)
    qc_dyn.x(0)  # prepare q0 = |1>
    qc_dyn.barrier()

    for i in range(N):
        # Mid-circuit measure q0
        qc_dyn.measure(0, cr_mid[i])
        # if c_mid[i] == 1, flip q[i+1]
        with qc_dyn.if_test((cr_mid, 2**i)):  # bit i set
            qc_dyn.x(i + 1)
        qc_dyn.barrier()

    # Final measurement
    for j in range(n_qubits):
        qc_dyn.measure(j, cr_out[j])

    # --- Groverize version ---
    qr2 = QuantumRegister(n_qubits, 'q')
    cr_out2 = ClassicalRegister(n_qubits, 'c_out')

    qc_grov = QuantumCircuit(qr2, cr_out2)
    qc_grov.x(0)  # prepare q0 = |1>
    qc_grov.barrier()

    for i in range(N):
        qc_grov.cx(0, i + 1)  # controlled-X from q0 to q[i+1]

    qc_grov.barrier()
    for j in range(n_qubits):
        qc_grov.measure(j, cr_out2[j])

    # --- Transpile ---
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

    print(f"  Submitting...")
    job_dyn = sampler.run([qc_dyn_t])
    job_grov = sampler.run([qc_grov_t])

    print(f"  Dynamic job: {job_dyn.job_id}")
    print(f"  Groverize job: {job_grov.job_id}")

    print(f"  Waiting for results...")
    res_dyn = job_dyn.result()
    res_grov = job_grov.result()

    counts_dyn = res_dyn[0].data.c_out.get_counts()
    counts_grov = res_grov[0].data.c_out.get_counts()

    # --- Ideal outcome: all target qubits should be |1> ---
    ideal_bitstring = '1' * n_qubits  # e.g., N=3 → '1111'

    p_dyn = counts_dyn.get(ideal_bitstring, 0) / SHOTS
    p_grov = counts_grov.get(ideal_bitstring, 0) / SHOTS

    tvd_dyn = 1.0 - p_dyn  # TVD from ideal (simplified: ideal is deterministic)
    tvd_grov = 1.0 - p_grov

    results[N] = {
        'dynamic': {'counts': counts_dyn, 'p_ideal': p_dyn, 'tvd': tvd_dyn},
        'groverize': {'counts': counts_grov, 'p_ideal': p_grov, 'tvd': tvd_grov},
        'dynamic_depth': qc_dyn_t.depth(),
        'groverize_depth': qc_grov_t.depth(),
    }

    print(f"\n  Results for N={N}:")
    print(f"    Ideal bitstring: {ideal_bitstring}")
    print(f"    Dynamic   P({ideal_bitstring}) = {p_dyn:.4f}  TVD = {tvd_dyn:.4f}")
    print(f"    Groverize P({ideal_bitstring}) = {p_grov:.4f}  TVD = {tvd_grov:.4f}")
    print(f"    Advantage: {tvd_dyn - tvd_grov:+.4f} TVD")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("SCALING SUMMARY")
print(f"{'='*60}")
print(f"{'N':>3s} {'Dyn TVD':>10s} {'Grov TVD':>10s} {'Advantage':>10s} {'Dyn depth':>10s} {'Grov depth':>10s}")
for N in N_values:
    r = results[N]
    adv = r['dynamic']['tvd'] - r['groverize']['tvd']
    print(f"{N:>3d} {r['dynamic']['tvd']:>10.4f} {r['groverize']['tvd']:>10.4f} {adv:>+10.4f} {r['dynamic_depth']:>10d} {r['groverize_depth']:>10d}")

# Save raw data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"experiments/ibm_scaling_{timestamp}.json"
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nRaw data saved to {save_path}")
