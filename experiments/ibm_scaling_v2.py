"""IBM: scaling experiment v2 — FIXED dynamic circuit.

Bug fix: use separate classical registers for each mid-circuit measurement.
Previous version used single cr_mid register, causing condition check failure
when multiple bits were set (cr_mid == 2**i checks whole register, not bit i).
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

N_values = [1, 2, 3, 4]
SHOTS = 4000
results = {}

for N in N_values:
    n_qubits = N + 1
    print(f"\n{'='*60}")
    print(f"N = {N}: {n_qubits} qubits")

    # --- Dynamic version (FIXED: separate classical registers) ---
    qr = QuantumRegister(n_qubits, 'q')
    # One classical register per mid-circuit measurement
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

    # --- Groverize version (same as before) ---
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

    res_dyn = job_dyn.result()
    res_grov = job_grov.result()

    counts_dyn = res_dyn[0].data.c_out.get_counts()
    counts_grov = res_grov[0].data.c_out.get_counts()

    ideal_bitstring = '1' * n_qubits
    p_dyn = counts_dyn.get(ideal_bitstring, 0) / SHOTS
    p_grov = counts_grov.get(ideal_bitstring, 0) / SHOTS

    # Bootstrap CI
    np.random.seed(42)
    diffs = []
    for _ in range(5000):
        bd = np.random.choice(list(counts_dyn.keys()), SHOTS, p=[v/SHOTS for v in counts_dyn.values()])
        bg = np.random.choice(list(counts_grov.keys()), SHOTS, p=[v/SHOTS for v in counts_grov.values()])
        pd = np.mean(bd == ideal_bitstring)
        pg = np.mean(bg == ideal_bitstring)
        diffs.append(pd - pg)
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))

    results[N] = {
        'p_dynamic': p_dyn, 'p_groverize': p_grov,
        'dyn_depth': qc_dyn_t.depth(), 'grov_depth': qc_grov_t.depth(),
        'dyn_counts': counts_dyn, 'grov_counts': counts_grov,
        'diff_ci': ci,
    }

    print(f"  Dynamic   P({ideal_bitstring}) = {p_dyn:.4f}")
    print(f"  Groverize P({ideal_bitstring}) = {p_grov:.4f}")
    print(f"  Difference: {p_grov - p_dyn:+.4f}  95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")

# Summary
print(f"\n{'='*60}")
print("SCALING SUMMARY (FIXED)")
print(f"{'='*60}")
print(f"{'N':>3s} {'Dyn P':>8s} {'Grov P':>8s} {'Advantage':>10s} {'Dyn depth':>10s} {'Grov depth':>10s} {'95% CI':>20s}")
for N in N_values:
    r = results[N]
    adv = r['p_groverize'] - r['p_dynamic']
    ci = r['diff_ci']
    print(f"{N:>3d} {r['p_dynamic']:>8.4f} {r['p_groverize']:>8.4f} {adv:>+10.4f} {r['dyn_depth']:>10d} {r['grov_depth']:>10d} [{ci[0]:+.4f}, {ci[1]:+.4f}]")

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"experiments/ibm_scaling_v2_{ts}.json", 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to experiments/ibm_scaling_v2_{ts}.json")
