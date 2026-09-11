"""IBM: groverize with q0 centered — same physical qubits as naive.

Uses the same chain [0,1,2,...,15,19,35] but reorders the logical→physical
mapping so q0 is in the middle instead of at one end.

Naive:   q0→0  (end)     max distance = N
Centered: q0→9  (center)  max distance = N/2
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

# Same physical chain as before
chain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 35]
print(f"Chain: {chain} ({len(chain)} qubits)")

def make_centered_layout(chain, n_qubits):
    """Put q0 at center, radiate q1..qN outward."""
    mid = n_qubits // 2  # center index in the sub-chain
    sub = chain[:n_qubits]
    # q0 at position mid, q1..qN alternating left/right
    layout = [0] * n_qubits
    layout[0] = sub[mid]
    left_idx = mid - 1
    right_idx = mid + 1
    for i in range(1, n_qubits):
        if i % 2 == 1 and left_idx >= 0:
            layout[i] = sub[left_idx]
            left_idx -= 1
        elif right_idx < len(sub):
            layout[i] = sub[right_idx]
            right_idx += 1
        elif left_idx >= 0:
            layout[i] = sub[left_idx]
            left_idx -= 1
    return layout

N_values = list(range(1, 18))  # 1 to 17
SHOTS = 4000

print("\nBuilding centered groverize circuits...")
centered_circuits = []
centered_meta = []

for N in N_values:
    n_qubits = N + 1
    phys = make_centered_layout(chain, n_qubits)

    qr = QuantumRegister(n_qubits, 'q')
    cr_out = ClassicalRegister(n_qubits, 'c_out')
    qc = QuantumCircuit(qr, cr_out)
    qc.x(0)
    qc.barrier()
    for i in range(N):
        qc.cx(0, i + 1)
    qc.barrier()
    for j in range(n_qubits):
        qc.measure(j, cr_out[j])

    pm = generate_preset_pass_manager(
        optimization_level=1, backend=backend,
        initial_layout=phys,
    )
    qc_t = pm.run(qc)
    centered_circuits.append(qc_t)
    centered_meta.append({
        'N': N, 'layout': phys,
        'depth': qc_t.depth(),
        'gates': qc_t.size(),
    })
    # Show distance from q0 to farthest qubit
    q0_pos = phys.index(phys[0])  # position of q0 in layout
    max_dist = max(abs(i - q0_pos) for i in range(n_qubits))
    print(f"  N={N:2d}: layout={phys[:6]}..., depth={qc_t.depth():3d}, max_dist={max_dist}")

# Build naive for comparison (same chain, q0 at end)
print("\nBuilding naive groverize circuits...")
naive_circuits = []
naive_meta = []

for N in N_values:
    n_qubits = N + 1
    phys = chain[:n_qubits]  # q0 at one end

    qr = QuantumRegister(n_qubits, 'q')
    cr_out = ClassicalRegister(n_qubits, 'c_out')
    qc = QuantumCircuit(qr, cr_out)
    qc.x(0)
    qc.barrier()
    for i in range(N):
        qc.cx(0, i + 1)
    qc.barrier()
    for j in range(n_qubits):
        qc.measure(j, cr_out[j])

    pm = generate_preset_pass_manager(
        optimization_level=1, backend=backend,
        initial_layout=phys,
    )
    qc_t = pm.run(qc)
    naive_circuits.append(qc_t)
    naive_meta.append({'N': N, 'depth': qc_t.depth(), 'gates': qc_t.size()})

# Depth comparison
print("\n=== DEPTH COMPARISON ===")
print(f"{'N':>3s} {'Naive':>8s} {'Centered':>10s} {'Saved':>8s}")
for i, N in enumerate(N_values):
    nd = naive_meta[i]['depth']
    cd = centered_meta[i]['depth']
    print(f"{N:>3d} {nd:>8d} {cd:>10d} {nd-cd:>+8d}")

# Submit both
sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print(f"\nSubmitting naive...")
naive_job = sampler.run(naive_circuits)
print(f"Job: {naive_job.job_id()}")

print(f"Submitting centered...")
centered_job = sampler.run(centered_circuits)
print(f"Job: {centered_job.job_id()}")

print("Waiting...")
naive_result = naive_job.result()
centered_result = centered_job.result()

# Results
print("\n=== RESULTS ===")
print(f"{'N':>3s} {'Naive P':>10s} {'Centered P':>12s} {'Diff':>8s} {'Naive d':>9s} {'Center d':>10s}")

data = {}
for idx, N in enumerate(N_values):
    n_qubits = N + 1
    ideal = '1' * n_qubits

    counts_n = naive_result[idx].data.c_out.get_counts()
    counts_c = centered_result[idx].data.c_out.get_counts()
    p_n = counts_n.get(ideal, 0) / SHOTS
    p_c = counts_c.get(ideal, 0) / SHOTS

    data[N] = {
        'naive_p': p_n, 'centered_p': p_c,
        'naive_depth': naive_meta[idx]['depth'],
        'centered_depth': centered_meta[idx]['depth'],
        'centered_layout': centered_meta[idx]['layout'],
    }
    print(f"{N:>3d} {p_n:>10.4f} {p_c:>12.4f} {(p_c-p_n)*100:>+7.1f}% {naive_meta[idx]['depth']:>9d} {centered_meta[idx]['depth']:>10d}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"experiments/ibm_groverize_centered_{ts}.json"
with open(path, 'w') as f:
    json.dump(data, f, indent=2, default=str)
print(f"\nSaved to {path}")
