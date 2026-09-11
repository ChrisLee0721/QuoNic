"""IBM: optimized groverize — q0 in the middle of the chain.

Naive layout:  q0--q1--q2--...--qN     max distance = N
Optimized:     ...--q1--q0--q1'--...   max distance = N/2

Also uses optimization_level=3 for aggressive routing.
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

# ============================================================
# Build connected chain (same as before)
# ============================================================
coupling = backend.coupling_map
neighbors = {}
for edge in coupling.get_edges():
    a, b = edge
    neighbors.setdefault(a, []).append(b)
    neighbors.setdefault(b, []).append(a)

def find_chain(coupling, length, start=0):
    neighbors = {}
    for edge in coupling.get_edges():
        a, b = edge
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    chain = [start]
    visited = {start}
    while len(chain) < length:
        current = chain[-1]
        found = False
        for nb in neighbors.get(current, []):
            if nb not in visited:
                chain.append(nb)
                visited.add(nb)
                found = True
                break
        if not found:
            for i in range(len(chain) - 2, -1, -1):
                for nb in neighbors.get(chain[i], []):
                    if nb not in visited:
                        chain = chain[:i+1] + [nb] + chain[i+1:]
                        visited.add(nb)
                        found = True
                        break
                if found:
                    break
            if not found:
                return chain
    return chain[:length]

chain = find_chain(coupling, 51, start=0)
print(f"Chain length: {len(chain)}")

# ============================================================
# Optimized layout: q0 in the middle
# ============================================================
def make_centered_layout(chain, n_qubits):
    """Put q0 at the center of the chain, q1..qN radiating outward."""
    mid = len(chain) // 2  # center of the full chain
    # q0 goes at position mid
    # q1, q2, ... go left and right alternating
    layout = [0] * n_qubits  # logical qubit -> physical qubit
    layout[0] = chain[mid]   # q0 at center

    left = mid - 1
    right = mid + 1
    for i in range(1, n_qubits):
        if i % 2 == 1 and left >= 0:
            layout[i] = chain[left]
            left -= 1
        elif right < len(chain):
            layout[i] = chain[right]
            right += 1
        elif left >= 0:
            layout[i] = chain[left]
            left -= 1
        else:
            # fallback: use any remaining
            for c in chain:
                if c not in layout:
                    layout[i] = c
                    break
    return layout

N_values = list(range(1, 24))  # 1 to 23
SHOTS = 4000

print("\nBuilding circuits...")
grov_circuits = []
metadata = []

for N in N_values:
    n_qubits = N + 1
    phys_qubits = make_centered_layout(chain, n_qubits)

    # Groverize circuit
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

    # Transpile with optimization_level=3
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend,
        initial_layout=phys_qubits,
    )
    qc_t = pm.run(qc)
    grov_circuits.append(qc_t)

    metadata.append({
        'N': N, 'n_qubits': n_qubits,
        'phys_qubits': phys_qubits,
        'depth': qc_t.depth(),
        'gates': qc_t.size(),
        'cx_count': qc_t.count_ops().get('cx', 0),
    })
    print(f"  N={N:2d}: center={phys_qubits[0]}, depth={metadata[-1]['depth']:3d}, cx={metadata[-1]['cx_count']:4d}")

# ============================================================
# Also build naive layout (q0 at end) for comparison
# ============================================================
print("\nBuilding naive circuits for comparison...")
naive_circuits = []
naive_metadata = []

for N in N_values:
    n_qubits = N + 1
    phys_qubits = chain[:n_qubits]  # q0 at one end

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
        initial_layout=phys_qubits,
    )
    qc_t = pm.run(qc)
    naive_circuits.append(qc_t)

    naive_metadata.append({
        'N': N, 'depth': qc_t.depth(),
        'gates': qc_t.size(),
        'cx_count': qc_t.count_ops().get('cx', 0),
    })

# Print comparison
print("\n=== DEPTH COMPARISON ===")
print(f"{'N':>3s} {'Naive':>8s} {'Optimized':>10s} {'Ratio':>8s}")
for i, N in enumerate(N_values):
    naive_d = naive_metadata[i]['depth']
    opt_d = metadata[i]['depth']
    ratio = opt_d / naive_d if naive_d > 0 else 0
    print(f"{N:>3d} {naive_d:>8d} {opt_d:>10d} {ratio:>7.2f}x")

# ============================================================
# Submit optimized circuits
# ============================================================
sampler = Sampler(backend)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print(f"\nSubmitting {len(grov_circuits)} optimized groverize circuits...")
job = sampler.run(grov_circuits)
print(f"Job: {job.job_id()}")
print("Waiting...")
result = job.result()

# ============================================================
# Results
# ============================================================
print("\n=== OPTIMIZED GROVERIZE RESULTS ===")
print(f"{'N':>3s} {'P':>8s} {'Naive P':>10s} {'Opt depth':>10s} {'Naive depth':>12s}")

# Load naive results from previous experiment
try:
    with open('experiments/ibm_full_scaling_fixed_20260904_234348.json') as f:
        naive_data = json.load(f)
    has_naive = True
except:
    has_naive = False
    print("  (no naive data for comparison)")

data = {}
for idx, N in enumerate(N_values):
    n_qubits = N + 1
    ideal = '1' * n_qubits
    counts = result[idx].data.c_out.get_counts()
    p = counts.get(ideal, 0) / SHOTS

    naive_p = naive_data[str(N)]['p_groverize'] if has_naive and str(N) in naive_data else None
    naive_d = naive_metadata[idx]['depth']
    opt_d = metadata[idx]['depth']

    data[N] = {
        'p_optimized': p,
        'p_naive': naive_p,
        'opt_depth': opt_d,
        'naive_depth': naive_d,
        'phys_qubits': metadata[idx]['phys_qubits'],
        'counts': counts,
    }

    naive_str = f"{naive_p:.4f}" if naive_p is not None else "N/A"
    print(f"{N:>3d} {p:>8.4f} {naive_str:>10s} {opt_d:>10d} {naive_d:>12d}")

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
path = f"experiments/ibm_optimized_groverize_{ts}.json"
with open(path, 'w') as f:
    json.dump(data, f, indent=2, default=str)
print(f"\nSaved to {path}")
