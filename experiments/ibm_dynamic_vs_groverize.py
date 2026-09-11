import sys
sys.stdout.reconfigure(encoding='utf-8')

"""IBM Quantum: dynamic circuit vs groverize() comparison.

Same program: "if q0==1, flip q1"
- Version A: Dynamic (mid-circuit measurement + classical feedback)
- Version B: Groverize (controlled unitary, no mid-circuit measurement)

Compare TVD on the same IBM hardware.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import numpy as np

# Connect
service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token='xJA97lPiGPyWT4wXKb0k8gkBGfmou5lB9hTGEBaHJD2a',
    instance='auto'
)

backend = service.backend('ibm_marrakesh')
print(f"Backend: {backend.name}")
print(f"Pending jobs: {backend.status().pending_jobs}")

# ============================================================
# Version A: Dynamic circuit (mid-circuit measurement + if)
# ============================================================
print("\n=== Building Version A: Dynamic circuit ===")

qr = QuantumRegister(2, 'q')
cr_mid = ClassicalRegister(1, 'c_mid')
cr_out = ClassicalRegister(2, 'c_out')

qc_dynamic = QuantumCircuit(qr, cr_mid, cr_out)

# Prepare q0 in |1⟩
qc_dynamic.x(0)
qc_dynamic.barrier()

# Mid-circuit measurement of q0
qc_dynamic.measure(0, cr_mid[0])

# Classical feedback: if c_mid == 1, flip q1
with qc_dynamic.if_test((cr_mid, 1)):
    qc_dynamic.x(1)

# Final measurement
qc_dynamic.barrier()
qc_dynamic.measure(0, cr_out[0])
qc_dynamic.measure(1, cr_out[1])

print("Dynamic circuit built OK")

# ============================================================
# Version B: Groverize (controlled unitary, no mid-circuit)
# ============================================================
print("\n=== Building Version B: Groverize (controlled unitary) ===")

qr2 = QuantumRegister(2, 'q')
cr_out2 = ClassicalRegister(2, 'c_out')

qc_groverize = QuantumCircuit(qr2, cr_out2)

# Prepare q0 in |1⟩
qc_groverize.x(0)
qc_groverize.barrier()

# Instead of measure+feedback, use controlled-X (CX from q0 to q1)
qc_groverize.cx(0, 1)

# Final measurement
qc_groverize.barrier()
qc_groverize.measure(0, cr_out2[0])
qc_groverize.measure(1, cr_out2[1])

print("Groverize circuit built OK")

# ============================================================
# Transpile for hardware
# ============================================================
print("\n=== Transpiling ===")
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

qc_dynamic_t = pm.run(qc_dynamic)
qc_groverize_t = pm.run(qc_groverize)

print(f"Dynamic depth: {qc_dynamic_t.depth()}, gates: {qc_dynamic_t.size()}")
print(f"Groverize depth: {qc_groverize_t.depth()}, gates: {qc_groverize_t.size()}")

# ============================================================
# Submit both to IBM hardware
# ============================================================
SHOTS = 4000

print(f"\n=== Submitting to {backend.name} ({SHOTS} shots each) ===")

sampler = Sampler(backend)
sampler.options.default_shots = SHOTS

# Submit both
print("Submitting dynamic circuit...")
job_dynamic = sampler.run([qc_dynamic_t])
print(f"Dynamic job ID: {job_dynamic.job_id}")

print("Submitting groverize circuit...")
job_groverize = sampler.run([qc_groverize_t])
print(f"Groverize job ID: {job_groverize.job_id}")

# Wait for results
print("\nWaiting for results...")
result_dynamic = job_dynamic.result()
result_groverize = job_groverize.result()

# ============================================================
# Extract and compare
# ============================================================
print("\n=== Results ===")

# Dynamic result
pub_dynamic = result_dynamic[0]
print(f"Dynamic data attrs: {[a for a in dir(pub_dynamic.data) if not a.startswith('_')]}")

# Groverize result
pub_groverize = result_groverize[0]
print(f"Groverize data attrs: {[a for a in dir(pub_groverize.data) if not a.startswith('_')]}")

# Extract counts
# For dynamic: c_out has the final measurement results
# For groverize: c_out has the final measurement results
try:
    dynamic_counts = pub_dynamic.data.c_out.get_counts()
    print(f"Dynamic counts: {dynamic_counts}")
except Exception as e:
    print(f"Dynamic extraction error: {e}")
    # Try alternative
    try:
        dynamic_counts = pub_dynamic.data.meas.get_counts()
        print(f"Dynamic counts (meas): {dynamic_counts}")
    except:
        print(f"Dynamic data: {pub_dynamic.data}")

try:
    groverize_counts = pub_groverize.data.c_out.get_counts()
    print(f"Groverize counts: {groverize_counts}")
except Exception as e:
    print(f"Groverize extraction error: {e}")
    try:
        groverize_counts = pub_groverize.data.meas.get_counts()
        print(f"Groverize counts (meas): {groverize_counts}")
    except:
        print(f"Groverize data: {pub_groverize.data}")

# ============================================================
# Compute TVD
# ============================================================
print("\n=== TVD Comparison ===")

def compute_tvd_from_ideal(counts, shots):
    """Compute TVD from ideal distribution: q0=1, q1=1."""
    ideal = {'11': 1.0}
    tvd = 0.0
    for bitstring in set(list(counts.keys()) + list(ideal.keys())):
        observed = counts.get(bitstring, 0) / shots
        expected = ideal.get(bitstring, 0.0)
        tvd += abs(observed - expected)
    return tvd / 2.0

def compute_cross_tvd(counts_a, counts_b, shots):
    """Compute TVD between two distributions."""
    all_keys = set(list(counts_a.keys()) + list(counts_b.keys()))
    tvd = 0.0
    for k in all_keys:
        p_a = counts_a.get(k, 0) / shots
        p_b = counts_b.get(k, 0) / shots
        tvd += abs(p_a - p_b)
    return tvd / 2.0

print(f"Expected outcome: q0=1, q1=1 (bitstring '11')")
print(f"Dynamic P(11): {dynamic_counts.get('11', 0) / SHOTS:.4f}")
print(f"Groverize P(11): {groverize_counts.get('11', 0) / SHOTS:.4f}")
print(f"\nDynamic TVD from ideal: {compute_tvd_from_ideal(dynamic_counts, SHOTS):.4f}")
print(f"Groverize TVD from ideal: {compute_tvd_from_ideal(groverize_counts, SHOTS):.4f}")
print(f"Cross-TVD (dynamic vs groverize): {compute_cross_tvd(dynamic_counts, groverize_counts, SHOTS):.4f}")

# Winner
tvd_dynamic = compute_tvd_from_ideal(dynamic_counts, SHOTS)
tvd_groverize = compute_tvd_from_ideal(groverize_counts, SHOTS)
if tvd_groverize < tvd_dynamic:
    print(f"\n★ Groverize wins by {tvd_dynamic - tvd_groverize:.4f} TVD")
elif tvd_dynamic < tvd_groverize:
    print(f"\n★ Dynamic wins by {tvd_groverize - tvd_dynamic:.4f} TVD")
else:
    print(f"\n★ Tie")
