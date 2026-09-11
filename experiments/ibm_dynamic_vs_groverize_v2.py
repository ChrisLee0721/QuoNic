"""IBM Quantum: dynamic vs groverize — controlled experiment.

Controls:
1. Same backend (ibm_marrakesh)
2. Dynamical Decoupling DISABLED
3. Twirling DISABLED
4. Batch submission (minimize calibration drift)
5. 8000 shots for statistical significance
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import DynamicalDecouplingOptions, TwirlingOptions
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
# Build circuits
# ============================================================

# Version A: Dynamic
qr = QuantumRegister(2, 'q')
cr_mid = ClassicalRegister(1, 'c_mid')
cr_out = ClassicalRegister(2, 'c_out')
qc_dynamic = QuantumCircuit(qr, cr_mid, cr_out)
qc_dynamic.x(0)
qc_dynamic.barrier()
qc_dynamic.measure(0, cr_mid[0])
with qc_dynamic.if_test((cr_mid, 1)):
    qc_dynamic.x(1)
qc_dynamic.barrier()
qc_dynamic.measure(0, cr_out[0])
qc_dynamic.measure(1, cr_out[1])

# Version B: Groverize
qr2 = QuantumRegister(2, 'q')
cr_out2 = ClassicalRegister(2, 'c_out')
qc_groverize = QuantumCircuit(qr2, cr_out2)
qc_groverize.x(0)
qc_groverize.barrier()
qc_groverize.cx(0, 1)
qc_groverize.barrier()
qc_groverize.measure(0, cr_out2[0])
qc_groverize.measure(1, cr_out2[1])

# ============================================================
# Transpile
# ============================================================
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
qc_dynamic_t = pm.run(qc_dynamic)
qc_groverize_t = pm.run(qc_groverize)

print(f"Dynamic depth: {qc_dynamic_t.depth()}, gates: {qc_dynamic_t.size()}")
print(f"Groverize depth: {qc_groverize_t.depth()}, gates: {qc_groverize_t.size()}")

# ============================================================
# Configure Sampler with all mitigations DISABLED
# ============================================================
SHOTS = 8000

sampler = Sampler(backend)
sampler.options.default_shots = SHOTS

# Disable Dynamical Decoupling
sampler.options.dynamical_decoupling.enable = False

# Disable Twirling
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print(f"\n=== Configuration ===")
print(f"Shots: {SHOTS}")
print(f"Dynamical Decoupling: DISABLED")
print(f"Twirling: DISABLED")
print(f"Backend: {backend.name}")

# ============================================================
# Batch submit (minimize calibration drift)
# ============================================================
print(f"\n=== Batch submitting to {backend.name} ===")

# Submit both in one batch to minimize time between executions
job_dynamic = sampler.run([qc_dynamic_t])
print(f"Dynamic job ID: {job_dynamic.job_id}")

job_groverize = sampler.run([qc_groverize_t])
print(f"Groverize job ID: {job_groverize.job_id}")

# Wait for results
print("\nWaiting for results...")
result_dynamic = job_dynamic.result()
result_groverize = job_groverize.result()

# ============================================================
# Extract counts
# ============================================================
print("\n=== Results ===")

dynamic_counts = result_dynamic[0].data.c_out.get_counts()
groverize_counts = result_groverize[0].data.c_out.get_counts()

print(f"Dynamic counts: {dynamic_counts}")
print(f"Groverize counts: {groverize_counts}")

# ============================================================
# Statistical analysis
# ============================================================
print("\n=== Statistical Analysis ===")

def tvd_from_ideal(counts, shots):
    ideal = {'11': 1.0}
    tvd = sum(abs(counts.get(k, 0)/shots - ideal.get(k, 0)) for k in set(list(counts.keys()) + list(ideal.keys())))
    return tvd / 2.0

def cross_tvd(a, b, shots):
    keys = set(list(a.keys()) + list(b.keys()))
    return sum(abs(a.get(k, 0)/shots - b.get(k, 0)/shots) for k in keys) / 2.0

def wilson_ci(p, n, z=1.96):
    """95% confidence interval for a proportion."""
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return center - margin, center + margin

p_dyn = dynamic_counts.get('11', 0) / SHOTS
p_grov = groverize_counts.get('11', 0) / SHOTS

ci_dyn = wilson_ci(p_dyn, SHOTS)
ci_grov = wilson_ci(p_grov, SHOTS)

tvd_dyn = tvd_from_ideal(dynamic_counts, SHOTS)
tvd_grov = tvd_from_ideal(groverize_counts, SHOTS)
tvd_cross = cross_tvd(dynamic_counts, groverize_counts, SHOTS)

# Bootstrap confidence interval for TVD difference
np.random.seed(42)
N_BOOT = 10000
tvd_diffs = []
for _ in range(N_BOOT):
    # Resample dynamic
    choices_dyn = np.random.choice(list(dynamic_counts.keys()), size=SHOTS, p=[v/SHOTS for v in dynamic_counts.values()])
    boot_dyn = {}
    for c in choices_dyn:
        boot_dyn[c] = boot_dyn.get(c, 0) + 1
    # Resample groverize
    choices_grov = np.random.choice(list(groverize_counts.keys()), size=SHOTS, p=[v/SHOTS for v in groverize_counts.values()])
    boot_grov = {}
    for c in choices_grov:
        boot_grov[c] = boot_grov.get(c, 0) + 1
    tvd_diffs.append(tvd_from_ideal(boot_dyn, SHOTS) - tvd_from_ideal(boot_grov, SHOTS))

tvd_diff_mean = np.mean(tvd_diffs)
tvd_diff_ci = (np.percentile(tvd_diffs, 2.5), np.percentile(tvd_diffs, 97.5))

print(f"Dynamic  P(11) = {p_dyn:.4f}  95% CI: [{ci_dyn[0]:.4f}, {ci_dyn[1]:.4f}]")
print(f"Groverize P(11) = {p_grov:.4f}  95% CI: [{ci_grov[0]:.4f}, {ci_grov[1]:.4f}]")
print(f"\nDynamic  TVD = {tvd_dyn:.4f}")
print(f"Groverize TVD = {tvd_grov:.4f}")
print(f"Cross-TVD = {tvd_cross:.4f}")
print(f"\nTVD difference (dynamic - groverize):")
print(f"  Mean: {tvd_diff_mean:.4f}")
print(f"  95% CI: [{tvd_diff_ci[0]:.4f}, {tvd_diff_ci[1]:.4f}]")

if tvd_diff_ci[0] > 0:
    print(f"\n★ Groverize significantly better (p < 0.05)")
elif tvd_diff_ci[1] < 0:
    print(f"\n★ Dynamic significantly better (p < 0.05)")
else:
    print(f"\n  No significant difference (p >= 0.05)")

# Full distribution comparison
print(f"\n=== Full Distribution ===")
print(f"{'State':>6s} {'Dynamic':>10s} {'Groverize':>10s} {'Diff':>8s}")
for k in sorted(set(list(dynamic_counts.keys()) + list(groverize_counts.keys()))):
    d = dynamic_counts.get(k, 0) / SHOTS
    g = groverize_counts.get(k, 0) / SHOTS
    print(f"{k:>6s} {d:>10.4f} {g:>10.4f} {d-g:>+8.4f}")
