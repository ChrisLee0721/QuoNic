"""IBM: test if nested if_test works with N=2."""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(channel='ibm_quantum_platform', token='xJA97lPiGPyWT4wXKb0k8gkBGfmou5lB9hTGEBaHJD2a', instance='auto')
backend = service.backend('ibm_marrakesh')

# N=2 nested if_test
qr = QuantumRegister(1, 'q')
c0 = ClassicalRegister(1, 'c0')
c1 = ClassicalRegister(1, 'c1')
cr_out = ClassicalRegister(1, 'out')

qc = QuantumCircuit(qr, c0, c1, cr_out)
qc.x(0)
qc.barrier()

# Step 1: H + measure
qc.h(0)
qc.measure(0, c0[0])

# if c0==1: Step 2
with qc.if_test((c0, 1)):
    qc.h(0)
    qc.measure(0, c1[0])

qc.barrier()
qc.measure(0, cr_out[0])

print(f"Nested N=2 depth: {qc.depth()}, gates: {qc.size()}")

pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
qc_t = pm.run(qc)
print(f"Transpiled depth: {qc_t.depth()}, gates: {qc_t.size()}")

sampler = Sampler(backend)
sampler.options.default_shots = 2000
sampler.options.dynamical_decoupling.enable = False
sampler.options.twirling.enable_gates = False
sampler.options.twirling.enable_measure = False

print("Submitting...")
job = sampler.run([qc_t])
print(f"Job: {job.job_id()}")

try:
    result = job.result()
    counts = result[0].data.out.get_counts()
    print(f"Result: {counts}")
    p0 = counts.get('0', 0) / 2000
    p1 = counts.get('1', 0) / 2000
    print(f"P(0)={p0:.4f}, P(1)={p1:.4f}")
    print(f"Theoretical: P(0)=0.75, P(1)=0.25")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    print("IBM does not support nested if_test")
