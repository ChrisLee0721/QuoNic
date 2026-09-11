"""
Tuna-17: Core physical parameter measurement.
Measures the parameters needed for the decision boundary:
  T = F_meas * exp(-T_fb/T2)
  F_eff = F_gate
"""

import numpy as np
import json
import time
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_quantuminspire.qi_provider import QIProvider

OUTPUT_FILE = "tuna17_core_params.json"

def run_circuit(backend, qc, shots=4000, retries=3):
    for attempt in range(retries):
        try:
            qc_t = transpile(qc, backend, optimization_level=0, initial_layout=[0])
            job = backend.run(qc_t, shots=shots)
            return job.result().get_counts()
        except Exception as e:
            if '429' in str(e):
                time.sleep(30 * (attempt + 1))
            else:
                return None
    return None

def main():
    results = {'timestamp': datetime.now().isoformat(), 'backend': 'Tuna-17'}
    print(f"=== Tuna-17 Core Parameters ===\n")

    provider = QIProvider()
    backend = provider.get_backend('Tuna-17')
    print(f"Backend: {backend.name}\n")

    # === 1. F_meas (Readout Fidelity) ===
    print("1. Measuring F_meas (readout fidelity)...")

    # |0>
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    counts = run_circuit(backend, qc)
    p0_given_0 = counts.get('0', 0) / sum(counts.values()) if counts else None
    print(f"   P(0|0) = {p0_given_0}")

    time.sleep(5)

    # |1>
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.measure(0, 0)
    counts = run_circuit(backend, qc)
    p1_given_1 = counts.get('1', 0) / sum(counts.values()) if counts else None
    print(f"   P(1|1) = {p1_given_1}")

    if p0_given_0 and p1_given_1:
        F_meas = (p0_given_0 + p1_given_1) / 2
        results['F_meas'] = F_meas
        print(f"   F_meas = {F_meas:.6f}\n")

    time.sleep(10)

    # === 2. T1 (Relaxation Time) ===
    print("2. Measuring T1 (relaxation)...")

    delays = [0, 5, 10, 20, 40, 80, 160, 320]
    t1_data = {}

    for d in delays:
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        if d > 0:
            qc.delay(d, 0, unit='us')
        qc.measure(0, 0)

        counts = run_circuit(backend, qc, shots=2000)
        if counts:
            p1 = counts.get('1', 0) / sum(counts.values())
            t1_data[d] = p1
            print(f"   {d}μs: P(1) = {p1:.4f}")
        time.sleep(3)

    results['T1_data'] = t1_data

    # Fit T1
    if len(t1_data) >= 4:
        t_vals = list(t1_data.keys())[1:]  # Skip 0
        p_vals = [t1_data[t] for t in t_vals]
        try:
            # Fit P(1) = A * exp(-t/T1)
            log_p = np.log([p for p in p_vals if p > 0])
            t_arr = np.array([t for t, p in zip(t_vals, p_vals) if p > 0])
            slope, intercept = np.polyfit(t_arr, log_p, 1)
            T1 = -1 / slope if slope < 0 else None
            results['T1_fit'] = T1
            print(f"   T1 ≈ {T1:.1f} μs\n" if T1 else "   T1 fit failed\n")
        except:
            print("   T1 fit failed\n")

    time.sleep(10)

    # === 3. T2 (Dephasing Time, Ramsey) ===
    print("3. Measuring T2 (dephasing)...")

    delays = [0, 2, 5, 10, 20, 40, 80]
    t2_data = {}

    for d in delays:
        qc = QuantumCircuit(1, 1)
        qc.h(0)  # Superposition
        if d > 0:
            qc.delay(d, 0, unit='us')
        qc.h(0)  # Interference
        qc.measure(0, 0)

        counts = run_circuit(backend, qc, shots=2000)
        if counts:
            p0 = counts.get('0', 0) / sum(counts.values())
            t2_data[d] = p0
            print(f"   {d}μs: P(0) = {p0:.4f}")
        time.sleep(3)

    results['T2_data'] = t2_data

    # Fit T2
    if len(t2_data) >= 4:
        t_vals = list(t2_data.keys())[1:]
        p_vals = [t2_data[t] for t in t_vals]
        try:
            # Fit P(0) = 0.5 + A * exp(-t/T2)
            dev = [p - 0.5 for p in p_vals]
            if all(d > 0 for d in dev):
                log_dev = np.log(dev)
                t_arr = np.array(t_vals)
                slope, intercept = np.polyfit(t_arr, log_dev, 1)
                T2 = -1 / slope if slope < 0 else None
                results['T2_fit'] = T2
                print(f"   T2 ≈ {T2:.1f} μs\n" if T2 else "   T2 fit failed\n")
            else:
                print("   T2 fit failed (deviation <= 0)\n")
        except:
            print("   T2 fit failed\n")

    time.sleep(10)

    # === 4. F_gate (Single-Qubit) ===
    print("4. Measuring F_gate (single-qubit)...")

    gate_data = {}
    for n in [1, 2, 5, 10, 20]:
        qc = QuantumCircuit(1, 1)
        for _ in range(2 * n):
            qc.h(0)
        qc.measure(0, 0)

        counts = run_circuit(backend, qc)
        if counts:
            p0 = counts.get('0', 0) / sum(counts.values())
            gate_data[f'H_{2*n}'] = p0
            print(f"   H^{2*n}: P(0) = {p0:.4f}")
        time.sleep(3)

    results['F_gate_data'] = gate_data

    # Fit F_gate
    if len(gate_data) >= 3:
        try:
            ns = []
            devs = []
            for key, p0 in gate_data.items():
                n = int(key.split('_')[1]) // 2
                if p0 > 0.5:
                    ns.append(n)
                    devs.append(np.log(p0 - 0.5))
            if len(ns) >= 2:
                slope, _ = np.polyfit(ns, devs, 1)
                F_gate = np.exp(slope)
                results['F_gate_fit'] = F_gate
                print(f"   F_gate ≈ {F_gate:.6f}\n")
        except:
            print("   F_gate fit failed\n")

    # === Summary ===
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if 'F_meas' in results:
        print(f"F_meas = {results['F_meas']:.6f}")
    if 'T1_fit' in results:
        print(f"T1 = {results['T1_fit']:.1f} μs")
    if 'T2_fit' in results:
        print(f"T2 = {results['T2_fit']:.1f} μs")
    if 'F_gate_fit' in results:
        print(f"F_gate = {results['F_gate_fit']:.6f}")

    # Calculate T (measurement tax)
    if 'F_meas' in results and 'T2_fit' in results:
        T_fb = 34  # Typical feedback latency in μs
        T = results['F_meas'] * np.exp(-T_fb / results['T2_fit'])
        results['T_fb_assumed'] = T_fb
        results['T_measurement_tax'] = T
        print(f"\nT = F_meas * exp(-T_fb/T2)")
        print(f"  = {results['F_meas']:.4f} * exp(-{T_fb}/{results['T2_fit']:.1f})")
        print(f"  = {T:.4f}")

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
