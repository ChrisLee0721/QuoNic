"""
Quandela F_gate extraction experiment.
Paste this into Quandela Hub or run locally with Perceval.

Circuit: H(q0) → [CZ(q0,q1) → H(q0) → H(q1)] × k → Measure
Odd k → ideal P(0) = 1.0
"""

import perceval as pcvl
from perceval.components import BS, PS, PERM, Circuit
import numpy as np

TOKEN = "YOUR_TOKEN_HERE"

def build_cz_circuit(k):
    """
    Build k-step CZ toggle circuit using linear optics.
    Each step = CZ + H on both qubits.
    In photonic encoding: CZ ≈ controlled-Z via linear optics.
    """
    # 4 modes: 2 qubits × 2 polarization/modes
    # Using dual-rail encoding: qubit |0⟩ = mode 0, qubit |1⟩ = mode 1
    c = pcvl.Circuit(2)

    # H on qubit 0 ≈ BS (50:50 beamsplitter)
    # CZ ≈ PS(π) on control when target is |1⟩

    # Simplified: use Perceval's native gate set
    # H = BS (Hadamard in optical basis)
    # CZ = controlled phase gate

    # For a 2-mode circuit with dual-rail encoding:
    # qubit 0 = modes (0,1), qubit 1 = modes (2,3)
    # But for simplicity, use 4-mode circuit

    c = pcvl.Circuit(4, name=f"CZ_toggle_k={k}")

    # H on qubit 0 (modes 0,1)
    c.add((0, 1), BS.H())

    for _ in range(k):
        # CZ between qubit 0 and qubit 1
        # In dual-rail: CZ = diag(1, 1, 1, -1)
        # Implementation: PS(π) on mode 3, conditioned on modes 0,1
        # Simplified CZ using linear optics:
        c.add((2, 3), BS.H())
        c.add((1, 2), BS.BS(theta=np.pi/2, phi=np.pi))
        c.add((2, 3), BS.H())

        # H on qubit 0
        c.add((0, 1), BS.H())
        # H on qubit 1
        c.add((2, 3), BS.H())

    return c


def build_simple_circuit(k):
    """
    Simpler version: 2-mode circuit.
    Encode qubit as photon presence/absence in mode 0.
    |0⟩ = photon in mode 0, |1⟩ = photon in mode 1.
    """
    c = pcvl.Circuit(2, name=f"simple_k={k}")

    # Initial H: 50:50 BS
    c.add((0, 1), BS.H())

    for _ in range(k):
        # CZ approximation: phase shift on |11⟩ component
        # Using PS on mode 1
        c.add(0, PS(np.pi))
        c.add((0, 1), BS.H())
        c.add(0, PS(np.pi))
        c.add((0, 1), BS.H())

    return c


def run_experiment():
    """Run F_gate extraction on Quandela Cloud."""

    # Use QuandelaComputer
    comp = pcvl.QuandelaComputer('sim:altair', token=TOKEN)

    depths = [51, 101, 201, 301, 401, 501]
    results = []

    print(f"{'k':>6} {'P(0)':>8}")
    for k in depths:
        c = build_simple_circuit(k)

        # Create processor
        proc = pcvl.Processor("SLOS", c)

        # Sample
        sampler = pcvl.algorithm.Sampler(proc)
        sample_count = sampler.sample_count(4000)

        # Get P(0): probability of |0⟩ in first mode
        total = sum(sample_count.values())
        p0 = sum(v for k, v in sample_count.items() if str(k) == '|0,1>') / total

        results.append({'depth': k, 'p0': p0, 'counts': dict(sample_count)})
        print(f"{k:>6} {p0:>8.4f}")

    # Fit
    k_vals = np.array([r['depth'] for r in results])
    p0_vals = np.array([r['p0'] for r in results])

    valid = p0_vals > 0.5
    if valid.sum() >= 2:
        dev = p0_vals[valid] - 0.5
        slope, _ = np.polyfit(k_vals[valid], np.log(dev), 1)
        F = np.exp(slope / 2)
        print(f"\nF_per_step = {F:.6f}")
        print(f"Error per step = {(1-F)*100:.4f}%")


if __name__ == '__main__':
    run_experiment()
