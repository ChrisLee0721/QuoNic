"""
Quandela Hub Payload — F_gate Extraction
Paste this into Quandela Hub's code editor and run.
"""

import perceval as pcvl
import numpy as np

# === Build 2-qubit CZ toggle circuit ===
# Dual-rail encoding: qubit 0 = modes (0,1), qubit 1 = modes (2,3)
# |00⟩ = photon in mode 0 + photon in mode 2

def build_circuit(k):
    """Build k-step CZ+H toggle circuit on 4 modes."""
    c = pcvl.Circuit(4, name=f"CZ_toggle_k={k}")

    # H on qubit 0
    c.add((0, 1), pcvl.BS.H())

    for _ in range(k):
        # CZ gate (controlled-Z) via linear optics
        c.add((2, 3), pcvl.BS.H())
        c.add((1, 2), pcvl.BS.BS(theta=np.pi/2, phi=np.pi))
        c.add((2, 3), pcvl.BS.H())

        # H on qubit 0
        c.add((0, 1), pcvl.BS.H())
        # H on qubit 1
        c.add((2, 3), pcvl.BS.H())

    return c


# === Run on Quandela Cloud ===
# Change 'sim:altair' to 'altair' for real hardware
processor = pcvl.RemoteProcessor("sim:altair")

depths = [51, 101, 201, 301, 401, 501]
shots = 4000

print(f"{'k':>6} {'P(00)':>8} {'P(01)':>8} {'P(10)':>8} {'P(11)':>8}")
results = []

for k in depths:
    c = build_circuit(k)
    processor.set_circuit(c)

    sampler = pcvl.algorithm.Sampler(processor)
    sample_count = sampler.sample_count(shots)

    # Count states: |0,2⟩ = |00⟩, |0,3⟩ = |01⟩, |1,2⟩ = |10⟩, |1,3⟩ = |11⟩
    total = sum(sample_count.values())
    counts = {}
    for state, cnt in sample_count.items():
        s = str(state)
        counts[s] = cnt

    # P(qubit0=0) = P(|00⟩) + P(|01⟩) = states with photon in mode 0
    p0 = sum(cnt for state, cnt in sample_count.items()
             if '1' not in str(state).split(',')[0].strip(' |')) / total

    results.append({'k': k, 'p0': p0, 'counts': counts})
    print(f"{k:>6} {p0:>8.4f}")

# Fit F_gate
print("\n=== F_gate Extraction ===")
k_vals = np.array([r['k'] for r in results])
p0_vals = np.array([r['p0'] for r in results])

valid = p0_vals > 0.5
if valid.sum() >= 2:
    dev = p0_vals[valid] - 0.5
    slope, _ = np.polyfit(k_vals[valid], np.log(dev), 1)
    F = np.exp(slope / 2)
    print(f"F_per_step = {F:.6f}")
    print(f"Error per step = {(1-F)*100:.4f}%")
    print(f"Gate fidelity = {F*100:.4f}%")
