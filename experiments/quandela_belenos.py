"""
Quandela Belenos photonic QPU payload.
Correct format: command=sample_count, max_shots=100
"""

import json
from datetime import datetime


def create_belenos_payload(shots=1000, max_shots=100):
    """Create payload for Belenos photonic QPU."""

    # Simple photonic circuit: two beam splitters with phase
    program = '''
import perceval as pcvl
import numpy as np

# Create 2-mode photonic circuit
c = pcvl.Circuit(2)

# Beam splitter (creates superposition)
c.add(0, pcvl.BS())

# Phase shifter
c.add((0,), pcvl.PS(np.pi/2))

# Another beam splitter
c.add(0, pcvl.BS())

# Serialize
print(pcvl.Serializable(c).dump())
'''.strip()

    payload = {
        "backend": "qpu:belenos",
        "command": "sample_count",
        "input_state": "|1,1>",
        "max_shots": max_shots,
        "program": program,
        "shots": shots
    }

    return payload


def create_f_gate_payload(k=1, shots=1000, max_shots=100):
    """Create payload for F_gate extraction at depth k."""

    # For photonic, we can't do the same CZ+H circuit
    # But we can create a depth-k photonic circuit
    program = f'''
import perceval as pcvl
import numpy as np

# Create photonic circuit with k beam splitters
c = pcvl.Circuit(2)

# Add k stages of beam splitters + phase shifters
for i in range({k}):
    c.add(0, pcvl.BS())
    c.add((0,), pcvl.PS(np.pi/4))

# Final beam splitter
c.add(0, pcvl.BS())

print(pcvl.Serializable(c).dump())
'''.strip()

    payload = {
        "backend": "qpu:belenos",
        "command": "sample_count",
        "input_state": "|1,1>",
        "max_shots": max_shots,
        "program": program,
        "shots": shots
    }

    return payload


if __name__ == "__main__":
    print("=== Belenos Payload (Basic) ===")
    print(json.dumps(create_belenos_payload(), indent=2))

    print("\n=== Belenos Payload (F_gate k=1) ===")
    print(json.dumps(create_f_gate_payload(1), indent=2))

    print("\n=== Belenos Payload (F_gate k=5) ===")
    print(json.dumps(create_f_gate_payload(5), indent=2))

    # Save
    with open("experiments/quandela_belenos_payloads.json", "w") as f:
        json.dump({
            "basic": create_belenos_payload(),
            "f_gate_k1": create_f_gate_payload(1),
            "f_gate_k5": create_f_gate_payload(5),
        }, f, indent=2)

    print("\nSaved to experiments/quandela_belenos_payloads.json")
