"""
Clean Perceval payload for Quandela photonic quantum computing.
"""

import json
from datetime import datetime


def create_clean_payload():
    """Create a clean Perceval payload."""

    # Perceval program as a string
    perceval_program = '''
import perceval as pcvl
import numpy as np

# Create 2-mode circuit
c = pcvl.Circuit(2)

# Add beam splitters and phase shifters
c.add(0, pcvl.BS())           # 50:50 beam splitter
c.add((0,), pcvl.PS(np.pi/2)) # Phase shifter
c.add(0, pcvl.BS())           # Another beam splitter

# Serialize circuit
cgs = pcvl.Serializable(c)
'''

    payload = {
        "name": f"photonic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "shots": 1000,
        "input_state": "|1,1>",
        "program": perceval_program.strip(),
        "backend": "Simplon.1"  # Simulator
    }

    return payload


def create_hardware_payload():
    """Create payload for actual hardware (Bloom)."""

    perceval_program = '''
import perceval as pcvl
import numpy as np

# Simple photonic circuit for Bell-like state
c = pcvl.Circuit(2)
c.add(0, pcvl.BS())
c.add((0,), pcvl.PS(np.pi/4))
c.add(0, pcvl.BS())

cgs = pcvl.Serializable(c)
'''

    payload = {
        "name": f"bloom_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "shots": 1000,
        "input_state": "|1,1>",
        "program": perceval_program.strip(),
        "backend": "boson-sampler"  # Bloom hardware
    }

    return payload


if __name__ == "__main__":
    print("=== Clean Perceval Payload ===\n")

    payload = create_clean_payload()
    print(json.dumps(payload, indent=2))

    print("\n=== Hardware Payload (Bloom) ===\n")

    hw_payload = create_hardware_payload()
    print(json.dumps(hw_payload, indent=2))

    # Save
    with open("experiments/quandela_payload_clean.json", "w") as f:
        json.dump({
            "simulator": payload,
            "hardware": hw_payload
        }, f, indent=2)

    print("\nSaved to experiments/quandela_payload_clean.json")
