"""
Perceval payload for Quandela photonic quantum computing.
Constructs a simple experiment to test gate fidelity.
"""

import json
from datetime import datetime


def create_perceval_payload():
    """Create a Perceval payload for a simple photonic experiment."""

    # Perceval circuit definition
    # Basic photonic circuit: H-like operation using beam splitters
    # Note: Perceval uses linear optics, not gate-based model

    payload = {
        "name": f"f_gate_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "description": "F_gate extraction test on Quandela photonic QPU",
        "shots": 1000,
        "backend": "Simplon simulator",  # or specific QPU name
        "circuit": {
            # Perceval DSL circuit definition
            # This is a simple 2-mode photonic circuit
            "components": [
                # Input state: |1,1> (one photon in each mode)
                {"type": "source", "modes": [0, 1], "state": "|1,1>"},

                # Beam splitter (50:50)
                {"type": "BS", "mode_in": [0, 1], "mode_out": [0, 1],
                 "theta": 3.14159 / 4},

                # Phase shifter
                {"type": "PS", "mode": 0, "phi": 3.14159 / 2},

                # Second beam splitter
                {"type": "BS", "mode_in": [0, 1], "mode_out": [0, 1],
                 "theta": 3.14159 / 4},

                # Measurement
                {"type": "measure", "modes": [0, 1]}
            ],
            "output_modes": [0, 1]
        },
        "parameters": {
            "precision": "double",
            "platform": "bloom"  # Quandela's 8-qubit photonic QPU
        }
    }

    return payload


def create_perceval_python_code():
    """Generate Perceval Python code for the experiment."""

    code = '''
import perceval as pcvl
import numpy as np

# Create a simple photonic circuit
# Using beam splitters to create superposition

# Define the circuit
circuit = pcvl.Circuit(2)  # 2 modes

# Add components
circuit.add(0, pcvl.BS())  # Beam splitter on modes 0,1
circuit.add((0,), pcvl.PS(np.pi/2))  # Phase shifter
circuit.add(0, pcvl.BS())  # Another beam splitter

# Display circuit
print("Circuit:")
pcvl.pdisplay(circuit)

# Create a source (Fock state |1,1>)
source = pcvl.Source(dim=2)  # 2 photons

# Create processor
processor = pcvl.Processor("Simplon", circuit)
processor.with_input(pcvl.Source generates |1,1>)

# Simulate
backend = pcvl.BackendFactory().create_backend("clifford")
backend.set_circuit(circuit)

# Get probability distribution
probs = backend.prob_distribution()
print(f"Output distribution: {probs}")
'''

    return code


def create_simple_payload():
    """Create a minimal valid payload."""

    # Minimal Perceval payload format
    payload = {
        "shots": 1000,
        "cloud_submission": {
            "payload": {
                "name": "bell_state_test",
                "shots": 1000,
                "input_state": "|1,1>",  # One photon in each mode
                "program": """
# Perceval program
import perceval as pcvl
import numpy as np

# Create circuit
c = pcvl.Circuit(2)
c.add(0, pcvl.BS())  # 50:50 beam splitter
c.add((0,), pcvl.PS(np.pi/2))  # Phase
c.add(0, pcvl.BS())  # Another beam splitter

# Return circuit
pcvl.Serializable下乡(c)
"""
            }
        }
    }

    return payload


if __name__ == "__main__":
    print("=== Perceval Payload Generator ===\n")

    # Generate payloads
    payload1 = create_perceval_payload()
    print("Payload 1 (JSON format):")
    print(json.dumps(payload1, indent=2))

    print("\n" + "="*50 + "\n")

    payload2 = create_simple_payload()
    print("Payload 2 (Minimal format):")
    print(json.dumps(payload2, indent=2))

    # Save to file
    with open("experiments/quandela_payload.json", "w") as f:
        json.dump(payload1, f, indent=2)
    print(f"\nSaved to experiments/quandela_payload.json")
