"""
Quandela Perceval payload with command field.
"""

import json
from datetime import datetime


def create_payload_with_command(command="submit_job"):
    """Create payload with required command field."""

    perceval_program = '''
import perceval as pcvl
import numpy as np

c = pcvl.Circuit(2)
c.add(0, pcvl.BS())
c.add((0,), pcvl.PS(np.pi/2))
c.add(0, pcvl.BS())

cgs = pcvl.Serializable(c)
'''

    payload = {
        "command": command,
        "name": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "shots": 1000,
        "input_state": "|1,1>",
        "program": perceval_program.strip(),
        "backend": "Simplon.1"
    }

    return payload


def create_simple_command_payload():
    """Minimal payload with just command."""

    payload = {
        "command": "submit",
        "backend": "Simplon.1",
        "shots": 1000,
        "circuit": {
            "n_modes": 2,
            "components": [
                {"type": "BS", "modes": [0, 1]},
                {"type": "PS", "mode": 0, "phi": 1.5708},
                {"type": "BS", "modes": [0, 1]}
            ],
            "input_state": [1, 1]
        }
    }

    return payload


def create_run_command():
    """Payload with run command."""

    payload = {
        "command": "run",
        "backend": "Simplon.1",
        "shots": 1000,
        "program": "import perceval as pcvl\nc = pcvl.Circuit(2)\nc.add(0, pcvl.BS())\nprint(pcvl.Serializable(c).dump())"
    }

    return payload


if __name__ == "__main__":
    print("=== Payload with command: submit_job ===")
    print(json.dumps(create_payload_with_command("submit_job"), indent=2))

    print("\n=== Simple command payload ===")
    print(json.dumps(create_simple_command_payload(), indent=2))

    print("\n=== Run command payload ===")
    print(json.dumps(create_run_command(), indent=2))

    # Save all variants
    with open("experiments/quandela_payloads.json", "w") as f:
        json.dump({
            "submit_job": create_payload_with_command("submit_job"),
            "submit": create_simple_command_payload(),
            "run": create_run_command()
        }, f, indent=2)

    print("\nSaved all variants to experiments/quandela_payloads.json")
