"""Check Rigetti Cepheus device capabilities and submit F_gate experiment."""

import json
import time
from datetime import datetime
import numpy as np
import boto3
from braket.aws import AwsDevice
from braket.circuits import Circuit

DEVICE_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"

device = AwsDevice(DEVICE_ARN)
print(f"Device: {device.name}, Status: {device.status}")
print(f"Qubits: {device.properties.paradigm.qubitCount}")

# Check supported gates
actions = device.properties.action
for action_type, action_spec in actions.items():
    print(f"\nAction: {action_type}")
    if hasattr(action_spec, 'supportedOperations'):
        print(f"  Supported ops: {action_spec.supportedOperations}")
    if hasattr(action_spec, 'supportedGates'):
        gates = [str(g) for g in action_spec.supportedGates]
        print(f"  Supported gates: {gates}")

# Get coupling map
connectivity = device.properties.paradigm.connectivity
if hasattr(connectivity, 'connectivityGraph'):
    graph = connectivity.connectivityGraph
    # Find first connected pair
    for q0, neighbors in list(graph.items())[:5]:
        print(f"  Qubit {q0} connects to: {neighbors}")
    # Pick first pair
    q0 = int(list(graph.keys())[0])
    q1 = int(list(graph[q0].keys())[0]) if isinstance(graph[q0], dict) else int(graph[q0][0])
    print(f"\nUsing qubit pair: ({q0}, {q1})")
