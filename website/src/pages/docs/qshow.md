---
title: qshow()
---

# qshow()

Run the current circuit and display results.

## Signature

```python
qshow(backend='auto', shots=1024, noise=None)
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | str | Backend name ('auto', 'qiskit', 'cirq', 'native', etc.) |
| `shots` | int | Number of measurement shots |
| `noise` | float or NoiseModel | Noise level (0.0-1.0) or NoiseModel |

## Examples

```python
from quonic import qshow

qshow()                          # Auto-select best backend
qshow(backend='qiskit')          # Use Qiskit
qshow(shots=4096)                # More shots
qshow(noise=0.05)                # 5% noise
```
