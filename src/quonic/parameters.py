"""Symbolic parameters for parameterized quantum circuits.

Example::

    from quonic.parameters import Parameter
    theta = Parameter("theta")
    circuit.add(GateOperation("ry", (0,), (theta,)))
    bound = circuit.bind({theta: 0.5})
"""

from __future__ import annotations

from .ir import Circuit, GateOperation


class Parameter:
    """A symbolic parameter for a parameterized circuit."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Parameter) and self.name == other.name


def bind_params(circuit: Circuit, values: dict[Parameter, float]) -> Circuit:
    """Bind symbolic parameters to concrete values.

    Args:
        circuit: circuit with Parameter objects in GateOperation.params
        values: mapping from Parameter to float value

    Returns:
        A new Circuit with all parameters replaced by values.
    """
    param_map = {p: v for p, v in values.items()}

    c = Circuit()
    c.allocate(circuit.num_qubits)
    for op in circuit.ops:
        if isinstance(op, GateOperation) and op.params:
            new_params = []
            for p in op.params:
                if isinstance(p, Parameter) and p in param_map:
                    new_params.append(param_map[p])
                else:
                    new_params.append(p)
            c.add(GateOperation(op.name, op.qubits, tuple(new_params)))
        else:
            c.add(op)
    return c


def bind_batch(
    circuit: Circuit, param_sets: list[dict[Parameter, float]]
) -> list[Circuit]:
    """Bind multiple sets of parameters to create multiple circuits.

    Args:
        circuit: parameterized circuit
        param_sets: list of parameter mappings

    Returns:
        List of bound circuits, one per parameter set.
    """
    return [bind_params(circuit, ps) for ps in param_sets]
