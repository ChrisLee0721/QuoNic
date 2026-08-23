"""Backend-independent intermediate representation (IR).

qgate() first records user operations into backend-independent GateOperation / Circuit,
then qshow() hands them to a concrete backend (Qiskit / Cirq / ...) to translate and execute.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateOperation:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class CRegCondition:
    """A multi-bit classical register equality test: the branch/loop condition
    ``creg == value`` for a register of ``width`` bits.

    ``value`` is the integer register value in [0, 2**width).
    """

    creg: str
    width: int
    value: int


@dataclass(frozen=True)
class ClassicalIfOperation:
    """Classical control flow: apply one of two branch gates depending on the control source.

    Unlike qif's quantum superposition, this produces a classical mixed state (incoherent entanglement).
    control may be:
      - int: measure that qubit (measure first, then branch)
      - str: read the measurement result already stored in the named single-bit creg (then when == 1)
      - CRegCondition: read the named multi-bit register, then when register == value
    then_op / else_op are single-bit branch gates.
    """

    control: int | str | CRegCondition
    then_op: GateOperation
    else_op: GateOperation

    @property
    def name(self) -> str:
        return "cif"

    @property
    def params(self) -> tuple[()]:
        return ()

    @property
    def qubits(self) -> tuple[int, ...]:
        qs = set()
        if isinstance(self.control, int):
            qs.add(self.control)
        qs.update(self.then_op.qubits)
        qs.update(self.else_op.qubits)
        return tuple(sorted(qs))


@dataclass(frozen=True)
class CMeasureOperation:
    """Measure qubit and store the result in the ``bit``-th position of the named
    classical register creg (bit defaults to 0 for the single-bit case)."""

    qubit: int
    creg: str
    bit: int = 0

    @property
    def name(self) -> str:
        return "cmeasure"

    @property
    def params(self) -> tuple[()]:
        return ()

    @property
    def qubits(self) -> tuple[int, ...]:
        return (self.qubit,)


@dataclass(frozen=True)
class ClassicalWhileOperation:
    """Classical feedback loop: repeat body until the creg register value equals until.

    ``width`` is the number of bits of the creg register (1 for a single bit);
    ``until`` is the integer register value in [0, 2**width). body is a tuple of
    ops (usually ending with creg.measure(...) to update the condition), the core
    of repeat-until-success (RUS) dynamic circuits.
    """

    creg: str
    until: int
    body: tuple[object, ...]
    width: int = 1

    @property
    def name(self) -> str:
        return "cwhile"

    @property
    def params(self) -> tuple[()]:
        return ()

    @property
    def qubits(self) -> tuple[int, ...]:
        qs = set()
        for op in self.body:
            qs.update(op.qubits)
        return tuple(sorted(qs))


_MEASURE_NAMES = ("measure", "cmeasure")


class Circuit:
    def __init__(self) -> None:
        self.ops: list[object] = []
        self.num_qubits: int = 0
        self.requires_grad: bool = False

    def add(self, op: object) -> None:
        self.ops.append(op)
        for q in op.qubits:
            self.num_qubits = max(self.num_qubits, q + 1)

    def allocate(self, n_qubits: int) -> None:
        # pre-reserve qubits (without emitting a gate), so QInt etc. can occupy indices even without initial gates
        self.num_qubits = max(self.num_qubits, n_qubits)

    def measured_qubits(self) -> set:
        measured = set()
        for op in self.ops:
            if op.name == "measure":
                measured.add(op.qubits[0])
            elif op.name == "cmeasure":
                measured.add(op.qubit)
            elif op.name == "cif" and isinstance(op.control, int):
                measured.add(op.control)
        return measured

    def unmeasured_qubits(self) -> list[int]:
        measured = self.measured_qubits()
        return [q for q in range(self.num_qubits) if q not in measured]

    def is_empty(self) -> bool:
        return not self.ops

    def gate_count(self) -> int:
        """Logical gate count (excluding measurement gates)."""
        return sum(1 for op in self.ops if op.name not in _MEASURE_NAMES)

    def depth(self) -> int:
        """Circuit depth: the longest dependency chain of non-measurement gates (multi-qubit gates synchronized per-qubit clock)."""
        clocks = [0] * self.num_qubits
        for op in self.ops:
            if op.name in _MEASURE_NAMES:
                continue
            d = max(clocks[q] for q in op.qubits) + 1
            for q in op.qubits:
                clocks[q] = d
        return max(clocks) if clocks else 0

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #

    def __iter__(self):
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __getitem__(self, index):
        return self.ops[index]

    def __repr__(self) -> str:
        preview = []
        for op in self.ops[:5]:
            if isinstance(op, GateOperation):
                qargs = ",".join(str(q) for q in op.qubits)
                preview.append(f"{op.name}({qargs})")
            else:
                preview.append(type(op).__name__)
        suffix = ", ..." if len(self.ops) > 5 else ""
        return f"Circuit(n={self.num_qubits}, ops=[{', '.join(preview)}{suffix}])"

    def copy(self) -> Circuit:
        """Return a deep copy of this circuit."""
        import copy
        return copy.deepcopy(self)

    def filter(self, qubits=None, name=None) -> Circuit:
        """Return a sub-circuit containing only ops matching the filter criteria.

        Args:
            qubits: keep ops whose qubits are a subset of this set
            name: keep ops whose name matches this string
        """
        c = Circuit()
        for op in self.ops:
            if qubits is not None:
                op_qubits = set(op.qubits) if hasattr(op, "qubits") else set()
                if not op_qubits.issubset(set(qubits)):
                    continue
            if name is not None and op.name != name:
                continue
            c.add(op)
        return c

    def slice(self, start: int = 0, end: int | None = None) -> Circuit:
        """Return a sub-circuit from ops[start:end]."""
        c = Circuit()
        for op in self.ops[start:end]:
            c.add(op)
        return c

    def inverse(self) -> Circuit:
        """Return the inverse circuit (reversed ops, each gate adjoint)."""
        from .compiler import _adjoint
        c = Circuit()
        for op in reversed(self.ops):
            if isinstance(op, GateOperation):
                c.add(_adjoint(op))
            else:
                c.add(op)
        return c

    def __add__(self, other: Circuit) -> Circuit:
        """Concatenate two circuits."""
        c = self.copy()
        for op in other.ops:
            c.add(op)
        return c

    # ------------------------------------------------------------------ #
    #  Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        ops = []
        for op in self.ops:
            if isinstance(op, GateOperation):
                ops.append({
                    "type": "gate",
                    "name": op.name,
                    "qubits": list(op.qubits),
                    "params": list(op.params),
                })
            elif isinstance(op, CMeasureOperation):
                ops.append({
                    "type": "cmeasure",
                    "qubit": op.qubit,
                    "creg": op.creg,
                    "bit": op.bit,
                })
        return {
            "num_qubits": self.num_qubits,
            "requires_grad": self.requires_grad,
            "ops": ops,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Circuit:
        """Deserialize from a dict."""
        c = cls()
        c.allocate(d["num_qubits"])
        c.requires_grad = d.get("requires_grad", False)
        for op_d in d["ops"]:
            if op_d["type"] == "gate":
                c.add(GateOperation(
                    op_d["name"],
                    tuple(op_d["qubits"]),
                    tuple(op_d.get("params", ())),
                ))
            elif op_d["type"] == "cmeasure":
                c.add(CMeasureOperation(
                    op_d["qubit"],
                    op_d["creg"],
                    op_d.get("bit", 0),
                ))
        return c

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Circuit:
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(s))

    def to_qasm3(self) -> str:
        """Export to OpenQASM 3.0 string."""
        lines = [f"qreg q[{self.num_qubits}];"]
        for op in self.ops:
            if isinstance(op, GateOperation):
                if op.name == "measure":
                    lines.append(f"measure q[{op.qubits[0]}];")
                else:
                    qargs = ", ".join(f"q[{q}]" for q in op.qubits)
                    if op.params:
                        pargs = ", ".join(str(p) for p in op.params)
                        lines.append(f"{op.name}({pargs}) {qargs};")
                    else:
                        lines.append(f"{op.name} {qargs};")
        return "\n".join(lines)
