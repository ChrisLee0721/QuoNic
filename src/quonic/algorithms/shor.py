"""Shor's algorithm: quantum period finding + continued fractions to factor a
large integer.

Find a non-trivial factor of an odd composite N (not a prime power).

Principle:
    1. Pick a random a coprime to N; its order r satisfies a^r ≡ 1 (mod N).
    2. Quantum period finding: use QPE to estimate the eigenphase s/r of the
       modular multiplication operator U_a |y> = |a·y mod N>.
    3. Continued fractions recover r from the phase j/2^t.
    4. If r is even and a^{r/2} ≠ -1 (mod N), then gcd(a^{r/2} ± 1, N) yields a factor.

Example:
    from quonic.algorithms import shor

    result = shor(15)          # returns 3 or 5
    print(result.value)
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from .._i18n import tr
from ..backends import run_circuit
from ..ir import Circuit, GateOperation
from ..qft import add_iqft, add_qft
from ..result import Result

# ---------------------------------------------------------------------------
# Basic quantum gates
# ---------------------------------------------------------------------------

def _crz(circuit: Circuit, c: int, t: int, phi: float) -> None:
    # Controlled Rz(phi): Rz(phi/2); CX; Rz(-phi/2); CX
    circuit.add(GateOperation("rz", (t,), (phi / 2,)))
    circuit.add(GateOperation("cx", (c, t)))
    circuit.add(GateOperation("rz", (t,), (-phi / 2,)))
    circuit.add(GateOperation("cx", (c, t)))


def _toffoli(circuit: Circuit, c1: int, c2: int, t: int) -> None:
    circuit.add(GateOperation("ccx", (c1, c2, t)))


def _cswap(circuit: Circuit, control: int, a: int, b: int) -> None:
    # Fredkin gate: controlled swap (3 Toffolis)
    _toffoli(circuit, control, a, b)
    _toffoli(circuit, control, b, a)
    _toffoli(circuit, control, a, b)


# ---------------------------------------------------------------------------
# QFT addition (Draper addition)
# ---------------------------------------------------------------------------

def _add_const(circuit: Circuit, qubits: Sequence[int], k: int) -> None:
    """|x> -> |x + k mod 2^m>, where m = len(qubits)."""
    m = len(qubits)
    add_qft(circuit, qubits)
    for j in range(m):
        circuit.add(GateOperation("rz", (qubits[j],), (2 * math.pi * k / 2 ** (j + 1),)))
    add_iqft(circuit, qubits)


def _cadd_const(circuit: Circuit, qubits: Sequence[int], k: int, control: int) -> None:
    """Controlled addition: |x> -> |x + k mod 2^m> when control=1. control is not in qubits."""
    m = len(qubits)
    add_qft(circuit, qubits)
    for j in range(m):
        _crz(circuit, control, qubits[j], 2 * math.pi * k / 2 ** (j + 1))
    add_iqft(circuit, qubits)


# ---------------------------------------------------------------------------
# Modular arithmetic
# ---------------------------------------------------------------------------

def _modinv(a: int, m: int) -> int | None:
    """Modular inverse a^{-1} mod m (a coprime to m), via the extended Euclidean algorithm."""
    a %= m
    if math.gcd(a, m) != 1:
        return None
    t, newt = 0, 1
    r, newr = m, a
    while newr != 0:
        q = r // newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    if t < 0:
        t += m
    return t


def cadd_mod(
    circuit: Circuit,
    qubits: Sequence[int],
    flag: int,
    b: int,
    N: int,
    control: int,
) -> None:
    """Controlled in-place modular addition: |x> -> |(x + b) mod N> when control=1,
    with flag returned to |0>.

    qubits has length n+1 (n = bit width of N), and x, b ∈ [0, N).
    flag is an extra auxiliary qubit (|0> at both start and end) that records the
    borrow flag of "x + b < N".
    """
    n = len(qubits) - 1
    M = 2 ** (n + 1)
    _cadd_const(circuit, qubits, b, control)      # x -> x + b
    _cadd_const(circuit, qubits, M - N, control)  # x + b -> x + b - N
    circuit.add(GateOperation("cx", (qubits[n], flag)))  # flag = underflow flag
    _cadd_const(circuit, qubits, N, flag)         # if underflow, add N back
    # Invert flag (using the relation flag = "(x+b) mod N >= b"):
    _cadd_const(circuit, qubits, M - b, control)  # x -> x - b
    circuit.add(GateOperation("cx", (qubits[n], flag)))
    circuit.add(GateOperation("cx", (control, flag)))
    _cadd_const(circuit, qubits, b, control)      # restore x


def _cmul_mod(
    circuit: Circuit,
    q: int,
    reg: Sequence[int],
    c: int,
    N: int,
    scratch: Sequence[int],
    anc: int,
    flag: int,
) -> None:
    """Controlled in-place modular multiplication: |reg> -> |c·reg mod N> when q=1,
    with scratch/anc/flag all returned to |0>.

    reg and scratch each have n+1 qubits; anc is the Toffoli auxiliary qubit and
    flag is the modular addition flag.
    """
    n = len(reg) - 1
    cinv = _modinv(c, N)
    # 1) scratch = c·reg mod N
    for i in range(n):
        a_i = (2 ** i * c) % N
        _toffoli(circuit, q, reg[i], anc)
        cadd_mod(circuit, scratch, flag, a_i, N, anc)
        _toffoli(circuit, q, reg[i], anc)
    # 2) controlled swap reg <-> scratch
    for i in range(n + 1):
        _cswap(circuit, q, reg[i], scratch[i])
    # 3) scratch -= c^{-1}·reg (clear scratch)
    for i in range(n):
        b_i = (N - (2 ** i * cinv) % N) % N
        _toffoli(circuit, q, reg[i], anc)
        cadd_mod(circuit, scratch, flag, b_i, N, anc)
        _toffoli(circuit, q, reg[i], anc)


def _mod_exp(
    circuit: Circuit,
    exponent: Sequence[int],
    reg: Sequence[int],
    a: int,
    N: int,
    scratch: Sequence[int],
    anc: int,
    flag: int,
) -> None:
    """|reg> -> |a^x mod N>, where x is the exponent register (LSB at exponent[0]).

    Consistent with qpe.py's no-swap IQFT convention: the j-th phase bit controls
    U^{2^{t-1-j}}.
    """
    t = len(exponent)
    for j in range(t):
        c = pow(a, 2 ** (t - 1 - j), N)
        _cmul_mod(circuit, exponent[j], reg, c, N, scratch, anc, flag)


# ---------------------------------------------------------------------------
# Classical part: continued fractions and factor extraction
# ---------------------------------------------------------------------------

def _convergents(x: float, max_q: int) -> Iterator[tuple[int, int]]:
    """Generate continued-fraction convergents p/q of a real number x (q <= max_q)."""
    if x <= 0:
        return
    p0, p1 = 0, 1
    q0, q1 = 1, 0
    r = x
    for _ in range(1000):
        a = math.floor(r + 1e-12)
        p = a * p1 + p0
        q = a * q1 + q0
        if q > max_q:
            return
        yield (p, q)
        p0, p1 = p1, p
        q0, q1 = q1, q
        frac = r - a
        if abs(frac) < 1e-12:
            return
        r = 1.0 / frac


def _period_from_phase(j: int, t: int, a: int, N: int) -> int | None:
    """Recover the order r of a from the phase j/2^t using continued fractions."""
    phi = j / (2 ** t)
    if phi == 0:
        return None
    for _, q in _convergents(phi, N):
        if q and pow(a, q, N) == 1:
            return q
    return None


def _factor_from_period(a: int, r: int | None, N: int) -> int | None:
    """Extract a factor from the order r; return None if r is odd or a^{r/2}≡-1."""
    if r is None or r % 2 != 0:
        return None
    x = pow(a, r // 2, N)
    if x == N - 1:
        return None
    for cand in (math.gcd(x - 1, N), math.gcd(x + 1, N)):
        if 1 < cand < N:
            return cand
    return None


def _perfect_power_factor(N: int) -> int | None:
    """If N is a perfect power b^k, return b; otherwise return None."""
    for b in range(2, N.bit_length() + 1):
        root = round(N ** (1.0 / b))
        for r in (root - 1, root, root + 1):
            if r >= 2 and r ** b == N:
                return r
    return None


def _run_once(
    N: int, a: int, t: int, backend: str, shots: int
) -> tuple[int | None, int | None, int | None, dict[str, int]]:
    """Run one round of quantum period finding; return (factor, j, r, exp_counts)."""
    n = (N - 1).bit_length()
    exponent = list(range(t))
    base = t
    reg = list(range(base, base + n + 1))
    base += n + 1
    scratch = list(range(base, base + n + 1))
    base += n + 1
    anc = base
    flag = base + 1

    circuit = Circuit()
    circuit.add(GateOperation("x", (reg[0],)))  # reg = |1>
    for q in exponent:
        circuit.add(GateOperation("h", (q,)))
    _mod_exp(circuit, exponent, reg, a, N, scratch, anc, flag)
    add_iqft(circuit, exponent)

    result = run_circuit(circuit, backend=backend, shots=shots)

    # Only care about the exponent register (rightmost t bits); sum over the other bits
    exp_counts = {}
    for bitstring, count in result.counts.items():
        e = bitstring[-t:]
        exp_counts[e] = exp_counts.get(e, 0) + count

    for e in sorted(exp_counts, key=exp_counts.get, reverse=True):
        j = int(e, 2)
        r = _period_from_phase(j, t, a, N)
        factor = _factor_from_period(a, r, N)
        if factor is not None:
            return factor, j, r, exp_counts
    return None, None, None, exp_counts


def shor(
    N: int,
    a: int | None = None,
    t: int | None = None,
    backend: str = "auto",
    shots: int = 1024,
    attempts: int = 8,
) -> Result:
    """Factor the integer N (odd composite and not a prime power), returning a
    non-trivial factor.

    Args:
        N: The integer to factor.
        a: Random base (chosen randomly by default).
        t: Number of precision bits for period finding, default 2 · bit width.
        backend / shots: Sampling parameters.
        attempts: Number of retries on failure.

    Returns: Result (kind="value"); result.value is a non-trivial factor of N.
    """
    N = int(N)
    if N < 2:
        raise ValueError(tr("err.shor_n", N=N))

    if N % 2 == 0:
        return Result.from_value(2, factor_of=N, method="even")

    pp = _perfect_power_factor(N)
    if pp is not None:
        return Result.from_value(pp, factor_of=N, method="perfect_power")

    if t is None:
        t = 2 * (N - 1).bit_length()

    fixed_a = a is not None
    for _ in range(attempts):
        x = a if fixed_a else random.randint(2, N - 1)
        g = math.gcd(x, N)
        if 1 < g < N:
            return Result.from_value(g, factor_of=N, method="gcd", a=x)

        factor, j, r, counts = _run_once(N, x, t, backend, shots)
        if factor is not None:
            return Result.from_value(
                factor, factor_of=N, a=x, period=r, phase_j=j, counts=counts
            )
        if fixed_a:
            break

    raise RuntimeError(tr("err.shor_failed", N=N))
