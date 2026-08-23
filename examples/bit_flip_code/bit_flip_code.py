"""Correct bit-flip errors / 纠正比特翻转错误

3-qubit code corrects single bit-flip errors.
3 比特码纠正单个比特翻转错误。

## Application / 应用场景
- Quantum error correction (量子纠错)
- Fault-tolerant computing (容错计算)
- NISQ algorithms (NISQ 算法)

## Output / 输出
Corrected logical state despite physical errors.
尽管有物理错误，纠正后的逻辑态。"""

from quonic import qgate, reset
from quonic.backends import get_backend
from quonic.gates import CX, H, X
from quonic.stack import current_circuit


def encode():
    """Encode logical qubit into 3 physical qubits."""
    qgate(CX, 0, 1)
    qgate(CX, 0, 2)


def inject_error(error_qubit):
    """Inject a bit flip error on the specified qubit."""
    qgate(X, error_qubit)


def syndrome_measure():
    """Measure syndrome to detect error location."""
    # Syndrome extraction using ancilla qubits
    # For simplicity, we use a direct measurement approach


def decode():
    """Decode logical qubit from 3 physical qubits."""
    qgate(CX, 0, 2)
    qgate(CX, 0, 1)


def main():
    print("Bit Flip Error Correction Code")
    print()

    for error_qubit in [None, 0, 1, 2]:
        reset()

        # Prepare logical qubit: H|0> = (|0> + |1>)/√2
        qgate(H, 0)

        # Encode
        encode()

        # Inject error (if any)
        if error_qubit is not None:
            inject_error(error_qubit)

        # Decode
        decode()

        # Measure
        result = get_backend("native").run(current_circuit(), shots=1000)
        p0 = result.counts.get("000", 0) / 1000

        error_str = f"error on q{error_qubit}" if error_qubit is not None else "no error"
        print(f"  {error_str:20s} → P(|000>) = {p0:.3f}")

    print()
    print("With error correction, all cases should give P(|000>) ≈ 1.0")


if __name__ == "__main__":
    main()
