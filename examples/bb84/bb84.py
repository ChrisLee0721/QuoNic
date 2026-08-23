"""Quantum key distribution / 量子密钥分发

BB84 protocol for secure key exchange using quantum mechanics.
BB84 协议利用量子力学实现安全密钥交换。

## Application / 应用场景
- Secure communication (安全通信)
- Quantum cryptography (量子密码学)
- Key distribution (密钥分发)

## Output / 输出
Shared secret key between Alice and Bob.
Alice 和 Bob 共享的密钥。"""

import random

from quonic import qgate, reset
from quonic.backends import get_backend
from quonic.gates import H, X
from quonic.stack import current_circuit


def bb84_round(alice_basis, alice_bit, bob_basis):
    """Run one round of BB84.

    Args:
        alice_basis: 0=Z, 1=X
        alice_bit: 0 or 1
        bob_basis: 0=Z, 1=X

    Returns:
        Bob's measurement result.
    """
    reset()

    # Alice prepares
    if alice_bit == 1:
        qgate(X, 0)
    if alice_basis == 1:  # X basis
        qgate(H, 0)

    # Bob measures
    if bob_basis == 1:  # X basis
        qgate(H, 0)

    result = get_backend("native").run(current_circuit(), shots=1)
    return int(next(iter(result.counts.keys())))


def main():
    n_rounds = 20
    alice_bases = [random.randint(0, 1) for _ in range(n_rounds)]
    alice_bits = [random.randint(0, 1) for _ in range(n_rounds)]
    bob_bases = [random.randint(0, 1) for _ in range(n_rounds)]

    # Run protocol
    bob_results = []
    for i in range(n_rounds):
        r = bb84_round(alice_bases[i], alice_bits[i], bob_bases[i])
        bob_results.append(r)

    # Sifting: keep only matching bases
    key = []
    for i in range(n_rounds):
        if alice_bases[i] == bob_bases[i]:
            key.append(alice_bits[i])

    print("BB84 Quantum Key Distribution")
    print(f"  Rounds: {n_rounds}")
    print(f"  Alice bases: {alice_bases}")
    print(f"  Alice bits:  {alice_bits}")
    print(f"  Bob bases:   {bob_bases}")
    print(f"  Bob results: {bob_results}")
    print(f"  Matching bases: {sum(1 for i in range(n_rounds) if alice_bases[i] == bob_bases[i])}")
    print(f"  Key: {key}")


if __name__ == "__main__":
    main()
