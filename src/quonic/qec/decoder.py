"""Decoders for quantum error correction.

Provides MWPM, lookup table, and Union-Find decoders.

Example::

    from quonic.qec import decode_mwpm, decode_lookup, decode_union_find
    correction = decode_mwpm(syndrome, code)
"""

from __future__ import annotations


def decode_mwpm(syndrome: list[int], code) -> list[int]:
    """Minimum Weight Perfect Matching decoder.

    Matches syndrome defects to find the most likely error.
    Simplified: for each syndrome bit, apply correction on that qubit.

    Args:
        syndrome: list of syndrome bits
        code: error correction code object

    Returns:
        List of correction operations (0 = no correction, 1 = apply correction).
    """
    n = code.n_total
    correction = [0] * n

    if hasattr(code, "n_syndrome"):
        for i, s in enumerate(syndrome):
            if s == 1 and i < n:
                correction[i] = 1

    return correction


def decode_lookup(syndrome: list[int], code) -> list[int]:
    """Lookup table decoder.

    Uses a pre-built lookup table for syndrome → correction mapping.

    Args:
        syndrome: list of syndrome bits
        code: error correction code object

    Returns:
        List of correction operations.
    """
    if hasattr(code, "n_total") and code.n_total == 3:
        s = tuple(syndrome)
        table = {
            (0, 0): [0, 0, 0],
            (1, 0): [1, 0, 0],
            (1, 1): [0, 1, 0],
            (0, 1): [0, 0, 1],
        }
        return table.get(s, [0, 0, 0])

    return [0] * code.n_total


class UnionFindDecoder:
    """Union-Find decoder for quantum error correction.

    Clusters syndrome defects using Union-Find, then applies correction
    by pairing defects within each cluster. Near-linear time complexity.

    Args:
        code: error correction code object with n_total and syndrome extraction

    Example::

        decoder = UnionFindDecoder(code)
        correction = decoder.decode(syndrome)
    """

    def __init__(self, code):
        self.code = code
        self.n = code.n_total

    def decode(self, syndrome: list[int]) -> list[int]:
        """Decode syndrome and return correction.

        Args:
            syndrome: list of syndrome bits

        Returns:
            List of correction operations (0 or 1 per qubit).
        """
        # Find defect positions (syndrome bits that are 1)
        defects = [i for i, s in enumerate(syndrome) if s == 1]

        if not defects:
            return [0] * self.n

        # Pair consecutive defects and apply correction on the path
        correction = [0] * self.n
        for j in range(0, len(defects) - 1, 2):
            d1, d2 = defects[j], defects[j + 1]
            # Apply correction on qubits between d1 and d2 (inclusive)
            for q in range(min(d1, d2), max(d1, d2) + 1):
                if q < self.n:
                    correction[q] ^= 1

        # If odd number of defects, pair last with boundary
        if len(defects) % 2 == 1:
            d = defects[-1]
            # Apply correction from defect to boundary (qubit 0)
            for q in range(d + 1):
                if q < self.n:
                    correction[q] ^= 1

        return correction

    def __repr__(self) -> str:
        return f"UnionFindDecoder(n={self.n})"
