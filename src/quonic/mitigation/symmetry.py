"""Symmetry verification — post-select counts satisfying symmetry constraints.

Discards measurement outcomes that violate known symmetries (particle number,
parity, etc.) and renormalizes the remaining counts.

Example::

    from quonic.mitigation import symmetry_verify

    counts = {"000": 400, "011": 300, "101": 200, "110": 100}
    # Keep only states with even parity (even number of 1s)
    filtered = symmetry_verify(counts, parity="even")
"""

from __future__ import annotations


def symmetry_verify(
    counts: dict[str, int],
    particle_number: int | None = None,
    parity: str | None = None,
    custom: callable | None = None,
) -> dict[str, int]:
    """Post-select measurement counts by symmetry constraints.

    Args:
        counts: Measurement histogram ``{bitstring: count}``.
        particle_number: If set, keep only bitstrings with exactly this many 1s.
        parity: ``"even"`` or ``"odd"`` — keep only bitstrings with the
            specified parity of 1-bits.
        custom: A callable ``f(bitstring) -> bool``.  Keep only bitstrings
            where *custom* returns ``True``.

    Returns:
        Filtered counts dict.  Keys that violate all active constraints are
        removed.  The result is **not** renormalized (raw counts preserved).
    """
    filtered: dict[str, int] = {}
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        if particle_number is not None and bits.count("1") != particle_number:
            continue
        if parity is not None:
            n_ones = bits.count("1")
            if parity == "even" and n_ones % 2 != 0:
                continue
            if parity == "odd" and n_ones % 2 != 1:
                continue
        if custom is not None and not custom(bits):
            continue
        filtered[bits] = count
    return filtered


def symmetry_verify_and_renormalize(
    counts: dict[str, int],
    **kwargs,
) -> dict[str, int]:
    """Like :func:`symmetry_verify` but renormalizes to original shot count.

    Scales the filtered counts so that their sum equals the original total.
    """
    total_before = sum(counts.values())
    filtered = symmetry_verify(counts, **kwargs)
    total_after = sum(filtered.values())
    if total_after == 0:
        return filtered
    scale = total_before / total_after
    return {k: round(v * scale) for k, v in filtered.items()}
