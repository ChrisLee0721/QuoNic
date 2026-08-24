"""Symmetry verification tests."""

from __future__ import annotations

from quonic.mitigation import symmetry_verify, symmetry_verify_and_renormalize


def test_particle_number_filter():
    counts = {"000": 100, "011": 200, "101": 300, "110": 400, "111": 500}
    result = symmetry_verify(counts, particle_number=2)
    assert set(result.keys()) == {"011", "101", "110"}
    assert result["011"] == 200


def test_parity_even():
    counts = {"00": 100, "01": 200, "10": 300, "11": 400}
    result = symmetry_verify(counts, parity="even")
    assert set(result.keys()) == {"00", "11"}


def test_parity_odd():
    counts = {"00": 100, "01": 200, "10": 300, "11": 400}
    result = symmetry_verify(counts, parity="odd")
    assert set(result.keys()) == {"01", "10"}


def test_custom_filter():
    counts = {"000": 100, "001": 200, "010": 300, "100": 400}
    result = symmetry_verify(counts, custom=lambda b: b.startswith("0"))
    assert set(result.keys()) == {"000", "001", "010"}


def test_combined_filters():
    counts = {"000": 100, "011": 200, "101": 300, "111": 400}
    result = symmetry_verify(counts, particle_number=2, parity="even")
    assert set(result.keys()) == {"011", "101"}


def test_empty_result():
    counts = {"00": 100, "11": 200}
    result = symmetry_verify(counts, particle_number=3)
    assert result == {}


def test_renormalize():
    counts = {"00": 100, "01": 200, "10": 300, "11": 400}
    result = symmetry_verify_and_renormalize(counts, parity="even")
    total = sum(result.values())
    assert total == 1000  # original total preserved
