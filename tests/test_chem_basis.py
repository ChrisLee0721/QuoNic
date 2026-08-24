"""Basis set registry tests."""

from __future__ import annotations

import pytest

from quonic.chem import list_bases, validate_basis


def test_validate_basis_sto3g():
    assert validate_basis("sto-3g") == "sto-3g"


def test_validate_basis_case_insensitive():
    assert validate_basis("STO-3G") == "sto-3g"
    assert validate_basis("Cc-PvDz") == "cc-pvdz"


def test_validate_basis_unknown():
    with pytest.raises(ValueError, match="Unknown basis"):
        validate_basis("made-up-basis")


def test_list_bases():
    bases = list_bases()
    assert len(bases) > 10
    assert "sto-3g" in bases
    assert "cc-pvdz" in bases
    assert bases == sorted(bases)
