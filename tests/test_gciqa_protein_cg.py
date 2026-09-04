"""Tests for GCIQA protein coarse-graining."""


from gciqa.coarsegrain_adapters import CoarseGrainingStrategy
from gciqa.pdb import MetalIon, ResidueInfo
from gciqa.protein_cg import ProteinCoarseGraining


class TestProteinCoarseGrainingInterface:
    def test_is_strategy(self):
        strategy = ProteinCoarseGraining()
        assert isinstance(strategy, CoarseGrainingStrategy)


class TestProteinCoarseGraining:
    def _make_simple_protein(self):
        """Create a simple protein with 2 residues and 1 metal ion."""
        atoms = ["N", "C", "O", "N", "C", "O", "ZN"]
        coords = [
            (0.0, 0.0, 0.0),  # Res 1
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),  # Res 2
            (6.0, 0.0, 0.0),
            (7.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),  # Metal
        ]
        residues = [
            ResidueInfo(name="ALA", number=1, chain="A", atom_indices=[0, 1, 2]),
            ResidueInfo(name="GLY", number=2, chain="A", atom_indices=[3, 4, 5]),
        ]
        metal_ions = [
            MetalIon(element="ZN", coord=(3.0, 0.0, 0.0), index=6),
        ]
        return atoms, coords, residues, metal_ions

    def test_basic_coarse_grain(self):
        atoms, coords, residues, metal_ions = self._make_simple_protein()
        strategy = ProteinCoarseGraining()
        cg = strategy.coarse_grain(atoms, coords, residues=residues, metal_ions=metal_ions)

        # Should have super-atoms (residues + metal)
        assert cg.n_super_atoms >= 2  # At least 2 residues
        assert cg.n_full_atoms == 7

    def test_metal_preserved(self):
        atoms, coords, residues, metal_ions = self._make_simple_protein()
        strategy = ProteinCoarseGraining()
        cg = strategy.coarse_grain(atoms, coords, residues=residues, metal_ions=metal_ions)

        # Metal ion (index 6) should be in its own super-atom
        metal_super = cg.atom_to_super[6]
        # The metal's super-atom should contain only the metal
        assert cg.super_to_atoms[metal_super] == [6]

    def test_coordinating_residue_separate(self):
        """Residues coordinating a metal should be separate super-atoms."""
        # Zn at (3,0,0), His NE2 at (2.5,0,0) — within coordination distance
        atoms = ["N", "C", "NE2", "N", "C", "O", "ZN"]
        coords = [
            (0.0, 0.0, 0.0),  # Res 1 (His)
            (1.0, 0.0, 0.0),
            (2.5, 0.0, 0.0),  # Coordinating atom
            (5.0, 0.0, 0.0),  # Res 2 (not coordinating)
            (6.0, 0.0, 0.0),
            (7.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),  # Metal
        ]
        residues = [
            ResidueInfo(name="HIS", number=1, chain="A", atom_indices=[0, 1, 2]),
            ResidueInfo(name="ALA", number=2, chain="A", atom_indices=[3, 4, 5]),
        ]
        metal_ions = [MetalIon(element="ZN", coord=(3.0, 0.0, 0.0), index=6)]

        strategy = ProteinCoarseGraining(metal_coordination_dist=2.5)
        cg = strategy.coarse_grain(atoms, coords, residues=residues, metal_ions=metal_ions)

        # His should be a separate super-atom (coordinating residue)
        his_super = cg.atom_to_super[0]
        ala_super = cg.atom_to_super[3]
        assert his_super != ala_super

    def test_no_residues_fallback(self):
        """Without residue info, each atom becomes its own super-atom."""
        atoms = ["C", "N", "O"]
        coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        strategy = ProteinCoarseGraining()
        cg = strategy.coarse_grain(atoms, coords)
        assert cg.n_super_atoms == 3

    def test_water_merge(self):
        """Water molecules should merge into nearby residues."""
        atoms = ["N", "C", "O", "O", "H", "H"]
        coords = [
            (0.0, 0.0, 0.0),  # Res 1
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.5, 1.0, 0.0),  # Water (near res 1)
            (2.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ]
        residues = [
            ResidueInfo(name="ALA", number=1, chain="A", atom_indices=[0, 1, 2]),
            ResidueInfo(name="HOH", number=2, chain="A", atom_indices=[3, 4, 5]),
        ]
        strategy = ProteinCoarseGraining(water_merge_dist=3.5)
        cg = strategy.coarse_grain(atoms, coords, residues=residues)

        # Water should be merged into residue 1's super-atom
        res1_super = cg.atom_to_super[0]
        water_super = cg.atom_to_super[3]
        assert res1_super == water_super

    def test_preserve_sites_returns_cg(self):
        """preserve_sites should return a CoarseGraining (no-op for protein)."""
        atoms, coords, residues, metal_ions = self._make_simple_protein()
        strategy = ProteinCoarseGraining()
        cg = strategy.coarse_grain(atoms, coords, residues=residues, metal_ions=metal_ions)

        # preserve_sites is a no-op for protein (metals already preserved)
        cg2 = strategy.preserve_sites(cg, [(3.0, 0.0, 0.0)])
        assert cg2.n_super_atoms == cg.n_super_atoms
