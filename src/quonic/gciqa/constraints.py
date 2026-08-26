"""Geometric constraint encoding for GCIQA.

Defines constraints on molecular geometry that the Grover oracle
uses to mark valid conformations. Constraints are purely geometric
(no energy values).

Example::

    constraints = ConstraintSet([
        GeometricConstraint.bond("C1", "N2", min_dist=1.3, max_dist=1.5),
        GeometricConstraint.angle("C1", "N2", "C3", min_deg=100, max_deg=140),
        GeometricConstraint.pocket(center=(10, 20, 30), radius=8.0),
        GeometricConstraint.no_clash("C1", "O2", min_dist=2.5),
    ])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConstraintType(Enum):
    """Types of geometric constraints."""
    BOND = "bond"
    ANGLE = "angle"
    DIHEDRAL = "dihedral"
    POCKET = "pocket"
    NO_CLASH = "no_clash"
    SURFACE_DIST = "surface_dist"
    HYDROGEN_BOND = "hydrogen_bond"


@dataclass(frozen=True)
class GeometricConstraint:
    """A single geometric constraint on molecular structure.

    Attributes:
        type: Type of constraint.
        atoms: Atom indices or names involved.
        params: Constraint parameters (min/max distances, angles, etc).
        weight: Priority weight (higher = more important).
    """

    type: ConstraintType
    atoms: tuple[str, ...]
    params: dict[str, float]
    weight: float = 1.0

    @classmethod
    def bond(
        cls,
        atom1: str,
        atom2: str,
        min_dist: float,
        max_dist: float,
        weight: float = 1.0,
    ) -> GeometricConstraint:
        """Distance constraint between two atoms.

        Args:
            atom1: First atom name/index.
            atom2: Second atom name/index.
            min_dist: Minimum distance (Angstrom).
            max_dist: Maximum distance (Angstrom).
            weight: Priority weight.
        """
        return cls(
            type=ConstraintType.BOND,
            atoms=(atom1, atom2),
            params={"min_dist": min_dist, "max_dist": max_dist},
            weight=weight,
        )

    @classmethod
    def angle(
        cls,
        atom1: str,
        atom2: str,
        atom3: str,
        min_deg: float,
        max_deg: float,
        weight: float = 1.0,
    ) -> GeometricConstraint:
        """Angle constraint between three atoms.

        Args:
            atom1: First atom.
            atom2: Central atom.
            atom3: Third atom.
            min_deg: Minimum angle (degrees).
            max_deg: Maximum angle (degrees).
            weight: Priority weight.
        """
        return cls(
            type=ConstraintType.ANGLE,
            atoms=(atom1, atom2, atom3),
            params={"min_deg": min_deg, "max_deg": max_deg},
            weight=weight,
        )

    @classmethod
    def dihedral(
        cls,
        atom1: str,
        atom2: str,
        atom3: str,
        atom4: str,
        min_deg: float,
        max_deg: float,
        weight: float = 1.0,
    ) -> GeometricConstraint:
        """Dihedral angle constraint between four atoms.

        Args:
            atom1-atom4: The four atoms defining the dihedral.
            min_deg: Minimum dihedral (degrees).
            max_deg: Maximum dihedral (degrees).
            weight: Priority weight.
        """
        return cls(
            type=ConstraintType.DIHEDRAL,
            atoms=(atom1, atom2, atom3, atom4),
            params={"min_deg": min_deg, "max_deg": max_deg},
            weight=weight,
        )

    @classmethod
    def pocket(
        cls,
        center: tuple[float, float, float],
        radius: float,
        atom: str = "*",
        weight: float = 1.0,
    ) -> GeometricConstraint:
        """Constraint: atom must be within a spherical pocket.

        Args:
            center: Pocket center (x, y, z) in Angstrom.
            radius: Pocket radius in Angstrom.
            atom: Atom name/index, or "*" for all atoms.
            weight: Priority weight.
        """
        return cls(
            type=ConstraintType.POCKET,
            atoms=(atom,),
            params={
                "cx": center[0], "cy": center[1], "cz": center[2],
                "radius": radius,
            },
            weight=weight,
        )

    @classmethod
    def no_clash(
        cls,
        atom1: str,
        atom2: str,
        min_dist: float = 2.0,
        weight: float = 1.0,
    ) -> GeometricConstraint:
        """Steric clash avoidance: atoms must be farther than min_dist.

        Args:
            atom1: First atom.
            atom2: Second atom.
            min_dist: Minimum allowed distance (Angstrom).
            weight: Priority weight.
        """
        return cls(
            type=ConstraintType.NO_CLASH,
            atoms=(atom1, atom2),
            params={"min_dist": min_dist},
            weight=weight,
        )

    @classmethod
    def surface_distance(
        cls,
        atom: str,
        receptor_surface: Any,
        min_dist: float = 0.0,
        max_dist: float = 5.0,
        weight: float = 1.0,
    ) -> GeometricConstraint:
        """Distance from atom to receptor surface.

        Args:
            atom: Ligand atom name/index.
            receptor_surface: Surface representation (mesh/point cloud).
            min_dist: Minimum distance to surface.
            max_dist: Maximum distance to surface.
            weight: Priority weight.
        """
        return cls(
            type=ConstraintType.SURFACE_DIST,
            atoms=(atom,),
            params={"min_dist": min_dist, "max_dist": max_dist},
            weight=weight,
        )

    @classmethod
    def hydrogen_bond(
        cls,
        donor: str,
        acceptor: str,
        max_dist: float = 3.5,
        min_angle: float = 120.0,
        weight: float = 1.0,
    ) -> GeometricConstraint:
        """Hydrogen bond geometry constraint.

        Args:
            donor: Donor atom name/index.
            acceptor: Acceptor atom name/index.
            max_dist: Max donor-acceptor distance (Angstrom).
            min_angle: Min D-H...A angle (degrees).
            weight: Priority weight.
        """
        return cls(
            type=ConstraintType.HYDROGEN_BOND,
            atoms=(donor, acceptor),
            params={"max_dist": max_dist, "min_angle": min_angle},
            weight=weight,
        )

    def evaluate(self, coordinates: dict[str, tuple[float, float, float]]) -> bool:
        """Check if a conformation satisfies this constraint.

        Args:
            coordinates: Dict mapping atom names to (x, y, z) positions.

        Returns:
            True if constraint is satisfied.
        """
        import math

        if self.type == ConstraintType.BOND:
            a1, a2 = self.atoms
            if a1 not in coordinates or a2 not in coordinates:
                return False
            d = _distance(coordinates[a1], coordinates[a2])
            return self.params["min_dist"] <= d <= self.params["max_dist"]

        elif self.type == ConstraintType.ANGLE:
            a1, a2, a3 = self.atoms
            if not all(a in coordinates for a in (a1, a2, a3)):
                return False
            angle = _angle(coordinates[a1], coordinates[a2], coordinates[a3])
            return self.params["min_deg"] <= angle <= self.params["max_deg"]

        elif self.type == ConstraintType.DIHEDRAL:
            a1, a2, a3, a4 = self.atoms
            if not all(a in coordinates for a in (a1, a2, a3, a4)):
                return False
            dih = _dihedral(
                coordinates[a1], coordinates[a2],
                coordinates[a3], coordinates[a4],
            )
            return self.params["min_deg"] <= dih <= self.params["max_deg"]

        elif self.type == ConstraintType.POCKET:
            cx = (self.params["cx"], self.params["cy"], self.params["cz"])
            r = self.params["radius"]
            atom = self.atoms[0]
            if atom == "*":
                return all(
                    _distance(pos, cx) <= r for pos in coordinates.values()
                )
            if atom not in coordinates:
                return False
            return _distance(coordinates[atom], cx) <= r

        elif self.type == ConstraintType.NO_CLASH:
            a1, a2 = self.atoms
            if a1 not in coordinates or a2 not in coordinates:
                return False
            d = _distance(coordinates[a1], coordinates[a2])
            return d >= self.params["min_dist"]

        elif self.type == ConstraintType.HYDROGEN_BOND:
            # Simplified: just check distance
            donor, acceptor = self.atoms
            if donor not in coordinates or acceptor not in coordinates:
                return False
            d = _distance(coordinates[donor], coordinates[acceptor])
            return d <= self.params["max_dist"]

        return True


@dataclass
class ConstraintSet:
    """A collection of geometric constraints.

    Attributes:
        constraints: List of constraints.
    """

    constraints: list[GeometricConstraint] = field(default_factory=list)

    def add(self, constraint: GeometricConstraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)

    def evaluate(self, coordinates: dict[str, tuple[float, float, float]]) -> tuple[bool, float]:
        """Evaluate all constraints on a conformation.

        Args:
            coordinates: Atom positions.

        Returns:
            Tuple of (all_satisfied, weighted_score).
            Score = sum of weights for satisfied constraints.
        """
        satisfied = []
        score = 0.0
        for c in self.constraints:
            ok = c.evaluate(coordinates)
            satisfied.append(ok)
            if ok:
                score += c.weight
        return all(satisfied), score

    def __len__(self) -> int:
        return len(self.constraints)

    def __iter__(self):
        return iter(self.constraints)


def _distance(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    """Euclidean distance between two points."""
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2) ** 0.5


def _angle(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    """Angle p1-p2-p3 in degrees."""
    import math
    v1 = (p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])
    v2 = (p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2])
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    n1 = (v1[0]**2 + v1[1]**2 + v1[2]**2) ** 0.5
    n2 = (v2[0]**2 + v2[1]**2 + v2[2]**2) ** 0.5
    if n1 * n2 < 1e-12:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def _dihedral(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
    p4: tuple[float, float, float],
) -> float:
    """Dihedral angle p1-p2-p3-p4 in degrees."""
    import math

    def _sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    def _cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def _dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def _norm(a):
        return (a[0]**2 + a[1]**2 + a[2]**2) ** 0.5

    b1 = _sub(p2, p1)
    b2 = _sub(p3, p2)
    b3 = _sub(p4, p3)

    n1 = _cross(b1, b2)
    n2 = _cross(b2, b3)

    n1_norm = _norm(n1)
    n2_norm = _norm(n2)
    b2_norm = _norm(b2)

    if n1_norm < 1e-12 or n2_norm < 1e-12 or b2_norm < 1e-12:
        return 0.0

    cos_d = _dot(n1, n2) / (n1_norm * n2_norm)
    cos_d = max(-1.0, min(1.0, cos_d))
    angle = math.degrees(math.acos(cos_d))

    # Sign
    sign = _dot(_cross(n1, n2), b2)
    if sign < 0:
        angle = -angle

    return angle
