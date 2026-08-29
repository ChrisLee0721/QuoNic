"""Metal coordination templates for GCIQA constraint generation.

Provides predefined geometric templates for common metal ion coordination
in proteins. These templates define expected coordination numbers, geometries,
and metal-ligand distances for generating GCIQA constraints.

Templates are通用 (general-purpose), not data-specific. They encode
well-established coordination chemistry knowledge from the PDB.

Usage:
    from gciqa.metal_templates import (
        get_metal_template,
        generate_metal_constraints,
        auto_detect_geometry,
    )

    # Get Zn²⁺ tetrahedral template
    template = get_metal_template("ZN", geometry="tetrahedral")

    # Generate constraints for a metal ion
    constraints = generate_metal_constraints(metal_ion, protein, template)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .constraints import ConstraintSet, GeometricConstraint

if TYPE_CHECKING:
    from .pdb import MetalIon, ProteinStructure


# Distance ranges: (min, max) in Angstroms
# Based on typical metal-ligand distances from PDB statistics
METAL_COORDINATION = {
    "ZN": {
        "tetrahedral": {
            "coordination_number": 4,
            "geometry": "tetrahedral",
            "distances": {
                "N": (1.9, 2.2),   # His NE2/ND1
                "O": (1.9, 2.2),   # Glu/Asp/water
                "S": (2.2, 2.5),   # Cys SG
            },
            "angles": {
                "N-Zn-N": (100, 120),
                "N-Zn-O": (100, 120),
                "O-Zn-O": (100, 120),
                "S-Zn-S": (100, 120),
                "N-Zn-S": (100, 120),
            },
        },
        "octahedral": {
            "coordination_number": 6,
            "geometry": "octahedral",
            "distances": {
                "N": (2.0, 2.3),
                "O": (2.0, 2.3),
                "S": (2.3, 2.6),
            },
            "angles": {
                "N-Zn-N": (85, 95),
                "N-Zn-O": (85, 95),
                "O-Zn-O": (85, 95),
            },
        },
    },
    "FE": {
        "octahedral": {
            "coordination_number": 6,
            "geometry": "octahedral",
            "distances": {
                "N": (1.9, 2.2),   # His
                "O": (1.9, 2.2),   # Glu/Asp/water
                "S": (2.2, 2.5),   # Cys/Met
            },
            "angles": {
                "N-Fe-N": (85, 95),
                "N-Fe-O": (85, 95),
                "O-Fe-O": (85, 95),
            },
        },
        "tetrahedral": {
            "coordination_number": 4,
            "geometry": "tetrahedral",
            "distances": {
                "N": (1.9, 2.2),
                "O": (1.9, 2.2),
                "S": (2.2, 2.5),
            },
            "angles": {
                "N-Fe-N": (100, 120),
                "N-Fe-S": (100, 120),
                "S-Fe-S": (100, 120),
            },
        },
    },
    "CU": {
        "tetrahedral": {
            "coordination_number": 4,
            "geometry": "tetrahedral",
            "distances": {
                "N": (1.9, 2.2),
                "O": (1.9, 2.2),
                "S": (2.2, 2.5),
            },
            "angles": {
                "N-Cu-N": (100, 120),
                "N-Cu-O": (100, 120),
                "O-Cu-O": (100, 120),
            },
        },
        "linear": {
            "coordination_number": 2,
            "geometry": "linear",
            "distances": {
                "N": (1.8, 2.1),
                "O": (1.8, 2.1),
                "S": (2.1, 2.4),
            },
            "angles": {
                "N-Cu-N": (170, 180),
                "N-Cu-S": (170, 180),
            },
        },
    },
    "MG": {
        "octahedral": {
            "coordination_number": 6,
            "geometry": "octahedral",
            "distances": {
                "O": (1.9, 2.3),   # Water, Glu, Asp
                "N": (2.0, 2.4),
            },
            "angles": {
                "O-Mg-O": (85, 95),
                "N-Mg-O": (85, 95),
            },
        },
    },
    "CA": {
        "pentagonal_bipyramidal": {
            "coordination_number": 7,
            "geometry": "pentagonal_bipyramidal",
            "distances": {
                "O": (2.2, 2.7),   # Water, Glu, Asp, backbone
                "N": (2.4, 2.8),
            },
            "angles": {
                "O-Ca-O": (65, 75),   # Pentagon
                "O-Ca-O": (140, 180), # Axial
            },
        },
        "octahedral": {
            "coordination_number": 6,
            "geometry": "octahedral",
            "distances": {
                "O": (2.2, 2.6),
                "N": (2.4, 2.8),
            },
            "angles": {
                "O-Ca-O": (85, 95),
            },
        },
    },
    "MN": {
        "octahedral": {
            "coordination_number": 6,
            "geometry": "octahedral",
            "distances": {
                "N": (2.0, 2.3),
                "O": (2.0, 2.3),
                "S": (2.3, 2.6),
            },
            "angles": {
                "N-Mn-N": (85, 95),
                "N-Mn-O": (85, 95),
                "O-Mn-O": (85, 95),
            },
        },
    },
    "CO": {
        "octahedral": {
            "coordination_number": 6,
            "geometry": "octahedral",
            "distances": {
                "N": (1.9, 2.2),
                "O": (1.9, 2.2),
                "S": (2.2, 2.5),
            },
            "angles": {
                "N-Co-N": (85, 95),
                "N-Co-O": (85, 95),
            },
        },
    },
    "NI": {
        "octahedral": {
            "coordination_number": 6,
            "geometry": "octahedral",
            "distances": {
                "N": (1.9, 2.2),
                "O": (1.9, 2.2),
                "S": (2.2, 2.5),
            },
            "angles": {
                "N-Ni-N": (85, 95),
                "N-Ni-O": (85, 95),
            },
        },
        "square_planar": {
            "coordination_number": 4,
            "geometry": "square_planar",
            "distances": {
                "N": (1.8, 2.1),
                "O": (1.8, 2.1),
                "S": (2.1, 2.4),
            },
            "angles": {
                "N-Ni-N": (85, 95),
                "N-Ni-O": (85, 95),
            },
        },
    },
}

# Default geometry for each metal (most common in proteins)
DEFAULT_GEOMETRY = {
    "ZN": "tetrahedral",
    "FE": "octahedral",
    "CU": "tetrahedral",
    "MG": "octahedral",
    "CA": "pentagonal_bipyramidal",
    "MN": "octahedral",
    "CO": "octahedral",
    "NI": "octahedral",
}


@dataclass
class MetalTemplate:
    """A metal coordination template."""
    element: str
    geometry: str
    coordination_number: int
    distances: dict[str, tuple[float, float]]
    angles: dict[str, tuple[float, float]]


def get_metal_template(element: str, geometry: str = "auto") -> MetalTemplate:
    """Get a metal coordination template.

    Args:
        element: Metal element symbol (e.g., "ZN", "FE")
        geometry: Coordination geometry ("tetrahedral", "octahedral", etc.)
                  If "auto", uses the default geometry for the metal.

    Returns:
        MetalTemplate with coordination parameters

    Raises:
        ValueError: If the metal or geometry is not supported
    """
    element = element.upper()
    if element not in METAL_COORDINATION:
        raise ValueError(
            f"Unsupported metal: {element}. "
            f"Supported: {list(METAL_COORDINATION.keys())}"
        )

    if geometry == "auto":
        geometry = DEFAULT_GEOMETRY.get(element, "octahedral")

    templates = METAL_COORDINATION[element]
    if geometry not in templates:
        # Fall back to default geometry for this metal
        geometry = DEFAULT_GEOMETRY.get(element, "octahedral")
        if geometry not in templates:
            available = list(templates.keys())
            raise ValueError(
                f"Unsupported geometry '{geometry}' for {element}. "
                f"Available: {available}"
            )

    params = templates[geometry]
    return MetalTemplate(
        element=element,
        geometry=geometry,
        coordination_number=params["coordination_number"],
        distances=params["distances"],
        angles=params["angles"],
    )


def auto_detect_geometry(
    metal_ion: MetalIon,
    protein: ProteinStructure,
    max_dist: float = 2.5,
) -> str:
    """Auto-detect coordination geometry from nearby ligands.

    Examines atoms within coordination distance of the metal ion
    and infers the most likely coordination geometry.

    Args:
        metal_ion: The metal ion to analyze
        protein: The protein structure
        max_dist: Maximum metal-ligand distance (Å)

    Returns:
        Detected geometry string (e.g., "tetrahedral", "octahedral")
    """
    # Find coordinating atoms
    coord_atoms = []
    for i, (x, y, z) in enumerate(protein.coords):
        if i == metal_ion.index:
            continue
        dist = math.sqrt(
            (x - metal_ion.coord[0]) ** 2
            + (y - metal_ion.coord[1]) ** 2
            + (z - metal_ion.coord[2]) ** 2
        )
        if dist <= max_dist:
            coord_atoms.append(i)

    n_coord = len(coord_atoms)

    # Infer geometry from coordination number
    if n_coord == 0:
        return DEFAULT_GEOMETRY.get(metal_ion.element.upper(), "octahedral")
    if n_coord <= 2:
        return "linear"
    elif n_coord == 3:
        return "trigonal"
    elif n_coord == 4:
        # Could be tetrahedral or square planar
        # Check angles to distinguish
        if len(coord_atoms) >= 4:
            angles = _compute_coordination_angles(metal_ion, protein, coord_atoms[:4])
            avg_angle = sum(angles) / len(angles) if angles else 109.5
            if avg_angle < 100:
                return "square_planar"
        return "tetrahedral"
    elif n_coord == 5:
        return "trigonal_bipyramidal"
    elif n_coord == 6:
        return "octahedral"
    elif n_coord == 7:
        return "pentagonal_bipyramidal"
    else:
        return "octahedral"  # Default fallback


def _compute_coordination_angles(
    metal_ion: MetalIon,
    protein: ProteinStructure,
    coord_atoms: list[int],
) -> list[float]:
    """Compute angles between coordinating atoms at the metal center."""
    angles = []
    mx, my, mz = metal_ion.coord

    for i in range(len(coord_atoms)):
        for j in range(i + 1, len(coord_atoms)):
            ax, ay, az = protein.coords[coord_atoms[i]]
            bx, by, bz = protein.coords[coord_atoms[j]]

            # Vectors from metal to ligands
            v1 = (ax - mx, ay - my, az - mz)
            v2 = (bx - mx, by - my, bz - mz)

            # Angle between vectors
            dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
            norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
            norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)

    return angles


def generate_metal_constraints(
    metal_ion: MetalIon,
    protein: ProteinStructure,
    template: MetalTemplate,
    max_dist: float = 2.5,
) -> ConstraintSet:
    """Generate GCIQA constraints for a metal coordination site.

    Creates bond constraints between the metal ion and its coordinating
    atoms based on the coordination template.

    Args:
        metal_ion: The metal ion
        protein: The protein structure
        template: Coordination template with distance/angle parameters
        max_dist: Maximum metal-ligand distance for finding coordinators

    Returns:
        ConstraintSet with bond constraints for the coordination site
    """
    constraints = []

    # Find coordinating atoms
    coord_atoms = []
    for i, (x, y, z) in enumerate(protein.coords):
        if i == metal_ion.index:
            continue
        dist = math.sqrt(
            (x - metal_ion.coord[0]) ** 2
            + (y - metal_ion.coord[1]) ** 2
            + (z - metal_ion.coord[2]) ** 2
        )
        if dist <= max_dist:
            element = protein.atoms[i]
            coord_atoms.append((i, element))

    # Generate bond constraints for each coordinating atom
    metal_key = str(metal_ion.index)
    for atom_idx, element in coord_atoms:
        atom_key = str(atom_idx)
        element_upper = element.upper()

        # Get distance range for this element type
        if element_upper in template.distances:
            min_dist, max_dist_range = template.distances[element_upper]
        else:
            # Use a generic range if element not in template
            min_dist, max_dist_range = (1.8, 2.5)

        constraints.append(
            GeometricConstraint.bond(
                metal_key, atom_key,
                min_dist=min_dist,
                max_dist=max_dist_range,
            )
        )

    return ConstraintSet(constraints)


def get_available_metals() -> list[str]:
    """Get list of supported metal elements."""
    return list(METAL_COORDINATION.keys())


def get_available_geometries(element: str) -> list[str]:
    """Get list of available geometries for a metal element."""
    element = element.upper()
    if element not in METAL_COORDINATION:
        return []
    return list(METAL_COORDINATION[element].keys())
