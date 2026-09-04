"""General ligand detection using spatial hashing + Union-Find.

Pure geometry + graph theory. No chemistry knowledge needed.

Algorithm:
1. Build molecular graph: atoms within 2.0A → covalent bond (edge)
2. Find connected components: Union-Find
3. Largest component = protein body
4. Other components with ≥3 atoms = ligands
5. Single atoms = water/metal (excluded)
6. Ligand centroid = pocket center

Level 2 refinement:
- Dense probe grid (0.5A) around coarse pocket center
- Coordination tendency score = f(density, coordination_atoms, hydrophobicity)
- Data-driven thresholds from distance distribution

Complexity: O(N) with spatial hashing.

Example::

    from gciqa.ligand_detect import detect_ligands, refine_pocket

    ligands = detect_ligands(atoms, coords)
    for lig in ligands:
        refined = refine_pocket(lig.centroid, atoms, coords, residue_ids)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

# Physical constants (not tunable)
_COVALENT_BOND_THRESHOLD = 2.0  # Angstrom — max covalent bond length
_MIN_LIGAND_ATOMS = 3            # Minimum atoms to be a ligand (not water/metal)
_MAX_LIGAND_ATOMS = 50           # Maximum atoms (exclude oligomers/nucleic acids)
_GRID_CELL_SIZE = _COVALENT_BOND_THRESHOLD  # Spatial hash grid cell size

# Coordination geometry constants (from chemistry, not tunable)
_COORD_ELEMENTS = {"N", "O", "S"}  # Common coordinating atoms

# Level 2 refinement parameters (derived from grid resolution)
_REFINE_GRID_SPACING = 0.5  # Angstrom — probe point spacing
_REFINE_BOX_HALF = 2.0      # Angstrom — half-size of refinement box


@dataclass
class Ligand:
    """Detected ligand.

    Attributes:
        atom_indices: Indices of atoms in the ligand.
        centroid: Center of mass (geometric centroid).
        n_atoms: Number of atoms.
    """

    atom_indices: list[int]
    centroid: tuple[float, float, float]
    n_atoms: int


def detect_ligands(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
) -> list[Ligand]:
    """Detect ligands in a protein structure.

    Pure geometry + graph theory. No chemistry knowledge needed.

    Args:
        atoms: Element symbols (e.g., ["C", "N", "O", ...]).
        coords: Atomic coordinates [(x, y, z), ...].

    Returns:
        List of detected Ligands, sorted by size (largest first).
    """
    n = len(atoms)
    if n == 0:
        return []

    # Step 1: Build molecular graph using spatial hashing
    # O(N) — each atom checks only neighboring grid cells
    parent = list(range(n))
    rank = [0] * n

    # Spatial hashing: assign each atom to a grid cell
    cell_map: dict[tuple[int, int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(coords):
        cx = int(x / _GRID_CELL_SIZE)
        cy = int(y / _GRID_CELL_SIZE)
        cz = int(z / _GRID_CELL_SIZE)
        cell_map.setdefault((cx, cy, cz), []).append(i)

    # For each atom, check neighboring 3x3x3 cells
    threshold_sq = _COVALENT_BOND_THRESHOLD ** 2
    for (cx, cy, cz), atoms_in_cell in cell_map.items():
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    neighbor_cell = (cx + dx, cy + dy, cz + dz)
                    if neighbor_cell not in cell_map:
                        continue
                    for i in atoms_in_cell:
                        for j in cell_map[neighbor_cell]:
                            if j <= i:
                                continue  # Avoid duplicate pairs
                            xi, yi, zi = coords[i]
                            xj, yj, zj = coords[j]
                            dist_sq = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2
                            if dist_sq < threshold_sq:
                                _union(parent, rank, i, j)

    # Step 2: Find connected components
    # O(N α(N)) ≈ O(N)
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(parent, i)
        components.setdefault(root, []).append(i)

    # Step 3: Largest component = protein body
    comp_sizes = [(len(indices), root, indices) for root, indices in components.items()]
    comp_sizes.sort(reverse=True)

    if not comp_sizes:
        return []

    protein_indices = comp_sizes[0][2]
    set(protein_indices)

    # Step 4: Find ligands (non-protein components with ≥3 atoms)
    ligands = []
    for size, root, indices in comp_sizes[1:]:
        if size < _MIN_LIGAND_ATOMS:
            continue  # Water/metal (single atoms or small ions)
        if size > _MAX_LIGAND_ATOMS:
            continue  # Oligomers/nucleic acids

        # Compute centroid
        cx = sum(coords[i][0] for i in indices) / size
        cy = sum(coords[i][1] for i in indices) / size
        cz = sum(coords[i][2] for i in indices) / size

        ligands.append(Ligand(
            atom_indices=indices,
            centroid=(cx, cy, cz),
            n_atoms=size,
        ))

    # Sort by size (largest first)
    ligands.sort(key=lambda lig: lig.n_atoms, reverse=True)
    return ligands



def detect_pockets(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Detect binding pockets using data-driven coordination shell detection.

    No hardcoded metal-ligand distance. No geometry assumptions.
    Only uses: covalent bond threshold (2.0A), coordination elements (N/O/S).

    Algorithm:
    1. Find N/O/S triplets (ligand-ligand pairs within 2*bond_threshold)
    2. For each triplet, sweep sphere radius from bond_threshold/2 to bond_threshold
    3. At each sphere intersection point, collect N/O/S distance distribution
    4. Score: peak in distance histogram = coordination shell
    5. Peak position IS the metal-ligand distance (derived, not assumed)

    Args:
        atoms: Element symbols.
        coords: Atomic coordinates.

    Returns:
        List of pocket centers (x, y, z), sorted by quality.
    """
    from collections import defaultdict

    n = len(atoms)
    if n == 0:
        return []

    nos_indices = [i for i, a in enumerate(atoms) if a in _COORD_ELEMENTS]
    if len(nos_indices) < 3:
        return []

    # Spatial hash for N/O/S atoms
    cell_size = _COVALENT_BOND_THRESHOLD * 2  # 4.0A
    nos_cells: dict[tuple[int, int, int], list[int]] = {}
    for i in nos_indices:
        x, y, z = coords[i]
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        cz = int(z / cell_size)
        nos_cells.setdefault((cx, cy, cz), []).append(i)

    # Find N/O/S pairs within 2*bond_threshold (ligand-ligand distance)
    # From triangle inequality: if M-L = d, then L-L <= 2d
    # So L-L <= 2*bond_threshold covers all possible coordination geometries
    ll_max = _COVALENT_BOND_THRESHOLD * 2  # 4.0A
    ll_max_sq = ll_max ** 2
    # Minimum L-L: must be > 0, use small value
    ll_min_sq = 1.0  # 1.0A minimum

    pairs = []
    for (cx, cy, cz), indices in nos_cells.items():
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cell = (cx + dx, cy + dy, cz + dz)
                    if cell not in nos_cells:
                        continue
                    for i in indices:
                        for j in nos_cells[cell]:
                            if j <= i:
                                continue
                            d_sq = sum((a - b) ** 2 for a, b in zip(coords[i], coords[j]))
                            if ll_min_sq <= d_sq <= ll_max_sq:
                                pairs.append((i, j))

    if not pairs:
        return []

    # Build adjacency and find triangles
    adj = defaultdict(set)
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)

    triangles = set()
    for i, j in pairs:
        common = adj[i] & adj[j]
        for k in common:
            if k <= i or k <= j:
                continue
            triangles.add(tuple(sorted([i, j, k])))

    if not triangles:
        return []

    # Sweep sphere radius: from bond_threshold/2 to bond_threshold
    # This covers all possible M-L distances (derived from bond topology)
    r_min = _COVALENT_BOND_THRESHOLD / 2.0  # 1.0A
    r_max = _COVALENT_BOND_THRESHOLD         # 2.0A
    r_steps = 10

    all_candidates = []  # (pt, score, derived_ml_dist)

    for step in range(r_steps):
        sphere_r = r_min + (r_max - r_min) * step / (r_steps - 1)

        for i, j, k in triangles:
            points = _sphere_intersection(coords[i], coords[j], coords[k], radius=sphere_r)
            for pt in points:
                # Collect N/O/S distance distribution
                nos_dists = []
                pcx = int(pt[0] / cell_size)
                pcy = int(pt[1] / cell_size)
                pcz = int(pt[2] / cell_size)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            cell = (pcx + dx, pcy + dy, pcz + dz)
                            if cell not in nos_cells:
                                continue
                            for idx in nos_cells[cell]:
                                d_sq = sum((a - b) ** 2 for a, b in zip(pt, coords[idx]))
                                nos_dists.append((math.sqrt(d_sq), idx))

                if len(nos_dists) < 3:
                    continue

                # Find coordination shell: cluster distances
                # Bin width derived from coordination-range distances only
                dists_only = [d for d, _ in nos_dists]
                dists_only.sort()

                # Adaptive bin width: use std dev of distances within
                # coordination range (bond_threshold/2 to bond_threshold*1.5)
                coord_min = _COVALENT_BOND_THRESHOLD / 2.0
                coord_max = _COVALENT_BOND_THRESHOLD * 1.5
                coord_range_dists = [d for d in dists_only if coord_min < d < coord_max]
                if len(coord_range_dists) > 1:
                    mean_d = sum(coord_range_dists) / len(coord_range_dists)
                    std_d = math.sqrt(sum((d - mean_d)**2 for d in coord_range_dists) / len(coord_range_dists))
                    bin_width = max(_COVALENT_BOND_THRESHOLD / 10.0, 2 * std_d)
                else:
                    bin_width = _COVALENT_BOND_THRESHOLD / 10.0

                # Find densest cluster of distances
                best_cluster_size = 0
                best_cluster_mean = 0
                best_cluster_indices = []

                for d in dists_only:
                    cluster = []
                    cluster_indices = []
                    for dist, idx in nos_dists:
                        if abs(dist - d) < bin_width:
                            cluster.append(dist)
                            cluster_indices.append(idx)
                    if len(cluster) > best_cluster_size:
                        best_cluster_size = len(cluster)
                        best_cluster_mean = sum(cluster) / len(cluster)
                        best_cluster_indices = list(cluster_indices)

                # Score: cluster size (more atoms at same distance = better)
                # This IS the coordination number, derived from data
                if best_cluster_size >= 3:
                    # Check for steric clashes: non-coord atoms too close
                    coord_set = set(best_cluster_indices)
                    min_clash_dist = _COVALENT_BOND_THRESHOLD / 2.0  # 1.0A
                    has_clash = False
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                cell = (pcx + dx, pcy + dy, pcz + dz)
                                if cell not in nos_cells:
                                    continue
                                for idx in nos_cells[cell]:
                                    if idx in coord_set:
                                        continue
                                    d_sq = sum((a - b) ** 2 for a, b in zip(pt, coords[idx]))
                                    if math.sqrt(d_sq) < min_clash_dist:
                                        has_clash = True

                    if not has_clash:
                        # Compute variance of coordination distances
                        coord_dists = [d for d, idx in nos_dists if idx in coord_set]
                        mean_d = sum(coord_dists) / len(coord_dists)
                        variance = sum((d - mean_d)**2 for d in coord_dists) / len(coord_dists)

                        # Score: cluster_size / (1 + variance)
                        score = best_cluster_size / (1.0 + variance * 10)
                        all_candidates.append((pt, score, best_cluster_mean))

    if not all_candidates:
        return []

    # Sort by score (higher = better)
    all_candidates.sort(key=lambda x: -x[1])

    # Deduplicate nearby candidates
    deduplicated = []
    for pt, score, ml_dist in all_candidates:
        is_dup = False
        for existing in deduplicated:
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pt, existing)))
            if d < 1.0:
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(pt)

    return deduplicated


def _sphere_intersection(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    radius: float = 2.0,
) -> list[tuple[float, float, float]]:
    """Find intersection of 3 spheres centered at A, B, C with given radius.

    Returns 0 or 2 points (the metal position candidates).
    """
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    r_sq = radius * radius

    # Vector from A to B
    abx, aby, abz = bx-ax, by-ay, bz-az
    d_ab = math.sqrt(abx*abx + aby*aby + abz*abz)

    if d_ab < 1e-10 or d_ab > 2*radius:
        return []

    # Unit vector from A to B
    ex = abx / d_ab
    ey = aby / d_ab
    ez = abz / d_ab

    # Distance from A to intersection circle plane
    x = d_ab / 2.0

    # Vector from A to C
    acx, acy, acz = cx-ax, cy-ay, cz-az

    # Project C onto AB line
    i = ex*acx + ey*acy + ez*acz

    # Vector from projection point to C
    jx = acx - i*ex
    jy = acy - i*ey
    jz = acz - i*ez
    j = math.sqrt(jx*jx + jy*jy + jz*jz)

    if j < 1e-10:
        return []

    # Unit vector in j direction
    jnx = jx / j
    jny = jy / j
    jnz = jz / j

    # Distance from intersection circle center to sphere-sphere intersection
    discriminant = r_sq - x*x
    if discriminant < 0:
        return []

    y = math.sqrt(discriminant)

    # Now find intersection with third sphere
    # Circle center in 3D: A + x*ex
    # Circle radius: y
    # Need: |P - C|² = r²
    # P = A + x*ex + y*(cos(t)*jnx + sin(t)*cross(ex,jnx))

    # Cross product ex × jnx
    cx_ = ey*jnz - ez*jny
    cy_ = ez*jnx - ex*jnz
    cz_ = ex*jny - ey*jnx

    # Solve for t: |A + x*ex + y*(cos(t)*jnx + sin(t)*cross) - C|² = r²
    # Let D = A + x*ex - C
    dx = ax + x*ex - cx
    dy = ay + x*ey - cy
    dz = az + x*ez - cz

    # |D + y*(cos(t)*jnx + sin(t)*cross)|² = r²
    # Expand: |D|² + 2y*(D·(cos(t)*jnx + sin(t)*cross)) + y² = r²
    # Let: D·jnx = a1, D·cross = a2
    a1 = dx*jnx + dy*jny + dz*jnz
    a2 = dx*cx_ + dy*cy_ + dz*cz_

    # |D|² + 2y*(a1*cos(t) + a2*sin(t)) + y² = r²
    d_sq = dx*dx + dy*dy + dz*dz
    # 2y*(a1*cos(t) + a2*sin(t)) = r² - y² - d_sq
    rhs = r_sq - y*y - d_sq

    # a1*cos(t) + a2*sin(t) = rhs / (2y)
    if abs(y) < 1e-10:
        return []

    k = rhs / (2*y)

    # a1*cos(t) + a2*sin(t) = k
    # This has solutions if k² ≤ a1² + a2²
    a1a2_sq = a1*a1 + a2*a2
    if k*k > a1a2_sq:
        return []

    # Solve: cos(t - phi) = k / sqrt(a1² + a2²)
    # where phi = atan2(a2, a1)
    phi = math.atan2(a2, a1)
    acos_val = k / math.sqrt(a1a2_sq)
    acos_val = max(-1.0, min(1.0, acos_val))
    delta = math.acos(acos_val)

    t1 = phi + delta
    t2 = phi - delta

    results = []
    for t in [t1, t2]:
        px = ax + x*ex + y*(math.cos(t)*jnx + math.sin(t)*cx_)
        py = ay + x*ey + y*(math.cos(t)*jny + math.sin(t)*cy_)
        pz = az + x*ez + y*(math.cos(t)*jnz + math.sin(t)*cz_)
        results.append((px, py, pz))

    return results


def _find(parent: list[int], i: int) -> int:
    """Find with path compression."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # Path compression
        i = parent[i]
    return i


def _union(parent: list[int], rank: list[int], i: int, j: int) -> None:
    """Union by rank."""
    ri = _find(parent, i)
    rj = _find(parent, j)
    if ri == rj:
        return
    if rank[ri] < rank[rj]:
        ri, rj = rj, ri
    parent[rj] = ri
    if rank[ri] == rank[rj]:
        rank[ri] += 1


def refine_pocket(
    coarse_center: tuple[float, float, float],
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    residue_ids: list[int] | None = None,
) -> tuple[float, float, float]:
    """Level 2 pocket refinement using coordination tendency score.

    Takes a coarse pocket center and refines it by scattering dense probe
    points (0.5A spacing) in a 20x20x20 A box around the center, then
    selecting the probe with the highest coordination tendency score.

    Score factors:
    - Low density (high weight): fewer protein atoms nearby = better pocket
    - Coordination atoms (high weight): N, O, S within 1.5-3.0A
    - Hydrophobic atoms (low weight): C atoms reduce score

    Args:
        coarse_center: Initial pocket center (x, y, z).
        atoms: Element symbols.
        coords: Atomic coordinates.
        residue_ids: Optional residue IDs for hydrophobicity estimation.

    Returns:
        Refined pocket center (x, y, z).
    """
    cx, cy, cz = coarse_center

    # Use wide coordination shell range for refinement
    # This covers all common metal-ligand distances (1.5-3.0A)
    coord_min = 1.5
    coord_max = 3.0
    coord_min ** 2
    coord_max_sq = coord_max ** 2

    # Spatial hash protein atoms
    cell_size = coord_max + 0.5
    atom_cells: dict[tuple[int, int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(coords):
        cell = (int(x / cell_size), int(y / cell_size), int(z / cell_size))
        atom_cells.setdefault(cell, []).append(i)

    # Scatter probe points in box around coarse center
    best_score = -float('inf')
    best_point = coarse_center
    half = _REFINE_BOX_HALF
    spacing = _REFINE_GRID_SPACING

    x = cx - half
    while x <= cx + half:
        y = cy - half
        while y <= cy + half:
            z = cz - half
            while z <= cz + half:
                px, py, pz = x, y, z
                pcx = int(px / cell_size)
                pcy = int(py / cell_size)
                pcz = int(pz / cell_size)

                # Collect distances, separated by atom type
                nos_dists = []   # N/O/S (potential coordinators)
                other_count = 0  # Non-N/O/S atoms in steric shell
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            cell = (pcx + dx, pcy + dy, pcz + dz)
                            if cell not in atom_cells:
                                continue
                            for idx in atom_cells[cell]:
                                ax, ay, az = coords[idx]
                                dist_sq = (px-ax)**2 + (py-ay)**2 + (pz-az)**2
                                if dist_sq < coord_max_sq:
                                    if atoms[idx] in _COORD_ELEMENTS:
                                        nos_dists.append(math.sqrt(dist_sq))
                                    elif dist_sq < 2.0 ** 2:  # Steric clash shell
                                        other_count += 1

                # Score: prefer tight N/O/S coordination at ~2.0A
                coord_score = 0.0
                for d in nos_dists:
                    deviation = abs(d - 2.0)
                    coord_score += math.exp(-(deviation / 0.3) ** 2)

                # Penalize steric clashes only (not coordination atoms)
                density_penalty = other_count * 0.5

                # Distance from coarse center (keep close)
                dist_from_coarse = math.sqrt(
                    (px-cx)**2 + (py-cy)**2 + (pz-cz)**2
                )

                score = coord_score - density_penalty - 0.5 * dist_from_coarse

                if score > best_score:
                    best_score = score
                    best_point = (px, py, pz)

                z += spacing
            y += spacing
        x += spacing

    return best_point


def geometric_pocket_volume(
    center: tuple[float, float, float],
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    ml_dist: float | None = None,
    tolerance: float = 0.3,
) -> dict:
    """Analyze the coordination shell using a geometric probe.

    Probe shape is derived from constraints, not assumed.
    If ml_dist is None, automatically derived from N/O/S distance clustering.

    Args:
        center: Pocket center (x, y, z).
        atoms: Element symbols.
        coords: Atomic coordinates.
        ml_dist: Metal-ligand distance. If None, derived from data.
        tolerance: Distance tolerance for matching (default 0.3A).

    Returns:
        Dict with coord_count, coord_indices, coord_dists,
        shell_free_frac, shell_volume, free_volume, derived_ml_dist.
    """
    cx, cy, cz = center

    # Spatial hash protein atoms
    hash_cell = _COVALENT_BOND_THRESHOLD * 2 + 2.0
    atom_cells: dict[tuple[int, int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(coords):
        cell = (int(x / hash_cell), int(y / hash_cell), int(z / hash_cell))
        atom_cells.setdefault(cell, []).append(i)

    # Collect all N/O/S distances from center
    nos_dists = []  # (distance, index)
    search_r = _COVALENT_BOND_THRESHOLD * 2  # 4.0A
    pcx = int(cx / hash_cell)
    pcy = int(cy / hash_cell)
    pcz = int(cz / hash_cell)
    for adx in range(-1, 2):
        for ady in range(-1, 2):
            for adz in range(-1, 2):
                cell = (pcx + adx, pcy + ady, pcz + adz)
                if cell not in atom_cells:
                    continue
                for idx in atom_cells[cell]:
                    if atoms[idx] not in _COORD_ELEMENTS:
                        continue
                    ax, ay, az = coords[idx]
                    d = math.sqrt((cx-ax)**2 + (cy-ay)**2 + (cz-az)**2)
                    if d < search_r:
                        nos_dists.append((d, idx))

    # Derive ml_dist from distance clustering if not given
    if ml_dist is None:
        if len(nos_dists) < 3:
            return {
                "coord_count": 0, "coord_indices": [], "coord_dists": [],
                "shell_free_frac": 0, "shell_volume": 0, "free_volume": 0,
                "derived_ml_dist": None,
            }
        # Find densest cluster of distances
        # Bin width derived from coordination-range distances only
        dists_only = sorted(d for d, _ in nos_dists)
        coord_min = _COVALENT_BOND_THRESHOLD / 2.0
        coord_max = _COVALENT_BOND_THRESHOLD * 1.5
        coord_range_dists = [d for d in dists_only if coord_min < d < coord_max]
        if len(coord_range_dists) > 1:
            mean_d = sum(coord_range_dists) / len(coord_range_dists)
            std_d = math.sqrt(sum((d - mean_d)**2 for d in coord_range_dists) / len(coord_range_dists))
            bin_width = max(_COVALENT_BOND_THRESHOLD / 10.0, 2 * std_d)
        else:
            bin_width = _COVALENT_BOND_THRESHOLD / 10.0

        best_count = 0
        best_center = dists_only[0]
        for d in dists_only:
            count = sum(1 for dd in dists_only if abs(dd - d) < bin_width)
            if count > best_count:
                best_count = count
                best_center = d
        ml_dist = best_center

    # Find coordination atoms (N/O/S within tolerance of ml_dist)
    coord_indices = []
    coord_dists = []
    coord_min_sq = (ml_dist - tolerance) ** 2
    coord_max_sq = (ml_dist + tolerance) ** 2

    for d, idx in nos_dists:
        d_sq = d * d
        if coord_min_sq <= d_sq <= coord_max_sq:
            coord_indices.append(idx)
            coord_dists.append(d)

    # Check angular coverage: how much of shell is blocked by non-coord atoms
    n_points = 200
    phi = (1 + math.sqrt(5)) / 2
    block_dist_sq = (tolerance * 2) ** 2

    coord_set = set(coord_indices)

    free_count = 0
    for i in range(n_points):
        y_pos = 1 - (2 * i / (n_points - 1))
        radius_at_y = math.sqrt(max(0, 1 - y_pos * y_pos))
        theta = 2 * math.pi * i / phi
        nx = radius_at_y * math.cos(theta)
        ny = y_pos
        nz = radius_at_y * math.sin(theta)

        px = cx + ml_dist * nx
        py = cy + ml_dist * ny
        pz = cz + ml_dist * nz

        pcx2 = int(px / hash_cell)
        pcy2 = int(py / hash_cell)
        pcz2 = int(pz / hash_cell)
        blocked = False
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                for adz in range(-1, 2):
                    cell = (pcx2 + adx, pcy2 + ady, pcz2 + adz)
                    if cell not in atom_cells:
                        continue
                    for idx in atom_cells[cell]:
                        if idx in coord_set:
                            continue
                        ax, ay, az = coords[idx]
                        d_sq = (px-ax)**2 + (py-ay)**2 + (pz-az)**2
                        if d_sq < block_dist_sq:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    break
            if blocked:
                break

        if not blocked:
            free_count += 1

    shell_free_frac = free_count / n_points
    shell_area = 4 * math.pi * ml_dist ** 2
    shell_thickness = 2 * tolerance
    shell_volume = shell_area * shell_thickness

    return {
        "coord_count": len(coord_indices),
        "coord_indices": coord_indices,
        "coord_dists": coord_dists,
        "shell_free_frac": shell_free_frac,
        "shell_volume": shell_volume,
        "free_volume": shell_free_frac * shell_volume,
        "derived_ml_dist": ml_dist,
    }


def pocket_volume(
    center: tuple[float, float, float],
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    probe_r: float = 0.5,
    shell_r: float = 4.0,
) -> float:
    """Calculate free volume around a point (volume of concavity).

    Scatters probe points on a grid in a sphere of radius shell_r around
    center. Counts points that don't collide with any protein atom.
    Free volume = count × cell_volume.

    Atomic radius derived from covalent bond threshold (not empirical):
    atom_r = _COVALENT_BOND_THRESHOLD / 2 = 1.0A.

    Collision: probe collides if dist(probe, atom) < probe_r + atom_r.

    Args:
        center: Pocket center (x, y, z).
        atoms: Element symbols.
        coords: Atomic coordinates.
        probe_r: Probe radius in Angstrom (grid spacing = 2*probe_r).
        shell_r: Shell radius to measure within.

    Returns:
        Free volume in cubic Angstrom.
    """
    cx, cy, cz = center
    spacing = 2.0 * probe_r
    cell_vol = spacing ** 3
    atom_r = _COVALENT_BOND_THRESHOLD / 2.0  # 1.0A — derived from bond topology

    # Spatial hash protein atoms
    hash_cell = shell_r + probe_r + atom_r + 1.0
    atom_cells: dict[tuple[int, int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(coords):
        cell = (int(x / hash_cell), int(y / hash_cell), int(z / hash_cell))
        atom_cells.setdefault(cell, []).append(i)

    clash_dist = probe_r + atom_r
    clash_dist_sq = clash_dist ** 2
    shell_r_sq = shell_r ** 2

    count = 0
    x = cx - shell_r
    while x <= cx + shell_r:
        y = cy - shell_r
        while y <= cy + shell_r:
            z = cz - shell_r
            while z <= cz + shell_r:
                dx, dy, dz = x - cx, y - cy, z - cz
                if dx*dx + dy*dy + dz*dz > shell_r_sq:
                    z += spacing
                    continue

                pcx = int(x / hash_cell)
                pcy = int(y / hash_cell)
                pcz = int(z / hash_cell)
                collision = False
                for adx in range(-1, 2):
                    for ady in range(-1, 2):
                        for adz in range(-1, 2):
                            cell = (pcx + adx, pcy + ady, pcz + adz)
                            if cell not in atom_cells:
                                continue
                            for idx in atom_cells[cell]:
                                ax, ay, az = coords[idx]
                                d_sq = (x-ax)**2 + (y-ay)**2 + (z-az)**2
                                if d_sq < clash_dist_sq:
                                    collision = True
                                    break
                            if collision:
                                break
                        if collision:
                            break
                    if collision:
                        break

                if not collision:
                    count += 1

                z += spacing
            y += spacing
        x += spacing

    return count * cell_vol


def can_accommodate(
    center: tuple[float, float, float],
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    target_r: float = 2.0,
) -> bool:
    """Check if a pocket can accommodate a sphere of given radius.

    Args:
        center: Pocket center (x, y, z).
        atoms: Element symbols.
        coords: Atomic coordinates.
        target_r: Target radius to accommodate (e.g., metal coordination radius).

    Returns:
        True if free volume >= target sphere volume.
    """
    free_vol = pocket_volume(center, atoms, coords, probe_r=0.5, shell_r=target_r + 1.0)
    target_vol = (4.0 / 3.0) * math.pi * target_r ** 3
    return free_vol >= target_vol


@dataclass
class SiteResult:
    """Result of constraint-driven site search.

    Attributes:
        positions: Atom positions {name: (x, y, z)}.
        score: Constraint satisfaction score (0-1).
        satisfied: Whether all constraints are satisfied.
        coord_atoms: Indices of protein atoms near the site (if protein given).
    """

    positions: dict[str, tuple[float, float, float]]
    score: float
    satisfied: bool
    coord_atoms: list[int] | None = None


def find_sites(
    constraints,
    atoms: list[str] | None = None,
    coords: list[tuple[float, float, float]] | None = None,
    n_samples: int = 100,
) -> list[SiteResult]:
    """Constraint-driven site search.

    User provides constraints, system finds positions satisfying them.
    With protein → search within protein (steric clash check).
    Without protein → generate conformations in free space.

    Supports two constraint styles:

    1. Explicit bond constraints (user specifies element types)::

        GeometricConstraint.bond('Zn', 'N', 1.8, 2.2)

    2. Coordination constraint (system auto-detects elements)::

        GeometricConstraint.coordination('Zn', n_ligands=4, min_dist=1.8, max_dist=2.5)

    Args:
        constraints: ConstraintSet with user-defined geometric constraints.
        atoms: Optional protein atom elements.
        coords: Optional protein atom coordinates.
        n_samples: Max number of results.

    Returns:
        List of SiteResult, sorted by score (highest first).
    """
    from .constraints import ConstraintSet

    if isinstance(constraints, ConstraintSet):
        cs = constraints
    else:
        cs = ConstraintSet(list(constraints))

    # Check for coordination constraint (auto-detect elements)
    coord_constraint = None
    for c in cs:
        if c.type.value == "coordination":
            coord_constraint = c
            break

    if coord_constraint is not None:
        metal = coord_constraint.atoms[0]
        n_ligands = int(coord_constraint.params["n_ligands"])
        dmin = coord_constraint.params["min_dist"]
        dmax = coord_constraint.params["max_dist"]

        if atoms is not None and coords is not None:
            return _find_sites_coordination(
                metal, n_ligands, dmin, dmax, cs, atoms, coords, n_samples,
            )
        else:
            # Free space: generate generic ligands
            ligand_constraints = [(f"L{i}", dmin, dmax) for i in range(n_ligands)]
            return _find_sites_free_space(metal, ligand_constraints, cs, None, n_samples)

    # Explicit bond constraints
    bond_constraints = [c for c in cs if c.type.value == "bond"]
    if not bond_constraints:
        return []

    # Identify the metal atom (appears in most bond constraints)
    atom_counts: dict[str, int] = {}
    for c in bond_constraints:
        for a in c.atoms:
            atom_counts[a] = atom_counts.get(a, 0) + 1
    metal = max(atom_counts, key=lambda a: atom_counts[a])

    # Collect ligand constraints: each bond constraint defines one metal-ligand bond
    ligand_constraints = []
    for c in bond_constraints:
        a1, a2 = c.atoms
        if a1 == metal:
            ligand_constraints.append((a2, c.params["min_dist"], c.params["max_dist"]))
        elif a2 == metal:
            ligand_constraints.append((a1, c.params["min_dist"], c.params["max_dist"]))

    if not ligand_constraints:
        return []

    # Find pocket center if specified (atom="*" or atom=metal)
    pocket_center = None
    for c in cs:
        if c.type.value == "pocket" and (c.atoms[0] == "*" or c.atoms[0] == metal):
            pocket_center = (c.params["cx"], c.params["cy"], c.params["cz"])
            break

    # Two modes: with protein or free space
    if atoms is not None and coords is not None:
        return _find_sites_with_protein(
            metal, ligand_constraints, cs, atoms, coords, pocket_center, n_samples,
        )
    else:
        return _find_sites_free_space(
            metal, ligand_constraints, cs, pocket_center, n_samples,
        )


def _find_sites_free_space(
    metal: str,
    ligand_constraints: list[tuple[str, float, float]],
    cs: object,
    pocket_center: tuple[float, float, float] | None,
    n_samples: int,
) -> list[SiteResult]:
    """Generate conformations in free space (no protein)."""
    # Assign unique names for duplicate atom types
    type_counter: dict[str, int] = {}
    unique_constraints = []
    for lig_type, dmin, dmax in ligand_constraints:
        count = type_counter.get(lig_type, 0)
        unique_name = f"{lig_type}_{count}" if count > 0 or sum(1 for t, _, _ in ligand_constraints if t == lig_type) > 1 else lig_type
        type_counter[lig_type] = count + 1
        unique_constraints.append((unique_name, dmin, dmax))

    # Build bond graph with unique names
    bond_graph: dict[str, list[tuple[str, float, float]]] = {}
    for lig_name, dmin, dmax in unique_constraints:
        bond_graph.setdefault(metal, []).append((lig_name, dmin, dmax))
        bond_graph.setdefault(lig_name, []).append((metal, dmin, dmax))

    all_atoms = {metal} | {name for name, _, _ in unique_constraints}
    root_pos = pocket_center or (0.0, 0.0, 0.0)

    conformations = _place_atoms(metal, root_pos, bond_graph, all_atoms)

    # Build renamed constraint set for evaluation
    from .constraints import ConstraintSet as CS
    from .constraints import GeometricConstraint
    name_usage: dict[str, int] = {}
    eval_constraints = []
    for c in cs:
        if c.type.value == "bond":
            a1, a2 = c.atoms
            u1 = metal if a1 == metal else _assign_unique_name(a1, ligand_constraints, name_usage)
            u2 = metal if a2 == metal else _assign_unique_name(a2, ligand_constraints, name_usage)
            eval_constraints.append(GeometricConstraint.bond(u1, u2, c.params["min_dist"], c.params["max_dist"]))
        elif c.type.value == "pocket":
            eval_constraints.append(c)
        elif c.type.value == "no_clash":
            a1, a2 = c.atoms
            u1 = metal if a1 == metal else _assign_unique_name(a1, ligand_constraints, name_usage)
            u2 = metal if a2 == metal else _assign_unique_name(a2, ligand_constraints, name_usage)
            eval_constraints.append(GeometricConstraint.no_clash(u1, u2, c.params["min_dist"]))
    eval_cs = CS(eval_constraints)

    results = []
    for conf in conformations:
        satisfied, score = eval_cs.evaluate(conf)
        results.append(SiteResult(
            positions=conf,
            score=score,
            satisfied=satisfied,
            coord_atoms=None,
        ))

    results.sort(key=lambda r: (-r.satisfied, -r.score))
    return results[:n_samples]


def _find_unique_name(
    name: str,
    metal: str,
    unique_constraints: list[tuple[str, float, float]],
    ligand_constraints: list[tuple[str, float, float]],
) -> str:
    """Map original atom name to unique name."""
    if name == metal:
        return metal
    for uname, _, _ in unique_constraints:
        base = uname.rsplit('_', 1)[0] if '_' in uname else uname
        if base == name:
            return uname
    return name


def _assign_unique_name(
    atom_type: str,
    ligand_constraints: list[tuple[str, float, float]],
    usage: dict[str, int],
) -> str:
    """Assign unique name for an atom type, tracking usage.

    For 'N' with3 constraints: first call returns 'N_0', second 'N_1', third 'N_2'.
    For 'O' with1 constraint: returns 'O' (no suffix needed).
    """
    total_of_type = sum(1 for t, _, _ in ligand_constraints if t == atom_type)
    idx = usage.get(atom_type, 0)
    usage[atom_type] = idx + 1
    if total_of_type > 1:
        return f"{atom_type}_{idx}"
    return atom_type


def _find_sites_coordination(
    metal: str,
    n_ligands: int,
    dmin: float,
    dmax: float,
    cs: object,
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    n_samples: int,
) -> list[SiteResult]:
    """Find conformations satisfying coordination constraint.

    For each metal candidate position, sample different combinations of
    coordination atoms from the protein to generate diverse conformations.
    Each conformation = metal position + n_ligands chosen from nearby N/O/S atoms.
    """
    import random
    rng = random.Random(42)

    cell_size = dmax * 2
    coord_cells: dict[tuple[int, int, int], list[int]] = {}
    for i, a in enumerate(atoms):
        if a in _COORD_ELEMENTS:
            x, y, z = coords[i]
            cell = (int(x / cell_size), int(y / cell_size), int(z / cell_size))
            coord_cells.setdefault(cell, []).append(i)

    if len(coord_cells) < n_ligands:
        return []

    # Find metal candidates
    candidates = _find_metal_candidates_coordination(
        atoms, coords, coord_cells, n_ligands, dmin, dmax,
    )

    if not candidates:
        return []

    # Build protein spatial hash
    protein_cells: dict[tuple[int, int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(coords):
        cell = (int(x / cell_size), int(y / cell_size), int(z / cell_size))
        protein_cells.setdefault(cell, []).append(i)

    results = []
    seen_confs = set()  # For deduplication by atom indices

    for metal_pos in candidates:
        mx, my, mz = metal_pos

        # Find ALL coordination atoms within range
        coord_atoms = []
        mcell = (int(mx / cell_size), int(my / cell_size), int(mz / cell_size))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cell = (mcell[0] + dx, mcell[1] + dy, mcell[2] + dz)
                    if cell not in coord_cells:
                        continue
                    for idx in coord_cells[cell]:
                        ax, ay, az = coords[idx]
                        d = math.sqrt((mx-ax)**2 + (my-ay)**2 + (mz-az)**2)
                        if dmin <= d <= dmax:
                            coord_atoms.append((idx, d))

        if len(coord_atoms) < n_ligands:
            continue

        # Sort by distance
        coord_atoms.sort(key=lambda x: x[1])

        # Generate diverse conformations by sampling different subsets
        # Each conformation = different combination of n_ligands atoms
        n_available = len(coord_atoms)
        max_combos = min(50, max(10, n_samples // max(1, len(candidates))))

        # Always include the best (closest) combination
        combos_to_try = [tuple(range(n_ligands))]

        # Sample random combinations
        if n_available > n_ligands:
            for _ in range(max_combos):
                combo = tuple(sorted(rng.sample(range(n_available), n_ligands)))
                if combo not in combos_to_try:
                    combos_to_try.append(combo)

        for combo in combos_to_try:
            selected = [coord_atoms[i] for i in combo]
            used_indices = {idx for idx, _ in selected}

            # Skip if this exact set of atoms was already seen
            conf_key = tuple(sorted(used_indices))
            if conf_key in seen_confs:
                continue
            seen_confs.add(conf_key)

            # Build conformation
            conf = {metal: metal_pos}
            ligand_constraints = []
            elem_counter: dict[str, int] = {}
            for idx, d in selected:
                elem = atoms[idx]
                count = elem_counter.get(elem, 0)
                total = sum(1 for j, _ in selected if atoms[j] == elem)
                name = f"{elem}_{count}" if total > 1 else elem
                elem_counter[elem] = count + 1
                conf[name] = coords[idx]
                ligand_constraints.append((elem, dmin, dmax))

            # Check steric clashes (skip coordination atoms)
            clash_dist = _COVALENT_BOND_THRESHOLD / 2.0
            clash_dist_sq = clash_dist ** 2
            has_clash = False
            for atom_name, (px, py, pz) in conf.items():
                if atom_name == metal:
                    continue
                pcx = int(px / cell_size)
                pcy = int(py / cell_size)
                pcz = int(pz / cell_size)
                for adx in range(-1, 2):
                    for ady in range(-1, 2):
                        for adz in range(-1, 2):
                            cell = (pcx + adx, pcy + ady, pcz + adz)
                            if cell not in protein_cells:
                                continue
                            for idx in protein_cells[cell]:
                                if idx in used_indices:
                                    continue
                                ax, ay, az = coords[idx]
                                d_sq = (px-ax)**2 + (py-ay)**2 + (pz-az)**2
                                if d_sq < clash_dist_sq:
                                    has_clash = True
                                    break
                            if has_clash:
                                break
                        if has_clash:
                            break
                    if has_clash:
                        break
                if has_clash:
                    break

            if has_clash:
                continue

            # Score: distance quality
            dmid = (dmin + dmax) / 2
            dist_quality = sum(1.0 - abs(d - dmid) / (dmax - dmin) for _, d in selected) / len(selected)
            score = float(len(selected)) + dist_quality

            results.append(SiteResult(
                positions=conf,
                score=score,
                satisfied=True,
                coord_atoms=list(used_indices),
            ))

    # Sort by score
    results.sort(key=lambda r: -r.score)

    # Deduplicate by conformation RMSD
    deduplicated = []
    for r in results:
        is_dup = False
        for existing in deduplicated:
            common = sorted(set(r.positions.keys()) & set(existing.positions.keys()))
            if not common:
                continue
            sum_sq = sum(
                sum((r.positions[a][i] - existing.positions[a][i])**2 for i in range(3))
                for a in common
            )
            rmsd = math.sqrt(sum_sq / len(common))
            if rmsd < 0.5:
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(r)
        if len(deduplicated) >= n_samples:
            break

    return deduplicated


def _find_metal_candidates_coordination(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    coord_cells: dict[tuple[int, int, int], list[int]],
    n_ligands: int,
    dmin: float,
    dmax: float,
) -> list[tuple[float, float, float]]:
    """Find metal positions by scanning N/O/S atom neighborhoods.

    For each N/O/S atom, find nearby N/O/S atoms within 2*dmax.
    If enough atoms are within the distance range of a central point,
    that point is a metal candidate. The central point is the centroid
    of the nearby atoms (more robust than sphere intersection).
    """
    cell_size = dmax * 2
    nos_indices = []
    for indices in coord_cells.values():
        nos_indices.extend(indices)

    if len(nos_indices) < 3:
        return []

    search_sq = (dmax * 2) ** 2
    candidates = []

    for idx in nos_indices:
        px, py, pz = coords[idx]
        pcx = int(px / cell_size)
        pcy = int(py / cell_size)
        pcz = int(pz / cell_size)

        # Find nearby N/O/S atoms
        nearby = []
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                for adz in range(-1, 2):
                    cell = (pcx + adx, pcy + ady, pcz + adz)
                    if cell not in coord_cells:
                        continue
                    for j in coord_cells[cell]:
                        if j == idx:
                            continue
                        d_sq = sum((a-b)**2 for a, b in zip(coords[idx], coords[j]))
                        if d_sq <= search_sq:
                            d = math.sqrt(d_sq)
                            nearby.append((j, d))

        if len(nearby) < n_ligands - 1:
            continue

        # Try each subset of nearby atoms as coordination shell
        # Use centroid as metal position estimate
        # For efficiency, just try all (n_ligands-1)-subsets
        for combo in itertools.combinations(range(len(nearby)), min(n_ligands - 1, len(nearby))):
            subset_indices = [nearby[i][0] for i in combo]
            subset_indices.append(idx)  # Include the central atom

            # Centroid
            cx = sum(coords[i][0] for i in subset_indices) / len(subset_indices)
            cy = sum(coords[i][1] for i in subset_indices) / len(subset_indices)
            cz = sum(coords[i][2] for i in subset_indices) / len(subset_indices)
            center = (cx, cy, cz)

            # Verify: count atoms within range from centroid
            coord_count = 0
            ccx = int(cx / cell_size)
            ccy = int(cy / cell_size)
            ccz = int(cz / cell_size)
            for adx in range(-1, 2):
                for ady in range(-1, 2):
                    for adz in range(-1, 2):
                        cell = (ccx + adx, ccy + ady, ccz + adz)
                        if cell not in coord_cells:
                            continue
                        for k in coord_cells[cell]:
                            d_sq = sum((a-b)**2 for a, b in zip(center, coords[k]))
                            d = math.sqrt(d_sq)
                            if dmin <= d <= dmax:
                                coord_count += 1
            if coord_count >= n_ligands - 1:
                candidates.append(center)

            # Also try sphere intersection for precision
            if len(subset_indices) >= 3:
                p1, p2, p3 = coords[subset_indices[0]], coords[subset_indices[1]], coords[subset_indices[2]]
                dmid = (dmin + dmax) / 2
                for pt in _sphere_intersection(p1, p2, p3, radius=dmid):
                    coord_count = 0
                    pcx2 = int(pt[0] / cell_size)
                    pcy2 = int(pt[1] / cell_size)
                    pcz2 = int(pt[2] / cell_size)
                    for adx in range(-1, 2):
                        for ady in range(-1, 2):
                            for adz in range(-1, 2):
                                cell = (pcx2 + adx, pcy2 + ady, pcz2 + adz)
                                if cell not in coord_cells:
                                    continue
                                for k in coord_cells[cell]:
                                    d_sq = sum((a-b)**2 for a, b in zip(pt, coords[k]))
                                    d = math.sqrt(d_sq)
                                    if dmin <= d <= dmax:
                                        coord_count += 1
                    if coord_count >= n_ligands - 1:
                        candidates.append(pt)

    # Deduplicate
    dedup = []
    for pt in candidates:
        is_dup = False
        for existing in dedup:
            d = math.sqrt(sum((a-b)**2 for a, b in zip(pt, existing)))
            if d < 1.0:
                is_dup = True
                break
        if not is_dup:
            dedup.append(pt)

    return dedup


def _find_sites_with_protein(
    metal: str,
    ligand_constraints: list[tuple[str, float, float]],
    cs: object,
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    pocket_center: tuple[float, float, float] | None,
    n_samples: int,
) -> list[SiteResult]:
    """Find sites within a protein structure.

    Strategy: for each candidate metal position (from N/O/S triplet sphere
    intersection), check if the coordination geometry satisfies constraints.
    """
    # Get metal-ligand distance range from constraints
    all_dmin = [dmin for _, dmin, _ in ligand_constraints]
    all_dmax = [dmax for _, _, dmax in ligand_constraints]
    ml_min = min(all_dmin)
    ml_max = max(all_dmax)
    (ml_min + ml_max) / 2

    # Find coordination atoms in protein (N, O, S)
    coord_indices = [i for i, a in enumerate(atoms) if a in _COORD_ELEMENTS]
    if len(coord_indices) < 3:
        return []

    # Spatial hash for coordination atoms
    cell_size = ml_max * 2
    coord_cells: dict[tuple[int, int, int], list[int]] = {}
    for i in coord_indices:
        x, y, z = coords[i]
        cell = (int(x / cell_size), int(y / cell_size), int(z / cell_size))
        coord_cells.setdefault(cell, []).append(i)

    # Find metal candidates via sphere intersection of coordination atom triplets
    candidates = _find_metal_candidates_from_constraints(
        atoms, coords, coord_cells, ligand_constraints, pocket_center,
    )

    if not candidates:
        return []

    # Build protein spatial hash for clash detection
    protein_cells: dict[tuple[int, int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(coords):
        cell = (int(x / cell_size), int(y / cell_size), int(z / cell_size))
        protein_cells.setdefault(cell, []).append(i)

    # For each candidate, find coordination atoms and evaluate constraints
    results = []
    for metal_pos in candidates:
        # Find coordination atoms within ml_min..ml_max
        mx, my, mz = metal_pos
        coord_atoms = []
        mcell = (int(mx / cell_size), int(my / cell_size), int(mz / cell_size))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    cell = (mcell[0] + dx, mcell[1] + dy, mcell[2] + dz)
                    if cell not in coord_cells:
                        continue
                    for idx in coord_cells[cell]:
                        ax, ay, az = coords[idx]
                        d = math.sqrt((mx-ax)**2 + (my-ay)**2 + (mz-az)**2)
                        if ml_min <= d <= ml_max:
                            coord_atoms.append((idx, d))

        # Check if coordination count matches constraints
        # Count by element type
        type_counts: dict[str, int] = {}
        for idx, d in coord_atoms:
            elem = atoms[idx]
            type_counts[elem] = type_counts.get(elem, 0) + 1

        # Check if we have enough of each type
        required_types: dict[str, int] = {}
        for lig_type, _, _ in ligand_constraints:
            required_types[lig_type] = required_types.get(lig_type, 0) + 1

        can_satisfy = True
        for elem, count in required_types.items():
            if type_counts.get(elem, 0) < count:
                can_satisfy = False
                break

        if not can_satisfy:
            continue

        # Build conformation: metal + best coordination atoms
        # Assign unique names for duplicate types
        type_counter: dict[str, int] = {}
        conf = {metal: metal_pos}
        used_indices = set()
        for lig_type, dmin, dmax in ligand_constraints:
            # Find best matching atom (within distance range, correct type, not used)
            best_idx = None
            best_d = float('inf')
            for idx, d in coord_atoms:
                if idx in used_indices:
                    continue
                if atoms[idx] != lig_type:
                    continue
                if not (dmin <= d <= dmax):
                    continue
                if d < best_d:
                    best_d = d
                    best_idx = idx
            if best_idx is not None:
                count = type_counter.get(lig_type, 0)
                unique_name = f"{lig_type}_{count}" if count > 0 or sum(1 for t, _, _ in ligand_constraints if t == lig_type) > 1 else lig_type
                type_counter[lig_type] = count + 1
                conf[unique_name] = coords[best_idx]
                used_indices.add(best_idx)

        # Evaluate constraints — map each constraint atom to its unique name
        from .constraints import ConstraintSet as CS
        from .constraints import GeometricConstraint
        # Build name mapping: track which unique names have been used per type
        name_usage: dict[str, int] = {}  # type -> count of used unique names
        eval_constraints = []
        for c in cs:
            if c.type.value == "bond":
                a1, a2 = c.atoms
                u1 = metal if a1 == metal else _assign_unique_name(a1, ligand_constraints, name_usage)
                u2 = metal if a2 == metal else _assign_unique_name(a2, ligand_constraints, name_usage)
                eval_constraints.append(GeometricConstraint.bond(u1, u2, c.params["min_dist"], c.params["max_dist"]))
            elif c.type.value == "pocket":
                eval_constraints.append(c)
            elif c.type.value == "no_clash":
                a1, a2 = c.atoms
                u1 = metal if a1 == metal else _assign_unique_name(a1, ligand_constraints, name_usage)
                u2 = metal if a2 == metal else _assign_unique_name(a2, ligand_constraints, name_usage)
                eval_constraints.append(GeometricConstraint.no_clash(u1, u2, c.params["min_dist"]))
        eval_cs = CS(eval_constraints)

        satisfied, score = eval_cs.evaluate(conf)

        # Check steric clashes with protein (skip coordination atoms)
        clash_dist = _COVALENT_BOND_THRESHOLD / 2.0
        clash_dist_sq = clash_dist ** 2
        has_clash = False
        for atom_name, (px, py, pz) in conf.items():
            if atom_name == metal:
                continue
            pcx = int(px / cell_size)
            pcy = int(py / cell_size)
            pcz = int(pz / cell_size)
            for adx in range(-1, 2):
                for ady in range(-1, 2):
                    for adz in range(-1, 2):
                        cell = (pcx + adx, pcy + ady, pcz + adz)
                        if cell not in protein_cells:
                            continue
                        for idx in protein_cells[cell]:
                            if idx in used_indices:
                                continue  # Skip coordination atoms
                            ax, ay, az = coords[idx]
                            d_sq = (px-ax)**2 + (py-ay)**2 + (pz-az)**2
                            if d_sq < clash_dist_sq:
                                has_clash = True
                                break
                        if has_clash:
                            break
                    if has_clash:
                        break
                if has_clash:
                    break
            if has_clash:
                break

        if has_clash:
            continue

        results.append(SiteResult(
            positions=conf,
            score=score,
            satisfied=satisfied,
            coord_atoms=list(used_indices),
        ))

    # Sort and deduplicate
    results.sort(key=lambda r: (-r.satisfied, -r.score))

    deduplicated = []
    for r in results:
        is_dup = False
        for existing in deduplicated:
            d = math.sqrt(sum((a-b)**2 for a, b in zip(r.positions[metal], existing.positions[metal])))
            if d < 1.0:
                is_dup = True
                break
        if not is_dup:
            deduplicated.append(r)
        if len(deduplicated) >= n_samples:
            break

    return deduplicated


def _find_metal_candidates_from_constraints(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    coord_cells: dict[tuple[int, int, int], list[int]],
    ligand_constraints: list[tuple[str, float, float]],
    pocket_center: tuple[float, float, float] | None = None,
) -> list[tuple[float, float, float]]:
    """Find metal positions from coordination atom sphere intersections.

    For each N/O/S atom, find neighboring N/O/S atoms, then compute
    sphere intersection to find metal position.
    """
    all_dmin = [dmin for _, dmin, _ in ligand_constraints]
    all_dmax = [dmax for _, _, dmax in ligand_constraints]
    ml_min = min(all_dmin)
    ml_max = max(all_dmax)
    ml_mid = (ml_min + ml_max) / 2

    cell_size = ml_max * 2
    nos_indices = []
    for cell, indices in coord_cells.items():
        nos_indices.extend(indices)

    if len(nos_indices) < 3:
        return []

    # For each N/O/S atom, find neighbors and compute sphere intersections
    search_sq = (ml_max * 2) ** 2
    candidates = []

    for idx in nos_indices:
        px, py, pz = coords[idx]
        pcx = int(px / cell_size)
        pcy = int(py / cell_size)
        pcz = int(pz / cell_size)
        neighbors = []
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                for adz in range(-1, 2):
                    cell = (pcx + adx, pcy + ady, pcz + adz)
                    if cell not in coord_cells:
                        continue
                    for j in coord_cells[cell]:
                        if j <= idx:
                            continue
                        d_sq = sum((a-b)**2 for a, b in zip(coords[idx], coords[j]))
                        if d_sq <= search_sq:
                            neighbors.append(j)

        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                ni, nj = neighbors[i], neighbors[j]
                points = _sphere_intersection(
                    coords[idx], coords[ni], coords[nj], radius=ml_mid
                )
                for pt in points:
                    # Verify coordination count
                    coord_count = 0
                    pt_cx = int(pt[0] / cell_size)
                    pt_cy = int(pt[1] / cell_size)
                    pt_cz = int(pt[2] / cell_size)
                    for adx in range(-1, 2):
                        for ady in range(-1, 2):
                            for adz in range(-1, 2):
                                cell = (pt_cx + adx, pt_cy + ady, pt_cz + adz)
                                if cell not in coord_cells:
                                    continue
                                for k in coord_cells[cell]:
                                    d_sq = sum((a-b)**2 for a, b in zip(pt, coords[k]))
                                    d = math.sqrt(d_sq)
                                    if ml_min <= d <= ml_max:
                                        coord_count += 1
                    if coord_count >= 3:
                        # If pocket specified, check distance
                        if pocket_center is not None:
                            d_to_pocket = math.sqrt(sum((a-b)**2 for a, b in zip(pt, pocket_center)))
                            if d_to_pocket > ml_max * 5:
                                continue
                        candidates.append(pt)

    # Deduplicate
    dedup = []
    for pt in candidates:
        is_dup = False
        for existing in dedup:
            d = math.sqrt(sum((a-b)**2 for a, b in zip(pt, existing)))
            if d < 1.0:
                is_dup = True
                break
        if not is_dup:
            dedup.append(pt)

    return dedup


def _find_metal_candidates(
    atoms: list[str],
    coords: list[tuple[float, float, float]],
    bond_graph: dict[str, list[tuple[str, float, float]]],
) -> list[tuple[float, float, float]]:
    """Find metal candidate positions using constraint-driven approach.

    For each N/O/S atom, find neighboring N/O/S atoms within ml distance.
    Use sphere intersection from triplets, then verify coordination count.

    Args:
        atoms: Element symbols.
        coords: Atomic coordinates.
        bond_graph: Bond constraints {atom: [(neighbor, dmin, dmax), ...]}.

    Returns:
        List of candidate metal positions.
    """

    # Get metal-ligand distance range from bond constraints
    all_dmin = []
    all_dmax = []
    for neighbors in bond_graph.values():
        for nb, dmin, dmax in neighbors:
            all_dmin.append(dmin)
            all_dmax.append(dmax)
    if not all_dmin:
        return []

    ml_min = min(all_dmin)
    ml_max = max(all_dmax)
    ml_mid = (ml_min + ml_max) / 2

    # Find N/O/S atoms
    nos_indices = [i for i, a in enumerate(atoms) if a in _COORD_ELEMENTS]
    if len(nos_indices) < 3:
        return []

    # Spatial hash
    cell_size = ml_max * 2
    nos_cells: dict[tuple[int, int, int], list[int]] = {}
    for i in nos_indices:
        x, y, z = coords[i]
        cell = (int(x / cell_size), int(y / cell_size), int(z / cell_size))
        nos_cells.setdefault(cell, []).append(i)

    # For each N/O/S atom, find neighbors within 2*ml_max
    # Then for each pair of neighbors, compute sphere intersection
    search_sq = (ml_max * 2) ** 2
    candidates = []

    for idx in nos_indices:
        # Find all N/O/S neighbors within 2*ml_max
        px, py, pz = coords[idx]
        pcx = int(px / cell_size)
        pcy = int(py / cell_size)
        pcz = int(pz / cell_size)
        neighbors = []
        for adx in range(-1, 2):
            for ady in range(-1, 2):
                for adz in range(-1, 2):
                    cell = (pcx + adx, pcy + ady, pcz + adz)
                    if cell not in nos_cells:
                        continue
                    for j in nos_cells[cell]:
                        if j <= idx:
                            continue
                        d_sq = sum((a-b)**2 for a, b in zip(coords[idx], coords[j]))
                        if d_sq <= search_sq:
                            neighbors.append(j)

        # For each pair of neighbors, compute sphere intersection
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                ni, nj = neighbors[i], neighbors[j]
                points = _sphere_intersection(
                    coords[idx], coords[ni], coords[nj], radius=ml_mid
                )
                for pt in points:
                    # Verify: count N/O/S atoms within ml_min..ml_max
                    coord_count = 0
                    pt_cx = int(pt[0] / cell_size)
                    pt_cy = int(pt[1] / cell_size)
                    pt_cz = int(pt[2] / cell_size)
                    for adx in range(-1, 2):
                        for ady in range(-1, 2):
                            for adz in range(-1, 2):
                                cell = (pt_cx + adx, pt_cy + ady, pt_cz + adz)
                                if cell not in nos_cells:
                                    continue
                                for k in nos_cells[cell]:
                                    d_sq = sum((a-b)**2 for a, b in zip(pt, coords[k]))
                                    d = math.sqrt(d_sq)
                                    if ml_min <= d <= ml_max:
                                        coord_count += 1
                    if coord_count >= 3:
                        candidates.append(pt)

    # Deduplicate
    dedup = []
    for pt in candidates:
        is_dup = False
        for existing in dedup:
            d = math.sqrt(sum((a-b)**2 for a, b in zip(pt, existing)))
            if d < 1.0:
                is_dup = True
                break
        if not is_dup:
            dedup.append(pt)

    return dedup


def _place_atoms(
    root: str,
    root_pos: tuple[float, float, float],
    bond_graph: dict[str, list[tuple[str, float, float]]],
    all_atoms: set[str],
    n_distance_samples: int = 3,
    max_conformations: int = 200,
) -> list[dict[str, tuple[float, float, float]]]:
    """Place atoms using distance geometry (sphere intersections).

    Strategy: for each atom, generate positions per distance sample.
    Single-neighbor atoms: random point on sphere (unique per atom, not shared).
    Multi-neighbor atoms: sphere intersection (deterministic).
    Expansion is linear: O(max_conformations × n_distance_samples × n_atoms).

    Args:
        root: Root atom name.
        root_pos: Root atom position.
        bond_graph: {atom: [(neighbor, dmin, dmax), ...]}.
        all_atoms: All atom names.
        n_distance_samples: Number of distance samples per bond range.
        max_conformations: Max conformations to keep at each step.

    Returns:
        List of conformations {atom_name: (x, y, z)}.
    """
    import random
    rng = random.Random(42)  # Reproducible

    conformations = [{root: root_pos}]
    queue_order = [root]
    visited = {root}

    # BFS placement order
    temp_queue = [root]
    while temp_queue:
        current = temp_queue.pop(0)
        for neighbor, _, _ in bond_graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue_order.append(neighbor)
                temp_queue.append(neighbor)

    # Place atoms one by one, prune at each step
    for atom in queue_order[1:]:
        placed_nbs = []
        for nb, dmin, dmax in bond_graph.get(atom, []):
            if any(nb in conf for conf in conformations):
                dists = [dmin + (dmax - dmin) * i / max(1, n_distance_samples - 1)
                         for i in range(n_distance_samples)]
                placed_nbs.append((nb, dists))

        if not placed_nbs:
            continue

        dist_lists = [dists for _, dists in placed_nbs]
        nb_names = [nb for nb, _ in placed_nbs]

        # Incremental expansion: for each conformation, add positions for this atom
        # For single-neighbor atoms: 1 random sphere point per distance sample
        # For multi-neighbor atoms: sphere intersection (deterministic)
        new_conformations = []
        for conf in conformations:
            for combo in itertools.product(*dist_lists):
                spheres = [(conf[nb], d) for nb, d in zip(nb_names, combo)]

                if len(spheres) == 1:
                    # Single neighbor: 1 random point on sphere (unique per atom)
                    positions = [_random_sphere_point(spheres[0][0], spheres[0][1], rng)]
                else:
                    # Multiple neighbors: intersect spheres
                    positions = _intersect_spheres_multi(spheres)

                for pos in positions:
                    new_conf = dict(conf)
                    new_conf[atom] = pos
                    new_conformations.append(new_conf)

        # Prune: keep only max_conformations
        if len(new_conformations) > max_conformations:
            new_conformations = new_conformations[:max_conformations]

        conformations = new_conformations

        if not conformations:
            return []

    for conf in conformations:
        for a in all_atoms:
            if a not in conf:
                conf[a] = root_pos

    return conformations


def _sample_sphere_points(
    center: tuple[float, float, float], radius: float, n: int = 8
) -> list[tuple[float, float, float]]:
    """Generate n evenly-spaced points on a sphere surface."""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        x = center[0] + radius * math.sin(theta) * math.cos(phi)
        y = center[1] + radius * math.sin(theta) * math.sin(phi)
        z = center[2] + radius * math.cos(theta)
        points.append((x, y, z))
    return points


def _sphere_intersection_3r(
    c1: tuple[float, float, float], r1: float,
    c2: tuple[float, float, float], r2: float,
    c3: tuple[float, float, float], r3: float,
) -> list[tuple[float, float, float]]:
    """Intersect 3 spheres with individual radii (0 or 2 points)."""
    return _intersect_spheres_multi([(c1, r1), (c2, r2), (c3, r3)])


def _random_sphere_point(
    center: tuple[float, float, float], radius: float, rng: object
) -> tuple[float, float, float]:
    """Generate one random point on a sphere surface (uniform distribution)."""
    import math
    # Marsaglia method for uniform sphere sampling
    while True:
        u = rng.random() * 2 - 1
        v = rng.random() * 2 - 1
        s = u * u + v * v
        if s < 1:
            break
    factor = radius * math.sqrt(1 - s) * 2
    return (
        center[0] + factor * u,
        center[1] + factor * v,
        center[2] + radius * (1 - 2 * s),
    )


def _intersect_spheres_multi(
    spheres: list[tuple[tuple[float, float, float], float]]
) -> list[tuple[float, float, float]]:
    """Analytical sphere-sphere intersection.

    2 spheres: circle intersection (8 sample points).
    3+ spheres: exact intersection (0 or 2 points).
    """
    if len(spheres) < 2:
        return []

    c1, r1 = spheres[0]
    c2, r2 = spheres[1]

    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dz = c2[2] - c1[2]
    d = math.sqrt(dx*dx + dy*dy + dz*dz)

    if d < 1e-10 or d > r1 + r2 + 1e-10 or d < abs(r1 - r2) - 1e-10:
        return []

    a = (r1*r1 - r2*r2 + d*d) / (2*d)
    h_sq = r1*r1 - a*a
    if h_sq < -1e-10:
        return []
    h = math.sqrt(max(0, h_sq))

    px = c1[0] + a * dx / d
    py = c1[1] + a * dy / d
    pz = c1[2] + a * dz / d

    # Orthonormal basis for circle plane
    wx, wy, wz = dx/d, dy/d, dz/d
    if abs(wx) < 0.9:
        ux, uy, uz = 1.0, 0.0, 0.0
    else:
        ux, uy, uz = 0.0, 1.0, 0.0
    dot = ux*wx + uy*wy + uz*wz
    ux -= dot * wx
    uy -= dot * wy
    uz -= dot * wz
    un = math.sqrt(ux*ux + uy*uy + uz*uz)
    if un < 1e-10:
        return [(px, py, pz)]
    ux /= un
    uy /= un
    uz /= un
    vx = wy*uz - wz*uy
    vy = wz*ux - wx*uz
    vz = wx*uy - wy*ux

    if len(spheres) == 2:
        if h < 1e-10:
            return [(px, py, pz)]
        points = []
        for i in range(8):
            angle = i * math.pi / 4
            x = px + h * (ux * math.cos(angle) + vx * math.sin(angle))
            y = py + h * (uy * math.cos(angle) + vy * math.sin(angle))
            z = pz + h * (uz * math.cos(angle) + vz * math.sin(angle))
            points.append((x, y, z))
        return points

    # 3+ spheres: intersect circle with 3rd sphere
    c3, r3 = spheres[2]
    fx = px - c3[0]
    fy = py - c3[1]
    fz = pz - c3[2]
    A = h * (ux*fx + uy*fy + uz*fz)
    B = h * (vx*fx + vy*fy + vz*fz)
    C = r3*r3 - fx*fx - fy*fy - fz*fz - h*h

    R = math.sqrt(A*A + B*B)
    if R < 1e-10:
        if abs(C) < 1e-5:
            points = []
            for i in range(8):
                angle = i * math.pi / 4
                x = px + h * (ux * math.cos(angle) + vx * math.sin(angle))
                y = py + h * (uy * math.cos(angle) + vy * math.sin(angle))
                z = pz + h * (uz * math.cos(angle) + vz * math.sin(angle))
                points.append((x, y, z))
            return points
        return []

    cos_val = C / (2 * R)
    if abs(cos_val) > 1 + 1e-10:
        return []
    cos_val = max(-1.0, min(1.0, cos_val))
    delta = math.acos(cos_val)
    phi = math.atan2(B, A)

    points = []
    for t in [phi + delta, phi - delta]:
        x = px + h * (ux * math.cos(t) + vx * math.sin(t))
        y = py + h * (uy * math.cos(t) + vy * math.sin(t))
        z = pz + h * (uz * math.cos(t) + vz * math.sin(t))
        points.append((x, y, z))

    return points
