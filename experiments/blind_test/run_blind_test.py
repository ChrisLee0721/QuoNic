"""Blind metal site prediction test.

Given ONLY a protein structure and metal element type (no knowledge of
metal position, coordinating atoms, or pocket center), predict where
the metal ion is. Compare against crystal structure ground truth.

This is the first honest evaluation of GCIQA's predictive capability.
Previous "0.070A error" was a circular measurement (constraints derived
from known structure, error = grid quantization).
"""
import json
import math
import random
import sys
import time
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gciqa.ligand_detect import detect_pockets, refine_pocket
from gciqa.metal_templates import METAL_COORDINATION, get_metal_template
from gciqa.pdb import find_metal_ions, parse_pdb

COORD_ELEMENTS = {"N", "O", "S"}
# Metals we can test (have templates)
TEMPLATED_METALS = set(METAL_COORDINATION.keys())


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def find_coordinating_atoms(pred_pos, atoms, coords, dmin, dmax):
    """Find N/O/S atoms within [dmin, dmax] of predicted position."""
    result = []
    for i, (a, c) in enumerate(zip(atoms, coords)):
        if a not in COORD_ELEMENTS:
            continue
        d = distance(pred_pos, c)
        if dmin <= d <= dmax:
            result.append((i, a, d))
    return result


def score_candidate(pred_pos, atoms, coords, template):
    """Score a candidate metal position by coordination quality."""
    # Use the broadest distance range from the template
    dmin_all = 999
    dmax_all = 0
    for elem, (dmin, dmax) in template.distances.items():
        dmin_all = min(dmin_all, dmin)
        dmax_all = max(dmax_all, dmax)

    # Find coordinating atoms
    coord_atoms = find_coordinating_atoms(pred_pos, atoms, coords, dmin_all * 0.8, dmax_all * 1.2)

    if not coord_atoms:
        return 0.0, []

    # Score: coordination count * distance quality
    # Distance quality: how close are distances to template center
    n_coord = len(coord_atoms)
    target_d = (dmin_all + dmax_all) / 2
    dist_scores = []
    for _, elem, d in coord_atoms:
        if elem.upper() in template.distances:
            tmin, tmax = template.distances[elem.upper()]
            t_center = (tmin + tmax) / 2
            dist_scores.append(1.0 / (1.0 + abs(d - t_center)))
        else:
            dist_scores.append(1.0 / (1.0 + abs(d - target_d)))

    avg_dist_score = sum(dist_scores) / len(dist_scores) if dist_scores else 0
    score = n_coord * avg_dist_score

    return score, coord_atoms


def blind_predict(pdb_path, metal_element, top_k=5):
    """Run blind metal site prediction.

    Args:
        pdb_path: Path to PDB file
        metal_element: Metal element (e.g., "ZN", "FE")
        top_k: Number of candidates to return

    Returns:
        dict with predictions and ground truth
    """
    t0 = time.time()

    # Parse PDB
    protein = parse_pdb(str(pdb_path))
    atoms = protein.atoms
    coords = protein.coords

    # Ground truth: find all metal ions of this type
    true_metals = find_metal_ions(protein, metal_element)
    if not true_metals:
        return {"error": f"No {metal_element} found in structure", "pdb": str(pdb_path)}

    # Get chemistry prior (template)
    # Try default geometry first, fall back to any available
    template = None
    if metal_element in METAL_COORDINATION:
        geoms = METAL_COORDINATION[metal_element]
        # Pick the first available geometry
        for geom_name, geom_data in geoms.items():
            template = get_metal_template(metal_element, geom_name)
            break

    if template is None:
        return {"error": f"No template for {metal_element}", "pdb": str(pdb_path)}

    # Step 1: Blind pocket detection (no knowledge of metal position)
    try:
        pockets = detect_pockets(atoms, coords)
    except Exception as e:
        return {"error": f"detect_pockets failed: {e}", "pdb": str(pdb_path)}

    if not pockets:
        return {
            "error": "No pockets detected",
            "pdb": str(pdb_path),
            "n_atoms": len(atoms),
            "true_metals": [{"coord": m.coord} for m in true_metals],
        }

    # Step 2: Refine top pocket candidates
    refined = []
    for pocket in pockets[:20]:  # Refine top 20
        try:
            r = refine_pocket(pocket, atoms, coords)
            refined.append(r)
        except Exception:
            continue

    if not refined:
        refined = pockets[:top_k]

    # Step 3: Score each candidate
    scored = []
    seen = set()
    for pos in refined:
        # Deduplicate (within 0.5A)
        key = (round(pos[0], 0), round(pos[1], 0), round(pos[2], 0))
        if key in seen:
            continue
        seen.add(key)

        score, coord_atoms = score_candidate(pos, atoms, coords, template)
        scored.append({
            "pos": pos,
            "score": score,
            "n_coordinating": len(coord_atoms),
            "coordinating_atoms": [(i, a, round(d, 2)) for i, a, d in coord_atoms[:6]],
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    predictions = scored[:top_k]

    t1 = time.time()

    # Validate against ground truth
    results_per_metal = []
    for metal in true_metals:
        true_pos = metal.coord
        best_dist = float("inf")
        best_idx = -1
        for j, pred in enumerate(predictions):
            d = distance(pred["pos"], true_pos)
            if d < best_dist:
                best_dist = d
                best_idx = j

        # Also compute distances to ALL refined candidates (not just top-K)
        best_overall_dist = best_dist
        for pos in refined:
            d = distance(pos, true_pos)
            if d < best_overall_dist:
                best_overall_dist = d

        results_per_metal.append({
            "metal_element": metal_element,
            "true_coord": list(true_pos),
            "best_topk_dist": round(best_dist, 2),
            "best_overall_dist": round(best_overall_dist, 2),
            "best_topk_idx": best_idx,
            "hit_2A": best_dist < 2.0,
            "hit_5A": best_dist < 5.0,
            "hit_10A": best_dist < 10.0,
        })

    # Random baseline: sample same number of random points in bounding box
    all_x = [c[0] for c in coords]
    all_y = [c[1] for c in coords]
    all_z = [c[2] for c in coords]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    zmin, zmax = min(all_z), max(all_z)

    rng = random.Random(42)
    n_random = len(refined)
    random_dists = []
    for metal in true_metals:
        true_pos = metal.coord
        best_random = float("inf")
        for _ in range(n_random):
            rp = (
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax),
                rng.uniform(zmin, zmax),
            )
            d = distance(rp, true_pos)
            if d < best_random:
                best_random = d
        random_dists.append(round(best_random, 2))

    return {
        "pdb": str(pdb_path),
        "n_atoms": len(atoms),
        "metal_element": metal_element,
        "n_pockets_detected": len(pockets),
        "n_refined": len(refined),
        "n_predictions": len(predictions),
        "top_predictions": [
            {
                "pos": [round(x, 2) for x in p["pos"]],
                "score": round(p["score"], 3),
                "n_coordinating": p["n_coordinating"],
            }
            for p in predictions
        ],
        "metals": results_per_metal,
        "random_baseline_best_dist": random_dists,
        "time_s": round(t1 - t0, 2),
    }


def run_all():
    """Run blind test on all available metalloproteins."""
    data_dir = ROOT / "experiments" / "zn_metalloproteinase" / "data"
    pdb_files = sorted(data_dir.glob("*.pdb"))

    if not pdb_files:
        print(f"No PDB files found in {data_dir}")
        return

    all_results = []
    summary = {"total": 0, "hit_2A": 0, "hit_5A": 0, "hit_10A": 0, "miss": 0}

    print("=" * 70)
    print("BLIND METAL SITE PREDICTION TEST")
    print("=" * 70)
    print(f"PDB directory: {data_dir}")
    print(f"Found {len(pdb_files)} PDB files")
    print()

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.upper()

        # Determine which metal to look for based on the file
        # We'll try all templated metals and see which ones are present
        protein = parse_pdb(str(pdb_path))
        found_metals = set()
        for mi in protein.metal_ions:
            elem = mi.element.upper() if hasattr(mi, 'element') else mi.element
            if elem in TEMPLATED_METALS:
                found_metals.add(elem)

        if not found_metals:
            print(f"  {pdb_id}: No templated metals found, skipping")
            continue

        for metal_elem in sorted(found_metals):
            print(f"  {pdb_id} ({metal_elem}): ", end="", flush=True)
            result = blind_predict(pdb_path, metal_elem, top_k=5)
            all_results.append(result)

            if "error" in result:
                print(f"ERROR - {result['error']}")
                continue

            for m in result["metals"]:
                summary["total"] += 1
                d = m["best_topk_dist"]
                if m["hit_2A"]:
                    summary["hit_2A"] += 1
                    print(f"HIT {d:.1f}A (top-K)", end=" ")
                elif m["hit_5A"]:
                    summary["hit_5A"] += 1
                    print(f"NEAR {d:.1f}A (top-K)", end=" ")
                elif m["hit_10A"]:
                    summary["hit_10A"] += 1
                    print(f"FAR {d:.1f}A (top-K)", end=" ")
                else:
                    summary["miss"] += 1
                    print(f"MISS {d:.1f}A", end=" ")

                # Show overall best
                if m["best_overall_dist"] < m["best_topk_dist"]:
                    print(f"(best overall: {m['best_overall_dist']:.1f}A)", end=" ")

                # Random baseline
                rb = result.get("random_baseline_best_dist", [])
                if rb:
                    print(f"(random: {rb[0]:.1f}A)", end=" ")

            print(f"[{result['time_s']:.1f}s]")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = summary["total"]
    if total == 0:
        print("No metals tested.")
        return

    print(f"Total metal sites tested: {total}")
    print(f"Hit rate (<2A):  {summary['hit_2A']}/{total} = {summary['hit_2A']/total*100:.1f}%")
    print(f"Hit rate (<5A):  {summary['hit_5A']}/{total} = {(summary['hit_2A']+summary['hit_5A'])/total*100:.1f}%")
    print(f"Hit rate (<10A): {summary['hit_10A']}/{total} = {(summary['hit_2A']+summary['hit_5A']+summary['hit_10A'])/total*100:.1f}%")
    print(f"Miss (>10A):     {summary['miss']}/{total} = {summary['miss']/total*100:.1f}%")

    # Save results
    out_dir = ROOT / "experiments" / "blind_test"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_all()
