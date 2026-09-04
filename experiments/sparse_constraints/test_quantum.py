"""Test: Quantum mode vs Classical mode for GCIQA.

Compares:
1. Classical search (random sampling)
2. Quantum search (Grover via qiskit-aer)

On a small system where quantum simulation is feasible.
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gciqa import GCIQA, ConstraintSet, GeometricConstraint


def generate_chain(n, bl=2.0):
    pos = [(0.0, 0.0, 0.0)]
    for i in range(1, n):
        a = math.radians(110 + random.gauss(0, 15))
        pos.append((pos[-1][0] + bl*math.cos(a*(i%3)),
                     pos[-1][1] + bl*math.sin(a*(i%3)),
                     pos[-1][2] + random.gauss(0, 0.3)))
    return pos


def extract_constraints(pos, n_extra, noise=1.0):
    n = len(pos)
    cs = []
    for i in range(n-1):
        d = math.sqrt(sum((a-b)**2 for a,b in zip(pos[i],pos[i+1])))
        cs.append((i,i+1,max(0.5,d-noise),d+noise))
    used = set()
    for _ in range(n_extra*20):
        if len(cs)-(n-1) >= n_extra: break
        i,j = random.randint(0,n-1), random.randint(0,n-1)
        if i==j or abs(i-j)<=1 or (i,j) in used: continue
        used.add((i,j))
        d = math.sqrt(sum((a-b)**2 for a,b in zip(pos[i],pos[j])))
        cs.append((i,j,max(0.5,d-noise),d+noise))
    return cs


def solve_dg(n_pts, cs, n_samples=3000):
    adj = {i:[] for i in range(n_pts)}
    for i,j,dmin,dmax in cs:
        adj[i].append((j,dmin,dmax))
        adj[j].append((i,dmin,dmax))
    root = max(adj, key=lambda k: len(adj[k]))
    best_pos, best_v = None, float("inf")
    for _ in range(n_samples):
        pos = [None]*n_pts
        pos[root] = (0.,0.,0.)
        placed = {root}
        queue = [root]
        v = 0.0
        while queue and len(placed)<n_pts:
            cur = queue.pop(0)
            for nb,dmin,dmax in adj[cur]:
                if nb in placed: continue
                bp,bv = None, float("inf")
                for _ in range(100):
                    t,p = random.uniform(0,2*math.pi), math.acos(2*random.random()-1)
                    r = random.uniform(dmin,dmax)
                    x = pos[cur][0]+r*math.sin(p)*math.cos(t)
                    y = pos[cur][1]+r*math.sin(p)*math.sin(t)
                    z = pos[cur][2]+r*math.cos(p)
                    vv = 0.0
                    for o,od,ox in adj[nb]:
                        if o in placed and o!=cur:
                            dd = math.sqrt((x-pos[o][0])**2+(y-pos[o][1])**2+(z-pos[o][2])**2)
                            vv += (max(0,od-dd)+max(0,dd-ox))**2
                    if vv<bv: bv,bp = vv,(x,y,z)
                if bp:
                    pos[nb]=bp; placed.add(nb); queue.append(nb); v+=bv
        for i in range(n_pts):
            if pos[i] is None: pos[i]=(random.uniform(-10,10),)*3
        if v<best_v: best_v,best_pos = v,pos
    return best_pos


def rmsd(pred,ref):
    n=len(pred)
    pc=[sum(p[i] for p in pred)/n for i in range(3)]
    rc=[sum(r[i] for r in ref)/n for i in range(3)]
    return math.sqrt(sum(sum((p[i]-r[i])**2 for i in range(3)) for p,r in zip(
        [(p[0]-pc[0],p[1]-pc[1],p[2]-pc[2]) for p in pred],
        [(r[0]-rc[0],r[1]-rc[1],r[2]-rc[2]) for r in ref]))/n)


def run_gciqa(n_pts, cs, init, cr, bits, use_quantum, n_shots=1000):
    gcs = [GeometricConstraint.bond(str(i),str(j),min_dist=dmin,max_dist=dmax) for i,j,dmin,dmax in cs]
    ps = list(init.values())
    cx,cy,cz = [sum(p[i] for p in ps)/len(ps) for i in range(3)]
    mr = max(math.sqrt((p[0]-cx)**2+(p[1]-cy)**2+(p[2]-cz)**2) for p in ps)
    gcs.append(GeometricConstraint.pocket(center=(cx,cy,cz),radius=max(mr*1.5,3.0)))

    g = GCIQA(n_super_atoms=n_pts, constraints=ConstraintSet(gcs), coord_range=cr,
              bits_per_coord=bits, alpha=0.7, convergence_threshold=0.5,
              use_quantum=use_quantum, initial_conformation=init)
    g._perturbation_pct = 0.05

    t0 = time.time()
    try:
        r = g.run(max_iterations=3, n_shots=n_shots, n_clusters=3)
        elapsed = time.time() - t0
        if r.best_conformation:
            rp = [r.best_conformation[str(i)] for i in range(n_pts)]
            return rp, elapsed
    except Exception:
        elapsed = time.time() - t0
        return None, elapsed
    return None, elapsed


def main():
    print("="*70)
    print("Quantum vs Classical Mode Comparison")
    print("="*70)

    # Small system: 3 points, 3 bits → 27 qubits (feasible for Aer)
    n_pts = 3
    n_ext = 1
    bits = 3  # 27 qubits

    random.seed(42)
    true = generate_chain(n_pts)
    cons = extract_constraints(true, n_ext)

    # DG initial conformation
    dg = solve_dg(n_pts, cons, n_samples=3000)
    r0 = rmsd(dg, true)

    xs=[p[0] for p in dg]; ys=[p[1] for p in dg]; zs=[p[2] for p in dg]
    m=3.0
    cr=((min(min(xs),min(ys),min(zs))-m, max(max(xs),max(ys),max(zs))+m))
    init = {str(i):p for i,p in enumerate(dg)}

    n_qubits = n_pts * 3 * bits
    print(f"\n  System: {n_pts} points, {n_ext} extra constraints")
    print(f"  Bits per coord: {bits}")
    print(f"  Qubits needed: {n_qubits}")
    print(f"  DG RMSD: {r0:.3f}A")
    print(f"  Coord range: {cr}")

    # Classical mode
    print("\n  --- Classical Mode ---")
    rp_c, t_c = run_gciqa(n_pts, cons, init, cr, bits, use_quantum=False, n_shots=1000)
    if rp_c:
        r_c = rmsd(rp_c, true)
        print(f"  RMSD: {r_c:.3f}A, Time: {t_c:.1f}s")
    else:
        print(f"  FAILED, Time: {t_c:.1f}s")

    # Quantum mode
    print("\n  --- Quantum Mode (qiskit-aer) ---")
    rp_q, t_q = run_gciqa(n_pts, cons, init, cr, bits, use_quantum=True, n_shots=1000)
    if rp_q:
        r_q = rmsd(rp_q, true)
        print(f"  RMSD: {r_q:.3f}A, Time: {t_q:.1f}s")
    else:
        print(f"  FAILED, Time: {t_q:.1f}s")

    # Speedup
    if t_c > 0 and t_q > 0:
        print(f"\n  Speedup: {t_c/t_q:.1f}x {'(quantum faster)' if t_q < t_c else '(classical faster)'}")


if __name__ == "__main__":
    main()
