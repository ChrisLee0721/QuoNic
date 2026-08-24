"""Molecular VQE with quonic.chem / 使用 quonic.chem 的分子 VQE

Complete pipeline: geometry -> SCF -> Hamiltonian -> VQE.
完整流程：几何结构 -> SCF -> 哈密顿量 -> VQE。

Requirements / 依赖:
    pip install 'quonic[chem]'

Usage / 运行:
    python chem_vqe.py
"""

from quonic.algorithms import vqe
from quonic.chem import Molecule, molecular_hamiltonian

# ── Define H2 molecule / 定义 H2 分子 ─────────────────────────

mol = Molecule.from_xyz(
    """
2
H2 at equilibrium bond length
H  0.0  0.0  0.0
H  0.0  0.0  0.74
""",
    basis="sto-3g",
)
print(f"Molecule: {mol}")
print(f"Electrons: {mol.n_electrons}")

# ── Generate qubit Hamiltonian / 生成量子比特哈密顿量 ──────────

ham_result = molecular_hamiltonian(mol)
print(f"\nNumber of qubits:  {ham_result.metadata['n_qubits']}")
print(f"SCF energy:        {ham_result.metadata['mf_energy']:.6f} Hartree")
print(f"Hamiltonian terms: {len(ham_result.metadata['hamiltonian'])}")

# ── Run VQE / 运行 VQE ────────────────────────────────────────

vqe_result = vqe(
    ham_result.metadata["hamiltonian"],
    ham_result.metadata["n_qubits"],
    maxiter=500,
    record_history=True,
)
print(f"\nVQE energy:   {vqe_result.value:.6f} Hartree")
print(f"Optimal params: {[f'{p:.4f}' for p in vqe_result.metadata['params'][:4]]}...")

# ── Convergence plot (optional) / 收敛曲线（可选） ──────────────

try:
    import matplotlib.pyplot as plt

    history = vqe_result.metadata.get("history", [])
    if history:
        plt.figure(figsize=(8, 4))
        plt.plot(history)
        plt.xlabel("Iteration")
        plt.ylabel("Energy (Hartree)")
        plt.title("VQE Convergence — H2 / STO-3G")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("vqe_convergence.png", dpi=150)
        print("\nConvergence plot saved to vqe_convergence.png")
except ImportError:
    pass
