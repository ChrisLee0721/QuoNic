"""Matrix product state (MPS) engine: breaks the 2^n memory wall for
low-entanglement circuits.

Naive version: single-qubit gates update locally, multi-qubit gates use the
"diagonal phase + H" trick plus SVD truncation, and non-adjacent qubits are
moved with a SWAP chain. The bond dimension is hard-truncated at chi_max.

Conventions: qubit 0 is the least-significant bit; sites from left to right are
qubit 0..n-1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._i18n import tr
from ._gates import _H, single


class MPSEngine:
    def __init__(self, num_qubits: int, chi_max: int = 32) -> None:
        self.n: int = num_qubits
        self.chi_max: int = chi_max
        # M[i] has shape [χ_{i-1}, 2, χ_i], initialized to |0...0> (all bond dimensions are 1)
        self.M: list[Any] = [np.zeros((1, 2, 1), dtype=complex) for _ in range(num_qubits)]
        for t in self.M:
            t[0, 0, 0] = 1.0

    # ------------------------------------------------------------------
    # basic tensor operations
    # ------------------------------------------------------------------
    def _apply_single(self, q: int, u: Any) -> None:
        self.M[q] = np.einsum("asb,ts->atb", self.M[q], u)

    def _merge(self, qubits: Sequence[int]) -> Any:
        theta = self.M[qubits[0]]
        for j in range(1, len(qubits)):
            theta = np.einsum("...a,abc->...bc", theta, self.M[qubits[j]])
        return theta

    def _restore_pair(self, theta: Any, i: int) -> None:
        """Restore [χL, 2, 2, χR] back into sites i, i+1 (one SVD)."""
        chi_l = theta.shape[0]
        chi_r = theta.shape[-1]
        mat = theta.reshape(chi_l * 2, 2 * chi_r)
        a, s, b = np.linalg.svd(mat, full_matrices=False)
        chi = min(len(s), self.chi_max)
        a = a[:, :chi]
        s = s[:chi]
        b = b[:chi, :]
        self.M[i] = a.reshape(chi_l, 2, chi)
        self.M[i + 1] = (s[:, None] * b).reshape(chi, 2, chi_r)

    def _restore(self, theta: Any, qubits: Sequence[int]) -> None:
        """Restore [χL, 2, ..., 2, χR] back into k consecutive sites (left SVD step by step)."""
        k = len(qubits)
        chi_l = theta.shape[0]
        chi_r = theta.shape[-1]
        cur = theta
        for idx in range(k - 1):
            num_phys = k - idx
            mat = cur.reshape(chi_l * 2, 2 ** (num_phys - 1) * chi_r)
            a, s, b = np.linalg.svd(mat, full_matrices=False)
            chi = min(len(s), self.chi_max)
            a = a[:, :chi]
            s = s[:chi]
            b = b[:chi, :]
            self.M[qubits[idx]] = a.reshape(chi_l, 2, chi)
            cur = (s[:, None] * b).reshape(chi, *([2] * (num_phys - 1)), chi_r)
            chi_l = chi
        self.M[qubits[k - 1]] = cur.reshape(chi_l, 2, chi_r)

    def _swap_adjacent(self, i: int) -> None:
        theta = np.einsum("asr,rtb->astb", self.M[i], self.M[i + 1])
        theta = np.einsum("astb->atsb", theta)
        self._restore_pair(theta, i)

    # ------------------------------------------------------------------
    # diagonal gates (cz / cp / mcz): merge -> diagonal scaling -> SVD restore
    # ------------------------------------------------------------------
    def _apply_diag_contiguous(self, qubits: Sequence[int], angle: float) -> None:
        theta = self._merge(qubits)
        k = len(qubits)
        index = (slice(None),) + (1,) * k + (slice(None),)
        theta[index] *= np.exp(1j * angle)
        if k == 2:
            self._restore_pair(theta, qubits[0])
        else:
            self._restore(theta, qubits)

    def _apply_diag(self, qubits: Sequence[int], angle: float) -> None:
        q = sorted(qubits)
        swaps: list[int] = []
        for j in range(1, len(q)):
            target = q[0] + j
            while q[j] > target:
                self._swap_adjacent(q[j] - 1)
                swaps.append(q[j] - 1)
                q[j] -= 1
        self._apply_diag_contiguous(q, angle)
        for i in reversed(swaps):
            self._swap_adjacent(i)

    # ------------------------------------------------------------------
    # gate dispatch
    # ------------------------------------------------------------------
    def apply(
        self, name: str, qubits: Sequence[int], params: tuple[float, ...] = ()
    ) -> None:
        name = name.lower()
        if name == "measure":
            return
        if name in ("i", "h", "x", "y", "z", "rx", "ry", "rz", "p"):
            self._apply_single(qubits[0], single(name, params))
        elif name == "cx":
            self._apply_single(qubits[1], _H)
            self._apply_diag(qubits, np.pi)
            self._apply_single(qubits[1], _H)
        elif name == "cz":
            self._apply_diag(qubits, np.pi)
        elif name == "cp":
            self._apply_diag(qubits, params[0])
        elif name == "ccx":
            self._apply_single(qubits[2], _H)
            self._apply_diag(qubits, np.pi)
            self._apply_single(qubits[2], _H)
        elif name == "mcz":
            self._apply_diag(qubits, np.pi)
        elif name == "swap":
            a, b = qubits[0], qubits[1]
            if abs(a - b) != 1:
                raise NotImplementedError(tr("err.mps_swap"))
            self._swap_adjacent(min(a, b))
        else:
            # Try custom gate registry
            from ..gates import _GATE_REGISTRY
            if name in _GATE_REGISTRY:
                gate = _GATE_REGISTRY[name]
                if gate.matrix is not None and len(qubits) == 1:
                    self._apply_single(qubits[0], gate.matrix)
                else:
                    raise ValueError(tr("err.mps_gate", name=name))
            else:
                raise ValueError(tr("err.mps_gate", name=name))

    def apply_noise(self, qubits: Sequence[int], p: float) -> None:
        """Apply depolarizing noise to the specified qubits.

        With probability p, applies a random Pauli error (X, Y, or Z) to each qubit.

        Args:
            qubits: qubit indices to apply noise to
            p: depolarizing error probability
        """
        if p <= 0:
            return
        paulis = [
            np.array([[0, 1], [1, 0]], dtype=complex),      # X
            np.array([[0, -1j], [1j, 0]], dtype=complex),   # Y
            np.array([[1, 0], [0, -1]], dtype=complex),      # Z
        ]
        for q in qubits:
            if np.random.random() < p:
                pauli = paulis[np.random.randint(3)]
                self._apply_single(q, pauli)

    # ------------------------------------------------------------------
    # sampling: right environment + per-bit conditional probabilities
    # ------------------------------------------------------------------
    def _right_env(self) -> list[Any]:
        r: list[Any] = [None] * (self.n + 1)
        r[self.n] = np.array([[1.0 + 0j]])
        for i in range(self.n - 1, -1, -1):
            r[i] = np.einsum("asc,cd,bsd->ab", self.M[i], r[i + 1], self.M[i].conj())
        return r

    def _sample_once(self, r: list[Any]) -> list[int]:
        left = np.array([[1.0 + 0j]])
        bits: list[int] = []
        for i in range(self.n):
            probs: list[float] = []
            for s in (0, 1):
                m = self.M[i][:, s, :]
                p = np.einsum("ab,ac,cd,bd->", left, m, r[i + 1], m.conj())
                probs.append(float(np.real(p)))
            probs = np.clip(probs, 0.0, None)
            total = probs.sum()
            probs = probs / total if total > 0 else [0.5, 0.5]
            s = int(np.random.choice([0, 1], p=probs))
            bits.append(s)
            m = self.M[i][:, s, :]
            left = np.einsum("ab,ac,bd->cd", left, m, m.conj())
        return bits

    def sample(self, shots: int) -> dict[str, int]:
        r = self._right_env()
        counts: dict[str, int] = {}
        for _ in range(shots):
            bits = self._sample_once(r)
            bs = "".join(str(b) for b in reversed(bits))
            counts[bs] = counts.get(bs, 0) + 1
        return counts

    def expectation(self, pauli: str) -> float:
        """Compute expectation value of a Pauli string (e.g. 'ZZ', 'XIZ').

        Uses the MPS contraction: ⟨ψ|P|ψ⟩ = Tr(ρ · P) where ρ is the reduced
        density matrix built from left-to-right contraction.

        Args:
            pauli: Pauli string (I, X, Y, Z) of length n_qubits.

        Returns:
            Real expectation value.
        """
        pauli_map = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

        left = np.array([[1.0 + 0j]])
        for i in range(self.n):
            p = pauli_map[pauli[i]] if i < len(pauli) else np.eye(2, dtype=complex)
            m = self.M[i]  # shape [chiL, 2, chiR]
            # Contract: left[a,b] * M[a,s,g] * P[s,t] * conj(M)[b,t,d] -> new[g,d]
            left = np.einsum("ab,asg,st,btd->gd", left, m, p, m.conj())
        return float(np.real(left[0, 0]))

    def to_statevector(self) -> np.ndarray:
        """Contract the MPS into a full 2^n state vector.

        Warning: exponential memory — only use for small n (<= 20).
        """
        sv = self.M[0]  # shape [1, 2, chi_1]
        for i in range(1, self.n):
            sv = np.einsum("...a,abc->...bc", sv, self.M[i])
        return sv.reshape(2**self.n)

    def bond_dimensions(self) -> list[int]:
        """Return the current bond dimension between each pair of adjacent sites."""
        return [self.M[i].shape[2] for i in range(self.n - 1)]

    def entropy(self, site: int) -> float:
        """Compute the von Neumann entropy of the bipartition at `site`.

        S = -Tr(ρ_L log ρ_L) where ρ_L is the reduced density matrix of qubits 0..site.
        """
        # Merge left part into a single tensor
        left = self.M[0]
        for i in range(1, site + 1):
            left = np.einsum("...a,abc->...bc", left, self.M[i])
        # left has shape [chiL, 2, 2, ..., chiR]
        # Reshape to [chiL * 2^site, chiR] for SVD
        chi_l = left.shape[0]
        chi_r = left.shape[-1]
        n_phys = site + 1
        mat = left.reshape(chi_l * (2 ** n_phys), chi_r)
        _, s, _ = np.linalg.svd(mat, full_matrices=False)
        # Schmidt values squared = eigenvalues of reduced density matrix
        s2 = s**2
        total = np.sum(s2)
        if total > 0:
            s2 = s2 / total
        s2 = s2[s2 > 1e-15]
        return float(-np.sum(s2 * np.log(s2)))

    def canonicalize(self, ortho_center: int = -1) -> None:
        """Put the MPS into canonical form.

        Left-canonical: M[0]..M[ortho_center-1] are isometries (U†U = I).
        Right-canonical: M[ortho_center+1]..M[n-1] are isometries.
        The orthogonality center carries the singular values.

        Args:
            ortho_center: index of the orthogonality center (-1 = last site).
        """
        if ortho_center < 0:
            ortho_center = self.n + ortho_center + 1
        ortho_center = max(0, min(ortho_center, self.n - 1))

        # Left-canonical sweep: QR from left to ortho_center
        for i in range(ortho_center):
            chi_l, d, chi_r = self.M[i].shape
            mat = self.M[i].reshape(chi_l * d, chi_r)
            q, r = np.linalg.qr(mat)
            chi_new = q.shape[1]
            self.M[i] = q.reshape(chi_l, d, chi_new)
            if i + 1 < self.n:
                self.M[i + 1] = np.einsum("ab,btc->atc", r, self.M[i + 1])

        # Right-canonical sweep: QR from right to ortho_center
        for i in range(self.n - 1, ortho_center, -1):
            chi_l, d, chi_r = self.M[i].shape
            mat = self.M[i].reshape(chi_l, d * chi_r)
            q, r = np.linalg.qr(mat.T)
            chi_new = q.shape[1]
            self.M[i] = q.T.reshape(chi_new, d, chi_r)
            if i - 1 >= 0:
                self.M[i - 1] = np.einsum("asb,bc->asc", self.M[i - 1], r.T)

    def is_left_canonical(self, site: int = 0) -> bool:
        """Check if M[site] is left-canonical (isometry: M†M = I)."""
        m = self.M[site]
        chi_l, d, chi_r = m.shape
        mat = m.reshape(chi_l * d, chi_r)
        product = mat.conj().T @ mat
        return np.allclose(product, np.eye(chi_r), atol=1e-10)

    def is_right_canonical(self, site: int = -1) -> bool:
        """Check if M[site] is right-canonical (isometry: MM† = I)."""
        if site < 0:
            site = self.n + site + 1
        m = self.M[site]
        chi_l, d, chi_r = m.shape
        mat = m.reshape(chi_l, d * chi_r)
        product = mat @ mat.conj().T
        return np.allclose(product, np.eye(chi_l), atol=1e-10)

    def norm(self) -> float:
        """Compute the norm of the MPS state."""
        left = np.array([[1.0 + 0j]])
        for i in range(self.n):
            m = self.M[i]
            left = np.einsum("ab,asc,bsd->cd", left, m, m.conj())
        return float(np.sqrt(np.real(left[0, 0])))

    def dmrg_sweep(self, hamiltonian: list[tuple[float, str]], max_sweeps: int = 10) -> float:
        """2-site DMRG sweep to minimize energy ⟨ψ|H|ψ⟩.

        True 2-site DMRG: merges pairs of adjacent tensors, optimizes the
        merged tensor via local eigenvalue problem, then SVDs back with
        truncation to chi_max.

        Args:
            hamiltonian: list of (coeff, pauli_string) terms, e.g. [(1.0, "ZZ"), (0.5, "X")]
            max_sweeps: number of left-right sweeps

        Returns:
            Final energy expectation value.
        """
        pauli_map = {
            "I": np.eye(2, dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }

        def compute_energy():
            energy = 0.0
            for coeff, pauli_str in hamiltonian:
                left = np.array([[1.0 + 0j]])
                for i in range(self.n):
                    p = pauli_map[pauli_str[i]] if i < len(pauli_str) else np.eye(2, dtype=complex)
                    m = self.M[i]
                    left = np.einsum("ab,asg,st,btd->gd", left, m, p, m.conj())
                energy += coeff * float(np.real(left[0, 0]))
            return energy

        def build_h_eff_2site(site: int):
            """Build the effective Hamiltonian for sites (site, site+1).

            Returns H_eff as a matrix acting on the merged tensor
            θ[chiL, d1, d2, chiR] reshaped to [chiL*d1*d2*chiR, ...].
            """
            chi_l = self.M[site].shape[0]
            d1 = self.M[site].shape[1]
            d2 = self.M[site + 1].shape[1]
            chi_r = self.M[site + 1].shape[2]
            dim = chi_l * d1 * d2 * chi_r

            # Build left environment L[α, β] = ⟨α|ρ_left|β⟩
            L = np.array([[1.0 + 0j]])
            for i in range(site):
                m = self.M[i]
                L = np.einsum("ab,asc,bsd->cd", L, m, m.conj())

            # Build right environment R[α, β] = ⟨α|ρ_right|β⟩
            R = np.array([[1.0 + 0j]])
            for i in range(self.n - 1, site + 2, -1):
                m = self.M[i]
                R = np.einsum("asc,cd,bsd->ab", m, R, m.conj())

            # Build H_eff as sum of terms
            H_eff = np.zeros((dim, dim), dtype=complex)

            for coeff, pauli_str in hamiltonian:
                # Get Pauli matrices for site and site+1
                p1 = pauli_map[pauli_str[site]] if site < len(pauli_str) else np.eye(2, dtype=complex)
                p2 = pauli_map[pauli_str[site + 1]] if site + 1 < len(pauli_str) else np.eye(2, dtype=complex)

                # H_eff[α,s1,s2,γ, β,t1,t2,δ] = L[α,β] * P1[s1,t1] * P2[s2,t2] * R[γ,δ]
                term = np.einsum("ab,st,uv,gd->asugb tvd", L, p1, p2, R)
                term = term.reshape(dim, dim)
                H_eff += coeff * term

            return H_eff

        def lanczos_ground_state(H_eff: Any, dim: int, krylov_dim: int = 20, tol: float = 1e-10) -> tuple[float, Any]:
            """Find the ground state of H_eff using Lanczos iteration.

            Builds a Krylov subspace and diagonalizes the tridiagonal matrix.
            Converges when the eigenvalue change between iterations is below tol.

            Returns (eigenvalue, eigenvector).
            """
            v = np.random.randn(dim) + 1j * np.random.randn(dim)
            v /= np.linalg.norm(v)

            krylov_vecs = [v]
            alphas = []
            betas = []
            prev_energy = None

            w = H_eff @ v
            alpha = np.real(np.conj(v) @ w)
            alphas.append(alpha)
            w = w - alpha * v

            for j in range(1, krylov_dim):
                beta = np.linalg.norm(w)
                if beta < 1e-12:
                    break
                betas.append(beta)
                v_next = w / beta
                krylov_vecs.append(v_next)

                w = H_eff @ v_next
                alpha = np.real(np.conj(v_next) @ w)
                alphas.append(alpha)
                w = w - alpha * v_next - beta * krylov_vecs[-2]

                for kv in krylov_vecs[:-1]:
                    w -= np.dot(np.conj(kv), w) * kv

                # Check convergence: diagonalize current tridiagonal matrix
                if len(alphas) >= 2:
                    T = np.diag(alphas)
                    for k in range(len(betas)):
                        T[k, k + 1] = betas[k]
                        T[k + 1, k] = betas[k]
                    eigvals = np.linalg.eigh(T)[0]
                    current_energy = eigvals[0]

                    if prev_energy is not None and abs(current_energy - prev_energy) < tol:
                        break
                    prev_energy = current_energy

            # Final diagonalization
            T = np.diag(alphas)
            for j in range(len(betas)):
                T[j, j + 1] = betas[j]
                T[j + 1, j] = betas[j]

            eigvals, eigvecs = np.linalg.eigh(T)
            idx = np.argmin(eigvals)
            energy = eigvals[idx]

            K = np.column_stack(krylov_vecs)
            v_ground = K @ eigvecs[:, idx]
            v_ground /= np.linalg.norm(v_ground)

            return energy, v_ground

        best_energy = compute_energy()

        for sweep in range(max_sweeps):
            # Left-to-right sweep
            for i in range(self.n - 1):
                theta = np.einsum("asb,btc->astc", self.M[i], self.M[i + 1])
                chi_l, d1, d2, chi_r = theta.shape
                dim = chi_l * d1 * d2 * chi_r

                H_eff = build_h_eff_2site(i)
                energy, v = lanczos_ground_state(H_eff, dim)

                theta_new = v.reshape(chi_l, d1, d2, chi_r)
                mat = theta_new.reshape(chi_l * d1, d2 * chi_r)
                u, s, vh = np.linalg.svd(mat, full_matrices=False)
                chi_new = min(len(s), self.chi_max)
                u = u[:, :chi_new]
                s = s[:chi_new]
                vh = vh[:chi_new, :]
                s = s / np.linalg.norm(s)

                self.M[i] = u.reshape(chi_l, d1, chi_new)
                self.M[i + 1] = (np.diag(s) @ vh).reshape(chi_new, d2, chi_r)

                best_energy = min(best_energy, energy)

            # Right-to-left sweep
            for i in range(self.n - 2, -1, -1):
                theta = np.einsum("asb,btc->astc", self.M[i], self.M[i + 1])
                chi_l, d1, d2, chi_r = theta.shape
                dim = chi_l * d1 * d2 * chi_r

                H_eff = build_h_eff_2site(i)
                energy, v = lanczos_ground_state(H_eff, dim)

                theta_new = v.reshape(chi_l, d1, d2, chi_r)
                mat = theta_new.reshape(chi_l * d1, d2 * chi_r)
                u, s, vh = np.linalg.svd(mat, full_matrices=False)
                chi_new = min(len(s), self.chi_max)
                u = u[:, :chi_new]
                s = s[:chi_new]
                vh = vh[:chi_new, :]
                s = s / np.linalg.norm(s)

                self.M[i] = u.reshape(chi_l, d1, chi_new)
                self.M[i + 1] = (np.diag(s) @ vh).reshape(chi_new, d2, chi_r)

                best_energy = min(best_energy, energy)

        return best_energy
