"""Algorithm templates: 72 quantum algorithms.

Each template depends only on numpy / scipy and is not tied to a specific
backend; sampling algorithms can switch among the qiskit / cirq / pennylane
/ qulacs / tensorcircuit backends.
"""

from .amplitude_amplification import amplitude_amplification
from .amplitude_estimation import amplitude_estimation
from .bb84 import bb84
from .bernstein_vazirani import bernstein_vazirani
from .bit_flip_code import bit_flip_code
from .color_code import color_code
from .deutsch_jozsa import deutsch_jozsa
from .discrete_log import discrete_log
from .dqaoa import dqaoa
from .dynamics_simulation import dynamics_simulation
from .e91 import e91
from .eigenvalue_solver import quantum_eigenvalue
from .elliptic_curve import elliptic_curve
from .fermion_mapping import jordan_wigner_2site
from .ft_gates import ft_gate
from .grover import diffusion, grover, mark_state
from .hadamard_test import hadamard_test
from .hamiltonian_simulation import hamiltonian_simulation
from .hamiltonians import from_qiskit_nature
from .hamiltonians_ext import from_openfermion, from_pauli_string, from_pennylane
from .hhl import hhl
from .hsp import hsp
from .lattice import lattice_svp
from .matrix_inversion import quantum_matrix_inversion
from .molecule_vqe import molecule_vqe
from .oracle import oracle
from .phase_flip_code import phase_flip_code
from .qaoa import qaoa_maxcut
from .qaoa_generic import qaoa
from .qaoa_knapsack import qaoa_knapsack
from .qaoa_mis import qaoa_mis
from .qaoa_tsp import qaoa_tsp
from .qbm import qbm
from .qcnn import qcnn
from .qft_algo import qft
from .qgan import qgan
from .qgnn import qgnn
from .qng import qng
from .qnn import qnn
from .qpca import qpca
from .qpe import qpe
from .qrl import qrl
from .qsp import qsp
from .qsvm import qsvm
from .qtda import qtda
from .qtransformer import qtransformer
from .quantum_annealing import quantum_annealing
from .quantum_annealing_hybrid import quantum_annealing_hybrid
from .quantum_bayesian import quantum_bayesian
from .quantum_clustering import quantum_clustering
from .quantum_counting import quantum_counting
from .quantum_fitting import quantum_fitting
from .quantum_kernel import quantum_kernel
from .quantum_monte_carlo import quantum_monte_carlo
from .quantum_ode import quantum_ode
from .quantum_pde import quantum_pde
from .quantum_walk import quantum_walk
from .rejection_sampling import rejection_sampling
from .shor import shor
from .shor_code import shor_code
from .simon import simon
from .stabilizer import stabilizer
from .steane_code import steane_code
from .superdense_coding import superdense_coding
from .surface_code import surface_code
from .swap_test import swap_test
from .syndrome import syndrome
from .teleportation import teleportation
from .trotter import trotter
from .vqc import vqc
from .vqe import vqe
from .vqr import vqr

__all__ = [
    "amplitude_amplification",
    "amplitude_estimation",
    "bb84",
    "bernstein_vazirani",
    "bit_flip_code",
    "color_code",
    "deutsch_jozsa",
    "diffusion",
    "discrete_log",
    "dqaoa",
    "dynamics_simulation",
    "e91",
    "elliptic_curve",
    "from_openfermion",
    "from_pauli_string",
    "from_pennylane",
    "from_qiskit_nature",
    "ft_gate",
    "grover",
    "hadamard_test",
    "hamiltonian_simulation",
    "hhl",
    "hsp",
    "jordan_wigner_2site",
    "lattice_svp",
    "mark_state",
    "molecule_vqe",
    "oracle",
    "phase_flip_code",
    "qaoa",
    "qaoa_knapsack",
    "qaoa_maxcut",
    "qaoa_mis",
    "qaoa_tsp",
    "qbm",
    "qcnn",
    "qft",
    "qgan",
    "qgnn",
    "qng",
    "qnn",
    "qpca",
    "qpe",
    "qrl",
    "qsp",
    "qsvm",
    "qtda",
    "qtransformer",
    "quantum_annealing",
    "quantum_annealing_hybrid",
    "quantum_bayesian",
    "quantum_clustering",
    "quantum_counting",
    "quantum_eigenvalue",
    "quantum_fitting",
    "quantum_kernel",
    "quantum_matrix_inversion",
    "quantum_monte_carlo",
    "quantum_ode",
    "quantum_pde",
    "quantum_walk",
    "rejection_sampling",
    "shor",
    "shor_code",
    "simon",
    "stabilizer",
    "steane_code",
    "superdense_coding",
    "surface_code",
    "swap_test",
    "syndrome",
    "teleportation",
    "trotter",
    "vqc",
    "vqe",
    "vqr",
]
