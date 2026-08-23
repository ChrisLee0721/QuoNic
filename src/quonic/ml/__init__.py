"""Quantum Machine Learning — variational circuits, data encoding, and optimizers.

Example::

    from quonic.ml import Ansatz, angle_encode, SPSAOptimizer
    from quonic.ml import expectation_loss, train

    ansatz = Ansatz.hardware_efficient(n_qubits=4, layers=3)
    encoded = angle_encode(features)
    opt = SPSAOptimizer(maxiter=100)
    result = train(ansatz, encoded, opt, observable="ZZZZ")
"""

from .adjoint import adjoint_grad, adjoint_grad_exact, adjoint_grad_statevector
from .adjoint_gpu import adjoint_grad_gpu
from .ansatz import Ansatz
from .blackbox import blackbox_grad, natural_gradient
from .encoding import amplitude_encode, angle_encode, iqp_encode
from .hybrid import ClassicalLayer, HybridModel, HybridResult, QNNLayer
from .loss import cross_entropy_loss, expectation_loss, fidelity_loss
from .optimizer import AdamOptimizer, QNGOptimizer, SPSAOptimizer
from .pipeline import QMLPipeline, QMLResult
from .pulse_grad import pulse_fisher_information, pulse_gradient
from .trainer import param_shift_grad, train, train_batch
from .viz import (
    plot_circuit_analysis,
    plot_gradient_flow,
    plot_parameter_distribution,
    plot_training_convergence,
)

__all__ = [
    "AdamOptimizer",
    "Ansatz",
    "ClassicalLayer",
    "HybridModel",
    "HybridResult",
    "QMLPipeline",
    "QMLResult",
    "QNGOptimizer",
    "QNNLayer",
    "SPSAOptimizer",
    "adjoint_grad",
    "adjoint_grad_exact",
    "adjoint_grad_gpu",
    "adjoint_grad_statevector",
    "amplitude_encode",
    "angle_encode",
    "blackbox_grad",
    "cross_entropy_loss",
    "expectation_loss",
    "fidelity_loss",
    "iqp_encode",
    "natural_gradient",
    "param_shift_grad",
    "plot_circuit_analysis",
    "plot_gradient_flow",
    "plot_parameter_distribution",
    "plot_training_convergence",
    "pulse_fisher_information",
    "pulse_gradient",
    "train",
    "train_batch",
]
