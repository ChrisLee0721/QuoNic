"""QuoNic — quantum programming, as simple as writing Python."""

from . import gates
from ._i18n import get_language, set_language
from .analysis import CircuitReport, analyze
from .batch import run_batch
from .compare import qeq, qgt, qlt
from .compiler import RoutingError, compile, decompose, groverize, optimize, randomized_compiling
from .encoding import amplitude_encode, angle_encode
from .gradients import numerical_gradient, param_shift
from .mitigation import CDRResult, PECResult, cdr, pec, symmetry_verify
from .noise import NoiseModel, amplitude_damping, depolarizing, phase_damping, thermal_relaxation
from .parameters import Parameter, bind_batch, bind_params
from .qgate import qgate
from .qif import cif, controlled, creg, cwhile, qif
from .qint import QInt, mul
from .qshow import qshow, qshow_all, run_circuits
from .readout import ReadoutCalibration, calibrate
from .result import Result
from .stack import reset
from .stepper import StepExecutor
from .topology import CouplingMap
from .zne import ZNEResult, fold, zne

__version__ = "0.12.0"

__all__ = [
    "CDRResult",
    "CircuitReport",
    "CouplingMap",
    "NoiseModel",
    "PECResult",
    "Parameter",
    "QInt",
    "ReadoutCalibration",
    "Result",
    "RoutingError",
    "StepExecutor",
    "ZNEResult",
    "__version__",
    "amplitude_damping",
    "amplitude_encode",
    "analyze",
    "angle_encode",
    "bind_batch",
    "bind_params",
    "calibrate",
    "cdr",
    "cif",
    "compile",
    "controlled",
    "creg",
    "cwhile",
    "decompose",
    "depolarizing",
    "fold",
    "gates",
    "get_language",
    "groverize",
    "mul",
    "numerical_gradient",
    "optimize",
    "param_shift",
    "pec",
    "phase_damping",
    "qeq",
    "qgate",
    "qgt",
    "qif",
    "qlt",
    "qshow",
    "qshow_all",
    "randomized_compiling",
    "reset",
    "run_batch",
    "run_circuits",
    "set_language",
    "symmetry_verify",
    "thermal_relaxation",
    "zne",
]
