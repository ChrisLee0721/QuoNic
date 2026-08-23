"""Distributed quantum computing — multi-chip and quantum network support.

Example::

    from quonic.distributed import QuantumNetwork, EntanglementPair
    network = QuantumNetwork(n_nodes=3)
"""

from .entanglement import (
    EntanglementPair,
    create_bell_pair,
    distribute_entanglement,
    remote_cnot,
    teleport_state,
)
from .network import Node, QuantumNetwork
from .scheduler import ScheduleStep, TaskSchedule, schedule_task

__all__ = [
    "EntanglementPair",
    "Node",
    "QuantumNetwork",
    "ScheduleStep",
    "TaskSchedule",
    "create_bell_pair",
    "distribute_entanglement",
    "remote_cnot",
    "schedule_task",
    "teleport_state",
]
