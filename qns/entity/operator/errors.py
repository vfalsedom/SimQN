#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.


class CircuitExecutionError(Exception):
    """Base exception for node circuit execution."""


class UnsupportedCircuitInstructionError(CircuitExecutionError):
    """Raised when a circuit contains an unsupported instruction."""


class QubitBackendMismatchError(CircuitExecutionError):
    """Raised when a circuit processor receives a non-Qiskit state."""


class QubitBusyError(CircuitExecutionError):
    """Raised when a qubit is already reserved by another circuit job."""


class QubitOwnershipError(CircuitExecutionError):
    """Raised when a node tries to operate a qubit owned by another node."""
