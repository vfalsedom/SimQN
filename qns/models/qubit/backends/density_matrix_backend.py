#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

from typing import Any, List, Optional

import numpy as np
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, partial_trace, purity

from qns.models.qubit.backends.base import QuantumStateBackend


class QiskitStateBackend(QuantumStateBackend):
    """A Qiskit ``DensityMatrix`` state backend."""

    name = "qiskit"

    def __init__(self, num: int, state: Optional[np.ndarray] = None,
                 rho: Optional[np.ndarray] = None, density_matrix: Optional[DensityMatrix] = None):
        super().__init__(num)
        dims = (2,) * num
        if density_matrix is not None:
            self._density_matrix = density_matrix
        elif rho is not None:
            self._density_matrix = DensityMatrix(np.asarray(rho, dtype=complex), dims=dims)
        else:
            vector = np.asarray(state, dtype=complex).reshape(2 ** num)
            self._density_matrix = DensityMatrix(vector, dims=dims)

    @property
    def data(self) -> np.ndarray:
        return self._density_matrix.data

    @property
    def raw_state(self) -> DensityMatrix:
        return self._density_matrix

    def _matrix_qargs(self, qargs: List[int]) -> List[int]:
        """Map a SimQN big-endian matrix operand to Qiskit subsystems."""
        return [self.num - qarg - 1 for qarg in reversed(qargs)]

    def _circuit_qargs(self, qargs: List[int]) -> List[int]:
        """Map Qiskit operation qubit i to the corresponding SimQN qubit."""
        return [self.num - qarg - 1 for qarg in qargs]

    def tensor(self, other: QuantumStateBackend) -> "QiskitStateBackend":
        if isinstance(other, QiskitStateBackend):
            density_matrix = self._density_matrix.tensor(other._density_matrix)
        else:
            density_matrix = self._density_matrix.tensor(DensityMatrix(other.data))
        return QiskitStateBackend(self.num + other.num, density_matrix=density_matrix)

    def evolve(self, operation: Any,
               qargs: Optional[List[int]] = None) -> "QiskitStateBackend":
        is_matrix = isinstance(operation, (np.ndarray, list, tuple))
        native_operation = Operator(operation) if is_matrix else operation
        if qargs is None:
            native_qargs = None
        elif is_matrix:
            native_qargs = self._matrix_qargs(qargs)
        else:
            native_qargs = self._circuit_qargs(qargs)
        density_matrix = self._density_matrix.evolve(native_operation, qargs=native_qargs)
        return QiskitStateBackend(self.num, density_matrix=density_matrix)

    def stochastic_evolve(self, operators: List[np.ndarray], probabilities: List[float],
                          qargs: Optional[List[int]] = None) -> "QiskitStateBackend":
        kraus = Kraus([np.sqrt(probability) * np.asarray(operator, dtype=complex)
                       for operator, probability in zip(operators, probabilities)])
        native_qargs = None if qargs is None else self._matrix_qargs(qargs)
        density_matrix = self._density_matrix.evolve(kraus, qargs=native_qargs)
        return QiskitStateBackend(self.num, density_matrix=density_matrix)

    def probabilities(self, qarg: int, projectors: List[np.ndarray]) -> List[float]:
        ret = []
        for projector in projectors:
            projected = self.evolve(projector, [qarg])
            ret.append(float(np.real_if_close(np.trace(projected.data))))
        return ret

    def collapse(self, qarg: int, projector: np.ndarray,
                 probability: float) -> "QiskitStateBackend":
        projected = self.evolve(projector, [qarg])
        return QiskitStateBackend(self.num, rho=projected.data / probability)

    def partial_trace(self, qarg: int) -> "QiskitStateBackend":
        if self.num == 1:
            return QiskitStateBackend(0, rho=np.array([[np.trace(self.data)]], dtype=complex))
        native_qarg = self.num - qarg - 1
        density_matrix = partial_trace(self._density_matrix, [native_qarg])
        return QiskitStateBackend(self.num - 1, density_matrix=density_matrix)

    def is_pure_state(self, eps: float = 0.000_001) -> bool:
        return abs(float(np.real_if_close(purity(self._density_matrix))) - 1) <= eps

    def statevector(self) -> Optional[np.ndarray]:
        if not self.is_pure_state():
            return None
        vector = self._density_matrix.to_statevector().data
        return vector.reshape((2 ** self.num, 1))
