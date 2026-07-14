#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Any, List, Optional

import numpy as np

from qns.models.qubit.backends.base import QuantumStateBackend


def _index_to_bits(index: int, num: int) -> List[int]:
    return [(index >> (num - i - 1)) & 1 for i in range(num)]


def _bits_to_index(bits: List[int]) -> int:
    ret = 0
    for bit in bits:
        ret = (ret << 1) | bit
    return ret


def _operator_expand(operator: np.ndarray, num: int, qargs: List[int]) -> np.ndarray:
    """Expand an operator using SimQN's big-endian logical qubit ordering."""
    operator = np.asarray(operator, dtype=complex)
    local_num = len(qargs)
    if operator.shape != (2 ** local_num, 2 ** local_num):
        raise ValueError("operator size does not match qargs")
    if len(set(qargs)) != local_num or any(qarg < 0 or qarg >= num for qarg in qargs):
        raise ValueError("invalid qargs")

    full_operator = np.zeros((2 ** num, 2 ** num), dtype=complex)
    for column in range(2 ** num):
        column_bits = _index_to_bits(column, num)
        local_column = _bits_to_index([column_bits[qarg] for qarg in qargs])
        for local_row in range(2 ** local_num):
            value = operator[local_row, local_column]
            if value == 0:
                continue
            row_bits = column_bits.copy()
            local_row_bits = _index_to_bits(local_row, local_num)
            for idx, qarg in enumerate(qargs):
                row_bits[qarg] = local_row_bits[idx]
            full_operator[_bits_to_index(row_bits), column] = value
    return full_operator


class NumpyStateBackend(QuantumStateBackend):
    """The original NumPy density-matrix backend."""

    name = "numpy"

    def __init__(self, num: int, state: Optional[np.ndarray] = None,
                 rho: Optional[np.ndarray] = None):
        super().__init__(num)
        if rho is None:
            vector = np.asarray(state, dtype=complex).reshape((2 ** num, 1))
            self._rho = np.dot(vector, vector.T.conjugate())
        else:
            self._rho = np.asarray(rho, dtype=complex)

    @property
    def data(self) -> np.ndarray:
        return self._rho

    def tensor(self, other: QuantumStateBackend) -> "NumpyStateBackend":
        return NumpyStateBackend(self.num + other.num, rho=np.kron(self.data, other.data))

    def evolve(self, operation: Any,
               qargs: Optional[List[int]] = None) -> "NumpyStateBackend":
        full_operator = np.asarray(operation, dtype=complex)
        if qargs is not None:
            full_operator = _operator_expand(full_operator, self.num, qargs)
        rho = np.dot(full_operator, np.dot(self._rho, full_operator.T.conjugate()))
        return NumpyStateBackend(self.num, rho=rho)

    def stochastic_evolve(self, operators: List[np.ndarray], probabilities: List[float],
                          qargs: Optional[List[int]] = None) -> "NumpyStateBackend":
        rho = np.zeros_like(self._rho, dtype=complex)
        for operator, probability in zip(operators, probabilities):
            rho += probability * self.evolve(operator, qargs).data
        return NumpyStateBackend(self.num, rho=rho)

    def probabilities(self, qarg: int, projectors: List[np.ndarray]) -> List[float]:
        ret = []
        for projector in projectors:
            projected = self.evolve(projector, [qarg])
            ret.append(float(np.real_if_close(np.trace(projected.data))))
        return ret

    def collapse(self, qarg: int, projector: np.ndarray,
                 probability: float) -> "NumpyStateBackend":
        projected = self.evolve(projector, [qarg])
        return NumpyStateBackend(self.num, rho=projected.data / probability)

    def partial_trace(self, qarg: int) -> "NumpyStateBackend":
        if self.num == 1:
            return NumpyStateBackend(0, rho=np.array([[np.trace(self._rho)]], dtype=complex))
        reshaped = np.reshape(self._rho, [2, 2] * self.num)
        rho = np.trace(reshaped, axis1=qarg, axis2=self.num + qarg)
        rho = np.reshape(rho, (2 ** (self.num - 1), 2 ** (self.num - 1)))
        return NumpyStateBackend(self.num - 1, rho=rho)

    def is_pure_state(self, eps: float = 0.000_001) -> bool:
        return abs(np.trace(np.dot(self._rho, self._rho)) - 1) <= eps

    def statevector(self) -> Optional[np.ndarray]:
        if not self.is_pure_state():
            return None
        values, vectors = np.linalg.eigh(self._rho)
        vector = vectors[:, int(np.argmax(values))]
        return vector.reshape((2 ** self.num, 1))
