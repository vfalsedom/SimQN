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

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class QuantumStateBackend(ABC):
    """The interface implemented by numerical quantum state backends."""

    name: str

    def __init__(self, num: int):
        self.num = num

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """Return the density matrix represented by this backend."""

    @property
    def raw_state(self):
        """Return the backend-native state object."""
        return self.data

    @abstractmethod
    def tensor(self, other: "QuantumStateBackend") -> "QuantumStateBackend":
        """Return ``self tensor other``."""

    @abstractmethod
    def evolve(self, operator: np.ndarray,
               qargs: Optional[List[int]] = None) -> "QuantumStateBackend":
        """Evolve the state with an operator."""

    @abstractmethod
    def stochastic_evolve(self, operators: List[np.ndarray], probabilities: List[float],
                          qargs: Optional[List[int]] = None) -> "QuantumStateBackend":
        """Evolve the state with a mixture of operators."""

    @abstractmethod
    def probabilities(self, qarg: int, projectors: List[np.ndarray]) -> List[float]:
        """Return measurement probabilities for a subsystem."""

    @abstractmethod
    def collapse(self, qarg: int, projector: np.ndarray,
                 probability: float) -> "QuantumStateBackend":
        """Return the normalized post-measurement state."""

    @abstractmethod
    def partial_trace(self, qarg: int) -> "QuantumStateBackend":
        """Trace one subsystem out of the state."""

    @abstractmethod
    def is_pure_state(self, eps: float = 0.000_001) -> bool:
        """Return whether this state is pure."""

    @abstractmethod
    def statevector(self) -> Optional[np.ndarray]:
        """Return a column statevector for a pure state, otherwise ``None``."""
