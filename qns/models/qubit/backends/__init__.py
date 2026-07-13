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

from typing import Optional

import numpy as np

from qns.models.qubit.backends.base import QuantumStateBackend
from qns.models.qubit.backends.numpy_backend import NumpyStateBackend
from qns.models.qubit.backends.qiskit_backend import QiskitStateBackend


_DEFAULT_BACKEND = "qiskit"
_BACKENDS = {
    "numpy": NumpyStateBackend,
    "qiskit": QiskitStateBackend,
}


def get_default_backend() -> str:
    return _DEFAULT_BACKEND


def set_default_backend(backend: str) -> None:
    global _DEFAULT_BACKEND
    if backend not in _BACKENDS:
        raise ValueError("unknown quantum state backend: {}".format(backend))
    _DEFAULT_BACKEND = backend


def create_backend(num: int, state: Optional[np.ndarray] = None,
                   rho: Optional[np.ndarray] = None, backend: Optional[str] = None) -> QuantumStateBackend:
    backend_name = backend if backend is not None else get_default_backend()
    try:
        backend_class = _BACKENDS[backend_name]
    except KeyError:
        raise ValueError("unknown quantum state backend: {}".format(backend_name))
    return backend_class(num=num, state=state, rho=rho)


__all__ = ["QuantumStateBackend", "NumpyStateBackend", "QiskitStateBackend",
           "create_backend", "get_default_backend", "set_default_backend"]
