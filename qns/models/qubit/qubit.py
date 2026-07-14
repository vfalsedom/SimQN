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

from qns.models.qubit.const import QUBIT_STATE_0, QUBIT_STATE_1, \
        QUBIT_STATE_P, QUBIT_STATE_N, QUBIT_STATE_L, QUBIT_STATE_R
from qns.models.qubit.backends import create_backend, get_default_backend
from qns.models.qubit.backends.base import QuantumStateBackend
from qns.models.core.backend import QuantumModel
from qns.models.qubit.errors import QStateBaseError, QStateQubitNotInStateError, \
                                    QStateSizeNotMatchError, OperatorNotMatchError, QGateStateJointError
from qns.utils.rnd import get_rand


class QState(object):
    """
    QState is the state of one (or multiple) qubits
    """
    def __init__(self, qubits: Optional[List["Qubit"]] = None, state: Optional[np.ndarray] = QUBIT_STATE_0,
                 rho: Optional[np.ndarray] = None, name: Optional[str] = None,
                 backend: Optional[str] = None, backend_state: Optional[QuantumStateBackend] = None):
        """
        Args:
            qubits (List[Qubit]): a list of qubits in this quantum state
            state: the state vector of this state, either ``state`` or ``rho`` can be used to present a state
            rho: the density matrix of this state, either ``state`` or ``rho`` can be used to present a state
            name (str): the name of this state
            backend (str): the quantum state backend, ``numpy`` or ``qiskit``
        """
        qubits = [] if qubits is None else qubits
        self.num = len(qubits)
        self.name = name
        self.qubits = qubits

        if backend_state is not None:
            if backend_state.num != self.num:
                raise QStateSizeNotMatchError
            self._backend = backend_state
            return

        if backend is None:
            backend_names = {qubit.state.backend_name for qubit in qubits
                             if hasattr(qubit, "state") and hasattr(qubit.state, "backend_name")}
            if len(backend_names) == 1:
                backend = backend_names.pop()
            elif len(backend_names) > 1:
                raise QGateStateJointError
            else:
                backend = get_default_backend()

        if rho is None:
            if state is None or len(state) != 2**self.num:
                raise QStateSizeNotMatchError
        else:
            if rho.shape != (2 ** self.num, 2 ** self.num):
                raise QStateSizeNotMatchError
            if abs(1 - rho.trace()) > 0.0000000001:
                raise QStateSizeNotMatchError
        self._backend = create_backend(self.num, state=state, rho=rho, backend=backend)

    @property
    def rho(self) -> np.ndarray:
        """Return the density matrix for compatibility with the NumPy backend API."""
        return self._backend.data

    @rho.setter
    def rho(self, rho: np.ndarray) -> None:
        if rho.shape != (2 ** self.num, 2 ** self.num) or abs(1 - rho.trace()) > 0.0000000001:
            raise QStateSizeNotMatchError
        self._backend = create_backend(self.num, rho=rho, backend=self.backend_name)

    @property
    def backend(self) -> QuantumStateBackend:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @classmethod
    def join(cls, state1: "QState", state2: "QState") -> "QState":
        if state1.backend_name != state2.backend_name:
            raise QGateStateJointError
        backend_state = state1.backend.tensor(state2.backend)
        return cls(state1.qubits + state2.qubits, backend_state=backend_state)

    def measure(self, qubit: "Qubit" = None, base: str = "Z") -> int:
        """
        Measure this qubit using Z basis
        Args:
            qubit (Qubit): the measuring qubit
            base: the measure base, "Z", "X" or "Y"

        Returns:
            0: QUBIT_STATE_0 state
            1: QUBIT_STATE_1 state
        """
        if base == "Z":
            M_0 = np.array([[1, 0], [0, 0]])
            M_1 = np.array([[0, 0], [0, 1]])
            S_0 = QUBIT_STATE_0
            S_1 = QUBIT_STATE_1
        elif base == "X":
            M_0 = 1/2 * np.array([[1, 1], [1, 1]])
            M_1 = 1/2 * np.array([[1, -1], [-1, 1]])
            S_0 = QUBIT_STATE_P
            S_1 = QUBIT_STATE_N
        elif base == "Y":
            M_0 = 1/2 * np.array([[1, -1j], [1j, 1]])
            M_1 = 1/2 * np.array([[1, 1j], [-1j, 1]])
            S_0 = QUBIT_STATE_R
            S_1 = QUBIT_STATE_L
        else:
            raise QStateBaseError

        try:
            idx = self.qubits.index(qubit)
        except ValueError:
            raise QStateQubitNotInStateError
        poss_0, poss_1 = self._backend.probabilities(idx, [M_0, M_1])
        total = poss_0 + poss_1
        poss_0 = min(1.0, max(0.0, poss_0 / total))
        poss_1 = 1.0 - poss_0
        rn = get_rand()

        if rn < poss_0:
            ret = 0
            ret_s = S_0
            collapsed = self._backend.collapse(idx, M_0, poss_0)
        else:
            ret = 1
            ret_s = S_1
            collapsed = self._backend.collapse(idx, M_1, poss_1)

        backend_name = self.backend_name
        self._backend = collapsed.partial_trace(idx)
        self.num -= 1
        self.qubits.remove(qubit)

        ns = QState([qubit], state=ret_s, backend=backend_name)
        qubit.state = ns
        return ret

    def operate(self, operator: np.ndarray):
        """
        transform using `operator`

        Args:
            operator (np.ndarray): the operator
        Raises:
            OperatorNotMatchError
        """
        if isinstance(operator, np.ndarray):
            if operator.shape != (2**self.num, 2**self.num):
                raise OperatorNotMatchError
        elif getattr(operator, "num_qubits", self.num) != self.num:
            raise OperatorNotMatchError
        self._backend = self._backend.evolve(operator)

    def operate_on(self, operator, qubits: List["Qubit"]):
        """Transform selected qubits using an operator in SimQN qubit order."""
        try:
            qargs = [self.qubits.index(qubit) for qubit in qubits]
        except ValueError:
            raise QStateQubitNotInStateError
        if isinstance(operator, np.ndarray):
            if operator.shape != (2 ** len(qargs), 2 ** len(qargs)):
                raise OperatorNotMatchError
        elif getattr(operator, "num_qubits", len(qargs)) != len(qargs):
            raise OperatorNotMatchError
        self._backend = self._backend.evolve(operator, qargs)

    def apply_operation(self, operation, qubits: List["Qubit"]) -> None:
        """Apply a backend-native operation to selected qubits."""
        self.operate_on(operation, qubits)

    def stochastic_operate(self, list_operators: List[np.ndarray] = [], list_p: List[float] = []):
        """
        A stochastic operate progess. It usually turns a pure state into a mixed state.

        Args:
            list_operators (List[np.ndarray]): a list of operators
            list_p (List[float]): a list of possibility
        Raises:
            OperatorNotMatchError
        """
        if len(list_operators) != len(list_p):
            raise OperatorNotMatchError("Not match number between operators and possibilities")

        sum = 0.0
        for p in list_p:
            if p < 0 or p > 1:
                raise OperatorNotMatchError("possibility not in range")
            sum += p

        if abs(1-sum) >= 1e-6:
            raise OperatorNotMatchError("Probabilities are not normalized")

        for operator in list_operators:
            if operator.shape != (2**self.num, 2**self.num):
                raise OperatorNotMatchError
        self._backend = self._backend.stochastic_evolve(list_operators, list_p)

    def stochastic_operate_on(self, list_operators: List[np.ndarray], list_p: List[float],
                              qubits: List["Qubit"]):
        if len(list_operators) != len(list_p):
            raise OperatorNotMatchError("Not match number between operators and possibilities")
        if any(p < 0 or p > 1 for p in list_p) or abs(1 - sum(list_p)) >= 1e-6:
            raise OperatorNotMatchError("Probabilities are not normalized")
        try:
            qargs = [self.qubits.index(qubit) for qubit in qubits]
        except ValueError:
            raise QStateQubitNotInStateError
        if any(operator.shape != (2 ** len(qargs), 2 ** len(qargs)) for operator in list_operators):
            raise OperatorNotMatchError
        self._backend = self._backend.stochastic_evolve(list_operators, list_p, qargs)

    def equal(self, other_state: "QState") -> bool:
        """
        compare two state vectors, return True if they are the same

        Args:
            other_state (QState): the second QState
        """
        return np.allclose(self.rho, other_state.rho)

    def is_pure_state(self, eps: float = 0.000_001) -> bool:
        """
        Args:
            eps: the accuracy

        Returns:
            bool, if the state is a pure state
        """
        return self._backend.is_pure_state(eps)

    def state(self) -> np.ndarray:
        """
        If the state is a pure state, return the state vector, or return None

        Returns:
            The pure state vector
        """
        return self._backend.statevector()

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qubit state "+self.name+">"
        return str(self.rho)


class Qubit(QuantumModel):
    """
    Represent a qubit
    """

    def __init__(self, state=QUBIT_STATE_0, rho: np.ndarray = None,
                 operate_decoherence_rate: float = 0, measure_decoherence_rate: float = 0,
                 name: Optional[str] = None, backend: Optional[str] = None,
                 owner: Optional[Any] = None):
        """
        Args:
            state (list): the initial state of a qubit, default is |0> = [1, 0]^T
            operate_decoherence_rate (float): the operate decoherence rate
            measure_decoherence_rate (float): the measure decoherence rate
            name (str): the qubit's name
            backend (str): the quantum state backend, ``numpy`` or ``qiskit``
            owner: the node that currently owns this qubit
        """

        self.name = name
        self.state = QState([self], state=state, rho=rho, backend=backend)
        self.operate_decoherence_rate = operate_decoherence_rate
        self.measure_decoherence_rate = measure_decoherence_rate
        self.owner = owner
        self._locked_by = None

    def measure(self):
        """
        Measure this qubit using Z basis

        Returns:
            0: QUBIT_STATE_0 state
            1: QUBIT_STATE_1 state
        """
        self.measure_error_model(decoherence_rate=self.measure_decoherence_rate)
        return self.state.measure(self)

    def measureX(self):
        """
        Measure this qubit using X basis.

        Returns:
            0: QUBIT_STATE_P state
            1: QUBIT_STATE_N state
        """
        self.measure_error_model(self.measure_decoherence_rate)
        return self.state.measure(self, "X")

    def measureY(self):
        """
        Measure this qubit using Y basis.
        Only for not entangled qubits.

        Returns:
            0: QUBIT_STATE_R state
            1: QUBIT_STATE_L state
        """
        self.measure_error_model(self.measure_decoherence_rate)
        return self.state.measure(self, "Y")

    def measureZ(self):
        """
        Measure this qubit using Z basis

        Returns:
            0: QUBIT_STATE_0 state
            1: QUBIT_STATE_1 state
        """
        self.measure_error_model(self.measure_decoherence_rate)
        return self.measure()

    def operate(self, operator: Any) -> None:
        """
        Perfrom a operate on this qubit

        Args:
            operator (Union[SingleQubitGate, np.ndarray]): an operator matrix, or a quantum gate in qubit.gate
        """
        self.operate_error_model(self.operate_decoherence_rate)
        from qns.models.qubit.gate import SingleQubitGate
        if isinstance(operator, SingleQubitGate):
            operator(self)
            return
        self.state.operate_on(operator, [self])

    def _operate_without_error(self, operator: Any) -> None:
        """
        Perfrom a operate on this qubit

        Args:
            operator (Union[SingleQubitGate, np.ndarray]): an operator matrix, or a quantum gate in qubit.gate
        """
        from qns.models.qubit.gate import SingleQubitGate
        if isinstance(operator, SingleQubitGate):
            operator(self)
            return
        self.state.operate_on(operator, [self])

    def stochastic_operate(self, list_operators: List[np.ndarray] = [], list_p: List[float] = []):
        """
        A stochastic operate on this qubit. It usually turns a pure state into a mixed state.

        Args:
            list_operators (List[np.ndarray]): a list of operators
            list_p (List[float]): a list of possibility
        Raises:
            OperatorNotMatchError
        """
        from qns.models.qubit.gate import SingleQubitGate
        operators_list = []
        for operator in list_operators:
            if isinstance(operator, SingleQubitGate):
                operators_list.append(operator._operator)
            else:
                operators_list.append(operator)
        self.state.stochastic_operate_on(operators_list, list_p, [self])

    @property
    def backend_name(self) -> str:
        return self.state.backend_name

    @property
    def is_locked(self) -> bool:
        return self._locked_by is not None

    def lock(self, job_id: str) -> None:
        from qns.entity.operator.errors import QubitBusyError
        if self._locked_by is not None and self._locked_by != job_id:
            raise QubitBusyError("qubit {} is locked by {}".format(self, self._locked_by))
        self._locked_by = job_id

    def unlock(self, job_id: str) -> None:
        if self._locked_by == job_id:
            self._locked_by = None

    def set_owner(self, owner: Optional[Any]) -> None:
        self.owner = owner

    def __repr__(self) -> str:
        if self.name is not None:
            return "<qubit "+self.name+">"
        return super().__repr__()

    def store_error_model(self, t: Optional[float] = 0, decoherence_rate: Optional[float] = 0, **kwargs):
        """
        The default error model for storing a qubit in quantum memory.
        The default behavior is doing nothing

        Args:
            t: the time stored in a quantum memory. The unit it second.
            decoherence_rate (float): the decoherence rate in Db.
            kwargs: other parameters
        """
        pass

    def transfer_error_model(self, length: Optional[float] = 0, decoherence_rate: Optional[float] = 0, **kwargs):
        """
        The default error model for transmitting this qubit
        The default behavior is doing nothing

        Args:
            length (float): the length of the channel
            decoherence_rate (float): the decoherence rate
            kwargs: other parameters
        """
        pass

    def operate_error_model(self, decoherence_rate: Optional[float] = 0, **kwargs):
        """
        The error model for operating a qubit.
        This function will change the quantum state.

        Args:
            decoherence_rate (float): the decoherency rate
            kwargs: other parameters
        """
        pass

    def measure_error_model(self, decoherence_rate: Optional[float] = 0, **kwargs):
        """
        The error model for measuring a qubit.
        This function will change the quantum state.

        Args:
            decoherence_rate (float): the decoherency rate
            kwargs: other parameters
        """
        pass
