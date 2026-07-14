import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix

from qns.models.epr.entanglement import BaseEntanglement
from qns.models.epr.mixed import MixedStateEntanglement
from qns.models.epr.werner import WernerStateEntanglement
from qns.models.qubit.backends import NumpyStateBackend, QiskitStateBackend, \
                                      get_default_backend, set_default_backend
from qns.models.qubit.const import OPERATOR_PAULI_I, OPERATOR_PAULI_X, \
                                   QUBIT_STATE_0, QUBIT_STATE_1, QUBIT_STATE_P, QUBIT_STATE_R
from qns.models.qubit.factory import QubitFactory
from qns.models.qubit.gate import CNOT, CR, CZ, H, RX, RY, RZ, Swap, Toffoli, U, X
from qns.models.qubit.errors import QGateStateJointError
from qns.models.qubit.qubit import Qubit
from qns.models.qubit.utils import joint
from qns.utils.rnd import set_seed


BACKENDS = ["numpy", "qiskit"]


@pytest.fixture(autouse=True)
def restore_default_backend():
    backend = get_default_backend()
    yield
    set_default_backend(backend)


def test_numpy_is_the_default_backend():
    assert get_default_backend() == "numpy"
    qubit = Qubit()
    assert qubit.backend_name == "numpy"
    assert isinstance(qubit.state.backend, NumpyStateBackend)

    factory_qubit = QubitFactory()()
    assert factory_qubit.backend_name == "numpy"
    assert isinstance(factory_qubit.state.backend, NumpyStateBackend)


@pytest.mark.parametrize("backend,backend_class", [
    ("numpy", NumpyStateBackend),
    ("qiskit", QiskitStateBackend),
])
def test_qubit_factory_selects_backend(backend, backend_class):
    factory = QubitFactory(backend=backend)
    q0 = factory()
    q1 = factory(backend="qiskit" if backend == "numpy" else "numpy")

    assert q0.backend_name == backend
    assert isinstance(q0.state.backend, backend_class)
    assert q1.backend_name != backend


def _gate_sequence(backend):
    q0 = Qubit(state=QUBIT_STATE_0, backend=backend, name="q0")
    q1 = Qubit(state=QUBIT_STATE_1, backend=backend, name="q1")
    q2 = Qubit(state=QUBIT_STATE_0, backend=backend, name="q2")

    H(q0)
    RX(q1, theta=0.31)
    RY(q2, theta=0.27)
    RZ(q0, theta=0.19)
    CNOT(q0, q1)
    CZ(q1, q2)
    CR(q0, q2, theta=0.23)
    Toffoli(q0, q1, q2)
    U(q2, 1j * OPERATOR_PAULI_I)
    Swap(q0, q2)
    return q0.state.rho


def test_numpy_and_qiskit_gate_results_are_equal():
    numpy_rho = _gate_sequence("numpy")
    qiskit_rho = _gate_sequence("qiskit")
    assert np.allclose(numpy_rho, qiskit_rho)


@pytest.mark.parametrize("backend", BACKENDS)
def test_controlled_gate_qubit_order(backend):
    control = Qubit(state=QUBIT_STATE_1, backend=backend)
    target = Qubit(state=QUBIT_STATE_0, backend=backend)
    CNOT(control, target)
    assert control.measure() == 1
    assert target.measure() == 1

    target = Qubit(state=QUBIT_STATE_0, backend=backend)
    control = Qubit(state=QUBIT_STATE_1, backend=backend)
    CNOT(control, target)
    assert control.measure() == 1
    assert target.measure() == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_toffoli_qubit_order(backend):
    q0 = Qubit(state=QUBIT_STATE_1, backend=backend)
    q1 = Qubit(state=QUBIT_STATE_1, backend=backend)
    q2 = Qubit(state=QUBIT_STATE_0, backend=backend)
    Toffoli(q0, q1, q2)
    assert q0.measure() == 1
    assert q1.measure() == 1
    assert q2.measure() == 1


@pytest.mark.parametrize("backend,state,measure,expected", [
    ("numpy", QUBIT_STATE_1, "measureZ", 1),
    ("qiskit", QUBIT_STATE_1, "measureZ", 1),
    ("numpy", QUBIT_STATE_P, "measureX", 0),
    ("qiskit", QUBIT_STATE_P, "measureX", 0),
    ("numpy", QUBIT_STATE_R, "measureY", 0),
    ("qiskit", QUBIT_STATE_R, "measureY", 0),
])
def test_measurement_bases(backend, state, measure, expected):
    qubit = Qubit(state=state, backend=backend)
    assert getattr(qubit, measure)() == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_measurement_splits_entangled_state(backend):
    set_seed(7)
    q0 = Qubit(backend=backend)
    q1 = Qubit(backend=backend)
    H(q0)
    CNOT(q0, q1)

    result0 = q0.measure()
    result1 = q1.measure()

    assert result0 == result1
    assert q0.state is not q1.state
    assert q0.state.num == 1
    assert q1.state.num == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_stochastic_operation_is_a_density_matrix_channel(backend):
    qubit = Qubit(backend=backend)
    qubit.stochastic_operate([OPERATOR_PAULI_I, OPERATOR_PAULI_X], [0.75, 0.25])
    expected = np.array([[0.75, 0], [0, 0.25]], dtype=complex)

    assert np.allclose(qubit.state.rho, expected)
    assert not qubit.state.is_pure_state()
    if backend == "qiskit":
        assert isinstance(qubit.state.backend.raw_state, DensityMatrix)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("entanglement", [
    BaseEntanglement(fidelity=0.8),
    MixedStateEntanglement(fidelity=0.8),
    WernerStateEntanglement(fidelity=0.8),
])
def test_entanglement_conversion_selects_backend(backend, entanglement):
    entanglement.is_decoherenced = False
    q0, q1 = entanglement.to_qubits(backend=backend)
    assert q0.backend_name == backend
    assert q1.backend_name == backend
    assert q0.state is q1.state


def test_default_backend_can_be_changed_for_legacy_code():
    set_default_backend("qiskit")
    qubit = Qubit()
    X(qubit)
    assert qubit.backend_name == "qiskit"
    assert isinstance(qubit.state.backend.raw_state, DensityMatrix)
    assert qubit.measure() == 1


def test_density_matrix_assignment_keeps_selected_backend():
    qubit = Qubit(backend="qiskit")
    qubit.state.rho = np.array([[0, 0], [0, 1]], dtype=complex)
    assert qubit.backend_name == "qiskit"
    assert qubit.measure() == 1


def test_states_using_different_backends_cannot_be_joined():
    q0 = Qubit(backend="numpy")
    q1 = Qubit(backend="qiskit")
    with pytest.raises(QGateStateJointError):
        joint(q0, q1)
