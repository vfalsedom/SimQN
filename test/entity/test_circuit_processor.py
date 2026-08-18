import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Kraus

from qns.entity.node.app import Application
from qns.entity.node.node import QNode
from qns.entity.operator.errors import CircuitExecutionError, QubitBackendMismatchError, \
                                       QubitOwnershipError
from qns.entity.operator.event import CircuitExecuteResponseEvent
from qns.entity.operator.processor import CircuitProcessor
from qns.models.qubit.const import QUBIT_STATE_0, QUBIT_STATE_1
from qns.models.qubit.qubit import Qubit
from qns.models.qubit.utils import joint
from qns.simulator.event import func_to_event
from qns.simulator.simulator import Simulator
from qns.simulator.trace import ProtocolTrace


def test_circuit_qubit_order_matches_simqn_big_endian_state() -> None:
    leading = Qubit(state=QUBIT_STATE_0, backend="qiskit")
    control = Qubit(state=QUBIT_STATE_1, backend="qiskit")
    target = Qubit(state=QUBIT_STATE_0, backend="qiskit")
    joint(leading, control)
    joint(control, target)

    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    CircuitProcessor().execute(circuit, [control, target])

    assert leading.measure() == 0
    assert control.measure() == 1
    assert target.measure() == 1


def test_qubit_mapping_does_not_depend_on_dictionary_order() -> None:
    control_register = QuantumRegister(1, "control")
    target_register = QuantumRegister(1, "target")
    circuit = QuantumCircuit(control_register, target_register)
    circuit.cx(control_register, target_register)
    control = Qubit(state=QUBIT_STATE_1, backend="qiskit")
    target = Qubit(state=QUBIT_STATE_0, backend="qiskit")

    CircuitProcessor().execute(circuit, {
        target_register[0]: target,
        control_register[0]: control,
    })

    assert control.measure() == 1
    assert target.measure() == 1


def test_qubit_mapping_requires_exact_circuit_coverage() -> None:
    circuit = QuantumCircuit(2)
    qubit_0 = Qubit(backend="qiskit")
    qubit_1 = Qubit(backend="qiskit")

    with pytest.raises(ValueError, match="does not cover"):
        CircuitProcessor().execute(circuit, {circuit.qubits[0]: qubit_0})

    foreign_qubit = QuantumCircuit(1).qubits[0]
    with pytest.raises(ValueError, match="outside the circuit"):
        CircuitProcessor().execute(circuit, {
            circuit.qubits[0]: qubit_0,
            circuit.qubits[1]: qubit_1,
            foreign_qubit: Qubit(backend="qiskit"),
        })


def test_qubit_mapping_rejects_duplicate_simqn_qubits() -> None:
    circuit = QuantumCircuit(2)
    qubit = Qubit(backend="qiskit")

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        CircuitProcessor().execute(circuit, {
            circuit.qubits[0]: qubit,
            circuit.qubits[1]: qubit,
        })


def test_measurement_and_if_test_update_live_qubit() -> None:
    qubit = Qubit(state=QUBIT_STATE_1, backend="qiskit")
    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "flag")
    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(qreg, creg)
    with circuit.if_test((creg, 1)):
        circuit.x(qreg)

    result = CircuitProcessor().execute(circuit, [qubit])

    assert result.get_register("flag") == 1
    assert result.get_bit(0) == 1
    assert qubit.measure() == 0


def test_circuit_processor_rejects_numpy_state() -> None:
    circuit = QuantumCircuit(1)
    circuit.x(0)
    with pytest.raises(QubitBackendMismatchError):
        CircuitProcessor().execute(circuit, [Qubit(backend="numpy")])


def test_circuit_executes_qiskit_quantum_channel() -> None:
    qubit = Qubit(backend="qiskit")
    identity = np.identity(2, dtype=complex)
    bit_flip = np.array([[0, 1], [1, 0]], dtype=complex)
    channel = Kraus([np.sqrt(0.75) * identity, np.sqrt(0.25) * bit_flip])
    circuit = QuantumCircuit(1)
    circuit.append(channel.to_instruction(), [0])

    CircuitProcessor().execute(circuit, [qubit])

    assert np.allclose(qubit.state.rho, np.array([[0.75, 0], [0, 0.25]], dtype=complex))


def test_strict_processor_checks_qubit_owner() -> None:
    alice = QNode("Alice")
    bob = QNode("Bob")
    qubit = Qubit(backend="qiskit", owner=bob)
    processor = CircuitProcessor(node=alice, strict_ownership=True)
    with pytest.raises(QubitOwnershipError):
        processor.execute(QuantumCircuit(1), [qubit])


class ResultApp(Application):
    def __init__(self):
        super().__init__()
        self.result = None
        self.add_handler(self.handle_result, [CircuitExecuteResponseEvent], [])

    def handle_result(self, node, event) -> None:
        self.result = event.result


def test_circuit_job_changes_state_at_completion_time() -> None:
    simulator = Simulator(0, 1, accuracy=1000)
    node = QNode("node")
    processor = CircuitProcessor("cpu", gate_duration=0.1, measure_duration=0.2)
    app = ResultApp()
    node.add_processor(processor)
    node.add_apps(app)
    node.install(simulator)

    qubit = Qubit(backend="qiskit", owner=node)
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    snapshots = []
    simulator.add_event(func_to_event(
        simulator.time(sec=0.15),
        lambda: snapshots.append(qubit.state.rho.copy()),
    ))
    processor.submit(circuit, [qubit], metadata={"round": 3})
    simulator.run()

    assert np.allclose(snapshots[0], np.array([[1, 0], [0, 0]], dtype=complex))
    assert app.result.get_bit(0) == 1
    assert app.result.metadata == {"round": 3}
    assert np.isclose(app.result.start_time, 0)
    assert np.isclose(app.result.finish_time, 0.3)


def test_submit_accepts_qubit_mapping_and_traces_binding() -> None:
    simulator = Simulator(0, 1, accuracy=1000)
    trace = ProtocolTrace()
    simulator.enable_trace(trace)
    node = QNode("node")
    processor = CircuitProcessor("cpu", gate_duration=0.1)
    node.add_processor(processor)
    node.install(simulator)
    control_register = QuantumRegister(1, "control")
    target_register = QuantumRegister(1, "target")
    circuit = QuantumCircuit(control_register, target_register)
    circuit.cx(control_register, target_register)
    control = Qubit(state=QUBIT_STATE_1, name="source", backend="qiskit", owner=node)
    target = Qubit(state=QUBIT_STATE_0, name="destination", backend="qiskit", owner=node)

    processor.submit(circuit, {
        target_register[0]: target,
        control_register[0]: control,
    })
    simulator.run()

    assert control.measure() == 1
    assert target.measure() == 1
    completed = trace.select(kind="circuit-completed")[0]
    assert [(item["circuit_qubit"], item["simqn_qubit"])
            for item in completed["qubit_mapping"]] == [
        ("control[0]", "source"),
        ("target[0]", "destination"),
    ]


def test_same_qubit_jobs_are_executed_in_queue_order() -> None:
    simulator = Simulator(0, 1, accuracy=1000)
    trace = ProtocolTrace()
    simulator.enable_trace(trace)
    node = QNode("node")
    processor = CircuitProcessor("cpu", gate_duration=0.1)
    node.add_processor(processor)
    node.install(simulator)
    qubit = Qubit(backend="qiskit", owner=node)
    circuit = QuantumCircuit(1)
    circuit.x(0)

    processor.submit(circuit, [qubit])
    processor.submit(circuit, [qubit])
    simulator.run()

    starts = trace.select(kind="circuit-started")
    assert [record["time"] for record in starts] == [0, 0.1]
    assert np.allclose(qubit.state.rho, np.array([[1, 0], [0, 0]], dtype=complex))
    assert not qubit.is_locked
    assert not processor.is_busy()


def test_job_beyond_simulation_end_does_not_lock_qubit() -> None:
    simulator = Simulator(0, 0.05, accuracy=1000)
    node = QNode("node")
    processor = CircuitProcessor("cpu", gate_duration=0.1)
    node.add_processor(processor)
    node.install(simulator)
    qubit = Qubit(backend="qiskit", owner=node)
    circuit = QuantumCircuit(1)
    circuit.x(0)
    processor.submit(circuit, [qubit])

    with pytest.raises(CircuitExecutionError, match="simulation end time"):
        simulator.run()

    assert not qubit.is_locked
    assert np.allclose(qubit.state.rho, np.array([[1, 0], [0, 0]], dtype=complex))


def test_submit_freezes_circuit_and_trace_snapshot() -> None:
    simulator = Simulator(0, 1, accuracy=1000)
    trace = ProtocolTrace()
    simulator.enable_trace(trace)
    node = QNode("node")
    processor = CircuitProcessor("cpu", gate_duration=0.1)
    node.add_processor(processor)
    node.install(simulator)
    qubit = Qubit(backend="qiskit", owner=node)
    circuit = QuantumCircuit(1)
    circuit.x(0)

    processor.submit(circuit, [qubit])
    circuit.x(0)
    simulator.run()

    completed = trace.select(kind="circuit-completed")
    assert len(completed[0]["circuit"].data) == 1
    assert np.allclose(qubit.state.rho, np.array([[0, 0], [0, 1]], dtype=complex))


def test_trace_builds_circuit_for_unnamed_qubits() -> None:
    simulator = Simulator(0, 1, accuracy=1000)
    trace = ProtocolTrace()
    simulator.enable_trace(trace)
    node = QNode("node")
    processor = CircuitProcessor("cpu")
    node.add_processor(processor)
    node.install(simulator)
    control = Qubit(backend="qiskit", owner=node)
    target = Qubit(backend="qiskit", owner=node)
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)

    processor.submit(circuit, [control, target])
    simulator.run()
    combined = trace.build_circuit()

    assert combined.num_qubits == 2
    completed = trace.select("circuit-completed")
    assert len(set(completed[0]["qubits"])) == 2


def test_trace_records_are_chronological_for_queued_jobs() -> None:
    simulator = Simulator(0, 1, accuracy=1000)
    trace = ProtocolTrace()
    simulator.enable_trace(trace)
    node = QNode("node")
    processor = CircuitProcessor("cpu", gate_duration=0.1)
    node.add_processor(processor)
    node.install(simulator)
    circuit = QuantumCircuit(1)
    circuit.x(0)

    processor.submit(circuit, [Qubit(backend="qiskit", owner=node)])
    processor.submit(circuit, [Qubit(backend="qiskit", owner=node)])
    simulator.run()

    times = [record["time"] for record in trace.records]
    assert times == sorted(times)
