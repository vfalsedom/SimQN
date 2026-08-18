from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qns.entity.operator import processor
from qns.entity.cchannel.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.node.app import Application
from qns.entity.node.node import QNode
from qns.entity.operator.event import CircuitExecuteResponseEvent
from qns.entity.operator.processor import CircuitProcessor
from qns.entity.qchannel.qchannel import QuantumChannel, RecvQubitPacket
from qns.models.qubit.qubit import Qubit
from qns.simulator.event import Event
from qns.simulator.simulator import Simulator
from qns.simulator.trace import ProtocolTrace
from qns.simulator.ts import Time
from qns.utils.table import format_table
from qns.utils.rnd import set_seed


TELEPORT_STATE = np.array([
    [np.sqrt(0.3)],
    [np.exp(0.37j) * np.sqrt(0.7)],
], dtype=complex)


def _epr_circuit() -> QuantumCircuit:
    alice_ebit = QuantumRegister(1, "A")
    bob_ebit = QuantumRegister(1, "B")
    circuit = QuantumCircuit(alice_ebit, bob_ebit, name="prepare-epr")
    circuit.h(alice_ebit)
    circuit.cx(alice_ebit, bob_ebit)
    return circuit


def _bell_circuit() -> QuantumCircuit:
    message = QuantumRegister(1, "Q")
    alice_ebit = QuantumRegister(1, "A")
    bit_a = ClassicalRegister(1, "a")
    bit_b = ClassicalRegister(1, "b")
    circuit = QuantumCircuit(message, alice_ebit, bit_a, bit_b,
                             name="bell-measurement")
    circuit.cx(message, alice_ebit)
    circuit.h(message)
    circuit.measure(alice_ebit, bit_a)
    circuit.measure(message, bit_b)
    return circuit


def _correction_circuit(bit_a: int, bit_b: int) -> QuantumCircuit:
    bob_ebit = QuantumRegister(1, "B")
    circuit = QuantumCircuit(bob_ebit, name="bob-correction")
    if bit_a:
        circuit.x(bob_ebit)
    if bit_b:
        circuit.z(bob_ebit)
    return circuit


class StartTeleportation(Event):
    def __init__(self, t: Time, dest: QNode, round_id: int, by: Application):
        super().__init__(t=t, name="teleportation-{}".format(round_id), by=by)
        self.dest = dest
        self.round_id = round_id

    def invoke(self) -> None:
        self.dest.handle(self)


class AliceProtocol(Application):
    def __init__(self, bob: QNode, qchannel: QuantumChannel,
                 cchannel: ClassicChannel, rounds: int, round_interval: float):
        super().__init__()
        self.bob = bob
        self.qchannel = qchannel
        self.cchannel = cchannel
        self.rounds = rounds
        self.round_interval = round_interval
        self.sessions: Dict[int, dict] = {}
        self.add_handler(self.start, [StartTeleportation], [self])
        self.add_handler(self.circuit_completed, [CircuitExecuteResponseEvent], [])
        self.add_handler(self.classic_received, [RecvClassicPacket], [])

    def install(self, node: QNode, simulator: Simulator) -> None:
        super().install(node, simulator)
        for round_id in range(self.rounds):
            simulator.add_event(StartTeleportation(
                simulator.ts + simulator.time(sec=round_id * self.round_interval),
                node, round_id, self,
            ))

    def start(self, node: QNode, event: StartTeleportation) -> None:
        vector = TELEPORT_STATE.reshape(2)
        session = {
            "message": Qubit(state=TELEPORT_STATE.copy(), name="Q-{}".format(event.round_id),
                             backend="qiskit", owner=node),
            "alice_ebit": Qubit(name="A-{}".format(event.round_id),
                                backend="qiskit", owner=node),
            "bob_ebit": Qubit(name="B-{}".format(event.round_id),
                              backend="qiskit", owner=node),
            "expected_rho": np.outer(vector, vector.conjugate()),
            "start_time": self.get_simulator().current_time.sec,
        }
        session["bob_ebit"].round_id = event.round_id
        session["bob_ebit"].expected_rho = session["expected_rho"]
        session["bob_ebit"].teleport_start_time = session["start_time"]
        self.sessions[event.round_id] = session
        self.get_simulator().record("protocol-round-start", node=node, round=event.round_id)
        circuit = _epr_circuit()
        node.get_processor("processor").submit(
            circuit,
            {
                circuit.qregs[0][0]: session["alice_ebit"],
                circuit.qregs[1][0]: session["bob_ebit"],
            },
            by=self,
            metadata={"stage": "prepare-epr", "round": event.round_id},
        )

    def circuit_completed(self, node: QNode,
                          event: CircuitExecuteResponseEvent) -> None:
        metadata = event.result.metadata
        round_id = metadata["round"]
        session = self.sessions[round_id]
        if metadata["stage"] == "prepare-epr":
            self.qchannel.send(session["bob_ebit"], self.bob)
        elif metadata["stage"] == "bell-measurement":
            bit_a = event.result.get_register("a")
            bit_b = event.result.get_register("b")
            self.cchannel.send(ClassicPacket(
                msg={"type": "measurement", "round": round_id,
                     "a": bit_a, "b": bit_b},
                src=node, dest=self.bob,
            ), self.bob)

    def classic_received(self, node: QNode, event: RecvClassicPacket) -> None:
        message = event.packet.get()
        if message.get("type") != "ebit-ack":
            return
        round_id = message["round"]
        session = self.sessions[round_id]
        circuit = _bell_circuit()
        node.get_processor("processor").submit(
            circuit,
            {
                circuit.qregs[0][0]: session["message"],
                circuit.qregs[1][0]: session["alice_ebit"],
            },
            by=self,
            metadata={"stage": "bell-measurement", "round": round_id},
        )


class BobProtocol(Application):
    def __init__(self, alice: QNode, cchannel: ClassicChannel):
        super().__init__()
        self.alice = alice
        self.cchannel = cchannel
        self.pending: Dict[int, Qubit] = {}
        self.results: List[dict] = []
        self.add_handler(self.qubit_received, [RecvQubitPacket], [])
        self.add_handler(self.classic_received, [RecvClassicPacket], [])
        self.add_handler(self.circuit_completed, [CircuitExecuteResponseEvent], [])

    def qubit_received(self, node: QNode, event: RecvQubitPacket) -> None:
        qubit = event.qubit
        self.pending[qubit.round_id] = qubit
        self.cchannel.send(ClassicPacket(
            msg={"type": "ebit-ack", "round": qubit.round_id},
            src=node, dest=self.alice,
        ), self.alice)

    def classic_received(self, node: QNode, event: RecvClassicPacket) -> None:
        message = event.packet.get()
        if message.get("type") != "measurement":
            return
        circuit = _correction_circuit(message["a"], message["b"])
        node.get_processor("processor").submit(
            circuit,
            {circuit.qregs[0][0]: self.pending[message["round"]]},
            by=self,
            metadata={"stage": "bob-correction", **message},
        )

    def circuit_completed(self, node: QNode,
                          event: CircuitExecuteResponseEvent) -> None:
        metadata = event.result.metadata
        if metadata["stage"] != "bob-correction":
            return
        qubit = self.pending.pop(metadata["round"])
        fidelity = float(np.real_if_close(
            np.trace(np.dot(qubit.expected_rho, qubit.state.rho))
        ))
        result = {
            "round": metadata["round"],
            "start_time": qubit.teleport_start_time,
            "complete_time": self.get_simulator().current_time.sec,
            "fidelity": fidelity,
            "success": np.allclose(qubit.state.rho, qubit.expected_rho),
            "backend": qubit.backend_name,
        }
        self.results.append(result)
        self.get_simulator().record(
            "protocol-round-completed", node=node, round=metadata["round"],
            fidelity=fidelity, success=result["success"],
        )


def simulate_teleportation(rounds: int = 2, quantum_delay: float = 0.01,
                           classic_delay: float = 0.005,
                           round_interval: float = 0.004,
                           trace: Optional[ProtocolTrace] = None) \
        -> Tuple[List[dict], ProtocolTrace]:
    if rounds < 1:
        raise ValueError("rounds must be positive")

    trace = ProtocolTrace() if trace is None else trace
    simulator = Simulator(0, 1, accuracy=1_000_000)
    simulator.enable_trace(trace)
    alice = QNode("Alice")
    bob = QNode("Bob")
    qchannel = QuantumChannel("Alice-Bob-q", delay=quantum_delay, bandwidth=0)
    cchannel = ClassicChannel("Alice-Bob-c", delay=classic_delay, bandwidth=0)
    alice.add_qchannel(qchannel)
    bob.add_qchannel(qchannel)
    alice.add_cchannel(cchannel)
    bob.add_cchannel(cchannel)
    alice.add_processor(CircuitProcessor(
        "processor", gate_duration=0.001, measure_duration=0.002,
        strict_ownership=True,
    ))
    bob.add_processor(CircuitProcessor(
        "processor", gate_duration=0.001, measure_duration=0.002,
        strict_ownership=True,
    ))
    alice_protocol = AliceProtocol(bob, qchannel, cchannel, rounds, round_interval)
    bob_protocol = BobProtocol(alice, cchannel)
    alice.add_apps(alice_protocol)
    bob.add_apps(bob_protocol)
    bob.install(simulator)
    alice.install(simulator)
    simulator.run()
    return sorted(bob_protocol.results, key=lambda item: item["round"]), trace


def test_teleportation_flow_results_and_visuals(tmp_path: Path) -> None:
    set_seed(23)
    results, trace = simulate_teleportation()

    assert len(results) == 2
    assert all(result["success"] for result in results)
    assert all(np.isclose(result["fidelity"], 1.0) for result in results)
    assert all(result["backend"] == "qiskit" for result in results)
    assert all(result["complete_time"] > result["start_time"] for result in results)

    flow_kinds = {
        "protocol-round-start", "circuit-completed", "quantum-received",
        "classic-received", "protocol-round-completed",
    }
    for round_id in range(2):
        flow = [
            (record["kind"], record.get("stage", record.get("label")))
            for record in trace.records
            if record.get("round") == round_id and record["kind"] in flow_kinds
        ]
        assert flow == [
            ("protocol-round-start", None),
            ("circuit-completed", "prepare-epr"),
            ("quantum-received", "B-{}".format(round_id)),
            ("classic-received", "ebit-ack"),
            ("circuit-completed", "bell-measurement"),
            ("classic-received", "measurement"),
            ("circuit-completed", "bob-correction"),
            ("protocol-round-completed", None),
        ]
        times = [record["time"] for record in trace.records
                 if record.get("round") == round_id]
        assert times == sorted(times)

    timeline_path = tmp_path / "timeline.png"
    timeline = trace.draw_timeline(timeline_path)
    assert timeline.axes and timeline_path.exists()
    plt.close(timeline)

    local_directory = tmp_path / "local-circuits"
    local_drawings = trace.draw_circuits(output="mpl", directory=local_directory)
    assert len(local_drawings) == 6
    assert len(list(local_directory.glob("*.png"))) == 6
    for drawing in local_drawings.values():
        assert drawing.axes
        plt.close(drawing)

    total_circuit = trace.build_circuit(round=0)
    assert total_circuit.num_qubits == 3
    assert total_circuit.num_clbits == 2
    assert {instruction.operation.name for instruction in total_circuit.data} >= {
        "h", "cx", "measure",
    }
    total_path = tmp_path / "total-circuit.png"
    total_drawing = trace.draw_circuit(round=0, filename=total_path)
    assert total_drawing.axes and total_path.exists()
    plt.close(total_drawing)


def main() -> None:
    set_seed(23)
    results, trace = simulate_teleportation(rounds=1)
    directory = Path(__file__).parent
    timeline = trace.draw_timeline(directory / "teleportation_timeline.png")
    plt.close(timeline)
    local_drawings = trace.draw_circuits(
        output="mpl", directory=directory / "teleportation_local_circuits",
    )
    for drawing in local_drawings.values():
        plt.close(drawing)
    total = trace.draw_circuit(
        round=0, filename=directory / "teleportation_circuit.png",
    )
    plt.close(total)
    print("\nGenerated artifacts")
    print("===================")
    print(format_table({
        "artifact": ["timeline", "local circuits", "total circuit"],
        "path": [
            str(directory / "teleportation_timeline.png"),
            str(directory / "teleportation_local_circuits"),
            str(directory / "teleportation_circuit.png"),
        ],
    }))
    print("\nTeleportation results")
    print("=====================")
    print(format_table(results))


if __name__ == "__main__":
    main()
