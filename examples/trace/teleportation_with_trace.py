"""用 Qiskit 后端和 ProtocolTrace 演示一轮量子隐形传态。"""

from typing import Dict, List, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qns.entity.cchannel.cchannel import (
    ClassicChannel,
    ClassicPacket,
    RecvClassicPacket,
)
from qns.entity.node.app import Application
from qns.entity.node.node import QNode
from qns.entity.operator.event import CircuitExecuteResponseEvent
from qns.entity.operator.processor import CircuitProcessor
from qns.entity.qchannel.qchannel import QuantumChannel, RecvQubitPacket
from qns.models.qubit import Qubit
from qns.simulator import Event, Simulator
from qns.simulator.trace import ProtocolTrace
from qns.simulator.ts import Time
from qns.utils import format_table


TELEPORT_STATE = np.array([
    [np.sqrt(0.3)],
    [np.exp(0.37j) * np.sqrt(0.7)],
], dtype=complex)


class StartTeleportation(Event):
    def __init__(self, t: Time, node: QNode, round_id: int, by: Application):
        super().__init__(t=t, name="teleportation-{}".format(round_id), by=by)
        self.node = node
        self.round_id = round_id

    def invoke(self) -> None:
        self.node.handle(self)


def prepare_epr() -> QuantumCircuit:
    alice = QuantumRegister(1, "A")
    bob = QuantumRegister(1, "B")
    circuit = QuantumCircuit(alice, bob, name="prepare-epr")
    circuit.h(alice)
    circuit.cx(alice, bob)
    return circuit


def bell_measurement() -> QuantumCircuit:
    message = QuantumRegister(1, "Q")
    alice = QuantumRegister(1, "A")
    bit_a = ClassicalRegister(1, "a")
    bit_b = ClassicalRegister(1, "b")
    circuit = QuantumCircuit(message, alice, bit_a, bit_b, name="bell-measurement")
    circuit.cx(message, alice)
    circuit.h(message)
    circuit.measure(alice, bit_a)
    circuit.measure(message, bit_b)
    return circuit


def correction(bit_a: int, bit_b: int) -> QuantumCircuit:
    bob = QuantumRegister(1, "B")
    circuit = QuantumCircuit(bob, name="bob-correction")
    if bit_a:
        circuit.x(bob)
    if bit_b:
        circuit.z(bob)
    return circuit


class AliceApp(Application):
    def __init__(self, bob: QNode, qchannel: QuantumChannel,
                 cchannel: ClassicChannel, rounds: int):
        super().__init__()
        self.bob = bob
        self.qchannel = qchannel
        self.cchannel = cchannel
        self.rounds = rounds
        self.sessions: Dict[int, dict] = {}
        self.add_handler(self.start, [StartTeleportation], [self])
        self.add_handler(self.circuit_completed, [CircuitExecuteResponseEvent], [])
        self.add_handler(self.classic_received, [RecvClassicPacket], [])

    def install(self, node: QNode, simulator: Simulator) -> None:
        super().install(node, simulator)
        for round_id in range(self.rounds):
            simulator.add_event(StartTeleportation(
                simulator.ts + simulator.time(sec=round_id * 0.01),
                node, round_id, self,
            ))

    def start(self, node: QNode, event: StartTeleportation) -> None:
        vector = TELEPORT_STATE.reshape(2)
        session = {
            "message": Qubit(
                state=TELEPORT_STATE.copy(),
                name="message-{}".format(event.round_id),
                backend="qiskit",
                owner=node,
            ),
            "alice_ebit": Qubit(
                name="alice-{}".format(event.round_id),
                backend="qiskit",
                owner=node,
            ),
            "bob_ebit": Qubit(
                name="bob-{}".format(event.round_id),
                backend="qiskit",
                owner=node,
            ),
            "expected_rho": np.outer(vector, vector.conjugate()),
        }
        session["bob_ebit"].round_id = event.round_id
        session["bob_ebit"].expected_rho = session["expected_rho"]
        self.sessions[event.round_id] = session
        self.get_simulator().record(
            "protocol-round-start", node=node, round=event.round_id,
        )

        circuit = prepare_epr()
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
        session = self.sessions[metadata["round"]]
        if metadata["stage"] == "prepare-epr":
            self.qchannel.send(session["bob_ebit"], self.bob)
        elif metadata["stage"] == "bell-measurement":
            self.cchannel.send(ClassicPacket(
                msg={
                    "type": "measurement",
                    "round": metadata["round"],
                    "a": event.result.get_register("a"),
                    "b": event.result.get_register("b"),
                },
                src=node,
                dest=self.bob,
            ), self.bob)

    def classic_received(self, node: QNode, event: RecvClassicPacket) -> None:
        message = event.packet.get()
        if message.get("type") != "ebit-ack":
            return
        session = self.sessions[message["round"]]
        circuit = bell_measurement()
        node.get_processor("processor").submit(
            circuit,
            {
                circuit.qregs[0][0]: session["message"],
                circuit.qregs[1][0]: session["alice_ebit"],
            },
            by=self,
            metadata={"stage": "bell-measurement", "round": message["round"]},
        )


class BobApp(Application):
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
            src=node,
            dest=self.alice,
        ), self.alice)

    def classic_received(self, node: QNode, event: RecvClassicPacket) -> None:
        message = event.packet.get()
        if message.get("type") != "measurement":
            return
        circuit = correction(message["a"], message["b"])
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
            np.trace(np.dot(qubit.expected_rho, qubit.state.rho)),
        ))
        result = {
            "round": metadata["round"],
            "fidelity": fidelity,
            "success": bool(np.allclose(qubit.state.rho, qubit.expected_rho)),
            "backend": qubit.backend_name,
        }
        self.results.append(result)
        self.get_simulator().record(
            "protocol-round-completed",
            node=node,
            round=metadata["round"],
            fidelity=fidelity,
            success=result["success"],
        )


def simulate_teleportation(rounds: int = 1) -> Tuple[List[dict], ProtocolTrace]:
    if rounds < 1:
        raise ValueError("rounds must be positive")

    simulator = Simulator(0, 1, accuracy=1_000_000)
    trace = simulator.enable_trace()
    alice = QNode("Alice")
    bob = QNode("Bob")
    qchannel = QuantumChannel("Alice-Bob-q", delay=0.01, bandwidth=0)
    cchannel = ClassicChannel("Alice-Bob-c", delay=0.005, bandwidth=0)
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
    alice_app = AliceApp(bob, qchannel, cchannel, rounds)
    bob_app = BobApp(alice, cchannel)
    alice.add_apps(alice_app)
    bob.add_apps(bob_app)
    alice.install(simulator)
    bob.install(simulator)
    simulator.run()
    return sorted(bob_app.results, key=lambda item: item["round"]), trace


def main() -> None:
    results, trace = simulate_teleportation()
    print(format_table(results))
    print("trace records:", len(trace.records))
    print("completed circuits:", len(trace.select("circuit-completed")))


if __name__ == "__main__":
    main()
