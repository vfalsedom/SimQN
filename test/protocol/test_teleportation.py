from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qns.entity.cchannel.cchannel import ClassicChannel, ClassicPacket, RecvClassicPacket
from qns.entity.node.app import Application
from qns.entity.node.node import QNode
from qns.models.qubit.const import QUBIT_STATE_0, QUBIT_STATE_1, QUBIT_STATE_P, QUBIT_STATE_R
from qns.models.qubit.gate import CNOT, H, X, Z
from qns.models.qubit.qubit import Qubit
from qns.simulator.event import Event
from qns.simulator.simulator import Simulator
from qns.simulator.ts import Time
from qns.utils.rnd import set_seed


TELEPORT_STATES = [
    QUBIT_STATE_0,
    QUBIT_STATE_1,
    QUBIT_STATE_P,
    QUBIT_STATE_R,
    np.array([[np.sqrt(0.3)], [np.exp(0.37j) * np.sqrt(0.7)]], dtype=complex),
]


def build_teleportation_circuit() -> QuantumCircuit:
    """Build the circuit diagram corresponding to the SimQN protocol below."""
    qubit = QuantumRegister(1, "Q")
    ebit0 = QuantumRegister(1, "A")
    ebit1 = QuantumRegister(1, "B")
    bit_a = ClassicalRegister(1, "a")
    bit_b = ClassicalRegister(1, "b")
    protocol = QuantumCircuit(qubit, ebit0, ebit1, bit_a, bit_b)

    protocol.h(ebit0)
    protocol.cx(ebit0, ebit1)
    protocol.barrier()

    protocol.cx(qubit, ebit0)
    protocol.h(qubit)
    protocol.barrier()

    protocol.measure(ebit0, bit_a)
    protocol.measure(qubit, bit_b)
    protocol.barrier()

    with protocol.if_test((bit_a, 1)):
        protocol.x(ebit1)
    with protocol.if_test((bit_b, 1)):
        protocol.z(ebit1)
    return protocol


def draw_teleportation_circuit(output: str = "mpl",
                               filename: Optional[Union[str, Path]] = None):
    """Draw the protocol; ``mpl`` returns a figure and ``text`` works in a terminal."""
    if filename is not None:
        filename = str(filename)
    return build_teleportation_circuit().draw(output=output, filename=filename)


class StartTeleportationEvent(Event):
    """Start one independent teleportation round at Alice."""

    def __init__(self, t: Time, dest: QNode, round_id: int, by: Application):
        super().__init__(t=t, name="teleportation-{}".format(round_id), by=by)
        self.dest = dest
        self.round_id = round_id

    def invoke(self) -> None:
        self.dest.handle(self)


class BobTeleportationApp(Application):
    """Store Bob's EPR halves and apply corrections after classic messages arrive."""

    def __init__(self):
        super().__init__()
        self.pending: Dict[int, Tuple[Qubit, np.ndarray, float]] = {}
        self.results: List[dict] = []
        self.timeline: List[dict] = []
        self.add_handler(self.handle_packet, [RecvClassicPacket], [])

    def reserve(self, round_id: int, qubit: Qubit,
                expected_rho: np.ndarray, start_time: float) -> None:
        self.pending[round_id] = (qubit, expected_rho, start_time)

    def handle_packet(self, node: QNode, event: RecvClassicPacket) -> None:
        message = event.packet.get()
        round_id = message["round"]
        qubit, expected_rho, start_time = self.pending.pop(round_id)

        if message["a"] == 1:
            X(qubit)
        if message["b"] == 1:
            Z(qubit)

        fidelity = float(np.real_if_close(np.trace(np.dot(expected_rho, qubit.state.rho))))
        result = {
            "round": round_id,
            "a": message["a"],
            "b": message["b"],
            "start_time": start_time,
            "complete_time": self.get_simulator().current_time.sec,
            "fidelity": fidelity,
            "success": np.allclose(qubit.state.rho, expected_rho),
            "backend": qubit.backend_name,
        }
        self.results.append(result)
        self.timeline.append({
            "round": round_id,
            "stage": "bob-correction",
            "time": self.get_simulator().current_time.sec,
        })


class AliceTeleportationApp(Application):
    """Prepare EPR pairs, perform Bell measurements, and send two classic bits."""

    def __init__(self, bob: QNode, cchannel: ClassicChannel,
                 bob_app: BobTeleportationApp, rounds: int,
                 backend: str, round_interval: float):
        super().__init__()
        self.bob = bob
        self.cchannel = cchannel
        self.bob_app = bob_app
        self.rounds = rounds
        self.backend = backend
        self.round_interval = round_interval
        self.timeline: List[dict] = []
        self.add_handler(self.start_round, [StartTeleportationEvent], [self])

    def install(self, node: QNode, simulator: Simulator) -> None:
        super().install(node, simulator)
        for round_id in range(self.rounds):
            start_time = simulator.ts + simulator.time(sec=round_id * self.round_interval)
            simulator.add_event(StartTeleportationEvent(
                t=start_time,
                dest=node,
                round_id=round_id,
                by=self,
            ))

    def start_round(self, node: QNode, event: StartTeleportationEvent) -> None:
        state = TELEPORT_STATES[event.round_id % len(TELEPORT_STATES)].copy()
        expected_vector = state.reshape(2)
        expected_rho = np.outer(expected_vector, expected_vector.conjugate())

        message_qubit = Qubit(state=state, name="Q-{}".format(event.round_id), backend=self.backend)
        alice_ebit = Qubit(name="A-{}".format(event.round_id), backend=self.backend)
        bob_ebit = Qubit(name="B-{}".format(event.round_id), backend=self.backend)

        H(alice_ebit)
        CNOT(alice_ebit, bob_ebit)
        self.bob_app.reserve(
            event.round_id,
            bob_ebit,
            expected_rho,
            self.get_simulator().current_time.sec,
        )

        CNOT(message_qubit, alice_ebit)
        H(message_qubit)
        bit_a = alice_ebit.measure()
        bit_b = message_qubit.measure()

        self.timeline.append({
            "round": event.round_id,
            "stage": "alice-measure",
            "time": self.get_simulator().current_time.sec,
            "a": bit_a,
            "b": bit_b,
        })
        packet = ClassicPacket(
            msg={"round": event.round_id, "a": bit_a, "b": bit_b},
            src=node,
            dest=self.bob,
        )
        self.cchannel.send(packet=packet, next_hop=self.bob)


def simulate_teleportation(rounds: int = 5, backend: str = "qiskit",
                           channel_delay: float = 0.05,
                           round_interval: float = 0.02) -> Tuple[List[dict], List[dict]]:
    """Run several overlapping teleportation rounds in SimQN discrete time."""
    if rounds < 1:
        raise ValueError("rounds must be positive")

    end_time = (rounds - 1) * round_interval + channel_delay + round_interval
    simulator = Simulator(start_second=0, end_second=end_time, accuracy=1_000_000)
    alice = QNode("Alice")
    bob = QNode("Bob")
    cchannel = ClassicChannel("Alice-Bob", delay=channel_delay, bandwidth=0)
    alice.add_cchannel(cchannel)
    bob.add_cchannel(cchannel)

    bob_app = BobTeleportationApp()
    alice_app = AliceTeleportationApp(
        bob=bob,
        cchannel=cchannel,
        bob_app=bob_app,
        rounds=rounds,
        backend=backend,
        round_interval=round_interval,
    )
    alice.add_apps(alice_app)
    bob.add_apps(bob_app)

    bob.install(simulator)
    alice.install(simulator)
    simulator.run()

    results = sorted(bob_app.results, key=lambda result: result["round"])
    timeline = sorted(
        alice_app.timeline + bob_app.timeline,
        key=lambda item: (item["time"], item["round"], item["stage"]),
    )
    return results, timeline


def test_teleportation_circuit_can_be_drawn() -> None:
    drawing = str(draw_teleportation_circuit(output="text"))
    assert "Q:" in drawing
    assert "A:" in drawing
    assert "B:" in drawing
    assert "If-0" in drawing

    figure = draw_teleportation_circuit(output="mpl")
    assert figure.axes
    plt.close(figure)


@pytest.mark.parametrize("backend", ["numpy", "qiskit"])
def test_discrete_time_teleportation(backend: str) -> None:
    set_seed(23)
    results, timeline = simulate_teleportation(
        rounds=5,
        backend=backend,
        channel_delay=0.05,
        round_interval=0.02,
    )

    assert len(results) == 5
    assert len(timeline) == 10
    assert all(result["success"] for result in results)
    assert all(np.isclose(result["fidelity"], 1.0) for result in results)
    assert all(result["backend"] == backend for result in results)
    assert all(np.isclose(result["complete_time"] - result["start_time"], 0.05)
               for result in results)
    assert results[0]["complete_time"] > results[1]["start_time"]


def main() -> None:
    """Run the example and save a Qiskit-rendered circuit next to this file."""
    set_seed(23)
    output_path = Path(__file__).with_name("teleportation_circuit.png")
    figure = draw_teleportation_circuit(output="mpl", filename=output_path)
    plt.close(figure)

    results, timeline = simulate_teleportation(rounds=5, backend="qiskit")
    for item in timeline:
        print("t={time:.2f}s round={round} {stage}".format(**item))
    print("circuit: {}".format(output_path))
    print("results: {}".format(results))


if __name__ == "__main__":
    main()
