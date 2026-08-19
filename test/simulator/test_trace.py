import json

from qiskit import QuantumCircuit

from qns.entity.node.node import QNode
from qns.simulator.simulator import Simulator
from qns.simulator.trace import ProtocolTrace


def test_simulator_trace_can_select_and_serialize_records(tmp_path) -> None:
    simulator = Simulator(0, 1)
    trace = simulator.enable_trace()
    circuit = QuantumCircuit(1)
    node = QNode("node")

    record = simulator.record(
        "circuit-completed", node=node, circuit=circuit,
        values=(1, True),
    )
    output = tmp_path / "trace.json"
    trace.to_json(output)

    assert record["sequence"] == 0
    assert trace.select(kind="circuit-completed", node="node") == [record]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == [{
        "sequence": 0,
        "time": 0.0,
        "kind": "circuit-completed",
        "node": "node",
        "values": [1, True],
    }]
    assert "circuit" not in payload[0]


def test_trace_record_is_noop_until_enabled() -> None:
    simulator = Simulator(0, 1)
    assert simulator.record("ignored") is None

    trace = ProtocolTrace()
    assert simulator.enable_trace(trace) is trace
    assert simulator.record("observed")["kind"] == "observed"
