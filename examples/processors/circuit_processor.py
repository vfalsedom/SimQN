"""演示 CircuitProcessor 的线路映射、排队和 CircuitResult。"""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qns.entity.node.app import Application
from qns.entity.node.node import QNode
from qns.entity.operator.event import CircuitExecuteResponseEvent
from qns.entity.operator.processor import CircuitProcessor
from qns.models.qubit import Qubit
from qns.models.qubit.const import QUBIT_STATE_1
from qns.simulator import Simulator


class ResultApp(Application):
    """接收线路响应并保存 CircuitResult。"""

    def __init__(self) -> None:
        super().__init__()
        self.result = None
        self.add_handler(self.receive_result, [CircuitExecuteResponseEvent], [])

    def receive_result(self, node: QNode, event: CircuitExecuteResponseEvent) -> None:
        self.result = event.result


def main() -> None:
    simulator = Simulator(0, 1, accuracy=1000)
    node = QNode("node")
    result_app = ResultApp()
    processor = CircuitProcessor(
        name="processor",
        gate_duration=0.1,
        measure_duration=0.02,
        strict_ownership=True,
    )
    node.add_processor(processor)
    node.add_apps(result_app)
    node.install(simulator)

    control_register = QuantumRegister(1, "control")
    target_register = QuantumRegister(1, "target")
    classical_register = ClassicalRegister(2, "measurement")
    circuit = QuantumCircuit(
        control_register, target_register, classical_register,
        name="controlled-not",
    )
    circuit.cx(control_register[0], target_register[0])
    circuit.measure(control_register[0], classical_register[0])
    circuit.measure(target_register[0], classical_register[1])

    control = Qubit(
        state=QUBIT_STATE_1,
        name="control-qubit",
        backend="qiskit",
        owner=node,
    )
    target = Qubit(name="target-qubit", backend="qiskit", owner=node)

    # 映射按线路量子比特精确覆盖；字典顺序不影响绑定结果。
    job_id = processor.submit(
        circuit,
        {
            target_register[0]: target,
            control_register[0]: control,
        },
        metadata={"stage": "cnot-demo"},
    )
    simulator.run()

    result = result_app.result
    if result is None:
        raise RuntimeError("circuit response was not delivered")

    print("job id          :", job_id)
    print("classical bits  :", result.classical_bits)
    print("classical regs  :", result.classical_registers)
    print("simulation time :", result.start_time, "->", result.finish_time)
    print("processor busy  :", processor.is_busy())


if __name__ == "__main__":
    main()
