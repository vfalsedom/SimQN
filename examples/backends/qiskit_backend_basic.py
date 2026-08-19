"""演示 NumPy 与 Qiskit 量子态后端的选择和兼容性。"""

import numpy as np

from qns.models.qubit import CNOT, H, Qubit, QubitFactory, X
from qns.models.qubit.backends import get_default_backend, set_default_backend


def build_bell_pair(backend: str):
    control = Qubit(name="control", backend=backend)
    target = Qubit(name="target", backend=backend)
    H(control)
    CNOT(control, target)
    return control, target


def main() -> None:
    numpy_control, numpy_target = build_bell_pair("numpy")
    qiskit_control, qiskit_target = build_bell_pair("qiskit")

    print("NumPy backend :", numpy_control.backend_name)
    print("Qiskit backend:", qiskit_control.backend_name)
    print("states equal   :", np.allclose(
        numpy_control.state.rho, qiskit_control.state.rho,
    ))
    print("Qiskit raw type:", type(qiskit_control.state.backend.raw_state).__name__)

    factory_qubit = QubitFactory(backend="qiskit")()
    print("factory backend:", factory_qubit.backend_name)

    old_default = get_default_backend()
    try:
        set_default_backend("qiskit")
        default_qubit = Qubit()
        X(default_qubit)
        print("new default    :", default_qubit.backend_name)
        print("default result :", default_qubit.measure())
    finally:
        set_default_backend(old_default)

    # 测量 Bell 对中的两个量子比特，结果应保持一致。
    print("Bell measurement:", numpy_control.measure(), numpy_target.measure())
    print("Qiskit Bell measurement:",
          qiskit_control.measure(), qiskit_target.measure())


if __name__ == "__main__":
    main()
