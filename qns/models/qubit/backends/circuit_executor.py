#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

from typing import Dict, List, Optional, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, ClassicalRegister
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import XGate

from qns.entity.operator.errors import QubitBackendMismatchError, \
                                       UnsupportedCircuitInstructionError
from qns.entity.operator.result import CircuitResult
from qns.models.qubit.qubit import Qubit
from qns.models.qubit.utils import joint


class CircuitExecutor:
    """Execute a Qiskit circuit against live SimQN Qiskit-backed qubits."""

    def execute(self, circuit: QuantumCircuit, qubits: Sequence[Qubit],
                job_id: str = "sync", metadata: Optional[dict] = None,
                start_time: float = 0, finish_time: float = 0) -> CircuitResult:
        if len(qubits) != circuit.num_qubits:
            raise ValueError("the number of SimQN qubits does not match the circuit")
        if len(set(qubits)) != len(qubits):
            raise ValueError("a circuit qubit mapping cannot contain duplicates")
        if any(qubit.backend_name != "qiskit" for qubit in qubits):
            raise QubitBackendMismatchError("circuit execution requires the qiskit backend")

        classical_values = [0] * circuit.num_clbits
        clbit_indices = list(range(circuit.num_clbits))
        self._execute_block(circuit, list(qubits), classical_values, clbit_indices)

        registers: Dict[str, int] = {}
        for register in circuit.cregs:
            value = 0
            for position, bit in enumerate(register):
                index = circuit.find_bit(bit).index
                value |= classical_values[index] << position
            registers[register.name] = value

        return CircuitResult(
            job_id=job_id,
            circuit=circuit,
            classical_bits=tuple(classical_values),
            classical_registers=registers,
            start_time=start_time,
            finish_time=finish_time,
            metadata={} if metadata is None else metadata.copy(),
        )

    def _execute_block(self, circuit: QuantumCircuit, qubits: List[Qubit],
                       classical_values: List[int], clbit_indices: List[int]) -> None:
        for instruction in circuit.data:
            operation = instruction.operation
            local_qubits = [qubits[circuit.find_bit(bit).index] for bit in instruction.qubits]
            local_clbits = [clbit_indices[circuit.find_bit(bit).index] for bit in instruction.clbits]

            if operation.name == "barrier":
                continue
            if operation.name == "measure":
                if len(local_qubits) != 1 or len(local_clbits) != 1:
                    raise UnsupportedCircuitInstructionError("invalid measure instruction")
                classical_values[local_clbits[0]] = local_qubits[0].measure()
                continue
            if operation.name == "reset":
                if len(local_qubits) != 1:
                    raise UnsupportedCircuitInstructionError("multi-qubit reset is not supported")
                if local_qubits[0].measure() == 1:
                    local_qubits[0].state.apply_operation(XGate(), [local_qubits[0]])
                continue
            if isinstance(operation, IfElseOp):
                branch = 0 if self._condition_is_true(
                    operation.condition, circuit, classical_values, clbit_indices
                ) else 1
                if branch < len(operation.blocks) and operation.blocks[branch] is not None:
                    block = operation.blocks[branch]
                    self._execute_block(block, local_qubits, classical_values, local_clbits)
                continue
            if operation.name in {"for_loop", "while_loop", "switch_case", "store"}:
                raise UnsupportedCircuitInstructionError(
                    "unsupported circuit instruction: {}".format(operation.name)
                )
            if len(local_qubits) == 0:
                continue

            for qubit in local_qubits[1:]:
                joint(local_qubits[0], qubit)
            local_qubits[0].state.apply_operation(operation, local_qubits)

    def _condition_is_true(self, condition, circuit: QuantumCircuit,
                           classical_values: List[int], clbit_indices: List[int]) -> bool:
        if not isinstance(condition, tuple) or len(condition) != 2:
            raise UnsupportedCircuitInstructionError("classical expressions are not supported")
        target, expected = condition
        if isinstance(target, Clbit):
            local_index = circuit.find_bit(target).index
            actual = classical_values[clbit_indices[local_index]]
        elif isinstance(target, ClassicalRegister):
            actual = 0
            for position, bit in enumerate(target):
                local_index = circuit.find_bit(bit).index
                actual |= classical_values[clbit_indices[local_index]] << position
        else:
            raise UnsupportedCircuitInstructionError("unsupported condition target")
        return actual == expected
