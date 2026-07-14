#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

from typing import Optional, Sequence
from uuid import uuid4

from qiskit import QuantumCircuit

from qns.entity.entity import Entity
from qns.entity.operator.errors import QubitBackendMismatchError, QubitOwnershipError
from qns.models.qubit.backends.circuit_executor import CircuitExecutor
from qns.models.qubit.qubit import Qubit
from qns.simulator.simulator import Simulator
from qns.simulator.ts import Time


class CircuitProcessor(Entity):
    """A node processor that executes Qiskit circuits on live SimQN qubits."""

    def __init__(self, name: str = None, node=None,
                 gate_duration: float = 0, measure_duration: float = 0,
                 reset_duration: float = 0, strict_ownership: bool = False):
        super().__init__(name=name)
        self.node = node
        self.gate_duration = gate_duration
        self.measure_duration = measure_duration
        self.reset_duration = reset_duration
        self.strict_ownership = strict_ownership
        self._executor = CircuitExecutor()
        self._next_available_time = None

    def install(self, simulator: Simulator) -> None:
        super().install(simulator)
        if self._next_available_time is None:
            self._next_available_time = simulator.ts

    def set_own(self, node) -> None:
        self.node = node

    def execute(self, circuit: QuantumCircuit, qubits: Sequence[Qubit],
                metadata: Optional[dict] = None):
        """Synchronously execute a circuit and immediately update the qubits."""
        self._validate_mapping(circuit, qubits)
        current_time = 0 if self._simulator is None else self._simulator.current_time.sec
        return self._executor.execute(
            circuit=circuit,
            qubits=qubits,
            metadata=metadata,
            start_time=current_time,
            finish_time=current_time,
        )

    def submit(self, circuit: QuantumCircuit, qubits: Sequence[Qubit],
               t: Optional[Time] = None, by=None,
               metadata: Optional[dict] = None) -> str:
        """Submit a stateful circuit job to the simulator event queue."""
        if self._simulator is None:
            raise RuntimeError("the circuit processor has not been installed")
        from qns.entity.operator.event import CircuitExecuteRequestEvent

        self._validate_mapping(circuit, qubits)
        job_id = uuid4().hex
        request_time = self._simulator.current_time if t is None else t
        request = CircuitExecuteRequestEvent(
            processor=self,
            circuit=circuit,
            qubits=list(qubits),
            job_id=job_id,
            metadata={} if metadata is None else metadata.copy(),
            t=request_time,
            by=by,
        )
        self._simulator.add_event(request)
        self._simulator.record(
            "circuit-submitted",
            time=request_time.sec,
            node=self.node,
            processor=self,
            circuit=circuit,
            qubits=[qubit.name for qubit in qubits],
            job_id=job_id,
            stage=request.metadata.get("stage", circuit.name),
            round=request.metadata.get("round"),
            metadata=request.metadata,
        )
        return job_id

    def handle(self, event) -> None:
        from qns.entity.operator.event import CircuitExecuteRequestEvent
        if isinstance(event, CircuitExecuteRequestEvent):
            self._start_job(event)

    def _start_job(self, request) -> None:
        from qns.entity.operator.event import CircuitExecuteCompleteEvent

        self._validate_mapping(request.circuit, request.qubits)
        locked = []
        try:
            for qubit in request.qubits:
                qubit.lock(request.job_id)
                locked.append(qubit)
        except Exception:
            for qubit in locked:
                qubit.unlock(request.job_id)
            raise

        current_time = self._simulator.current_time
        start_time = self._next_available_time if self._next_available_time > current_time else current_time
        finish_time = start_time + self._simulator.time(sec=self.estimate_duration(request.circuit))
        self._next_available_time = finish_time
        self._simulator.record(
            "circuit-started",
            time=start_time.sec,
            end_time=finish_time.sec,
            node=self.node,
            processor=self,
            circuit=request.circuit,
            qubits=[qubit.name for qubit in request.qubits],
            job_id=request.job_id,
            stage=request.metadata.get("stage", request.circuit.name),
            round=request.metadata.get("round"),
            metadata=request.metadata,
        )
        complete = CircuitExecuteCompleteEvent(
            processor=self,
            request=request,
            start_time=start_time,
            t=finish_time,
            by=self,
        )
        self._simulator.add_event(complete)

    def complete(self, event) -> None:
        from qns.entity.operator.event import CircuitExecuteResponseEvent

        request = event.request
        try:
            result = self._executor.execute(
                circuit=request.circuit,
                qubits=request.qubits,
                job_id=request.job_id,
                metadata=request.metadata,
                start_time=event.start_time.sec,
                finish_time=event.t.sec,
            )
        finally:
            for qubit in request.qubits:
                qubit.unlock(request.job_id)

        response = CircuitExecuteResponseEvent(
            node=self.node,
            result=result,
            request=request,
            t=event.t,
            by=self,
        )
        self._simulator.record(
            "circuit-completed",
            time=event.t.sec,
            node=self.node,
            processor=self,
            circuit=request.circuit,
            qubits=[qubit.name for qubit in request.qubits],
            job_id=request.job_id,
            stage=request.metadata.get("stage", request.circuit.name),
            round=request.metadata.get("round"),
            classical_registers=result.classical_registers,
            metadata=request.metadata,
        )
        self._simulator.add_event(response)

    def estimate_duration(self, circuit: QuantumCircuit) -> float:
        duration = 0.0
        for instruction in circuit.data:
            operation = instruction.operation
            if operation.name == "barrier":
                continue
            if operation.name == "measure":
                duration += self.measure_duration
            elif operation.name == "reset":
                duration += self.reset_duration
            elif operation.name == "delay":
                duration += self._delay_seconds(operation.duration, operation.unit)
            elif operation.name == "if_else":
                duration += max((self.estimate_duration(block) for block in operation.blocks
                                 if block is not None), default=0)
            else:
                duration += self.gate_duration
        return duration

    def is_busy(self) -> bool:
        if self._simulator is None or self._next_available_time is None:
            return False
        return self._next_available_time > self._simulator.current_time

    def _validate_qubits(self, qubits: Sequence[Qubit]) -> None:
        for qubit in qubits:
            if self.strict_ownership and qubit.owner is not self.node:
                raise QubitOwnershipError(
                    "processor {} cannot operate qubit owned by {}".format(self, qubit.owner)
                )
            if qubit.owner is not None and self.node is not None and qubit.owner is not self.node:
                raise QubitOwnershipError(
                    "processor {} cannot operate qubit owned by {}".format(self, qubit.owner)
                )

    def _validate_mapping(self, circuit: QuantumCircuit, qubits: Sequence[Qubit]) -> None:
        if len(qubits) != circuit.num_qubits:
            raise ValueError("the number of SimQN qubits does not match the circuit")
        if len(set(qubits)) != len(qubits):
            raise ValueError("a circuit qubit mapping cannot contain duplicates")
        if any(qubit.backend_name != "qiskit" for qubit in qubits):
            raise QubitBackendMismatchError("circuit execution requires the qiskit backend")
        self._validate_qubits(qubits)

    def _delay_seconds(self, duration, unit: str) -> float:
        factors = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
        if unit not in factors:
            raise ValueError("delay unit {} is not supported".format(unit))
        return float(duration) * factors[unit]

    def __repr__(self) -> str:
        if self.name is not None:
            return "<circuit processor {}>".format(self.name)
        return super().__repr__()
