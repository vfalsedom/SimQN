#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def build_circuit(trace, round: Optional[int] = None) -> QuantumCircuit:
    """Combine completed circuit jobs into the circuit path observed in a trace."""
    records = [record for record in trace.records
               if record["kind"] == "circuit-completed"
               and "circuit" in record
               and (round is None or record.get("round") == round)]
    records.sort(key=lambda record: (record["time"], record["sequence"]))
    if not records:
        raise ValueError("the trace does not contain completed circuit jobs")

    used_names = set()
    qubit_names = _ordered_qubit_names(records)

    qubit_registers = {
        name: QuantumRegister(1, _unique_name(_identifier(name, "q"), used_names))
        for name in qubit_names
    }
    classical_registers = []
    job_clbits = {}
    for record in records:
        circuit = record["circuit"]
        mapped_clbits = []
        if circuit.cregs:
            for register in circuit.cregs:
                name = _unique_name(_identifier(register.name, "c"), used_names)
                mapped_register = ClassicalRegister(len(register), name)
                classical_registers.append(mapped_register)
                mapped_clbits.extend(mapped_register)
        else:
            for index in range(circuit.num_clbits):
                name = _unique_name("c{}".format(index), used_names)
                mapped_register = ClassicalRegister(1, name)
                classical_registers.append(mapped_register)
                mapped_clbits.extend(mapped_register)
        job_clbits[record["job_id"]] = mapped_clbits

    combined = QuantumCircuit(*qubit_registers.values(), *classical_registers,
                              name="observed-protocol")
    for record in records:
        circuit = record["circuit"]
        mapped_qubits = [qubit_registers[name][0] for name in record["qubits"]]
        if len(mapped_qubits) != circuit.num_qubits:
            raise ValueError("the traced qubit mapping does not match its circuit")
        combined.compose(circuit, qubits=mapped_qubits,
                         clbits=job_clbits[record["job_id"]], inplace=True)
        label = "{}: {}".format(trace._name(record.get("node")),
                                record.get("stage", circuit.name))
        combined.barrier(*combined.qubits, label=label)
    return combined


def draw_circuit(trace, round: Optional[int] = None, output: str = "mpl",
                 filename: Optional[Union[str, Path]] = None):
    """Draw the complete circuit path reconstructed from traced jobs."""
    if filename is not None:
        filename = str(filename)
    return build_circuit(trace, round=round).draw(output=output, filename=filename)


def draw_circuits(trace, node: Optional[Union[str, Any]] = None,
                  output: str = "mpl", directory: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Draw the exact circuits completed by traced node processors."""
    node_name = trace._name(node) if node is not None else None
    target = None if directory is None else Path(directory)
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)

    drawings = {}
    for record in trace.records:
        if record["kind"] != "circuit-completed" or "circuit" not in record:
            continue
        if node_name is not None and trace._name(record.get("node")) != node_name:
            continue
        filename = None
        if target is not None:
            stage = str(record.get("stage", "circuit")).replace(" ", "-")
            filename = str(target / "round-{}-{}-{}-{}.png".format(
                record.get("round", "none"), trace._name(record.get("node")),
                stage, record["sequence"],
            ))
        drawings[record["job_id"]] = record["circuit"].draw(output=output, filename=filename)
    return drawings


def _identifier(value: Any, fallback: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", str(value))
    if not name:
        name = fallback
    if name[0].isdigit():
        name = "{}_{}".format(fallback, name)
    return name


def _ordered_qubit_names(records) -> list:
    first_seen = []
    following = {}
    indegree = {}
    for record in records:
        names = record["qubits"]
        for name in names:
            if name not in first_seen:
                first_seen.append(name)
                following[name] = set()
                indegree[name] = 0
        for left, right in zip(names, names[1:]):
            if right not in following[left]:
                following[left].add(right)
                indegree[right] += 1

    ready = [name for name in first_seen if indegree[name] == 0]
    ordered = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for other in first_seen:
            if other in following[name]:
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
    return ordered if len(ordered) == len(first_seen) else first_seen


def _unique_name(name: str, used_names: set) -> str:
    candidate = name
    suffix = 1
    while candidate in used_names:
        candidate = "{}_{}".format(name, suffix)
        suffix += 1
    used_names.add(candidate)
    return candidate
