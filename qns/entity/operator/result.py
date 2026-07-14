#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from qiskit import QuantumCircuit


@dataclass
class CircuitResult:
    """The result of one stateful circuit execution."""

    job_id: str
    circuit: QuantumCircuit
    classical_bits: Tuple[int, ...]
    classical_registers: Dict[str, int]
    start_time: float = 0
    finish_time: float = 0
    metadata: dict = field(default_factory=dict)

    def get_bit(self, index: int) -> int:
        return self.classical_bits[index]

    def get_register(self, name: str, default: Optional[int] = None) -> Optional[int]:
        return self.classical_registers.get(name, default)
