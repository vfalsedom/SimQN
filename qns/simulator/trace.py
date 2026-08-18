#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ProtocolTrace:
    """Collect processor, channel, and memory behavior for visualization."""

    def __init__(self):
        self._simulator = None
        self.records: List[dict] = []
        self._sequence = 0

    def bind(self, simulator) -> None:
        self._simulator = simulator

    def record(self, kind: str, time: Optional[float] = None, **data) -> dict:
        if time is None:
            time = 0 if self._simulator is None else self._simulator.current_time.sec
        record = {"sequence": self._sequence, "time": float(time), "kind": kind}
        record.update(data)
        self.records.append(record)
        self._sequence += 1
        return record

    def select(self, kind: Optional[str] = None, node: Optional[Union[str, Any]] = None) -> List[dict]:
        node_name = self._name(node) if node is not None else None
        return [record for record in self.records
                if (kind is None or record["kind"] == kind)
                and (node_name is None or self._name(record.get("node")) == node_name)]

    def draw_timeline(self, filename: Optional[Union[str, Path]] = None):
        from qns.visualization.timeline import draw_timeline
        return draw_timeline(self, filename=filename)

    def build_circuit(self, round: Optional[int] = None):
        from qns.visualization.circuit import build_circuit
        return build_circuit(self, round=round)

    def draw_circuit(self, round: Optional[int] = None, output: str = "mpl",
                     filename: Optional[Union[str, Path]] = None):
        from qns.visualization.circuit import draw_circuit
        return draw_circuit(self, round=round, output=output, filename=filename)

    def draw_circuits(self, node: Optional[Union[str, Any]] = None,
                      output: str = "mpl", directory: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        from qns.visualization.circuit import draw_circuits
        return draw_circuits(self, node=node, output=output, directory=directory)

    def to_json(self, filename: Union[str, Path], indent: int = 2) -> None:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump([self._serializable(record) for record in self.records],
                      file, ensure_ascii=False, indent=indent)

    def as_dicts(self) -> List[dict]:
        return [self._serializable(record) for record in self.records]

    def _serializable(self, value):
        if isinstance(value, dict):
            return {str(key): self._serializable(item) for key, item in value.items()
                    if key != "circuit"}
        if isinstance(value, (list, tuple)):
            return [self._serializable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return self._name(value)

    def _name(self, value) -> str:
        if isinstance(value, str):
            return value
        name = getattr(value, "name", None)
        return str(value) if name is None else str(name)
