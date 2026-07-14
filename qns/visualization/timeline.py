#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

from pathlib import Path
from typing import Optional, Union

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def draw_timeline(trace, filename: Optional[Union[str, Path]] = None):
    """Draw aligned processor and channel lanes from a protocol trace."""
    items = []
    lane_names = []
    for record in trace.records:
        if record["kind"] == "circuit-started":
            lane = trace._name(record.get("node"))
            category = "circuit"
        elif record["kind"] == "quantum-transfer":
            lane = "quantum:{}".format(trace._name(record.get("channel")))
            category = "quantum"
        elif record["kind"] == "classic-transfer":
            lane = "classic:{}".format(trace._name(record.get("channel")))
            category = "classic"
        else:
            continue
        if lane not in lane_names:
            lane_names.append(lane)
        items.append((lane, category, record))

    figure_height = max(2.8, 0.72 * len(lane_names) + 1.2)
    figure, axes = plt.subplots(figsize=(11, figure_height))
    positions = {lane: len(lane_names) - index - 1 for index, lane in enumerate(lane_names)}
    colors = {"circuit": "#3568b8", "quantum": "#8b5fbf", "classic": "#d17b25"}
    starts = [record["time"] for _, _, record in items]
    finishes = [record.get("end_time", record["time"]) for _, _, record in items]
    time_span = max(max(finishes, default=1) - min(starts, default=0), 0.000_001)

    for lane, category, record in items:
        start = record["time"]
        finish = record.get("end_time", start)
        width = max(finish - start, 0.000_001)
        y = positions[lane]
        axes.broken_barh([(start, width)], (y - 0.22, 0.44),
                         facecolors=colors[category], alpha=0.82)
        label = record.get("stage") or record.get("label") or category
        if width < time_span * 0.09:
            axes.annotate(str(label), (finish + time_span * 0.008, y), ha="left", va="center",
                          fontsize=8, color=plt.rcParams["text.color"], clip_on=False)
        else:
            axes.annotate(str(label), (start + width / 2, y), ha="center", va="center",
                          fontsize=8, color="white", clip_on=True)

    for record in trace.records:
        if record["kind"] not in {"circuit-completed", "quantum-received", "classic-received"}:
            continue
        if record["kind"] == "circuit-completed":
            lane = trace._name(record.get("node"))
        elif record["kind"] == "quantum-received":
            lane = "quantum:{}".format(trace._name(record.get("channel")))
        else:
            lane = "classic:{}".format(trace._name(record.get("channel")))
        if lane in positions:
            axes.plot(record["time"], positions[lane], marker="o", markersize=4,
                      color=colors["circuit" if record["kind"] == "circuit-completed"
                                   else "quantum" if record["kind"] == "quantum-received"
                                   else "classic"])

    axes.set_yticks([positions[lane] for lane in lane_names], lane_names)
    axes.set_xlabel("simulation time (s)")
    axes.set_ylabel("resource lane")
    axes.grid(axis="x", alpha=0.25)
    axes.margins(x=0.06)
    axes.legend(handles=[
        Patch(facecolor=colors["circuit"], label="node circuit"),
        Patch(facecolor=colors["quantum"], label="quantum transfer"),
        Patch(facecolor=colors["classic"], label="classic transfer"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
               label="completion / receive"),
    ], loc="upper right")
    figure.tight_layout()
    if filename is not None:
        figure.savefig(str(filename), dpi=160, bbox_inches="tight")
    return figure
