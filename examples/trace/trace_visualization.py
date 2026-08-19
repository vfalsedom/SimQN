"""演示 ProtocolTrace 的线路、时序图和 JSON 导出。"""

import argparse
from pathlib import Path

from matplotlib import pyplot as plt

try:
    from examples.trace.teleportation_with_trace import simulate_teleportation
except ImportError:
    from teleportation_with_trace import simulate_teleportation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="保存 PNG 和 JSON 产物的目录",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results, trace = simulate_teleportation()
    trace.to_json(args.output / "trace.json")
    timeline = trace.draw_timeline(args.output / "timeline.png")
    plt.close(timeline)

    circuit = trace.draw_circuit(
        round=0,
        output="mpl",
        filename=args.output / "combined-circuit.png",
    )
    plt.close(circuit)

    drawings = trace.draw_circuits(
        output="mpl",
        directory=args.output / "circuits",
    )
    for drawing in drawings.values():
        plt.close(drawing)

    print("results:", results)
    print("records:", len(trace.records))
    print("artifacts:", args.output)


if __name__ == "__main__":
    main()
