"""演示 hop_limit 请求生成和 format_table 结果格式化。"""

from qns.network import QuantumNetwork
from qns.network.route.dijkstra import DijkstraRouteAlgorithm
from qns.network.topology import LineTopology
from qns.utils import format_table
from qns.utils.rnd import set_seed


def main() -> None:
    network = QuantumNetwork(
        topo=LineTopology(nodes_number=6),
        route=DijkstraRouteAlgorithm(),
    )
    set_seed(7)
    network.random_requests(
        number=2,
        hop_limit=3,
        attr={"send_rate": 10},
    )

    rows = []
    for index, request in enumerate(network.requests, start=1):
        metric, _, path = network.query_route(request.src, request.dest)[0]
        rows.append({
            "request": index,
            "source": request.src.name,
            "destination": request.dest.name,
            "metric": metric,
            "path": " -> ".join(node.name for node in path),
            "send_rate": request.attr["send_rate"],
        })

    print(format_table(rows))


if __name__ == "__main__":
    main()
