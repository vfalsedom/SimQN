from qns.network.network import QuantumNetwork
from qns.network.route.dijkstra import DijkstraRouteAlgorithm
from qns.network.topology.linetopo import LineTopology
from qns.utils.rnd import set_seed


def test_random_requests_honors_hop_limit_and_builds_route() -> None:
    network = QuantumNetwork(
        topo=LineTopology(nodes_number=4),
        route=DijkstraRouteAlgorithm(),
    )
    set_seed(17)

    network.random_requests(number=1, hop_limit=3)

    assert len(network.requests) == 1
    request = network.requests[0]
    metric, next_hop, path = network.query_route(request.src, request.dest)[0]
    assert metric >= 3
    assert len(path) - 1 >= 3
    assert next_hop is path[1]


def test_random_requests_without_hop_limit_keeps_existing_behavior() -> None:
    network = QuantumNetwork(
        topo=LineTopology(nodes_number=4),
        route=DijkstraRouteAlgorithm(),
    )
    set_seed(19)

    network.random_requests(number=2)

    assert len(network.requests) == 2
    assert all(request.src is not request.dest for request in network.requests)
