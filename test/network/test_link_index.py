from qns.entity import QNode, QuantumChannel
from qns.network.network import QuantumNetwork
from qns.network.topology import LineTopology


def test_topology_index():
    net = QuantumNetwork(LineTopology(4))
    a, b, c, d = net.nodes
    assert net.get_link(a, b) is net.qchannels[0]
    assert net.get_link(b, a) is net.qchannels[0]
    assert net.get_link(a, c) is None
    assert set(net.get_neighbors(b)) == {a, c}
    assert set(net.get_neighbors(d)) == {c}
    assert net.get_qchannel(net.qchannels[0].name) is net.qchannels[0]
    assert a.get_qchannel(b) is net.get_link(a, b)


def test_add_channel_and_parallel_links():
    net = QuantumNetwork()
    a, b, c = QNode('a'), QNode('b'), QNode('c')
    for node in (a, b, c):
        net.add_node(node)
    assert not net.get_neighbors(a)
    assert not net.get_neighbors(QNode('unknown'))
    first = QuantumChannel('first', [a, b], bandwidth=0)
    net.add_qchannel(first)
    neighbors = net.get_neighbors(a)
    assert net.get_link(a, b) is first
    assert net.get_link(b, a) is first
    assert set(neighbors) == {b}
    net.add_qchannel(QuantumChannel('parallel', [b, a]))
    assert net.get_link(a, b) is first
    assert set(net.get_neighbors(a)) == {b}
    net.add_qchannel(QuantumChannel('next', [a, c]))
    assert set(neighbors) == {b, c}
    assert not hasattr(neighbors, 'add')


def test_rebuild_after_direct_changes():
    net = QuantumNetwork(LineTopology(3))
    a, b, c = net.nodes
    removed = net.qchannels.pop(0)
    net.rebuild_link_index()
    assert net.get_link(a, b) is None
    assert not net.get_neighbors(a)
    removed.node_list = [a, c]
    net.qchannels.append(removed)
    net.rebuild_link_index()
    assert net.get_link(a, c) is removed
    assert net.get_link(c, a) is removed
    assert set(net.get_neighbors(c)) == {a, b}
