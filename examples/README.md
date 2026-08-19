# SimQN 示例目录

示例按主题分目录。所有命令都应在仓库根目录、torch conda 环境中执行。

## 目录分类

| 目录 | 内容 |
| --- | --- |
| backends/ | NumPy/Qiskit 量子态后端。 |
| processors/ | Qiskit 线路处理器、线路映射、队列和执行结果。 |
| trace/ | 量子隐形传态、事件追踪、线路绘制、时序图和 JSON 导出。 |
| network/ | 经典包转发、纠缠分发拓扑、网络请求和表格输出。 |
| protocols/ | BB84、DEJMPS、纠缠交换和经典通信协议。 |
| experiments/ | 多进程实验和参数扫描。 |

## 新 API 示例

~~~bash
conda run -n torch python examples/backends/qiskit_backend_basic.py
conda run -n torch python examples/processors/circuit_processor.py
conda run -n torch python examples/trace/teleportation_with_trace.py
conda run -n torch python examples/network/network_requests_and_table.py
~~~

追踪可视化示例需要显式指定输出目录：

~~~bash
conda run -n torch python examples/trace/trace_visualization.py \
  --output /tmp/simqn-trace
~~~

输出目录包含：

- trace.json：可序列化的协议事件；
- timeline.png：处理器和信道资源时序图；
- combined-circuit.png：合并后的协议线路；
- circuits/：各节点实际完成的局部线路。

## 既有示例

- protocols/DEJMPS.py
- protocols/entanglement_swapping.py
- protocols/simple_bb84.py
- protocols/trusted_relay_bb84.py
- protocols/classic_communication.py
- network/classic_packet_forward.py
- network/entanglement_distribution_line_topo.py
- network/entanglement_distribution_random_topo.py
- experiments/parallel_experiments.py

部分既有示例包含较大的参数扫描，运行前请根据本机资源调整节点数量、仿真时长和迭代次数。

