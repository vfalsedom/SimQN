# 修改日志

按北京时间（Asia/Shanghai）记录修改内容。同一天的新变更追加到对应日期下；修改记录不代表已提交或已推送。

以下初始记录的代码修改日期依据文件最后修改时间整理，不代表能够追溯全部历史修改；自本日志建立后，在每次同步时持续补充实际修改和验证结果。

## 2026-10-19

- 远程提交 `3459406`：`qns/network/route/dijkstra.py` 在重建路由表前执行 `route_table.clear()`，清除失效路由；`qns/entity/node/node.py` 补充文件尾空行。修改日期依据该提交的北京时间 2026-10-19。
- 本次同步验证：使用 Conda `qu-code` 环境运行 `test/network/test_route.py`、`test/network/test_route_ex.py` 和 `test/network/test_link_index.py`，结果为 **7 passed**。
- `qns/network/network.py`：新增量子信道索引，提供 `get_link()`、`get_neighbors()` 和 `rebuild_link_index()`；初始化和添加信道时更新索引，支持双向查找，并保留平行信道中的首条信道。
- `test/network/test_link_index.py`：新增拓扑索引、动态添加信道、平行信道和手动重建索引的测试。
- `CHANGELOG.md`：建立本日志，记录当前修改及后续按日期维护的规则。
- 验证：使用 Conda `qu-code` 环境运行 `python -m pytest -q -p no:cacheprovider test/network/test_link_index.py`，结果为 **3 passed**。
- 同步批次：2026-10-19；本日志随代码修改纳入版本管理，目标分支为 `vfalsedom/SimQN` 的 `qiskit-backend`。推送结果以同步执行报告为准。

## 2026-10-18

- `requirements.txt`：将 `numpy>=1.23,<2` 改为 `numpy`，移除 NumPy 版本范围限制。
- 日期依据：该文件最后修改时间为 2026-10-18 03:59:40（北京时间）。
- 同步批次：2026-10-19；保留实际文件修改日期，与本次链路索引变更一并同步。
