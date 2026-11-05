# 当前运行状态（自动生成快照）

采样时间：2026-11-05T14:03:45+08:00（北京时间）。

此文件不是实时监控；刷新命令 `python3 /data/jaka-vla/scripts/status.py --write-doc`。
阶段标记可能滞后，须结合进程和已落盘结果判断。

| 项目 | 已观察到的状态 |
| --- | --- |
| 阶段标记 | 当前批次结束，待复核结果 |
| 流水线 PID / 进程状态 | 16186 / None |
| 匹配流水线且非僵尸进程 | False |
| 原始示教已保存 / 规划种子 | 50 / 50 |
| 训练进度（日志最近值） | 2000 / 2000 |
| 最终检查点权重文件存在 | True |
| 数据盘可用 | 108.56 GiB |
| GPU 快照 | NVIDIA GeForce RTX 4090, 16 MiB, 24564 MiB, 0 % |

转换验证报告：50 条、18744 帧，频率 [25] Hz，状态语义 ['measured_qpos']。

训练资源报告：completed=True；峰值分配 3.5749893188476562 GiB；包含加载/保存的用时 449.3256176880004 秒。

## 已写入的固定 K 汇总

| K | 成功/回合数 | 成功率 |
| --- | --- | --- |
| 4 | 0/3 | 0.0% |
| 8 | 0/3 | 0.0% |
| 16 | 0/3 | 0.0% |

仅展示汇总文件中已完成的项目；逐回合证据在 `results/pilot_k*/results.json`，不得将部分结果当作完整批次。

当前是同步、零模拟推理延迟评测；不是事件监测器、真实延迟或节卡真机实验。

## 证据入口

- `results/project_status.json`：本次机器可读快照。
- `results/pilot_status.txt`、`results/pilot.pid`：流水线标记；必须交叉核实。
- `logs/collect-pilot.log`、`logs/convert-pilot.log`、`logs/train-pilot.log`、`logs/eval-baselines.log`：各阶段日志。
- `results/handover_pilot_validation.json`、`checkpoints/smolvla_pilot_seed42/resource_profile.json`、`results/pilot_baselines.json`：阶段结束后生成。
- `docs/EXPERIMENT_LOG.md`：历史验证、实验参数和结论边界。
