---
name: jaka-vla-project-operations
description: 在 /data/jaka-vla 接续双臂 VLA 仿真项目，核查环境、数据、SmolVLA 训练和固定 K 评测，记录证据并安全恢复中断实验。
---

# 双臂 VLA 项目操作手册

本文件按用户要求保留小写 `skill.md`，供项目内显式阅读；未安装为全局技能，不能假定工具会自动发现。协作约定见 `AGENT.md`。

## 先查现场

```bash
cd /data/jaka-vla
git status --short
python3 scripts/status.py --write-doc
nvidia-smi
```

阅读 `MEMORY.md`、`docs/STATUS.md` 和实验日志。核对阶段、进程状态及结果，不仅看 PID 存在或 `pilot_status.txt`。`--write-doc` 只生成快照，不启动训练或持续监控。

## 环境选择

先执行 `source /data/jaka-vla/env.sh`。采集/仿真使用 `envs/robotwin/bin/python`；转换/训练/模型服务使用 `envs/vla/bin/python`。不要在同一进程混装两套 NumPy/PyTorch，也不要无依据升级上游代码。

环境改动后运行各自依赖检查，再用 `scripts/freeze_environment.sh` 更新包清单、源码哈希和 patch。离屏渲染问题用 `scripts/smoke_render.py` 验证真实 RGB 产物；不能只凭 Vulkan 列表判断。

## 数据流程

1. `scripts/configure_smoke.py` 生成 `jaka_smoke`（1 条）与 `jaka_pilot`（50 条）配置；`scripts/patch_sim_timing.py` 应用于已固定版本的 RoboTwin。
2. 采集入口为 RoboTwin 的 `scripts/collect_data.py handover_block jaka_pilot`。运行前检查现有进程与输出，不重复采集正在运行的批次。
3. 转换用 `scripts/convert_dataset.py --source <原始HDF5目录> --output <新目录> --repo-id <实验ID>`；在 VLA 环境执行。现有 pilot 的精确参数见 `scripts/pilot_pipeline.sh`。
4. 必须验证 `measured_qpos` 语义、25 Hz 时间戳、14 维形状、有限值与三路 RGB。通过后检查 `results/<数据集目录名>_validation.json`。
5. `_command_state_v0` 仅为历史排错产物，不与修正后的数据混训。新转换输出不能覆盖旧目录。

## 训练与评测

- `scripts/train_smoke.sh` 是已完成的 20 步资源检查；其输出目录已存在，不要直接重跑覆盖。
- `scripts/pilot_pipeline.sh` 是当前已启动的有界批次配置来源；不要为了查看进度再次执行。
- 训练使用 `scripts/run_training.py` 封装 LeRobot，记录 `resource_profile.json`。总步数完成且报告 `completed=true`、检查点可加载才算训练完成。
- 同时保存/重载策略、预处理与后处理器；不能丢失归一化统计。`scripts/check_checkpoint.py` 目前硬编码 smoke 路径，检查新批次前须明确换成对应数据与权重。
- `scripts/run_baselines.py` 固定同一 pilot 权重评测 K=4/8/16，每个 K 3 回合。模拟器和模型服务由 Unix socket 连接。
- 当前评测会暂停仿真等待模型；耗时属于墙钟测量，不能解释为仿真内实际推理延迟。每回合 JSON 与录像一起保留。
- 查看训练：`tail -n 20 logs/train-pilot.log`；查看评测：`cat results/pilot_baselines.json`（仅生成后可读）。

## 中断与失败恢复

1. 核对进程命令行、阶段、日志尾部、磁盘和部分产物，记录失败发生在哪一步。
2. 不重启整条流水线；选取未完成阶段，采用新的输出目录或上游明确支持的恢复方式。
3. 不只因 `run/policy.sock` 存在就删除它；先确认没有活跃模型服务及连接。
4. CUDA OOM 先检查额外 GPU 进程和 batch；修改配置后建立新实验记录，不能把不同配置结果合并。
5. 依赖错误只修复相应环境，保留版本；RGB 路径并不要求可选点云包 pytorch3d。
6. 不自动循环重试、扩预算或启动额外 GPU 作业。当前实验有序推进即可。

## 收尾记录

追加 `docs/EXPERIMENT_LOG.md`：实验 ID、时间、数据与源码、预算/种子、指标、产物、失败/限制。
更新 `MEMORY.md`；方法改变时更新 `docs/TECHNICAL_DESIGN.md`。
执行 `python3 scripts/status.py --write-doc`，校对 Git 差异，保存代码/配置/文档版本。
汇报时分别说明工程链路、学习策略效果和研究方法进展，不把计划写成结果。

## 查看已有录像

用户查看结果时先核实 `results/pilot_baselines.json` 和各回合MP4。操作步骤见 `docs/VIEW_RESULTS.md`。
`python3 scripts/build_review_bundle.py` 可从当前pilot产物生成离线查看页面与ZIP，输出在 `results/review_bundle/`、`results/jaka-vla-review.zip`。
生成页面不运行模型；页面是快照。服务端浏览默认使用只绑定127.0.0.1的HTTP服务与SSH转发，或直接下载ZIP离线打开。
