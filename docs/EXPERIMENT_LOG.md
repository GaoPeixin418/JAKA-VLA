# 实验与工程验证记录

时间均按北京时间。早期条目依据现存 JSON、日志和代码回填，日期为 2026-11-05；没有精确起止时间时不补造。
基础代码提交 `fbd1556`；各库/权重版本见 `manifests/sources.json`。本文件保留历史，当前快照见 `docs/STATUS.md`。

## E00 环境与渲染

- 目标：确认租用 GPU 能计算并实际渲染 RGB。
- 配置：Ubuntu22.04 / RTX4090 / 隔离仿真、模型环境；环境详见 `MEMORY.md`。
- 结果：CUDA 计算、Vulkan/SAPIEN 离屏图像及两个环境依赖检查通过。
- 证据：`logs/render-smoke.log`、`results/render_smoke.png`、`manifests/`。
- 限制：渲染通过不代表模型控制成功；可选 pytorch3d 未安装，不影响本轮 RGB 路径。

## E01 数据语义修正和单条示教

- 发现：采样间隔/FPS 混用及 primitive 边界不均匀；原本体接口读取指令目标而非实测位置。
- 修改：统一25Hz采样；读取实际qpos/夹爪开度；分开保存 commanded_state；记录状态与动作时间戳。
- 数据：`data/raw/jaka_smoke/handover_block/aloha_agilex/data/episode_0000000.hdf5`；转换为 `data/lerobot/handover_smoke`。
- 结果：377对、14维、三相机RGB，15.08秒；时间戳、形状、有限值校验通过。
- 证据：`results/handover_smoke_validation.json`、`scripts/patch_sim_timing.py`、上游patch。
- 排除：带 `_command_state_v0` 的旧数据/检查点不用于修正后的实验。

## E02 SmolVLA 推理、短训练与重载

- 模型：固定版本SmolVLA，450,046,176总参数 / 99,880,992可训练参数。
- 预算：单条修正数据，seed42，batch8，AMP，20步；命令见 `scripts/train_smoke.sh`。
- 结果：训练完成，加载/保存共28.03秒；PyTorch峰值分配3.575GiB，预留3.660GiB。检查点和归一化处理器重载成功，输出 `1×50×14`。
- 速度检查：合成输入预热后约0.26秒/块；不等于真实任务中整体控制延迟。
- 证据：`results/policy_smoke.json`、`results/checkpoint_smoke.json`、`checkpoints/smolvla_smoke/resource_profile.json`、`logs/train-smoke.log`。
- 结论：单4090足以运行当前小规模配置；不能据此保证策略任务效果或大批次资源需求。

## E03 专家回放与学习策略接口

- 控制：25Hz、每个指令线性插值10个物理步，使用实测状态。
- 专家回放：seed0成功，364步，14.56秒仿真时间，26.23秒墙钟时间。
- 学习策略：smoke权重，seed1000，10步接口检查完成；成功标记false，不作为完整任务失败率/成功率统计。
- 证据：`results/expert_replay/results.json` 与对应视频；`results/closed_loop_smoke/results.json`。
- 结论：控制与进程通信链路可运行；尚未验证学习策略完整任务表现。

## E04 首轮完整任务 pilot（进行中）

- 启动：2026-11-05，`scripts/pilot_pipeline.sh`，PID记录 `results/pilot.pid`；不重复启动。
- 数据：50条成功专家示教，输出 `data/raw/jaka_pilot/handover_block/aloha_agilex/data`，转换目标 `data/lerobot/handover_pilot`。
- 训练：seed42、batch8、2000步、AMP、workers4，最终输出 `checkpoints/smolvla_pilot_seed42/checkpoints/last/pretrained_model`。
- 评测：同一权重，H50，K4/8/16，各3回合，测试seeds1000–1002，每回合最多750步；同步零模拟推理延迟。
- 预算：采集5400秒、训练5400秒、每个K评测1800秒、评测封装6000秒；转换未设置单独timeout。出错停止，无自动重试或自动扩量。
- 2026-11-05 13:10核查：50/50原始HDF5已保存，转换正在进行；尚无学习策略成功率产物。
- 2026-11-05 13:16:53核查：转换验证完成，50条、18,744帧，全部25Hz、measured_qpos；训练达到1448/2000步，最近100步统计loss=0.029。GPU整卡快照4248MiB；此为瞬时占用，不是训练峰值或任务效果。
- 证据：`logs/collect-pilot.log`、`logs/convert-pilot.log`、`logs/train-pilot.log`（训练开始后）、`results/handover_pilot_validation.json`（转换结束后）、`results/pilot_baselines.json`（各K完成后逐项写入）。
- 待复核：转换总帧数、训练完整性/资源、全部9回合结果与录像；检查同一进程多回合环境重置。
- 解释边界：3回合/K只作初筛；即使获得成功也不是事件方法有效性、语言泛化或Sim2Real证据。

## D01 项目文档建档

- 2026-11-05：新增 `AGENT.md`、标准入口 `AGENTS.md`、`MEMORY.md`、`skill.md`，拆分技术设计、实验历史和可刷新状态快照。
- 约定每次改动/实验后更新记录；快照命令只读实验进度并写文档，不启动额外GPU任务，不改当前流水线。
- 所有凭据排除在文档与归档之外。

## 后续条目模板

记录：实验ID和时间 → 目的 → 代码/数据/配置/种子 → 训练及评测预算 → 实测结果和不确定性 → JSON/日志/检查点路径 → 结论边界 → 下一步。
失败实验注明退出阶段和原因，保留原始产物，不以删除或覆盖代替记录。
