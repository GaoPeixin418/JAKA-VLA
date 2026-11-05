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

## E04 首轮完整任务 pilot（已完成，模型0/9）

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

## E04 进度复核：2026-11-05T13:30:32+08:00

- 训练完成：2000/2000步，资源报告completed=true；检查点已加载执行完整闭环回合。训练/加载/保存总耗时449.33秒，峰值分配3.575GiB，峰值预留3.660GiB；最后记录loss=0.024。
- 已落盘评测：K=4 已完成 3/3 回合、成功 0；K=8 已完成 3/3 回合、成功 0；K=16 已完成 0/3 回合、成功 0。未完成组不报告成完整成功率，既定流水线仍在执行。
- K4的三个回合均运行至750控制步（30秒仿真时间），未满足任务成功判据；每回合188次策略调用，平均推理约0.259–0.266秒/块。
- 当前结论：训练和评测工程流程可运行，但基础策略尚未在已完成的测试中成功。原因未定位，不能据此判断事件重规划方法是否有效；该模块尚未实现。
- 后续：收齐剩余回合，先看失败录像并核查数据/动作对齐、状态分布和训练覆盖，再决定下一轮有限实验。本次仅查验和记档，未增加训练或测试预算。
- 证据：`checkpoints/smolvla_pilot_seed42/resource_profile.json`、`results/pilot_k*/results.json`、`results/pilot_progress_review.json`、各回合视频及训练日志。

## E04 完成复核：2026-11-05 13:35:40

- 流水线阶段为 `pilot_complete_review_results_before_scaling`，进程已退出；全部9回合结果与录像存在。K4=0/3、K8=0/3、K16=0/3。
- 各K使用同一2000步权重、相同种子和控制器；同步零模拟推理延迟。所有模型回合未满足任务成功判据，不能据此宣称自适应重规划有效。
- 已完成当前预算内的训练/评测；未启动额外训练。后续需要分析失败阶段与根因，基础策略仍待改善。
- 证据：`results/pilot_baselines.json`、三个 `results/pilot_k*/results.json`、对应9段MP4与训练资源报告。

## D02 录像与结果查看入口：2026-11-05

- 新增 `scripts/build_review_bundle.py`，将已存在的1段专家回放与9段模型测试视频、JSON证据整理为离线HTML页面和约3.6MB ZIP；不运行模型。
- 产物：`results/review_bundle/index.html`、`results/jaka-vla-review.zip`；操作说明：`docs/VIEW_RESULTS.md`。
- 验证：27个本地链接存在；10段录像均为H.264/yuv420p、25FPS；专家录像14.60秒，模型录像各30.04秒；ZIP完整性通过。
- 页面明确区分专家与模型结果，说明播放时间为仿真时间，未包含同步推理等待。没有启动常驻HTTP服务；用户可离线查看或自行通过SSH隧道查看。

## D01 项目文档建档

- 2026-11-05：新增 `AGENT.md`、标准入口 `AGENTS.md`、`MEMORY.md`、`skill.md`，拆分技术设计、实验历史和可刷新状态快照。
- 约定每次改动/实验后更新记录；快照命令只读实验进度并写文档，不启动额外GPU任务，不改当前流水线。
- 所有凭据排除在文档与归档之外。

## 后续条目模板

记录：实验ID和时间 → 目的 → 代码/数据/配置/种子 → 训练及评测预算 → 实测结果和不确定性 → JSON/日志/检查点路径 → 结论边界 → 下一步。
失败实验注明退出阶段和原因，保留原始产物，不以删除或覆盖代替记录。
