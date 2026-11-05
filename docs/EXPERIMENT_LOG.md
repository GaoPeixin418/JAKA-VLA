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

## E05 用户反馈后的离线诊断：2026-11-05

- 用户反馈：机械臂无目标运动、接近但未抓住/推倒方块、没有持物仍出现交接动作，以及普遍抖动。
- 诊断对象：原seed42、2000步权重；SHA256 `f92fb0495f9d48d584a945843eb567347ac8cc440bda00fb81b9d15769496842`。原训练、控制器和评测未修改。
- 全50条检查动作移位；对现有检查点做114次离线推理，用时46.11秒，不训练、不新增仿真评测回合。
- 结果：同一初始观测32次推理首步单关节标准差最高4.22°；16个训练观测×4采样的首步关节MAE1.94°，首步夹爪MAE0.097。专家观测序列上的块边界目标跳变明显大于专家目标。
- 发现：输入配置仍声明6维，但实测14维状态全部进入模型；归一化及RGB抽查未发现明显错接。控制器缺少跨块连续性约束/速率限制，并使用零关节速度目标，与专家控制不同；不能凭现有日志分解视频抖动原因。
- 首次诊断脚本因CPU/CUDA比较设备不一致而失败；修正诊断脚本后新建 `_v2` 目录成功，旧日志保留。绘图使用已有仿真环境，未新增依赖。
- 证据与范围：`docs/DIAGNOSIS_20261105.md`，`results/offline_audit_20261105_v2/report.json`、`supplement.json`、`offline_arrays.npz`、图表；114次推理都是训练来源观测诊断，不能作为泛化或改进成功率。
- 下一步建议：输入/轨迹记录核查 → 多专家回放控制验收 → 少轨迹抓取验收 → 连续性和训练增量对照 → 事件重规划。等待用户具体修改需求，本次不改变原基线。

## 后续实验记录格式

记录：实验ID和时间 → 目的 → 代码/数据/配置/种子 → 训练及评测预算 → 实测结果和不确定性 → JSON/日志/检查点路径 → 结论边界 → 下一步。
失败实验注明退出阶段和原因，保留原始产物，不以删除或覆盖代替记录。


## 2026-11-05 · Zu7 仿真迁移与修正（本轮）

授权：用户要求直接修正并更换节卡机型，后明确两台Zu7，允许暂定夹爪与安装方式。旧ALOHA结果全部保留；以下不与旧场景作单因素性能归因。

- 模型来源：JakaCobot/jaka_ros2@7e985b78fe3c7807507fb81e8119b76f794e3bae，详见manifests/jaka_zu7_source.json。安装、工具、驱动和控制参数见configs/robots/dual_zu7.json；实际真机尚未连接。
- 启动/首轮采集失败：起始姿态与桌面坐标不适配；修正基座/起始姿态及每臂桌面变换。首个循环人工终止，随后加max_seed_attempts，失败日志保留。
- 末端坐标：固定关节global_pose与工具子link旋转差180°，导致抬升方向错误；Zu7改用工具子link姿态。`probe-zu7-task-v2.log`已能抓取/抬升/交接，放置仍失败。
- `zu7_task_probe_v3`：桌面顶面/中心修正、掌部碰撞小球覆盖后预放置成功，最终低位放置仍失败。
- `zu7_task_probe_v4`：上移接取抓点到中心下1cm，双臂工具距离不足导致碰撞，放弃此设置。
- `zu7_task_probe_v5`：恢复上下±7cm抓点，接收台改8cm高。seed0完整专家成功。
- `collect-zu7-smoke-v3.log`：seeds0–2，3/3成功；原始数据3条，转换1232帧，25Hz、14维、measured_qpos、RGB和身份检查通过。数据目录data/lerobot/zu7_smoke。
- `zu7_expert_replay_v2`：1.5rad/s指令限速，3/3右臂空抓。示教最大目标速度约2.67rad/s，限速截断改变运动时序/关节路径。逐物理步记录支持该诊断。
- `collect-zu7-slow-smoke.log` / `zu7_task_probe_v6`：尝试velocity_scale=.35，12种子0成功，触发有界退出，未产出示教；不采用。
- `zu7_expert_replay_v3`：指令上限3rad/s、8rad/s²，夹爪2/s，omega28；3/3通过稳定放置验收，但实测速度峰值5.07/4.16/8.30rad/s，说明仍有物理碰撞，不能仅凭任务成功宣布抖动解决。
- `zu7_expert_contact_diagnostic`：定位右工具/手指撞左Link_2，单对峰值冲量19.67Ns。仿真接触日志只用于诊断，不作为策略输入。
- `smolvla_zu7_overfit3000_seed42`：3条示教、batch8、3000步、seed42、AMP，H50；训练含加载/保存645.44s，峰值分配3.575GiB、预留3.660GiB。显式STATE14、ACTION14、RGB3×240×320；从SmolVLA基座微调，不复用ALOHA检查点。
- `zu7_learned_seen3000`：K8、固定回合噪声、seeds0–2，0/3。seed0/1左空抓并终止，seed2最大抬升0.185m但预算650步内未放置。只验证拟合，未跑未见场景。输入夹爪反馈的空抓监测正常触发，专家成功率未计入模型结果。
- `patch_zu7_interarm.py`：cuRobo世界加入对方机械臂的实际网格和工具盒；每次专家规划刷新其姿态。为满足“对方静止”的假设，最后左臂先退回，再右臂放置。未实现双臂同时运动的联合规划。
- `zu7_task_probe_v7`：seed0完整专家成功，交叉机械臂接触采样0。
- `collect-zu7-collision-smoke.log`：新3条seeds0–2成功，转换1296帧，数据data/lerobot/zu7_collision_smoke。与先前1232帧批次隔离。
- `zu7_expert_collision_replay`：3/3稳定放置成功；实测速度最大2.665/3.924/2.499rad/s。seed0无双臂接触；seed1/2仍有右指与左掌轻碰（峰值3.945/3.878Ns），比原严重碰撞减轻，但不能报告全批次零碰撞。
- 新增0.12rad跟踪误差持续0.04s的失败终止检测，仅使用指令与本体感知；未实现自动恢复。所有指令速度/加速度上限只约束命令，不保证物理速度。
- 正在运行`smolvla_zu7_refine3000_seed42`：从前3000步Zu7权重继续在新1296帧上训练3000步，重新初始化AdamW和余弦调度，并非优化器状态无缝resume。累计训练预算6000步。

各次探测、采集、回放、训练均有独立输出及timeout；不覆盖旧录像/数据/权重。最终本轮结果补记于下方。


### 本轮收尾

- `smolvla_zu7_refine3000_seed42`训练完成，额外3000步，含加载/保存653.11s；峰值分配3.575GiB、预留3.660GiB，累计6000步。未继续扩大训练。
- `zu7_learned_seen6000`：K8、固定回合噪声、seeds0–2，0/3。seed0/1分别123/155步因空抓停止；seed2抬升0.138m后222步因跟踪异常停止。观测state/qpos跟踪p95约0.0015–0.0020rad；失败仍来自不正确的目标/交互，不能用小跟踪误差证明任务成功。
- `zu7_learned_seen6000_zero`：同权重/控制器、零初始噪声，0/3。seed0/1空抓；seed2抬升0.141m后跟踪异常停止。零噪声诊断表明仅减少采样随机性不够；不是默认SmolVLA采样性能。
- 学习策略尚未通过已见场景验收，因此本轮未扩到未见场景或论文统计对照。未实现自动重抓、事件自适应K或真机部署。
- 5项单元测试、Python编译及Git差异检查通过，实际专家回放/学习闭环有完整录像/轨迹/接触日志。输入/输出身份校验与数据验证通过。
- 更新AGENT.md、MEMORY.md、skill.md、README及技术/实验/查看文档；刷新STATUS.md和manifests，生成独立Zu7查看包。失败记录全部保留，后台作业结束。

## D03 项目限定的GitHub备份准备：2026-11-05

- 目的：支持在家查看和退租恢复；本次不训练、不新增评测。
- 用户授权仅JAKA-VLA同步到GaoPeixin418；不修改全局Git设置，不同步其他项目，不设置定时任务。
- 完成微调说明文档，备份脚本覆盖原始/LeRobot数据、所有检查点、优化器/归一化器、录像、成功/失败记录及Git历史。
- 分片不超过1500MiB；保存分片与原文件SHA256，恢复只写新目录；测试覆盖多分片/符号链接往返和损坏拒绝。
- GitHub登录和上传尚未完成。最终备份结果另行记录，不把计划写成已完成。

## GitHub备份完成（2026-11-05）

- 已验证账号GaoPeixin418，并创建私有仓库https://github.com/GaoPeixin418/JAKA-VLA；main分支保存项目代码、文档和历史。
- Release backup-2026-11-05已发布，21/21附件、8,357,873,370字节；GitHub返回的每个附件SHA256均与原备份一致。服务器端665个文件/链接完整校验、6项测试通过。
- 包含data/checkpoints/results/logs及代码/额外配置；不含环境、缓存、凭据或第三方下载资产。详见docs/BACKUP_RESTORE.md和manifests/github-backup-20261105.json。
- 传输途中服务器重启，已重连并补全GitHub上传；没有新训练或评测，原模型0/3结论不变。
- 本地额外全量副本仍在下载；出现LOCAL_BACKUP_COMPLETE.json才代表本地校验完成。GitHub备份已完成，不依赖该下载进度。
- 同步范围只限JAKA-VLA；无自动Git同步，无全局Git配置变更。
