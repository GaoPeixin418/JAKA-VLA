---

## 2026-11-12 关机前接续：17:06关机，三目标K25已完成

用户明确本日17:06关机，覆盖更早“今天无时限”的快照。16:38核实没有训练/评测运行；关机前仅整理、校验、备份和提交，不再开新训练。17:00仍按既有每小时检查点反馈；不要修改/重复创建heartbeat，不要在关机后自动启动实验。

当前学习对照为 `lab_pi05_three_targets_20261112/5744`：从stage3/14854权重初始化，重启优化器/LR，seed45，布局0三条示教1170帧，实际5744更新且1170锚点全覆盖，冻结参数哈希不变。K50已见0/3；同权重K25已见1/3（蓝）、开发2/3（蓝/黄），红块两处空抓。全部六回合均保留并独立复算250Hz持物判据，12段视频完整解码/本地哈希通过。开发1000未用于本轮训练但已参与开发选择，不能叫盲测或稳定成功率。成功达到0.6秒即结束，未验证长期持物、完整双臂交接或真机。当前硬件仍为归档仿真配置。

本日新增专家数据0，已有97条41206帧/33布局；最终留出2000–2019未用。stage2/3、single_scene、three_targets的精确权重差分/优化器/统计和本日完整实验证据六份归档均已本地逐文件校验，全部模型差分仍依赖11月11日完整qualification参考包。不能宣称权重已上传GitHub Release。

接续优先读 `docs/LAB_PI05_20261112.md` 和 `docs/NEXT_ITERATION_20261112.md`。下次先查进程和磁盘，不重启已结束实验。固定5744+K25对照，诊断最多15分钟，优先补末段接近/恢复状态覆盖；小批验收、短训、匹配六回合后再决定扩采。计划未执行，不能把计划数量计入数据。当前约13G可用空间，避免重复复制完整权重。



## 2026-11-12 15:25 · 首个π0.5单场景成功，三目标课程训练中

详见docs/LAB_PI05_20261112.md，以下优先于更早快照。single_scene从stage3/14854初始化，新的优化器/LR，episode0全部330帧，2000次更新，冻结哈希不变。同一模型/布局0蓝块/noise42，K8为0/1，K50为1/1；成功最大抬升0.05088m，连续合格持物0.604s，无非预定接触，物理证据复算与录像完整解码通过。这只证明已见单场景可学，裕量小，不代表泛化。stage3多目标模型K8及K50均seen0/3、development0/3，不能说改K单独解决了问题。

当前只运行lab_pi05_three_targets_20261112：从stage3/14854权重重新初始化优化器，seed45，episodes0/1/2共1170帧，最多6000更新/15:50截止，随后K50同布局三个目标验证；有成功再测开发。不要重复启动。新训练采样每遍无放回且保存准确索引，控制器/判据不变，未加入真值输入。本日新增专家数据0，最终留出未用。今天没有确认关机时间；用户每小时整点反馈，15:00已反馈，下次16:00。

stage2/3权重差分实际恢复71数组通过，本地40文件SHA256验证完成。single_scene差分实际恢复通过，本地正在传输；不能称全部新模型已本地备份。单场景成功/失败录像与真实输入已本地保存。新SSH控制连接不涉及项目配置变更，连接信息不写入记录。

## 2026-11-11 最终接续：归档硬件97条数据与π0.5短训评测

本轮已完成，**不要重复启动**任何11月11日采集/训练/评测流水线。用户22:00关机；此前19:37及更早历史预算均已失效。服务器当前无项目训练/仿真进程。最终结果详见 `docs/LAB_PI05_20261111.md`。

- 按用户授权先按归档：左右不同Zu7模型、完整安装变换/0.4863007m基座间距、右柜DO0左爪/DO1右爪、False开/True闭；现机未核对。三指活动机构、50mm行程、连续开度及三RGB/腕相机仍为仿真假设。硬件及最终相机均已本地备份。
- 新硬件a/b/b2/c/d/e共97条验收数据、41206帧、33布局，32个布局三个目标完整；红蓝黄32/32/33，9个颜色×位置组合都有覆盖。专家+同场景有界servo独立回放+无非预定接触全部通过。b的一次视频日志Unicode编码异常保留失败，UTF-8修复后b2重采，不重复计数。预检first3/a_inputcheck不混入新增数量。
- 全部64组同布局目标切换初始实测状态和三RGB逐值一致；LeRobot/OpenPI转换、统计、H50边界/增量、实际prompt和三类token检查通过。训练集 `lab_multitarget_20261111_merged`。
- 新π0.5从公开base初始化，H50/batch1/seed42，真实4854次更新（21:15时限，未跑满6000），1324.68秒，峰值8508MiB，冻结参数哈希不变。52,153,376可训练参数；训练样本仅约全帧11.8%，不能称收敛。checkpoint `checkpoints/openpi/pi05_zu7_lora_h50/lab_pi05_finetune_20261111/4854`。
- 独立重载输出50×14有限值通过；6回合K8/noise42/650步：已见布局0为0/3，开发布局1000为0/3。5空抓、1步数耗尽；未碰到目标，闭爪时水平偏差13.1–33.6cm。1回合手指自接触，物理关节峰值约0.709rad/s，指令限速不代表实机安全。最终留出2000–2019未用。
- 9次保存专家观测的固定拟合检查K8左关节RMSE0.01082–0.02155rad，H50 0.04051–0.15422rad。初始预测随指令变化但未正确抓取；低loss不是任务成功。没有修改控制器去掩盖失败。
- 所有原始批次、最终转换数据、统计、硬件和结果本地逐文件验证通过。最终训练数据224文件/2卷/2,926,684,774字节，SHA256 `9dd898f673f74ad20b6e78ea4aa2dc82af296d39b7c6fdfe817dda258b798081`。新权重精确delta19文件/508,501,731字节，SHA256 `2e80838faea93de378f3da6e5a34a430dd2f67f452d6a65b506f1c3e804575e4`；实际恢复71参数数组逐字节一致。**delta依赖完整qualification参考备份**（5,764,218,880字节，SHA256 `9438701e4e2007d936fa61361fff665c9477e04bfcd8fa8c9a3dc4a7f3f266a7`），不能单独恢复。最终评测64文件16,155,534字节本地验证通过。数据/权重未声称上传GitHub Release。
- 本地主输出 `outputs/20261111/review/learning.html`，全部6学习视频含失败；专家页index.html独立。6视频3236帧完整解码，本地哈希一致。恢复和备份清单在outputs/20261111，代码Git历史收尾备份另见最终清单。

下一轮先固定97条与控制器，增加有效训练覆盖并检查专家观测预测误差、在线图像差异与闭环偏移，再做明确开发集验证。扩采补齐训练布局及初态/恢复变化，300条设计目标未达到，97条充分性未建立。可靠抓取后再扩展动态干扰和完整双臂交接，不接真机直到安装/夹爪/相机及控制接口核实。失败证据必须保留，不将本次checkpoint称为更好策略。

## 2026-11-11 · 归档硬件、π0.5 工程验收与多目标数据（18:26 接续快照）

用户授权按 JAKA_Dual 归档推进，现机暂无法确认；用户于18:26明确将今天关机延至北京时间22:00，授权继续扩采并做新硬件微调评测；原19:37时限已被取代，不沿用历史实验预算。工程验收、专家成功与学习成功分别报告。

- 硬件证据：37 个文件及 SHA256；左右不同 Zu7 模型，完整 6DoF 安装变换，基座距离 0.4863007 m；右柜 DO0 左爪、DO1 右爪，False 开/True 闭。三指行程 50 mm 和连续开度、三路 RGB/腕相机均为仿真假设，Kinect 光学轴待核对。新数据身份 `dual_jaka_zu7_lab_archive`。详见 `docs/LAB_ARCHIVE_20261111.md`。
- π0.5 工程验收已完成：旧硬件 8 条 1076 帧，H50/batch1/seed42，3 次真实 LoRA+投影更新；52,153,376 可训练参数，总3,403,421,456；梯度有限非零，冻结哈希不变；整卡峰值8527 MiB。独立进程重载后旧 seed0 抓取0/1，不能声称改善。检查点 `checkpoints/openpi/pi05_zu7_lora_h50/pi05_qualification_20261111/3`；报告 `docs/PI05_QUALIFICATION_20261111.md`。
- 检查点本地备份已完成：`outputs/20261111/pi05/qualification_checkpoint_and_results.tar`，5,764,218,880 bytes，42文件逐项校验，SHA256 `9438701e4e2007d936fa61361fff665c9477e04bfcd8fa8c9a3dc4a7f3f266a7`。不是 GitHub Release 上传证明。
- 修复：CUDA fork、左右 URDF 混用、桌面模型不一致、保守球误入桌面、cuRobo 慢自碰撞内核残留输出。每臂当前444球；CPU/GPU FK <0.2微米，碰撞/无碰撞交替回归通过。初态只在第一物理步前重置。无归档依据且干涉手指的模拟相机碰撞外壳已移除，左右URDF哈希更新。
- 专家从抓取 IK 反向检查垂直路径，再连接初态；TOPPRA 后逐250Hz状态复查碰撞、限位、速度与加速度。旧区域部分构型触及J4下限，v2物块区域先经工作空间筛选。三个颜色、三个目标指令、位置/尺寸/yaw变化。外部模拟相机改到物体一侧，初始图像确认三目标可见；相机配置加入物理身份校验。
- `lab_probe_20261111_0_v26` 专家抓取成功：13.188秒仿真、抬升0.10007m、连续保持2.4s；`lab_replay_20261111_0_v26` 同场景有界回放成功，无非预定接触。v25虽抓取/回放成功但掌部碰目标，拒绝入库。v26属于最终外部相机调整前的诊断，也不混入最终批量数据。
- 正在运行 `results/lab_multitarget_20261111_a`：最多30次/10布局，18:55停止新尝试，逐条回放且拒绝非预定接触；18:20 已有首条通过。以progress.json为实时计数，不重复启动。训练布局0–99、开发1000–1009、最终留出2000–2019；300条是目标设计而非现有数量。当前未在新硬件数据上训练。
- 9项单元测试通过；多Parquet转换集成验证4轨迹/12数值帧/36RGB帧通过；π0.5 H50/增量/离散状态审计通过。环境清单现额外保存cuRobo与OpenPI patch。新参数差分备份工具只改变保存编码，合成bfloat16+float32重建字节一致测试通过；尚未在新训练检查点上使用。
- 本轮代码尚未最终提交。后续需收拢批次验收、转换统计、按时间有限微调/开发评测、备份新资料和结果，刷新状态并commit，验证仓库钩子推送。本地hardware备份v2仅覆盖最终外部相机调整之前，最终配置仍需补备份。



## 2026-11-09下午接续

新增的`classify_matched_grasp_events.py`只分析既有15条轨迹；结果为`results/zu7_matched_event_labels_20261109_v2.json`，使用水平位移口径。π0.5独立环境`/data/pi05-env`、固定源码`repos/openpi`和底模缓存`/data/pi05-cache/models`不影响SmolVLA。`convert_zu7_for_openpi.py`用OpenPI环境运行，可从LeRobot 3.0生成独立旧格式副本；现有8条1076帧副本为`data/openpi/zu7_grasp_coverage_v21_full`。`jaka_vla/openpi_zu7.py`定义三相机和12关节增量/2夹爪绝对变换，尚未训练。先读`docs/GRASP_EVENT_TAXONOMY_20261109.md`和`docs/PI05_PREFLIGHT_20261109.md`，不要重复下载/转换或将预检称为模型效果。

## 当前批次：2026-11-09配对诊断已结束

先读MEMORY最上方及`docs/MATCHED_CHECKPOINTS_20261109.md`。stable1500、coverage3000、recovery4000在相同新种子203–207、≤±0.04rad左臂初态扰动、K8/H50/noise42/250步下分别0/5、2/5、2/5。未新训练、未改策略或控制器。逐物理步判据、全部15条目标几何与跟踪诊断均已完成；失败混有接近偏差、提前推动物块和预算耗尽，不要简化为单一原因。下一步按“接近→接触/物块位移→闭爪→有效持物”标注失效，再设计数据/训练步数匹配的对照。

最新查看包为`reviews/2026-11-09/jaka-zu7-matched-checkpoints-20261109-v3.zip`，15段录像及JSON证据已完整性验证；只代表左臂抓取子任务。包内用户资料不含权重/训练集。不要重跑当前15个回合或启动新训练。

## 历史：2026-11-06 18:30批次已结束

先读MEMORY最上方及docs/RECOVERY_CURRICULUM_20261106.md最终结果。recovery4000训练/9回合和离线审计均已完成，原/新/干预恢复分别1/3、1/3、0/3；不得重复运行流水线或将新模型当成更好基线。模型和结果已本地完整校验，新权重未发布Release。

audit_grasp_evaluations.py从保存证据重算持物判据并汇总运动/归一化，不执行策略；audit_grasp_training_fit.py仅适用于当前14条、每条2次调用、26不同观测的28次离线诊断，输出拒绝覆盖，旧名义初态null视为空配置。它们均不能替代闭环评测。最新全景ZIP在reviews/2026-11-06/jaka-zu7-recovery-progress-20261106-1830.zip。

## 从失败中途状态采恢复示教

docs/RECOVERY_CURRICULUM_20261106.md记录18:30续进批次。probe_zu7_short_grasp.py --start-trace <已有失败trace> --start-step N --seed S [--collect]会确认物体未移动>5mm、夹爪仍开>85%，只在记录前重设机器人初态，再运行专家。source_trace/step/hash写入identity；回放自动恢复相同初态。run_zu7_eval.py --initial-condition-file <condition.json>可单独验证学习恢复，必须与普通初态分开报告。不得把初态干预说成完整策略前缀或自主在线恢复。

run_recovery_batch.py最多5条追加候选，加已完成seed0-step50共最多6条；train_grasp_recovery.py合并旧8条与通过恢复轨迹，从base最多4000步；evaluate_recovery4000.py最多9回合（已见3、新200–202三条、明确恢复初态3条），18:20后不启动新回合。接续先查状态，不重复启动。test_statistics.py覆盖近常量关节噪声放大回归。

## 归一化修正与初态覆盖

先读docs/NORMALIZATION_FIX_20261106.md。旧shortgrasp1500为0/3，新stable1500已完成且2/3，二者分别从base训练1500步。不能再次运行当成新结果。新数据先由stabilize_zu7_statistics.py复制并重算float64统计（关节std floor .05rad、夹爪.05），再prepare_training_v2.py，训练入口核查哈希并拒绝近零状态尺度。policy_server_v2会回传normalized_state_abs_max，eval报告normalization_checks；这些只作诊断，不改变动作。

probe_zu7_short_grasp.py --initial-offset .04给左臂设置可复现的初始偏移并稳定后才开始采样；eval专家回放自动读取源identity中的initial_condition；学习评测须显式使用同样幅度。仅初态重置，不是在线专家纠偏。run_grasp_coverage_batch.py有明确数量/种子/17:05停止采集约束，不重复启动已有批次。

## 短路径抓取课程（2026-11-06 16时后）

先读docs/SHORT_GRASP_20261106.md的本轮结果，避免重跑已结束预算。probe_zu7_short_grasp.py --seed N --output <新目录> [--collect]仅执行专家左抓取、抬升10cm、保持1秒。4个抓取面×2种等价夹爪roll，规划前段添加目标物体7×7×21cm避碰盒；末段允许目标接触，桌面/另一臂避碰保留。候选按完整两段的关节空间弧长选择并缓存执行，不重新随机规划。原选择器可加--original-selector进行同种子对照。

采集产物在<输出>/raw，状态/动作仍为实测/下一25Hz采样点指令。必须用eval_v2.py --expert-hdf5 <raw/data> --seed-start N --episodes 1 --review-video逐条回放，再由assemble_grasp_curriculum.py验收。抓取课程只有一条所有场景共用的准确指令，不能解释为语言泛化。eval_v2.py和run_zu7_eval.py新增--instruction，默认原完整任务句不变；success仍只表示完整交接，grasp_success单独报告。

## 全景录像与抓取阶段评价（2026-11-06下午）

`run_zu7_eval.py --review-video`会额外生成`seed_<seed>_multiview.mp4`，包括全局斜视、俯视、头部近景和左腕图像。`eval_v2.py`的专家回放也支持此参数。新录像补拍终态；片头2秒/片尾3秒明确标注静帧，原始仿真时长见results.json。不能为延长视频而继续执行失败动作或把静帧时间计为持物时间。

`grasp_success`仅表示左抓取子阶段，原`success`仍表示完整交接放置。接触与物体真值仅用于评价，不能反馈给策略或控制器。看新录像时同时报告检查点、测试种子、抓取/完整成功及终止原因。

`build_multiview_review.py --output <新目录> --archive <新ZIP> --experiments <实验目录名...>`生成可离线查看的半速/逐帧页面，包含完整录像、结果、数值轨迹和视频校验。默认拒绝覆盖旧包；只同步此项目。

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

## 当前主流程：两台 Zu7

旧ALOHA流程只作历史排错参考。读取`MEMORY.md`与`docs/ZU7_CORRECTION_20261105.md`后，核实最新结果，不重跑已完成批次。

- 机器人配置：`configs/robots/dual_zu7.json`；`scripts/setup_zu7.py`下载固定版本官方模型并生成SAPIEN/cuRobo配置。变更几何/安装必须建立新数据批次，保留旧身份清单。
- 上游修正入口：`patch_zu7_planner.py`、`patch_zu7_tool_frame.py`、`patch_zu7_task.py`、`patch_zu7_interarm.py`、`patch_collection_budget.py`。首次建立环境依次执行，结束时`freeze_environment.sh`记录父仓库及XPolicyLab patch。
- 收集：仿真环境运行`python scripts/collect_zu7.py --task-config zu7_collision_smoke`。现有3条已完成，直接重跑会拒绝覆盖。原始目录有robot_identity.json，HDF5有机器人和URDF哈希。
- 转换：VLA环境`convert_dataset_v2.py --source <绝对HDF5目录> --output <新LeRobot目录> --repo-id <本地ID>`，自动检查实测语义、均匀时间、14维、RGB、身份。
- 控制验收：仿真环境`eval_v2.py --expert-hdf5 <绝对data目录> --seed-start 0 --episodes 3 --output <新的results绝对目录>`。根据seed.txt选择对应示教，不能用一条示教回放不同随机场景。
- 训练：`prepare_training_v2.py`生成显式特征配置，再由`run_training_v2.py --config <配置JSON>`训练。已完成的输出拒绝覆盖。检查resource_profile.completed及每个检查点的project_identity.json。
- 学习闭环：`run_zu7_eval.py --name zu7_<新实验名> --checkpoint <绝对checkpoint路径> --seed-start 0 --episodes 3 --k 8`。包装器在两环境启动自己的服务/仿真并限制运行时间，结束时只清理自己启动的进程。
- 分析：`analyze_control_traces.py --results <实验目录>`。必须同时看指令和实测速度；低平均误差、指令限速、最终任务成功不能掩盖瞬时碰撞/速度尖峰。
- 新批次必须逐条按seed.txt回放，再用`curate_zu7_replays.py --source <原始根目录> --replays <各seed回放的父目录> --output <新筛选根目录>`筛选任务有效轨迹。失败示教不得进入训练集或其归一化统计；接受任务回放不等于安全验收。筛选输出保留SHA256及源文件链接，复制/备份时须保留原始源数据。
- 离线预测审计：`audit_zu7_predictions.py`；闭环偏离分析：`analyze_zu7_failed_rollouts.py`。这两类结果均不能替代学习策略任务成功率。
- 查看：`build_zu7_review.py`生成`results/zu7_review/index.html`和`results/jaka-zu7-review.zip`，与旧pilot页面分开。
- 查看本轮快照可显式使用`build_zu7_review.py --output /data/jaka-vla/results/zu7_review_20261106 --archive /data/jaka-vla/results/jaka-zu7-review-20261106.zip`；不会覆盖此前查看包，支持分seed保存的专家回放。
- 参数消融可改控制器并记录实际配置；物理rig、任务几何、URDF和25Hz采样兼容性仍需通过校验。固定噪声与random要作为明确实验条件报告。

## 旧 ALOHA 数据流程（历史）

1. `scripts/configure_smoke.py` 生成 `jaka_smoke`（1 条）与 `jaka_pilot`（50 条）配置；`scripts/patch_sim_timing.py` 应用于已固定版本的 RoboTwin。
2. 采集入口为 RoboTwin 的 `scripts/collect_data.py handover_block jaka_pilot`。运行前检查现有进程与输出，不重复采集正在运行的批次。
3. 转换用 `scripts/convert_dataset.py --source <原始HDF5目录> --output <新目录> --repo-id <实验ID>`；在 VLA 环境执行。现有 pilot 的精确参数见 `scripts/pilot_pipeline.sh`。
4. 必须验证 `measured_qpos` 语义、25 Hz 时间戳、14 维形状、有限值与三路 RGB。通过后检查 `results/<数据集目录名>_validation.json`。
5. `_command_state_v0` 仅为历史排错产物，不与修正后的数据混训。新转换输出不能覆盖旧目录。

## 旧 ALOHA 训练与评测（历史）

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

## 项目限定的GitHub备份

用户只授权本JAKA-VLA自动同步至GaoPeixin418/JAKA-VLA私有仓库，不据此同步其他项目或修改全局Git配置。该仓库的每个项目commit会由post-commit钩子推送到main；提交前先完成检查与记录。现有2026-11-05快照已上传并校验；不要因查看进度而重复打包或重复上传。训练依旧关闭push_to_hub/WandB。

恢复/更新备份先读docs/BACKUP_RESTORE.md、docs/HOME_ACCESS.md和manifests/github-backup-20261105.json。Git clone只得到代码；完整实验附件在Release。仅有文件存在不代表传输完成，使用restore_backup.py完整校验；独立本地副本须检查LOCAL_BACKUP_COMPLETE.json而非仅查看目录大小。


## 小体积录像查看包同步

用户已授权本项目同步。生成完整校验过的小体积查看包（本轮约4.7MiB）后，可将固定日期的ZIP复制到reviews/YYYY-MM-DD/，附SHA256SUMS和中文查看说明，随项目commit同步。不要覆盖历史日期包；先核验本轮结果与MP4及页面引用齐全。数据集、权重与大规模原始视频仍走独立备份，不加入Git。更新HOME_ACCESS与README入口，不能把本地存在说成GitHub已同步。

## 条件输入和专家路径的只读诊断

audit_zu7_paths.py读取已有轨迹，报告首次闭爪请求前的TCP长度/端点距离/构型变化。直线不是可执行避碰路径，失败端点也不可当成功路径对照。
audit_zu7_conditioning.py固定两检查点各6观测和seed42噪声，总84次离线调用，拒绝覆盖。只测输入敏感性，不执行动作/训练；不能把输出变化说成指令执行或泛化成功。本轮已完成，接续先读结果。
