# 短路径抓取修正与小批课程验证（2026-11-06）

用户在15:56要求继续推进，仍按此前16:30关机时间预留备份。本轮先修正专家抓取路径，再验收数据及尝试学习抓取子任务。机器人仍是两台官方Zu7模型与暂定平行夹爪，没有连接真机。

## 已确认的路径问题与修正

旧选择器主要按抓取朝向打分；其中choose_best_pose也没有更新now_step，无法按代码意图选择较短候选。新模块jaka_vla/short_grasp.py不修改上游文件，针对首次左抓取枚举4个抓取面和2种等价夹爪roll。对每个候选先规划到预抓取位置，再从这一规划的终点规划7cm受约束插入。按两段关节空间弧长选择，并缓存所选轨迹执行，避免重新规划改变IK分支。

第一版v1的seed6路径只有0.522m，却在1.228s开始用手指撞动目标，最终空抓。这条失败被完整保留，未用于训练。v2把目标物体以7×7×21cm盒加入前段规划世界，末段允许抓取接触，同时保留桌面、机器人自碰及另一臂几何。目标真值只用于专家，不输入学习策略。

## 专家物理验证

同一seed6，原选择器与v2使用同一初始场景、同一机器人、同一直接专家执行器。长度按250Hz实测TCP计算，到第一次闭爪请求低于95%为止；与之前25Hz审计的2.775m略有采样差异。

| 运行 | 抓取前TCP行程 | 峰值实测关节速度 | 连续有效夹持 | 结果 |
|---|---:|---:|---:|---|
| 原选择器seed6 | 2.777m | 2.032rad/s | 1.324s | 抓取成功 |
| v1短路径seed6 | 0.522m | 1.425rad/s | 0s | 撞动目标、空抓 |
| v2避碰短路径seed6 | 0.531m | 1.221rad/s | 1.336s | 抓取成功 |
| v2采集seed0 | 0.579m | 1.305rad/s | 1.344s | 抓取成功 |
| v2采集seed1 | 0.566m | 1.217rad/s | 1.344s | 抓取成功 |
| v2采集seed6 | 0.538m | 1.241rad/s | 1.316s | 抓取成功 |

seed6首次对照行程减少约80.9%，但耗时只从6.080s降至5.356s，因为新路径更保守。再次采集时渲染/调用上下文下规划略有变化，不能宣称逐位确定性；每次执行仍复用当次选中的两条规划。所有成功新专家均抬升约10cm，未检测到双臂接触，最大关节跟踪误差约0.0021–0.0023rad。峰值速度下降不能单独证明所有抖动已消除，更不能作为真机安全认证。

有效夹持判据：物体抬升>5cm、实测左开度15–85%、左右两指均对物体产生>1e-5Ns接触冲量、右臂未碰物体，连续至少0.6s。物体/接触真值只用于评价。

## 数据与学习实验

采集使用25Hz连续均匀采样，状态是实测关节/开度，动作为下一采样点指令。每条HDF5带机器人和URDF身份。用原有BoundedJointServo逐条回放seeds0/1/6，抓取均通过，连续有效夹持分别2.260/2.288/2.248s，无双臂接触。旧评测器继续执行至预算，所以原始报告保留完整任务success=false、step_budget_exhausted；这三次验证目标是抓取子任务，未做交接。

assemble_grasp_curriculum.py检查同seed、专家抓取、控制器回放抓取、控制保护和双臂接触后才组装接受集。3条共398帧（130/134/134），总计约15.92秒，位于data/raw/zu7_shortgrasp_accepted和data/lerobot/zu7_shortgrasp_accepted。所有场景共用一句任务准确指令：Use the left arm to grasp the red block, lift it, and hold it.

这是工程可学性检查所用的小批次，远不足以支持泛化结论。任务scope明确是left_grasp_lift_hold，不混入旧完整交接数据，不延续已有9000步检查点。新配置zu7_shortgrasp1500.json从固定SmolVLA base开始，冻结VLM、训练动作专家/状态投影，条件流匹配，batch8、lr1e-4、H50、AMP、seed42、1500步。完成状态和学习结果见本文后续记录与resource_profile.json。

评测仍为三路RGB、14维实测状态、文本→动作，K8，固定噪声，250Hz物理和25Hz控制，同步推理暂停仿真。新增--instruction及--task-scope选项，默认行为保持完整交接；抓取子任务达到0.6s判据后结束并明确标注。success仍只代表完整交接，grasp_success单独报告。三条训练种子均为已见场景；成功也不能称为未见场景泛化或语言理解已验证。

## 文件与复现

- 专家选择器：jaka_vla/short_grasp.py；验证/采集：scripts/probe_zu7_short_grasp.py。
- 失败及对照：results/zu7_shortpath_v1_seed6、zu7_shortpath_original_seed6、zu7_shortpath_v2_seed6。
- 新示教：results/zu7_shortpath_collect_seed0、seed1、seed6中的raw子目录。
- 控制器回放：results/zu7_shortpath_replay_seed0、seed1、seed6。
- 数据验收：scripts/assemble_grasp_curriculum.py及接受集acceptance.json。
- 配置：configs/zu7_shortgrasp1500.json与.identity.json。
- 训练：checkpoints/smolvla_zu7_shortgrasp1500_seed42；logs/train-zu7-shortgrasp1500.log。
- 查看包构建：scripts/build_short_grasp_review.py，支持专家与学习结果，保留失败视频、数值轨迹与SHA256；片头2秒和片尾3秒明确标注冻结画面，不计入有效持物时间。

后续应先固定这一干净抓取动作族，扩充独立场景并设置留出评测，再增加物体尺寸/形状、遮挡、抓取失败和恢复、多目标指令选择。完整交接需单独处理相对位姿与碰撞；当前研究目标“事件感知自适应重规划”尚未实现。

## 本轮最终学习结果与备份

1500步已完成，用时313.54秒，峰值CUDA分配3.575GiB。seeds6/0/1的K8闭环抓取为0/3；均未完成完整交接（未测）。逐条证据：[{"seed": 6, "grasp_success": false, "oracle_max_object_lift_m": 0.0, "failure_reason": {"type": "step_budget_exhausted", "budget": 200}}, {"seed": 0, "grasp_success": false, "oracle_max_object_lift_m": 0.0, "failure_reason": {"type": "step_budget_exhausted", "budget": 200}}, {"seed": 1, "grasp_success": false, "oracle_max_object_lift_m": 2.980232238769531e-07, "failure_reason": {"type": "step_budget_exhausted", "budget": 200}}]。seed6在接近中途停滞，8秒内未闭爪，末端仍约18cm偏离目标；小跟踪误差不代表策略目标正确。不能声称这轮获得新的学习抓取成功demo。下一步应对中途停滞状态做固定噪声离线动作/归一化审计，并采集带初始姿态扰动和纠偏的示教，形成独立留出评测；不继续仅以训练loss判断可用性。

查看包：reviews/2026-11-06/jaka-zu7-short-grasp-20261106.zip，包含原专家、失败v1、修正专家、控制器回放及三条学习结果。专家/数据83文件已在本地逐项SHA256核对。完整检查点约1.3GiB，原始整包传输因带宽不足改为增量方式。服务器在16:30后仍在线，已于16:35完成：197MiB模型差分与此前本地保存的independent3000模型重建新model.safetensors，SHA256逐字节一致；优化器、调度器、RNG、归一化、配置及LeRobot数据另行下载并逐项校验28个文件。专家/数据83文件和查看包63文件亦已校验。本地LOCAL_BACKUP_COMPLETE.json记录完成状态。权重与完整训练状态现已保存在服务器和本地；GitHub本轮只同步代码、文档及小型查看包，未上传本轮权重Release。scripts/restore_checkpoint_delta.py已在服务器和本地各完成一次精确重建校验。
