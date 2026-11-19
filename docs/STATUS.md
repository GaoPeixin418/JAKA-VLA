# 2026-11-19 当前接续状态

服务器已开机，11月18日新增29条专家轨迹已完成29/29完整数据检查及同场景控制器回放验收（12559帧），派生训练raw数据集已生成；LeRobot/OpenPI转换及入训尚未开始。原97条数据续训的3500步检查点仍未在新布局评测。恢复target1重对齐示教的控制器回放未完成，target0候选失败。当前优先完成项目代码/记录GitHub同步、独立数据与权重备份、97+29数据集版本化合并，再分阶段微调并固定未见布局评测。

最新接续细节见 [ITERATION_20261119.md](ITERATION_20261119.md)。以下保留此前状态快照作为历史记录，不代表当前实时进程状态。

---

# 2026-11-16 固定模型评测最终状态

本轮固定π0.5 three_targets/5744、K25/H50/noise42，共完成333/333回合；训练布局复现1/3（33.3%），开发集6/30（20.0%），原定留出15/60（25.0%），追加留出56/240（23.3%），全部独立留出71/300（23.7%）。完整独立留出布局100个，布局聚类bootstrap 95%区间19.3% - 28.0%。运行异常0，未完成0。

仅验证归档硬件仿真中的左臂抓取、抬升和0.6秒持物；同步推理暂停仿真，不代表长期持物、双臂交接、语言作用消融、抗推理延迟或真机结果。此次未训练、未调整控制器或成功判据。

本轮评测已结束，后续运行状态应另查实际进程；下方为旧状态工具的历史阶段快照。报告见BENCHMARK_20261116_RESULTS.md。

# 实验室归档硬件与 π0.5 当前状态

采样时间：2026-11-16T20:12:06+08:00（北京时间）。

按归档建模，尚未实机核对；夹爪行程、腕相机和光学坐标轴仍有假设。
硬件与规划证据见 `docs/LAB_ARCHIVE_20261111.md`；π0.5工程验收见 `docs/PI05_QUALIFICATION_20261111.md`。

活跃脚本：[]

| 新批次 | 当前结果 |
| --- | --- |
| results/lab_multitarget_20261111_a/progress.json | 12 条验收通过 / 12 次尝试；completed=True；deadline |
| results/lab_multitarget_20261111_b/progress.json | 1 条验收通过 / 2 次尝试；completed=True；persistent_worker_exception |
| results/lab_multitarget_20261111_b2/progress.json | 2 条验收通过 / 2 次尝试；completed=True；attempt_budget |
| results/lab_multitarget_20261111_c/progress.json | 30 条验收通过 / 30 次尝试；completed=True；attempt_budget |
| results/lab_multitarget_20261111_d/progress.json | 30 条验收通过 / 30 次尝试；completed=True；attempt_budget |
| results/lab_multitarget_20261111_e/progress.json | 22 条验收通过 / 22 次尝试；completed=True；deadline |
| results/lab_pi05_finetune_20261111/audit.json | 更新 4854 步；completed=True；仅说明训练工程状态 |
| results/lab_pi05_finetune_stage2_20261112/audit.json | 更新 9854 步；completed=True；仅说明训练工程状态 |
| results/lab_pi05_finetune_stage3_20261112/audit.json | 更新 14854 步；completed=True；仅说明训练工程状态 |
| results/lab_pi05_single_scene_20261112/audit.json | 更新 2000 步；completed=True；仅说明训练工程状态 |
| results/lab_pi05_three_targets_20261112/audit.json | 更新 5744 步；completed=True；仅说明训练工程状态 |
| results/pi05_qualification_20261111/audit.json | 更新 3 步；completed=True；仅说明训练工程状态 |
| 新硬件合并训练集 | 97 条 / 41206 帧 / 33 布局；数据充分性未确立 |
| results/lab_pi05_eval_20261111/summary.json | seen 合格抓取 0/3；development 合格抓取 0/3；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_single_scene_20261112/summary.json | seen 合格抓取 0/1；development 合格抓取 0/0；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_single_scene_k50_20261112/summary.json | seen 合格抓取 1/1；development 合格抓取 0/0；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_stage2b_20261112/summary.json | seen 合格抓取 0/3；development 合格抓取 0/3；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_stage3_20261112/summary.json | seen 合格抓取 0/3；development 合格抓取 0/3；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_stage3_k50_20261112/summary.json | seen 合格抓取 0/3；development 合格抓取 0/3；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_three_targets_dev_k25_20261112/summary.json | seen 合格抓取 0/0；development 合格抓取 2/3；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_three_targets_k25_20261112/summary.json | seen 合格抓取 1/3；development 合格抓取 0/0；completed=True；小样本开发诊断 |
| results/lab_pi05_eval_three_targets_seen_20261112/summary.json | seen 合格抓取 0/3；development 合格抓取 0/0；completed=True；小样本开发诊断 |

以下为旧硬件及 ALOHA 历史记录。

# 历史 Zu7 临时夹爪实验

采样时间：2026-11-16T20:12:06+08:00。已确认两台 Zu7；夹爪/安装为暂定仿真配置，未接真机。

活跃项目脚本：[]

| Zu7 实验产物 | 验收结果 |
| --- | --- |
| results/zu7_coverage3000_new_seed100_102/results.json | 抓取子任务 2/3；未测试完整交接 |
| results/zu7_coverage3000_seen_seed0_1/results.json | 抓取子任务 1/2；未测试完整交接 |
| results/zu7_coverage3000_seen_seed6/results.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_expert_collision_replay/results.json | 完整任务 3/3 成功 |
| results/zu7_expert_contact_diagnostic/results.json | 完整任务 1/1 成功 |
| results/zu7_expert_replay_v2/results.json | 完整任务 0/3 成功 |
| results/zu7_expert_replay_v3/results.json | 完整任务 3/3 成功 |
| results/zu7_grasp_review6000_seed0_1/results.json | 完整任务 0/2 成功；左抓取 0/2 |
| results/zu7_grasp_review6000_seed2/results.json | 完整任务 0/1 成功；左抓取 1/1 |
| results/zu7_grasp_review9000_seed6/results.json | 完整任务 0/1 成功；左抓取 0/1 |
| results/zu7_independent_expert_replay/seed_11/results.json | 完整任务 1/1 成功 |
| results/zu7_independent_expert_replay/seed_12/results.json | 完整任务 0/1 成功 |
| results/zu7_independent_expert_replay/seed_6/results.json | 完整任务 1/1 成功 |
| results/zu7_independent_expert_replay/seed_7/results.json | 完整任务 1/1 成功 |
| results/zu7_independent_expert_replay/seed_8/results.json | 完整任务 1/1 成功 |
| results/zu7_independent_expert_replay/seed_9/results.json | 完整任务 1/1 成功 |
| results/zu7_learned_independent9000_seed3_5/results.json | 完整任务 0/3 成功 |
| results/zu7_learned_seen3000/results.json | 完整任务 0/3 成功 |
| results/zu7_learned_seen6000/results.json | 完整任务 0/3 成功 |
| results/zu7_learned_seen6000_zero/results.json | 完整任务 0/3 成功 |
| results/zu7_learned_seen9000_seed6_8/results.json | 完整任务 0/3 成功 |
| results/zu7_learned_unseen6000_seed3_5/results.json | 完整任务 0/3 成功 |
| results/zu7_learned_unseen6000_seed3_5_k1/results.json | 完整任务 0/3 成功 |
| results/zu7_learned_unseen6000_seed3_5_k4/results.json | 完整任务 0/3 成功 |
| results/zu7_matched_coverage3000_seed203_207/results.json | 抓取子任务 2/5；未测试完整交接 |
| results/zu7_matched_recovery4000_seed203_207/results.json | 抓取子任务 2/5；未测试完整交接 |
| results/zu7_matched_stable1500_seed203_207/results.json | 抓取子任务 0/5；未测试完整交接 |
| results/zu7_multiview_expert_seed6/results.json | 完整任务 1/1 成功；左抓取 1/1 |
| results/zu7_perturbed_collect_seed13/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_collect_seed14/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_collect_seed15/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_collect_seed16/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_collect_seed17/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_replay_seed13/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_replay_seed14/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_replay_seed15/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_replay_seed16/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_perturbed_replay_seed17/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_pi05_qualification_20261111/results.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_recovery4000_intervened_seed0_step50/results.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_recovery4000_intervened_seed101_step45/results.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_recovery4000_intervened_seed6_step35/results.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_recovery4000_new_seed200_202/results.json | 抓取子任务 1/3；未测试完整交接 |
| results/zu7_recovery4000_seen_seed0_1/results.json | 抓取子任务 1/2；未测试完整交接 |
| results/zu7_recovery4000_seen_seed6/results.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_recovery_collect_seed0_step40/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_collect_seed0_step50/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_collect_seed101_step35/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_collect_seed101_step45/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_collect_seed6_step25/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_collect_seed6_step35/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_replay_seed0_step40/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_replay_seed0_step50/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_replay_seed101_step35/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_replay_seed101_step45/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_replay_seed6_step25/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_recovery_replay_seed6_step35/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_shortgrasp1500_seed0_1/results.json | 抓取子任务 0/2；未测试完整交接 |
| results/zu7_shortgrasp1500_seed6/results.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_shortpath_collect_seed0/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_shortpath_collect_seed1/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_shortpath_collect_seed6/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_shortpath_original_seed6/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_shortpath_replay_seed0/results.json | 完整任务 0/1 成功；左抓取 1/1 |
| results/zu7_shortpath_replay_seed1/results.json | 完整任务 0/1 成功；左抓取 1/1 |
| results/zu7_shortpath_replay_seed6/results.json | 完整任务 0/1 成功；左抓取 1/1 |
| results/zu7_shortpath_v1_seed6/result.json | 抓取子任务 0/1；未测试完整交接 |
| results/zu7_shortpath_v2_seed6/result.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_stable1500_seed0_1/results.json | 抓取子任务 1/2；未测试完整交接 |
| results/zu7_stable1500_seed6/results.json | 抓取子任务 1/1；未测试完整交接 |
| results/zu7_task_probe_v3/result.json | 完整任务 0/1 成功 |
| results/zu7_task_probe_v4/result.json | 完整任务 0/1 成功 |
| results/zu7_task_probe_v5/result.json | 完整任务 1/1 成功 |
| results/zu7_task_probe_v6/result.json | 完整任务 0/1 成功 |
| results/zu7_task_probe_v7/result.json | 完整任务 1/1 成功 |
| data/raw/zu7_collision_smoke/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 3 条原始示教 |
| data/raw/zu7_grasp_coverage/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 8 条原始示教 |
| data/raw/zu7_grasp_recovery/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 14 条原始示教 |
| data/raw/zu7_independent_accepted/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 5 条原始示教 |
| data/raw/zu7_independent_smoke/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 6 条原始示教 |
| data/raw/zu7_shortgrasp_accepted/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 3 条原始示教 |
| data/raw/zu7_slow_smoke/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 0 条原始示教 |
| data/raw/zu7_smoke/handover_block/jaka_zu7_proxy_jaka_zu7_proxy | 3 条原始示教 |
| checkpoints/smolvla_zu7_coverage3000_seed42/resource_profile.json | completed=True |
| checkpoints/smolvla_zu7_independent3000_seed42/resource_profile.json | completed=True |
| checkpoints/smolvla_zu7_overfit3000_seed42/resource_profile.json | completed=True |
| checkpoints/smolvla_zu7_recovery4000_seed42/resource_profile.json | completed=True |
| checkpoints/smolvla_zu7_refine3000_seed42/resource_profile.json | completed=True |
| checkpoints/smolvla_zu7_shortgrasp1500_seed42/resource_profile.json | completed=True |
| checkpoints/smolvla_zu7_stable1500_seed42/resource_profile.json | completed=True |
| train-zu7-coverage3000.log | 已观察到 3000/3000 步 |
| train-zu7-independent3000.log | 已观察到 3000/3000 步 |
| train-zu7-overfit3000.log | 已观察到 3000/3000 步 |
| train-zu7-recovery4000.log | 已观察到 4000/4000 步 |
| train-zu7-refine3000.log | 已观察到 3000/3000 步 |
| train-zu7-shortgrasp1500.log | 已观察到 1500/1500 步 |
| train-zu7-stable1500.log | 已观察到 1500/1500 步 |

下面为已结束的 ALOHA 历史 pilot 快照。不能当作当前 Zu7 实验状态。

# 当前运行状态（自动生成快照）

采样时间：2026-11-16T20:12:06+08:00（北京时间）。

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
| 数据盘可用 | 11.39 GiB |
| GPU 快照 | NVIDIA GeForce RTX 4080 SUPER, 16 MiB, 32760 MiB, 0 % |

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
