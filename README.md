# 双臂 VLA 事件感知重规划实验

## 最新结果：2026-11-16 固定模型性能评测

本轮固定π0.5 three_targets/5744、K25/H50/noise42，共完成333/333回合；训练布局复现1/3（33.3%），开发集6/30（20.0%），原定留出15/60（25.0%），追加留出56/240（23.3%），全部独立留出71/300（23.7%）。完整独立留出布局100个，布局聚类bootstrap 95%区间19.3% - 28.0%。运行异常0，未完成0。

[评测资料与全部录像](reviews/2026-11-16/README.md) · [汇报版PDF](reviews/2026-11-16/JAKA_pi05_性能评测报告.pdf) · [详细报告](docs/BENCHMARK_20261116_RESULTS.md)

仅验证归档硬件仿真中的左臂抓取、抬升和0.6秒持物；同步推理暂停仿真，不代表长期持物、双臂交接、语言作用消融、抗推理延迟或真机结果。此次未训练、未调整控制器或成功判据。

## 当前技术路线

- 场景：RoboTwin / SAPIEN，按实验室归档建模的两台JAKA Zu7；左右模型、基座安装、DO夹爪映射有归档依据。三指机构行程、连续开度及三RGB/腕相机仍为待实机核验的仿真假设。
- 模型：OpenPI π0.5，LoRA加投影微调；97条验收示教/41206帧/33训练布局已保存。当前检查点在已有权重上做三目标课程训练，不等于充分利用全数据后的最优模型。
- 输入：三路RGB、14维实测关节/夹爪状态和语言；输出H50×14维目标动作，K25执行，25Hz控制、250Hz物理仿真。
- 当前阶段：固定模型能力评估已扩大到多个布局；先提升基础抓取稳定性，再推进完整双臂交接、事件重规划和真机迁移。
- 专家成功、学习策略成功、研究方法效果和真机结果分别报告。历史SmolVLA试验保留在对应日期文档。

## 项目记录

- [AGENT.md](AGENT.md)、[AGENTS.md](AGENTS.md)：协作约定与记录责任。
- [MEMORY.md](MEMORY.md)：当前事实、已完成工作和接续步骤。
- [skill.md](skill.md)：环境、采集、转换、训练、评测和恢复操作。
- [修正报告](docs/ZU7_CORRECTION_20261105.md)：本轮改动、验收结果与限制。
- [技术方案](docs/TECHNICAL_DESIGN.md)、[实验日志](docs/EXPERIMENT_LOG.md)：方法与完整成功/失败记录。
- [状态快照](docs/STATUS.md)：用`python3 scripts/status.py --write-doc`刷新。
- [查看录像](docs/VIEW_RESULTS.md)：服务器原始产物和离线播放包。
- [微调说明](docs/FINETUNING_GUIDE.md)：训练方法、数据设计、已做实验和检查点内容。
- [备份与恢复](docs/BACKUP_RESTORE.md)：GitHub代码、Release实验附件和换电脑后的操作。
- [在家查看项目](docs/HOME_ACCESS.md)：浏览器、代码ZIP、GitHub Desktop及离线录像的逐步说明。

JAKA-VLA 在服务器项目仓库提交后自动推送到私有 GitHub `main`。同步钩子和安装入口见 `scripts/github_post_commit.sh`、`scripts/install_github_autosync.sh`；其他项目不会同步。

## 环境与配置

本轮服务器识别为RTX 4080 SUPER，报告显存32760MiB，约58GiB内存、148GiB数据盘。运行前`source env.sh`。
仿真使用`envs/robotwin/bin/python`；π0.5使用独立`/data/pi05-env/bin/python`及`scripts/pi05_env.sh`。历史SmolVLA使用`envs/vla/bin/python`。
当前归档硬件配置`configs/robots/dual_zu7_lab.json`，归档证据`configs/robots/lab_archive_20261111.json`；生成/注册入口`scripts/setup_lab_zu7.py`。旧配置保留用于历史实验。
版本、依赖清单和上游patch保存在`manifests/`。源网格按固定版本本地下载，不包含在代码归档。

## 实验管理

采集/训练/评测均有独立预算与超时，输出拒绝覆盖。旧ALOHA pilot（50条、2000步、0/9）保留为历史失败记录，不能混入Zu7数据或当成其训练结果。
训练集与已见场景评测用于检查拟合，未见场景和更多种子才能评价泛化。当前无论文级结论。
用户已授权本项目的私有GitHub备份：代码/文档和小体积录像包进仓库。2026-11-05完整实验快照在Release；11-06新增数据/最终权重/优化器在服务器及经过SHA256核验的本地副本，尚未发布新Release。训练不自动上传，WandB关闭。只有Git clone不能得到权重，具体备份状态见备份文档。
