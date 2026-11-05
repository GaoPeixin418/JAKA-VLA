# 双臂 VLA 事件感知重规划实验

项目根目录：`/data/jaka-vla`。当前平台为两台 JAKA Zu7 的仿真模型，使用暂定平行夹爪与安装配置；尚未连接真机。

## 当前路线

先把抓取—交接—放置的基础策略和控制链路做可靠，再研究根据协作事件调整重规划时机。专家成功、学习策略成功、语言泛化和真机结果分别报告。

- 场景：RoboTwin / SAPIEN；官方Zu7 URDF/网格；蓝色8cm接收台。
- 模型：LeRobot0.4.4 / SmolVLA 450M，约99.88M参数可训练，单4090运行。
- 输入：三路RGB、14维实测关节/夹爪状态、语言指令。
- 动作：H=50的14维目标块，25Hz控制、250Hz物理步，跨块衔接并输出一致的位置/速度目标。
- 专家规划已加入另一臂几何；最后退回/放置分开执行，保证规划时另一臂静止。学习动作并不自动获得规划器的碰撞保证。
- 记录逐步指令、实测运动、物体/TCP轨迹、接触冲量、空抓和跟踪异常。仿真真值只用于诊断/评价。
- 同步推理期间仿真暂停；当前结果不是抗推理延迟证据。

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

## 环境与配置

服务器RTX4090 24GB，约60GB系统内存，150GB数据盘。运行前`source env.sh`。
仿真使用`envs/robotwin/bin/python`；转换、训练和模型服务使用`envs/vla/bin/python`。
机器人主配置`configs/robots/dual_zu7.json`；生成/注册入口`scripts/setup_zu7.py`。
版本、依赖清单和上游patch保存在`manifests/`。源网格按固定版本本地下载，不包含在代码归档。

## 实验管理

采集/训练/评测均有独立预算与超时，输出拒绝覆盖。旧ALOHA pilot（50条、2000步、0/9）保留为历史失败记录，不能混入Zu7数据或当成其训练结果。
训练集与已见场景评测用于检查拟合，未见场景和更多种子才能评价泛化。当前无论文级结论。
用户已授权本项目的私有GitHub备份：代码/文档进仓库，数据/检查点/录像/日志进带SHA256校验的Release附件。训练不自动上传，WandB关闭。只有Git clone不能得到大文件，释放实例前须核对附件完整性与本地副本；具体状态见备份文档。
