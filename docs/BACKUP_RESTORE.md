# JAKA-VLA 备份与恢复

范围仅为 `/data/jaka-vla` 项目。用户账户 `GaoPeixin418` 下的[私有仓库 JAKA-VLA](https://github.com/GaoPeixin418/JAKA-VLA)保存代码、文档和小体积录像查看包；2026-11-05完整实验快照已上传Release。11-06新增数据/最终权重/优化器在服务器和经过SHA256核验的本地副本，尚未发布新Release。JAKA-VLA仓库启用提交后自动推送main，不为其他项目设置Git。

## 内容和边界

代码仓库保存完整项目Git历史、脚本、机器人配置、技术文档、训练配置、依赖版本与上游补丁。
2026-11-05 Release附件保存截至该快照的 `data/`、历史和Zu7 `checkpoints/`（含优化器/归一化器）、`results/`（含录像/诊断/失败记录）、`logs/`，以及代码Git bundle和上游新增配置的备份。每个分片最多1500MiB，附每个原文件及分片的SHA256。该Release不包含11-06新增实验。

不包含Python环境、下载/编译缓存、访问凭据、其他项目，以及可从固定来源重新获取的第三方仓库和模型网格。它是实验产物备份，不是可直接启动的服务器磁盘镜像。第三方下载服务未来不可用时仍需另行取得相应依赖。

## 在家查看

1. 登录自己的GitHub账号，进入私有JAKA-VLA仓库。直接阅读README、docs/FINETUNING_GUIDE.md、MEMORY.md和修正报告。
2. 下载代码可用页面 `Code → Download ZIP`；保留Git历史则运行 `git clone https://github.com/GaoPeixin418/JAKA-VLA.git`。私有仓库需要GitHub登录，网页已登录不等于终端已登录；可用GitHub Desktop克隆。
3. 录像无需GPU。进入 `reviews/2026-11-06/` 下载查看ZIP、完整解压并打开 `index.html`；最新包为 `jaka-zu7-recovery-progress-20261106-1830.zip`。recovery4000 原场景1/3、新场景1/3、重设中途初态恢复0/3。此前stable1500 原场景2/3另保留在17:30包。旧Release中的0/3为历史结果。见 `RECOVERY_CURRICULUM_20261106.md`。
4. `Code → Download ZIP` 和 `git clone` 均不含Release大文件。完整恢复还要下载Release全部附件。

## 校验并恢复实验文件

将某一次备份的全部附件放在同一目录，不要混用不同批次的清单。Python3标准库即可校验/解包，不需要CUDA。

```bash
python3 JAKA-VLA/scripts/restore_backup.py --backup /path/to/backup
python3 JAKA-VLA/scripts/restore_backup.py --backup /path/to/backup --restore-to /path/to/jaka-vla-restored
```

恢复目录必须不存在。第一条完整读取分片并检查所有原文件；第二条执行同样检查并恢复文件，保留检查点last符号链接。优先在Linux/WSL执行，Windows原生可能需要开启符号链接权限。仅需某一个模型时也可按清单顺序拼接该模型全部part后解压，但完整性仍以配套校验为准。

`backup-source/project-history.bundle`包含备份时的代码历史；可用它离线克隆：

```bash
git clone /path/to/jaka-vla-restored/backup-source/project-history.bundle /path/to/jaka-vla-code
```

## 在新GPU服务器重建运行环境

1. 首选Ubuntu22.04、NVIDIA驱动和4090级GPU；运行脚本当前仍有 `/data/jaka-vla` 绝对路径。按原路径恢复最省修改，其他目录需同步调整env.sh、配置和脚本中的路径。
2. 分别重建 `envs/robotwin`（Python3.10、Torch2.4、SAPIEN/cuRobo）和 `envs/vla`（Python3.12、Torch2.8+cu128、LeRobot0.4.4）；参考 `manifests/*-packages.txt`。清单含原镜像的一些本地构建路径，不是可直接无条件执行的pip安装锁文件。
3. 按 `manifests/sources.json` 获取RoboTwin、XPolicyLab、cuRobo和LeRobot的固定提交，应用 `manifests/*.patch`；上游额外配置另存于source-history内。执行 `scripts/setup_zu7.py` 获取固定版本官方模型并生成机器人描述。先阅读skill.md，避免重复应用同一补丁。
4. 将恢复的data、checkpoints、results、logs放到项目根目录下。检查最新检查点的pretrained_model和相邻training_state，不要只复制model.safetensors而丢失归一化器/配置。
5. 先进行环境/离屏渲染检查，再读取原有结果；不要为了恢复而直接重跑已完成训练。真机尚未接入。

## 生成新的备份

确认没有正在写入数据/权重/日志的实验，准备 `backup-source/` 后运行：

```bash
python3 scripts/create_backup.py --output /data/jaka-vla/backups/backup-NEW
```

输出目录必须是新目录，且不能放在data/checkpoints/results/logs/backup-source内部。脚本只打包，不上传、不删除数据。读取失败或源文件发生变化会停止；必须看到BACKUP_COMPLETE且独立校验通过才算备份成功。

## 2026-11-05 同步记录

- 用户授权仅本项目同步至GaoPeixin418，按私有仓库准备。
- 已完成账户验证，私有仓库main分支已上传完整项目历史。
- [Release backup-2026-11-05](https://github.com/GaoPeixin418/JAKA-VLA/releases/tag/backup-2026-11-05)已发布：21个附件，共8,357,873,370字节，全部与服务器SHA256一致。
- 8组归档、10个分片，原始内容10,619,080,561字节；服务器完整流式校验665个文件/链接通过，恢复往返/损坏测试通过。
- 服务器重启曾中断传输，已重连并补传；数据盘完整备份未丢失。
- 独立本地全量副本后续已完成665项核验并通过恢复演练，完成标记在本地 `outputs/jaka-vla/github-backup/LOCAL_BACKUP_COMPLETE.json`；不把旧的下载中状态当作当前状态。
- 逐附件大小、SHA256和下载地址见manifests/github-backup-20261105.json。

## 2026-11-06新增实验的本地备份

以下路径均在当前工作电脑 `/home/lightwheel/Documents/Codex/2026-11-05/wo/outputs/jaka-vla/` 下，尚未发布新的GitHub Release。复制到自己的另一台电脑时应复制整个对应目录和清单；Git clone只包含代码和小录像包。

| 相对目录 | 内容与完成标记 |
| --- | --- |
| `github-backup/backup-20261106/` | 当日上午新增数据/9000步最终检查点/训练状态和结果；LOCAL_BACKUP_COMPLETE.json |
| `20261106/short-grasp/` | 短路径课程原始数据、专家验收、1500步完整最终检查点和处理器；LOCAL_BACKUP_COMPLETE.json |
| `20261106/normalization-fix/` | stable1500 和 coverage3000 最终完整模型/训练状态/数据；LOCAL_BACKUP_COMPLETE.json、COVERAGE_BACKUP_COMPLETE.json。coverage中间2000步仅留服务器，最终3000步已完整保留 |
| `20261106/recovery-1830/` | 新14条课程、恢复专家、4000步最终完整模型/优化器/数据、9学习回合及审计。专家134、模型/转换数据33、结果153文件SHA256核验通过；有LOCAL/EXPERT/RESULTS/REVIEW各完成标记 |

恢复批次的 `backup/` 已含可直接恢复的完整检查点（包括 `model.safetensors`、training_state、处理器和last链接），不需要另下载冻结主干。`model-training.tar`使用差分节省传输，若仅从归档重建，须配套已备份的 `smolvla_zu7_independent3000_seed42.tar.gz.part001`，提取003000模型后运行 `restore_checkpoint_delta.py`，并核对 `recovery-model-files-20261106.json` 全部33项。该旧模型base SHA256为 `4af14506cb023158935ebd2f17a539add85a62b594b8ed06813cf4432a023471`，新完整模型SHA256为 `e9e2b2dbb4d0ae5d1a6440b697b183825411eb65d8bf8aa33c7de0ee37461ade`。

本地新增实验备份与11-05 Release是不同快照，不能用旧Release清单验证新模型。新模型的源文件SHA256清单也保存在仓库 `manifests/recovery-model-files-20261106.json`、`recovery-results-files-20261106.json`。

## 自动Git同步

本项目仓库在服务器启用了`.git/hooks/post-commit`。每次在`/data/jaka-vla`完成Git提交，钩子都会把该提交安全推送到本私有仓库的`main`分支。需先运行项目规定的测试、更新实验/技术记录并检查差异，然后创建commit；Git不会在文件保存时自动决定和创建commit。仓库本地`core.sshCommand`供`fetch`/`pull`更新远端，不影响其他仓库。

认证使用GitHub Deploy Key `jaka-vla project-only autosync 2026-11-05`，在GitHub设置中仅挂载本仓库且具读写权限；私钥仅在该服务器项目Git目录下、权限0600，不进入Git、备份和Release，也未改变全局Git配置。钩子推送失败会保留本地commit，需查网络/Deploy Key/远程main分支后显式重试。

其他仓库没有此钩子。其他项目或对话必须在用户明确提出时才可使用Git同步。

## 退租当前服务器

在 `/data/jaka-vla/backups` 与本地副本都确认完整、Release可校验恢复后，再结束服务器租约。最后一次项目提交应确认已出现在GitHub `main`。随后在仓库 `Settings → Deploy keys` 删除 `jaka-vla project-only autosync 2026-11-05`，撤销租用服务器的仓库写权限；Deploy Key不包含在备份中。
