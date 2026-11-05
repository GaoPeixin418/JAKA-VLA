# 查看当前 Zu7 结果

当前报告：`/data/jaka-vla/docs/ZU7_CORRECTION_20261105.md`。

优先下载服务器的`/data/jaka-vla/results/jaka-zu7-review.zip`，解压后打开`jaka-zu7-review/index.html`。页面分别展示失败诊断、专家回放和学习策略，视频与结果JSON均可离线查看。

```bash
scp -P <SSH端口> <用户名>@<SSH主机>:/data/jaka-vla/results/jaka-zu7-review.zip ./jaka-zu7-review.zip
```

在自己的电脑运行下载命令；连接信息使用租赁平台给出的值，不将密码写到命令或项目文档。

在服务器直接查看文件：

| 内容 | 路径 |
| --- | --- |
| 当前成果页面 | `/data/jaka-vla/results/zu7_review/index.html` |
| 双臂避碰专家回放 | `/data/jaka-vla/results/zu7_expert_collision_replay/seed_0.mp4`（另有1、2） |
| 前3000步学习测试 | `/data/jaka-vla/results/zu7_learned_seen3000/` |
| 累计6000步学习测试 | `/data/jaka-vla/results/zu7_learned_seen6000/`（生成后可读） |
| 逐步轨迹/接触 | 每个实验目录的`seed_*_trace.npz`、`seed_*_contacts.json` |
| 控制分析图 | 各已分析目录中的`seed_0_control.png` |
| 当前训练日志 | `/data/jaka-vla/logs/train-zu7-refine3000.log` |

每个目录的results.json是成功/失败依据；专家回放不能代表模型成功。录像按25FPS仿真时间播放，不包含模型推理等待。

浏览服务器当前页面可启动：

```bash
python3 -m http.server 18080 --bind 127.0.0.1 --directory /data/jaka-vla/results/zu7_review
```

本机建立SSH隧道后打开`http://127.0.0.1:18080/`，具体隧道命令见下方历史查看说明。无需开放公网HTTP端口。
重建当前页面用`python3 /data/jaka-vla/scripts/build_zu7_review.py`；只读取结果，不启动训练。

---

以下是旧ALOHA pilot的历史查看说明，与当前Zu7播放包分开。

# 查看实验结果与录像

旧ALOHA实验：50条示教、2000步SmolVLA训练；K=4/8/16各3回合全部完成，模型成功0/9。专家回放成功1/1，单独作为动作参考。此结论仅限当前pilot。

## 直接下载查看（推荐）

服务器下载包：`/data/jaka-vla/results/jaka-vla-review.zip`，约3.6MB。
用租赁平台文件管理器、SFTP或SCP下载，解压后用浏览器打开 `jaka-vla-review/index.html`。
保持 `index.html`、`videos/`、`evidence/` 的相对位置，即可离线播放10段录像和读取JSON证据，无需Python或模型环境。

在自己的电脑终端运行以下命令；将占位符替换为租赁平台提供的SSH连接信息：

```bash
scp -P <SSH端口> <用户名>@<SSH主机>:/data/jaka-vla/results/jaka-vla-review.zip ./jaka-vla-review.zip
```

SCP的端口参数是大写 `-P`；不要在远程终端运行这条下载到本机的命令。

## 通过SSH隧道在浏览器查看

先在服务器终端启动只监听本机的静态页面服务，保持该终端运行：

```bash
python3 -m http.server 18080 --bind 127.0.0.1 --directory /data/jaka-vla/results/review_bundle
```

然后在自己的电脑上打开另一个终端，建立SSH转发，保持运行：

```bash
ssh -p <SSH端口> -o ExitOnForwardFailure=yes -N -L 127.0.0.1:18080:127.0.0.1:18080 <用户名>@<SSH主机>
```

用本机浏览器打开 `http://127.0.0.1:18080/`。`-N`连接后无提示输出通常是正常状态。
看完后在两个终端分别按Ctrl+C停止转发和页面服务。不需要开放服务器公网HTTP端口。
这是查看服务；不启动训练，也不占用GPU。项目未自动开启常驻网页服务。

## 原始产物位置

| 内容 | 服务器路径 |
| --- | --- |
| 查看页面 | `/data/jaka-vla/results/review_bundle/index.html` |
| 成功专家回放 | `/data/jaka-vla/results/expert_replay/seed_0.mp4` |
| K=4模型测试 | `/data/jaka-vla/results/pilot_k4/seed_1000.mp4`，另有1001、1002 |
| K=8模型测试 | `/data/jaka-vla/results/pilot_k8/seed_1000.mp4`，另有1001、1002 |
| K=16模型测试 | `/data/jaka-vla/results/pilot_k16/seed_1000.mp4`，另有1001、1002 |
| 评测汇总 | `/data/jaka-vla/results/pilot_baselines.json` |
| 逐回合指标 | 各 `pilot_k*` 目录中的 `results.json` |
| 训练日志 | `/data/jaka-vla/logs/train-pilot.log` |
| 训练资源报告 | `/data/jaka-vla/checkpoints/smolvla_pilot_seed42/resource_profile.json` |
| 最终模型/处理器 | `/data/jaka-vla/checkpoints/smolvla_pilot_seed42/checkpoints/last/pretrained_model` |

## 录像怎么比较

先看专家回放了解完整抓取、交接、放置流程，再对照同一个测试seed的K4/K8/K16模型录像。专家seed0和模型测试seed1000–1002不是同一初始场景，不应按帧直接比较。
模型录像为25FPS、约30秒仿真时间；同步评测中推理暂停仿真，录像不包含推理等待，不能作为实时延迟表现。
最终success以任务判据的JSON结果为准；只看动作接近目标不等于任务成功。

## 查看进度或重建页面

```bash
python3 /data/jaka-vla/scripts/status.py --write-doc
python3 -m json.tool /data/jaka-vla/results/pilot_baselines.json
tail -n 30 /data/jaka-vla/logs/train-pilot.log
```

本轮查看包由 `python3 /data/jaka-vla/scripts/build_review_bundle.py` 生成，读取已存在的pilot数据并更新专用页面与ZIP，不运行模型。
页面和ZIP是生成时的快照；后续实验结果变化后需明确重建。下载包包含录像和结果，不包含权重、示教训练集或连接凭据。
