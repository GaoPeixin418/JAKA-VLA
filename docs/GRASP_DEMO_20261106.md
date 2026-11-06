# 2026-11-06 下午：学习策略抓取 demo 与全景录像

本轮重新验证了已有 SmolVLA 的一次成功左抓取，并改善录像可读性。没有新增训练，也没有解决整条双臂交接任务。6000步模型的训练/开发seeds0/1/2：抓取1/3、完整任务0/3。

## 先看哪段

下载 `reviews/2026-11-06/jaka-zu7-multiview-20261106.zip`，完整解压，打开 `jaka-zu7-multiview/index.html`。第一段是学习策略seed2，不是专家回放；后续有两条同模型失败、一条9000步模型失败及一条单独标注的专家参考。

服务器页面：`/data/jaka-vla/results/zu7_multiview_review_20261106/index.html`。

第一段原视频：`/data/jaka-vla/results/zu7_grasp_review6000_seed2/seed_2_multiview.mp4`。

全景大图为前方斜视，机器人命名的左臂出现在画面右侧；右侧还有俯视、原头相机、左腕相机。页面默认半速播放，支持逐帧、全屏、跳转抓取确认/终止前一秒。

## 实测结果

| 权重 / 模式 | Seed | 左抓取 | 完整任务 | 抬升峰值 | 连续有效夹持 | 有效仿真动作 / 视频时长 |
| --- | --- | --- | --- | --- | --- | --- |
| SmolVLA 6000步 | 2 | 通过 | 失败：跟踪异常保护 | 13.76cm | 3.36s | 8.868s / 13.92s |
| SmolVLA 6000步 | 0 | 失败 | 失败：空抓 | 0cm | 0s | 4.92s / 9.96s |
| SmolVLA 6000步 | 1 | 失败 | 失败：空抓 | 0.39cm | 0s | 6.20s / 11.24s |
| SmolVLA 9000步 | 6 | 失败 | 失败：空抓 | 0.09cm | 0s | 6.80s / 11.84s |
| 专家参考回放 | 6 | 通过 | 通过 | 15.99cm | 5.188s | 17.56s / 22.60s |

6000步seed2在6.108s达到抓取判据，之后仍继续执行完整交接任务；8.74s起右指接触左工具，8.868s持续跟踪误差触发终止。新视频保留该失败结尾，未剪成完整任务成功。

抓取判据：物体相对初始中心高度>5cm，实测左夹爪开度15%–85%，两左指同时与方块有>1e-5N·s正接触冲量，右臂不接触方块，连续保持0.6s。物体/接触真值只用于评价和录像标签，不进入策略或动作控制。该判据仅适配当前仿真物体/夹爪；真机需要可部署的抓取确认方式。

## 为什么旧视频短，新视频改变了什么

- 旧评测空抓/跟踪异常即终止，9000步seed6/7/8实际分别只有6.80/6.84/4.48秒；视频只有320×240头部近景。不是播放器截断。
- 新增1280×800多视角录像，使用独立全局/俯视相机；原三路策略输入与物理配置保持一致。
- 从初始状态一直录到真实终止，不省略后续失败；补拍终态，避免旧脚本最后一次动作未入镜。
- 片头2秒与片尾3秒为明确标注的冻结帧，仿真时钟不推进，不计作有效夹持。原视频25FPS，查看页默认半速，因此第一段完整观看约27.84秒。没有取消保护或继续执行空抓动作来增加时长。
- 同步推理期间仿真暂停，模型墙钟等待不计入视频；不能用视频速度宣称实时部署。

## 复现与来源

权重：`checkpoints/smolvla_zu7_refine3000_seed42/checkpoints/last/pretrained_model`。

权重SHA256：`85554712095c403bbad1d9b88cc878371719deff2e87712bdae012b0f0ae3c52`。

```bash
cd /data/jaka-vla
source env.sh
envs/robotwin/bin/python scripts/run_zu7_eval.py \
  --name zu7_grasp_review6000_seed2_new \
  --checkpoint /data/jaka-vla/checkpoints/smolvla_zu7_refine3000_seed42/checkpoints/last/pretrained_model \
  --seed-start 2 --episodes 1 --k 8 --max-steps 650 --review-video
```

代码拒绝覆盖已有目录。每回合有原近景、多视角、results.json、run_config.json（学习策略）、物理/控制步轨迹和接触记录；新增grasp_evidence数组，原数组格式保持不变。

四条学习策略复核与原相同种子/权重的 `physics_trace`、`control_trace`、`raw_chunks`、`blended_chunks` 数值完全一致（最大差0）。验收文件 `manifests/multiview-validation-20261106.json`。视频分辨率、帧数、FPS、时长和ZIP完整性由构建脚本校验。

这是已有模型在已知场景上的可复现抓取，不是新训练带来的成功率提升；9000步模型仍失败。下一步应定位交接时右指与左工具的碰撞，并解决初始位置变化下的抓取对准。
