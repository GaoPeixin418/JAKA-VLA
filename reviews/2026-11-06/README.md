# 2026-11-06 录像与结果查看包

最新全景包：`reviews/2026-11-06/jaka-zu7-multiview-20261106.zip`（约9.24MB）。在GitHub文件页面点Download raw file下载，完整解压后打开`jaka-zu7-multiview/index.html`。默认半速，可逐帧和全屏。首段为6000步SmolVLA的seed2抓取成功、后续交接失败；6000步seeds0/1/2抓取1/3、完整任务0/3。9000步失败和专家参考均分别标注。下文旧包作为历史保留。

下载 [jaka-zu7-review-20261106.zip](jaka-zu7-review-20261106.zip)，在文件页面点击 Download raw file 下载。先完整解压，再用浏览器打开 jaka-zu7-review/index.html；也可以用播放器直接打开各子目录的MP4。

本包包含36段录像：历史对照、最新六条专家回放和最新三条学习策略测试。最新学习策略在 zu7_learned_independent9000_seed3_5/seed_3.mp4、seed_4.mp4、seed_5.mp4；这三条均空抓、成功率0/3。专家回放5/6完成任务，不计入模型成功率。

页面按仿真时间播放，未包含推理等待；没有真机录像。此包为生成时的快照，最新诊断文字见 docs/GRASP_DIAGNOSIS_20261106.md。

文件大小和SHA256记录在SHA256SUMS；这是小体积查看包，不含训练数据集或模型权重。完整实验备份按 docs/BACKUP_RESTORE.md 操作。

最新短路径抓取查看包：[jaka-zu7-short-grasp-20261106.zip](jaka-zu7-short-grasp-20261106.zip)，下载解压打开index.html。专家与学习结果分别标注，包含失败。学习1500步抓取结果0/3，不能把专家3/3当成学习成果。

归一化修正后学习抓取2/3查看包：[jaka-zu7-normalization-fix-20261106.zip](jaka-zu7-normalization-fix-20261106.zip)。下载解压打开index.html；包含新成功和失败，均为已见场景抓取子任务。
