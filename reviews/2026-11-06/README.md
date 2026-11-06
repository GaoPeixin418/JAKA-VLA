# 2026-11-06 录像与结果查看包

最新全景包：[jaka-zu7-recovery-progress-20261106-1830.zip](jaka-zu7-recovery-progress-20261106-1830.zip)。点 Download raw file，完整解压后打开 `jaka-zu7-recovery-progress-20261106-1830/index.html`。10段全景视频（9段学习、1段专家参考）全部完整解码，59个包内文件SHA256、ZIP和页面相对链接通过。默认半速，可逐帧；seed0另附速度/跟踪/夹爪曲线，各组有运动指标。

recovery4000：原场景1/3、新 seeds200–202 为1/3、重设中途初态的恢复子任务0/3，未达到预期改善。学习成功为 seed1 和 seed200，仅为红色长方块抓取子任务；专家成功另标，不能混计。包SHA256：`0f376273b24ff9ac36286e49dcc1fc5513655cfccc924af41a50a9bc25f17527`。详见 [恢复课程记录](../../docs/RECOVERY_CURRICULUM_20261106.md)。

上一批：[jaka-zu7-grasp-progress-20261106-1730.zip](jaka-zu7-grasp-progress-20261106-1730.zip)。9段学习录像：stable1500 原场景2/3；coverage3000 原场景1/3、新 seeds100–102 为2/3，分别保留。

以下均为历史包：全景包 `jaka-zu7-multiview-20261106.zip` 展示早期6000步模型抓取1/3但交接失败；早期9000步和短路径1500步的失败仍保留。当前结果请按检查点和批次查看。

下载 [jaka-zu7-review-20261106.zip](jaka-zu7-review-20261106.zip)，在文件页面点击 Download raw file 下载。先完整解压，再用浏览器打开 jaka-zu7-review/index.html；也可以用播放器直接打开各子目录的MP4。

该历史包包含36段录像：历史对照、六条专家回放和三条学习策略测试。历史学习策略在 zu7_learned_independent9000_seed3_5/seed_3.mp4、seed_4.mp4、seed_5.mp4；这三条均空抓、成功率0/3。专家回放5/6完成任务，不计入模型成功率。

页面按仿真时间播放，未包含推理等待；没有真机录像。此包为生成时的快照，最新诊断文字见 docs/GRASP_DIAGNOSIS_20261106.md。

文件大小和SHA256记录在SHA256SUMS；这是小体积查看包，不含训练数据集或模型权重。完整实验备份按 docs/BACKUP_RESTORE.md 操作。

早期短路径抓取查看包：[jaka-zu7-short-grasp-20261106.zip](jaka-zu7-short-grasp-20261106.zip)，下载解压打开index.html。专家与学习结果分别标注，包含失败。该版修正归一化前的1500步学习抓取结果0/3，不能把专家3/3当成学习成果。
