# 在自己的电脑查看 JAKA-VLA

你已完成GitHub授权。后续查看不需要再输入这次的一次性代码，也不需要把密码或令牌发给任何人。

## 只看代码和技术文档

1. 在自己的电脑浏览器登录 `GaoPeixin418`。
2. 打开 https://github.com/GaoPeixin418/JAKA-VLA 。这是私有仓库，未登录或账号不对时可能显示404。
3. 首页是README。点击 `docs` → `FINETUNING_GUIDE.md` 查看微调方法、数据和训练文件；`ZU7_CORRECTION_20261105.md` 是修正报告；根目录 `MEMORY.md` 是进展与限制。
4. 如需保存到电脑，点绿色 `Code` → `Download ZIP`，下载后解压。用VS Code打开解压文件夹即可。只看代码不需要GPU、CUDA、Python或Git。

## 看最新录像（2026-11-06）

1. 登录 GaoPeixin418，打开 https://github.com/GaoPeixin418/JAKA-VLA/tree/main/reviews/2026-11-06 。
2. 点击 jaka-zu7-review-20261106.zip，在文件页点击 Download raw file（下载原始文件）。
3. 完整解压，打开 jaka-zu7-review/index.html，或者直接用播放器打开子目录里的MP4。
4. 最新学习策略目录是 zu7_learned_independent9000_seed3_5，包含seed3/4/5；均空抓、0/3。新增专家回放5/6完成任务，不能把专家成功当作模型成功。
5. 本小体积录像查看包已经随代码提交同步到仓库；Code → Download ZIP也会包含它。数据集和权重大文件仍单独备份，不包含在该查看包中。

## 看2026-11-05历史录像

1. 打开 https://github.com/GaoPeixin418/JAKA-VLA/releases/tag/backup-2026-11-05 。全部21个附件已上传并校验通过。
2. 在 `Assets` 展开附件列表，下载 `jaka-zu7-review.zip`。
3. 先解压整个ZIP，再用浏览器打开其中的 `index.html`；也可直接用播放器打开MP4。
4. 页面区分专家回放和学习模型。最新专家回放3/3成功，学习模型仍为0/3，不能把专家视频当作模型成果。

## 想保留完整历史、以后更新代码

可以安装GitHub Desktop，用浏览器登录同一账号；选择 `File → Clone repository`，在URL栏填写 `https://github.com/GaoPeixin418/JAKA-VLA.git`，选择本地目录后点Clone。只克隆这个仓库即可，不必将其他文件夹加入Git。

以后点 `Fetch origin` / `Pull origin` 获取本项目的新提交。本项目提交后由服务器上的仓库钩子推送到GitHub `main`；你本地的其他文件不会上传。

项目工作产生的新Git提交会由JAKA-VLA仓库自己的钩子推送到`main`；每次改动仍会先经过检查和项目记录。其他项目不会因此自动同步。

## 退租前保留完整训练成果

代码ZIP或Clone不含大文件。需要在Release中下载同一批次的全部附件，包括 `backup-index.json`、`SHA256SUMS`、所有 `*.manifest.json` 和 `*.tar.gz.partNNN`。将它们放在同一个文件夹，不要改分片名。

完整备份约8.35GB（约7.78GiB），恢复约10.62GB；另预留代码和运行环境空间。在Linux/WSL使用仓库中的 `scripts/restore_backup.py` 验证所有文件，并可恢复到新的空目录。详见仓库 `docs/BACKUP_RESTORE.md`。

备份包含项目自己的数据、模型/优化器/归一化器、录像、轨迹、日志和代码历史；不包含Python环境、下载缓存、第三方下载资产或凭据。重建仿真环境需要按已记录的版本重新安装依赖。

## 同步范围

仅JAKA-VLA。没有为其他项目设置Git同步，没有修改全局Git配置，也没有设置定时自动上传。以后要同步其他内容，仍需你明确提出。
