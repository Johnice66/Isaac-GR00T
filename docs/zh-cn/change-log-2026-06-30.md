# 项目修改日志（2026-06-30）

## 2026-06-30：整理 GitHub 发布范围

涉及文件：

- `.gitignore`
- `docs/zh-cn/README.md`
- `docs/zh-cn/change-log-2026-06-30.md`

主要改动：

- 增加 `datasets/*/data/` 和 `datasets/*/videos/` 忽略规则。
- 增加数据集备份文件 `*.bak`、`*.bak_*` 的忽略规则。
- 保留数据集目录中的转换脚本、`meta/config*.py`、`modality.json`、
  `episodes.jsonl`、统计量和其他轻量元数据。
- 为当前 AgiBot G01 项目修改建立独立 Git 分支，并准备发布到
  `Johnice66/Isaac-GR00T`。

修改原理：

- `datasets/task_5093` 本地目录约 4.3 GB，主体是 parquet 帧数据和 MP4 视频，
  不适合直接存入普通 Git 仓库，也会超过 GitHub 单文件及仓库传输限制。
- 配置、标注元数据和转换脚本体积较小，保留它们可以追踪 subtask 数据处理方式，
  并支持在持有原始数据的环境中复现配置。
- 数据集实体应通过专用对象存储、数据集平台或 Git LFS 管理，不与代码提交混合。

实现功能：

- 防止误提交数 GB 的训练数据和视频。
- 保留当前训练、subtask 转换、server 加速和真机 client 修改的代码与文档。
- 使项目分支可以通过标准 Git 推送到 GitHub。

验证记录：

- 发布前检查工作区中所有大于 95 MB 的文件。
- 使用 `git status --ignored` 和暂存区统计确认数据主体未进入提交。
- 推送完成后需要检查远程分支提交和文件清单。

已知限制和后续事项：

- GitHub 分支不包含原始 parquet 和相机视频，部署或重新训练时仍需单独准备数据集。
- checkpoint、TensorRT engine 和训练日志同样不应提交到普通 Git；当前工作区未包含这些产物。
