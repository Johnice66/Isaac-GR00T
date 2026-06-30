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

## 2026-06-30：同步最新 main 分支

涉及范围：

- 上游区间：`626af89..ab88b50`，共 7 个提交。
- 合并分支：`origin/main` 和 `johnice/main`，两者同步时指向同一提交 `ab88b50`。
- 自动合并文件：`.gitignore`、`gr00t/eval/run_gr00t_server.py`。
- 后续同步文件：`gr00t/eval/run_gr00t_server_new.py`。
- 上游实际改动覆盖训练、数据加载、模型处理、推理服务、评估、TensorRT、平台依赖和测试。

主要改动：

- 部署模式集中到 `gr00t/deployment/modes.py`，强化 ONNX/TensorRT 模式、BF16、静态 batch、
  engine 输入和精度校验，并补充 `dit_only` 验证。
- `PolicyServer` 和 `PolicyClient` 增加显式资源释放及上下文管理能力；当前自定义 server 已融合新的
  `with PolicyServer(...)` 生命周期管理，同时保留 `pytorch`、`torch_compile` 和 `tensorrt` 参数。
- 同步更新备用的 `run_gr00t_server_new.py`，避免它继续使用上游已经替换的旧 server 生命周期。
- 视频解码统一为 torchcodec，移除旧的 PyAV/ffmpeg/decord/OpenCV 后端选择逻辑；rollout 视频改为
  流式 ffmpeg 编码。
- 加强训练恢复、分布式初始化、最佳 checkpoint 保存、数据统计缓存指纹和 shard 边界处理。
- 加强 N1.7 图像输入校验、Qwen3-VL RoPE 重建和 flow matching 数值稳定性。
- 增加 RoboCasa365、SONIC G1 全身控制示例、TensorRT 契约测试以及多项训练和推理回归测试。

修改原理：

- 当前项目分支保留 AgiBot G01 专用训练、subtask 和真机部署修改，通过普通 merge 吸收上游更新，
  不改写已经推送的项目提交历史。
- 合并前使用 `git merge-tree --write-tree --messages` 预演三方合并；返回成功且仅报告两个文件自动合并，
  因此不需要人工冲突取舍。

实现功能：

- 当前分支获得最新的训练稳定性、推理资源管理、TensorRT 构建校验和测试覆盖。
- 保持现有 AgiBot server 的 `--inference-mode torch_compile` 与 TensorRT full pipeline 启动方式可用。

验证记录：

- 确认 `origin/main` 与 `johnice/main` 都指向 `ab88b50`。
- 合并预演和实际合并均无冲突。
- 合并后检查 `run_gr00t_server.py`，自定义加速参数和上游 `PolicyServer` 上下文管理同时存在。
- 对关键 server、client、数据转换和部署脚本执行 Python 语法检查。

已知限制和后续事项：

- 本地 macOS 环境不具备项目完整 CUDA/TensorRT 运行条件，不能替代服务器上的 GPU 集成验证。
- 上游已统一使用 torchcodec；服务器更新依赖后应重点验证本地 HEVC 数据集读取和真机相机处理流程。
- TensorRT 导出契约有所加强，已有 engine 若与新代码版本不匹配，应重新执行导出、构建和校验流程。
