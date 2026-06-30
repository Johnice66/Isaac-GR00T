<div align="center">
  <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1.7 Header">
</div>

# Isaac GR00T N1.7 中文说明

本仓库当前已经切到中文文档入口，方便直接面向 AgiBot G01 真机微调与部署项目使用。

英文原版根文档已保存在 [README.en.md](README.en.md)。仓库内第一方英文文档均补充了对应的 `*.zh-CN.md` 中文版本；第三方依赖目录 `external_dependencies/`、许可证和归属清单这类上游内容保持原样，不在这次翻译范围内。

## 当前项目文档

你的项目迁移文档已经放到最前面：

- [AgiBot G01 项目迁移文档](PROJECT_HANDOFF_2026-06-08_CN.md)

建议先读这一份，再回到仓库通用说明。它记录了当前项目的训练路径、subtask 方案、server/client 适配、TensorRT 与 `torch.compile` 路线、磁盘空间问题，以及当前主用 checkpoint。

## 仓库概览

Isaac GR00T N1.7 是一个通用 humanoid 视觉-语言-动作模型。仓库覆盖四条主线：

1. 数据准备：把机器人示教数据整理成 GR00T 兼容的 LeRobot v2 格式。
2. 微调训练：基于 `launch_finetune.py` 在自定义 embodiment 或任务上训练。
3. 推理评估：支持 open-loop、仿真评估、真机 server/client 推理。
4. 部署加速：支持 PyTorch、`torch.compile`、ONNX、TensorRT。

模型结构是视觉语言骨干 + 动作扩散头：

<div align="center">
  <img src="media/model-architecture.png" width="800" alt="model-architecture">
</div>

## 快速导航

- 项目总文档入口：[docs/zh-cn/README.md](docs/zh-cn/README.md)
- 常见问题：[FAQ.zh-CN.md](FAQ.zh-CN.md)
- 贡献说明：[CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)
- 数据格式准备：[getting_started/data_preparation.zh-CN.md](getting_started/data_preparation.zh-CN.md)
- 模态配置说明：[getting_started/data_config.zh-CN.md](getting_started/data_config.zh-CN.md)
- 新 embodiment 微调：[getting_started/finetune_new_embodiment.zh-CN.md](getting_started/finetune_new_embodiment.zh-CN.md)
- Policy API 与 server/client：[getting_started/policy.zh-CN.md](getting_started/policy.zh-CN.md)
- 真机部署建议：[getting_started/real_world_deployment.zh-CN.md](getting_started/real_world_deployment.zh-CN.md)
- 部署与推理加速：[scripts/deployment/README.zh-CN.md](scripts/deployment/README.zh-CN.md)

## 目录结构

```text
gr00t/              核心包：模型、数据、训练、评估、推理服务
getting_started/    上手文档
examples/           数据集和机器人示例
scripts/            部署、转换、辅助脚本
docs/zh-cn/         当前项目的中文工程文档与修改日志
demo_data/          小型示例数据
datasets/           当前分支额外加入的数据集与转换脚本
```

## 当前分支的项目注意事项

这部分是当前 AgiBot G01 项目最重要的现实约束，优先级高于通用 README 习惯：

- 远端训练/推理服务器不要默认使用 `uv run`。已知会触发依赖重同步，之前 flash-attn 下载源不稳定。
- 远端每次新 shell 都要先激活 venv，并设置 venv 内 cuDNN 的 `LD_LIBRARY_PATH`。
- 当前固定点任务只控制双臂 14 关节和左右夹爪，不控制底盘、头部、腰部。
- 如果 checkpoint 训练时用了 `config_subtask.py`，client 语言 key 必须发 `sub_task`，而不是 `annotation.human.task_description`。

完整说明见 [CLAUDE.md](CLAUDE.md) 和 [PROJECT_HANDOFF_2026-06-08_CN.md](PROJECT_HANDOFF_2026-06-08_CN.md)。

## 安装

### 基础依赖

- Python：dGPU 默认 3.10；Thor / Spark 使用 3.12
- CUDA：dGPU 12.8；Orin 12.6；Thor / Spark 13.0
- 包管理：推荐 `uv`
- 视频后端：当前只支持 `torchcodec`，需要 FFmpeg

### 克隆

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

如果之前没有带 submodule：

```bash
git submodule update --init --recursive
```

### dGPU 默认安装

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
uv sync --python 3.10
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

更详细的平台矩阵和 Jetson / Spark / Docker 说明见 [scripts/deployment/README.zh-CN.md](scripts/deployment/README.zh-CN.md)。

## 典型工作流

### 1. 数据准备

把数据整理成 GR00T LeRobot 格式，并补齐 `meta/modality.json`：

- [数据准备指南](getting_started/data_preparation.zh-CN.md)
- [模态配置指南](getting_started/data_config.zh-CN.md)

### 2. 微调

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ./demo_data/cube_to_bowl_5 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/SO100/so100_config.py \
  --num-gpus 1 \
  --output-dir /tmp/so100
```

详细参数说明见 [getting_started/finetune_new_embodiment.zh-CN.md](getting_started/finetune_new_embodiment.zh-CN.md)。

### 3. 推理服务

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/checkpoint \
  --embodiment-tag NEW_EMBODIMENT
```

客户端和 observation / action 格式说明见 [getting_started/policy.zh-CN.md](getting_started/policy.zh-CN.md)。

### 4. 部署加速

构建 TensorRT 全流程：

```bash
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path /path/to/checkpoint \
  --dataset-path /path/to/dataset \
  --embodiment-tag NEW_EMBODIMENT
```

部署对比、导出模式和 benchmark 见 [scripts/deployment/README.zh-CN.md](scripts/deployment/README.zh-CN.md)。

## 示例入口

- DROID：[examples/DROID/README.zh-CN.md](examples/DROID/README.zh-CN.md)
- LIBERO：[examples/LIBERO/README.zh-CN.md](examples/LIBERO/README.zh-CN.md)
- SimplerEnv：[examples/SimplerEnv/README.zh-CN.md](examples/SimplerEnv/README.zh-CN.md)
- RoboCasa：[examples/robocasa/README.zh-CN.md](examples/robocasa/README.zh-CN.md)
- RoboCasa GR1 Tabletop：[examples/robocasa-gr1-tabletop-tasks/README.zh-CN.md](examples/robocasa-gr1-tabletop-tasks/README.zh-CN.md)
- SO100：[examples/SO100/README.zh-CN.md](examples/SO100/README.zh-CN.md)
- GR00T WholeBodyControl：[examples/GR00TWholeBodyControl/README.zh-CN.md](examples/GR00TWholeBodyControl/README.zh-CN.md)

## 中文文档维护说明

本仓库当前采用：

- `README.md`：中文主入口
- `README.en.md`：根文档英文备份
- `*.zh-CN.md`：第一方英文文档的中文版本
- `docs/zh-cn/`：项目级中文工程说明和修改日志

每次改代码后，都要同步更新中文修改日志。具体规则见 [CLAUDE.md](CLAUDE.md)。
