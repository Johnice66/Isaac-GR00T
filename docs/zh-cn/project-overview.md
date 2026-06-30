# 项目总览

Isaac GR00T N1.7 是一个面向通用人形/半人形机器人技能的开源 VLA（Vision-Language-Action）模型仓库。它提供模型结构、数据处理、微调训练、推理服务、仿真评估和部署加速工具。

项目的核心问题是：给定语言指令、视觉观测和机器人状态，预测未来一段时间的连续动作序列，并支持多种机器人 embodiment。

## 核心能力

- 多模态输入：语言、图像、多路相机、机器人低维状态。
- 连续动作输出：以 action chunk 的形式输出多个未来控制步。
- 多 embodiment 支持：通过 embodiment tag、模态配置和 embodiment-specific 投影器区分机器人形态。
- 微调支持：从基础模型或已微调 checkpoint 出发，对新机器人/新任务训练。
- 推理服务：本地 `Gr00tPolicy` 或 ZeroMQ `PolicyServer`/`PolicyClient`。
- 评估工具：open-loop 数据集回放评估、仿真 closed-loop rollout。
- 部署加速：PyTorch eager、`torch.compile`、ONNX 导出、TensorRT pipeline。

## 顶层目录

```text
gr00t/              主 Python 包
  configs/          模型、数据、训练配置
  data/             LeRobot 数据读取、sharding、统计量、状态/动作处理
  model/            N1.7 模型、Qwen3 backbone、DiT 动作头
  experiment/       微调训练入口、Trainer、checkpoint 回调
  policy/           推理 Policy、服务端/客户端、ReplayPolicy
  eval/             open-loop 和仿真 closed-loop 评估
  deployment/       部署模式枚举等轻量运行时定义
examples/           DROID、LIBERO、SimplerEnv、SO100、RoboCasa 示例
getting_started/    用户指南和 notebook
scripts/            下载、修复、转换、部署导出和 TensorRT 工具
tests/              CPU/GPU、数据、模型、策略、部署回归测试
docker/             容器构建说明和 Dockerfile
external_dependencies/ 仿真环境和第三方项目
```

## 关键入口

常用入口如下：

```bash
# 安装依赖
uv sync --all-extras

# 微调
bash examples/finetune.sh \
  --base-model-path <checkpoint> \
  --dataset-path <lerobot_dataset> \
  --embodiment-tag <tag> \
  --output-dir <output>

# 直接启动推理服务
python gr00t/eval/run_gr00t_server.py \
  --model-path <checkpoint> \
  --embodiment-tag <tag>

# open-loop 评估
python gr00t/eval/open_loop_eval.py \
  --model-path <checkpoint> \
  --dataset-path <dataset> \
  --embodiment-tag <tag>

# TensorRT 一体化构建
python scripts/deployment/build_trt_pipeline.py \
  --model-path <checkpoint> \
  --dataset-path <dataset> \
  --embodiment-tag <tag>
```

## 主流程鸟瞰

训练路径：

```text
examples/finetune.sh
  -> gr00t/experiment/launch_finetune.py
  -> get_default_config() + FinetuneConfig 覆盖
  -> gr00t/experiment/experiment.py::run()
  -> MODEL_REGISTRY 创建 pipeline
  -> DatasetFactory 创建 ShardedMixtureDataset
  -> Processor 处理单步样本
  -> Gr00tTrainer 训练模型
  -> 保存 model + processor + experiment_cfg
```

推理路径：

```text
Gr00tPolicy
  -> AutoModel.from_pretrained(checkpoint)
  -> AutoProcessor.from_pretrained(checkpoint 或 checkpoint/processor)
  -> observation 校验
  -> processor 将 video/state/language 转成模型输入
  -> model.get_action()
  -> processor.decode_action() 反归一化并转回绝对动作
```

模型路径：

```text
语言 + 多图像
  -> Qwen3VLProcessor
  -> Qwen3Backbone 输出 backbone_features

状态 + noisy action + timestep + embodiment_id
  -> CategorySpecificMLP / MultiEmbodimentActionEncoder
  -> AlternateVLDiT / DiT
  -> action_decoder 输出 velocity
  -> flow matching denoising 得到 action chunk
```

## 代码中的几个关键概念

`EmbodimentTag` 不是普通字符串标签，而是贯穿数据列、模态配置、processor、模型投影器和 checkpoint 兼容性的契约。基础模型只支持预训练 tag；后训练/自定义 tag 通常需要匹配的微调 checkpoint。

`ModalityConfig` 决定某个 embodiment 使用哪些列、取哪些时间偏移、动作 horizon 是多少。例如 video/state/language 通常取当前帧，action 通常取未来多个 step。

`Processor` 是训练和推理之间的桥。训练时把 LeRobot step 转成模型输入；推理时把机器人实时 observation 转成同样格式；输出时再负责动作反归一化和 relative-to-absolute 转换。

`ShardedMixtureDataset` 是训练数据调度核心。它把多个 LeRobot 数据集按 mix ratio 混合，按 shard 迭代，并合并每个 embodiment 的统计量。

`Gr00tN1d7ActionHead` 是动作预测核心。它训练时学习从 noise 到 action 的 velocity；推理时从随机噪声出发，用少量 denoising step 积分出连续动作 chunk。

## 当前项目状态提示

这个仓库包含 public EA 形态的主代码，也包含本地工作区中的新增数据、脚本和实验文件。阅读项目时应把 `gr00t/`、`examples/`、`getting_started/`、`scripts/deployment/` 和 `tests/` 作为稳定主线；`datasets/`、临时测试脚本和 handoff 文档更像本地实验上下文。

