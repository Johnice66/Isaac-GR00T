# 近期项目修改日志（截至 2026-06-11）

本文记录本仓库近期围绕 AgiBot G01 + GR00T N1.7 训练、subtask 条件化、推理服务加速和真机 client 的新增修改。后续所有项目代码修改都应继续更新本文档，或新增同类修改日志并在 `docs/zh-cn/README.md` 中链接。

## 总览

近期修改的核心目标是把项目从“单次训练 + 固定任务文本 + 手工推理脚本”推进到以下状态：

- 支持使用 `config_subtask.py` 做 `sub_task` 条件化训练。
- 支持把 Genie/Genie Studio 的阶段标注转换成 GR00T loader 能读取的 `episodes.jsonl[*].sub_tasks`。
- 让 `run_gr00t_server.py` 可以显式选择 PyTorch eager、`torch.compile` 或 TensorRT 推理路线。
- 把真机 client 从硬编码实验脚本整理为可配置脚本，并能适配新版 server 和 `sub_task` 语言键。
- 把当前项目状态整理成中文迁移文档，便于切换聊天窗口或服务器环境。

## 修改记录

### 2026-06-03：推理 server 增加加速路线参数

涉及文件：

- `gr00t/eval/run_gr00t_server.py`
- `gr00t/eval/run_gr00t_server_new.py`（当前与 `run_gr00t_server.py` 内容一致，是同版本副本）

主要改动：

- 新增 `--inference-mode`，支持：
  - `pytorch`：默认 PyTorch eager 推理。
  - `torch_compile`：使用 `torch.compile` 编译 action head。
  - `tensorrt`：加载 TensorRT engine。
- 新增别名归一化：
  - `eager -> pytorch`
  - `compile`、`torch.compile -> torch_compile`
  - `trt -> tensorrt`
- 新增 `--compile-mode`，默认 `max-autotune`。
- 新增 `--trt-engine-path` 和 `--trt-mode`。
- TensorRT 支持模式：
  - `n17_full_pipeline`
  - `vit_llm_only`
  - `action_head`
  - `dit_only`
- 对 ReplayPolicy 增加保护：非 `pytorch` 推理模式只适用于 `Gr00tPolicy`，不适用于 dataset replay。

修改原理：

- GR00T 的真实推理加速发生在 server 侧，client 只负责发送 observation 和接收 action。
- `torch_compile` 路线通过替换 `policy.model.action_head.model.forward` 为 `torch.compile(...)` 后的 callable，减少 action head 推理开销。
- TensorRT 路线通过 `scripts/deployment/trt_model_forward.py` 中的 `setup_tensorrt_engines()` 把已构建好的 engine 装配到 policy 内部模块上。
- TensorRT artifact 不写入 checkpoint 目录，而是由 `build_trt_pipeline.py` 生成到部署目录，例如 `gr00t_trt_deployment/engines`。

实现功能：

- 可直接启动推荐的 `torch.compile` server：

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
  --model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000 \
  --embodiment-tag new_embodiment \
  --host 0.0.0.0 \
  --port 5555 \
  --inference-mode torch_compile
```

- 可选启动 TensorRT full pipeline server：

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
  --model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000 \
  --embodiment-tag new_embodiment \
  --host 0.0.0.0 \
  --port 5555 \
  --inference-mode tensorrt \
  --trt-engine-path /root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/engines \
  --trt-mode n17_full_pipeline
```

当前性能依据：

- A100 + 4 denoising steps benchmark 中，`torch.compile` E2E 约 `175 ms / 5.7 Hz`。
- TensorRT full pipeline E2E 约 `210 ms / 4.8 Hz`，verify cosine `1.000000 PASS`，但在当前数据处理瓶颈下总延迟不如 `torch.compile`。
- 因此当前默认部署建议是 `torch.compile`，TensorRT 保留为可选路线。

### 2026-06-04：新增真机 client v7

涉及文件：

- `test_0526_n17_safe_v7.py`

主要改动：

- 新增面向 AgiBot G01/G1 的 GR00T N1.7 真机 client。
- 使用仓库自带 `gr00t.policy.server_client.PolicyClient`，替换手写 ZMQ/msgpack 逻辑。
- 增加 CLI 参数，避免把 host、port、任务文本、控制频率、安全阈值、ROS topic 等关键参数硬编码在脚本中。
- 增加 server `ping()`、`get_modality_config()` 检查和首次 `get_action()` 测试。
- 支持 action 名称归一化：
  - split keys：`left_arm_joint_position` + `right_arm_joint_position`
  - combined key：`joint_position`
  - optional `action.` 前缀
- 保留并整理真机安全逻辑：
  - 14 维关节输出校验。
  - NaN/Inf 检查。
  - 单步关节变化限制。
  - 速度、加速度限制。
  - blend、final hold、EMA 等平滑参数。
- ROS topic 参数化：
  - 相机：`/camera/head_color`、`/camera/hand_left_color`、`/camera/hand_right_color`
  - 手臂状态：`/hal/arm_joint_state`
  - 手臂控制：`/wbc/arm_command`
  - 夹爪状态：`/hal/left_ee_data`、`/hal/right_ee_data`
  - 夹爪控制：`/wbc/left_ee_command`、`/wbc/right_ee_command`

修改原理：

- GR00T PolicyServer 已经定义了正式 client/server 协议，真机 client 应复用 `PolicyClient`，避免协议漂移。
- 真机 client 不需要知道 server 使用的是 PyTorch eager、`torch.compile` 还是 TensorRT；只需要连接正确 host/port，并发送与训练 config 匹配的 observation。
- 控制侧必须把模型输出的 action chunk 转成符合真机限制的逐帧命令，避免直接把稀疏目标点下发到真机。

实现功能：

- 先无控制验证 server、ROS 观测和 action shape：

```bash
python test_0526_n17_safe_v7.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "Fixed-point Non-generalized Door Opening" \
  --no-enable-control
```

- 再开启真机控制：

```bash
python test_0526_n17_safe_v7.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "Fixed-point Non-generalized Door Opening" \
  --enable-control \
  --control-freq 60 \
  --execute-horizon 8
```

### 2026-06-08：新增通用 subtask 数据准备脚本

涉及文件：

- `scripts/prepare_subtask_training_from_info.py`

主要改动：

- 新增脚本，用于把 `meta/info.json` 中的阶段/子任务标注转换成 GR00T loader 已支持的格式：

```json
{
  "episode_index": 0,
  "sub_tasks": [
    {"start": 0, "end": 80, "text": "approach the handle"}
  ]
}
```

- 支持多个 dataset path 一次处理。
- 支持从多种常见 schema 中保守识别阶段标注字段：
  - 容器字段：`sub_tasks`、`subtasks`、`stages`、`phases`、`segments` 等。
  - 起止字段：`start`、`start_frame`、`end`、`end_frame` 等。
  - 文本字段：`text`、`task`、`label`、`description`、`subtask`、`sub_task` 等。
- 写出同级训练配置 `meta/config_subtask.py`。
- 在写出 `config_subtask.py` 时，把 language modality 从 `annotation.human.task_description` 替换成 `sub_task`。
- 写入前备份原 `episodes.jsonl`。
- 提供 `--dry-run` 和 `--force`。

修改原理：

- GR00T 的 `LeRobotEpisodeLoader` 原生支持两个 language key：`task` 和 `sub_task`。
- 当训练 config 的 language key 是 `sub_task` 时，loader 会从 `episode_meta["sub_tasks"]` 中按 frame 区间生成每一帧的 language 文本。
- 因此训练 subtask-conditioned model 的关键不是改 parquet，而是：
  - `episodes.jsonl` 中存在 `sub_tasks`。
  - 训练 config 的 language modality key 是 `sub_task`。

实现功能：

```bash
python scripts/prepare_subtask_training_from_info.py \
  /root/gpufree-data/Isaac-GR00T/dataset/task_2609 \
  /root/gpufree-data/Isaac-GR00T/dataset/task_2178
```

训练时使用：

```bash
--modality-config-path /root/gpufree-data/Isaac-GR00T/dataset/task_2609/meta/config_subtask.py
```

注意事项：

- 只运行转换脚本但仍使用原始 `config.py` 训练，不会真的启用 `sub_task` 条件化。
- 必须确认训练 checkpoint 的 server modality config 中 language key 是 `sub_task`，client 才能发送 `sub_task`。

### 2026-06-08 至 2026-06-10：task_5093 子任务标注转换

涉及文件：

- `datasets/task_5093/convert_instruction_segments_to_subtasks.py`
- `datasets/task_5093/meta/episodes.jsonl`
- `datasets/task_5093/meta/episodes.jsonl.bak_20260610_210230`

主要改动：

- 新增 task_5093 专用转换脚本，把 `meta/info.json["instruction_segments"]` 写入 `meta/episodes.jsonl[*]["sub_tasks"]`。
- 对原始 `episodes.jsonl` 创建时间戳备份。
- 当前 `episodes.jsonl` 已包含 863 个 episode 的 `sub_tasks`。

当前检查结果：

- `episodes: 863`
- `missing_sub_tasks: 0`
- 每个 episode 都有 2 个 subtask。
- 唯一子任务文本：
  - `Press Lock Body Switch`
  - `Grasp Lock Body Handle and Pull`
- 覆盖检查通过：每条 episode 的 subtask 区间从 `0` 覆盖到 `length`，没有 gap/overlap。
- 有一个标注顺序异常需要人工复核：
  - `episode_index=26` 的顺序是 `Grasp Lock Body Handle and Pull -> Press Lock Body Switch`
  - 其他 862 条都是 `Press Lock Body Switch -> Grasp Lock Body Handle and Pull`

修改原理：

- 数据文件中字段叫复数 `sub_tasks`，代表一个 episode 内多个阶段。
- 训练/推理 config 中 language key 叫单数 `sub_task`，代表当前帧输入给模型的一条语言条件。
- loader 会根据 action horizon 把每个 `sub_tasks` 区间附近的帧映射到对应的 `sub_task` 文本。

### 2026-06-10：真机 client v8 增加 subtask language key 适配

涉及文件：

- `test_0526_n17_safe_v8.py`

主要改动：

- 基于 v7 增加 language key 参数化。
- 新增 `--language-key`：
  - 默认 `auto`
  - 可显式指定 `sub_task`
  - 可显式指定 `annotation.human.task_description`
- 新增 `--subtask`，作为 `--task` 的别名，用于 subtask-conditioned checkpoint。
- 启动时通过 `get_modality_config()` 读取 server 的 language modality keys。
- `--language-key auto` 时优先选择：
  1. `sub_task`
  2. `annotation.human.task_description`
  3. server 返回的第一个 language key
  4. fallback 到 `annotation.human.task_description`
- 构造 observation 时不再硬编码：

```python
{"annotation.human.task_description": [[task]]}
```

而是使用：

```python
{config.language_key: [[task]]}
```

- 在配置打印和 observation summary 中显示实际发送的 language key 和文本。

修改原理：

- `Gr00tPolicy` 推理时会校验 observation language key 是否与 checkpoint 的 modality config 匹配。
- 如果模型训练使用 `config_subtask.py`，server 期望的 language key 是 `sub_task`。
- 如果 client 仍发送 `annotation.human.task_description`，即使文本内容正确，也会因为 key 不匹配导致推理失败或输入语义不一致。

实现功能：

- subtask 模型第一阶段无控制测试：

```bash
python test_0526_n17_safe_v8.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --language-key sub_task \
  --subtask "Press Lock Body Switch" \
  --no-enable-control
```

- subtask 模型第二阶段：

```bash
python test_0526_n17_safe_v8.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --language-key sub_task \
  --subtask "Grasp Lock Body Handle and Pull" \
  --enable-control
```

当前限制：

- v8 只支持发送某一个当前 subtask 文本。
- v8 还不会自动判断机器人当前处于哪个阶段。
- 若要完整自动执行，需要继续增加子任务切换机制，例如：
  - 手动按键切换。
  - 按 cycle 数或时间切换。
  - 从 ROS topic 接收阶段。
  - 根据视觉/状态阈值判断阶段。

验证记录：

- `PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m py_compile test_0526_n17_safe_v8.py` 已通过语法检查。
- 本地直接跑 `--help` 时因本机缺少 `cv2` 失败，这是本地依赖问题，不是脚本语法问题。

### 2026-06-08：新增聊天迁移文档

涉及文件：

- `PROJECT_HANDOFF_2026-06-08.md`
- `PROJECT_HANDOFF_2026-06-08_CN.md`

主要改动：

- 记录当前项目训练、部署、server/client、subtask、TensorRT/torch.compile、磁盘空间和 checkpoint 状态。
- 中文版本用于新聊天窗口迁移，避免丢失重要上下文。

关键结论：

- 两数据集联合训练的可用 checkpoint 是：

```text
/root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000
```

- checkpoint 60000 的 3 个 safetensors shard 已验证可读。
- 训练末尾报错是磁盘满导致最终 parent output dir 保存失败，不是训练本身未完成。

### 2026-06-11：新增中文项目文档目录

涉及文件：

- `docs/README.md`
- `docs/zh-cn/README.md`
- `docs/zh-cn/project-overview.md`
- `docs/zh-cn/modules.md`
- `docs/zh-cn/principles-data-and-modality.md`
- `docs/zh-cn/principles-model.md`
- `docs/zh-cn/principles-training-inference.md`
- `docs/zh-cn/principles-deployment.md`

主要改动：

- 新增中文项目文档，解释 GR00T N1.7 的工程结构、训练/推理主流程、数据与模态机制、模型结构和部署加速路径。
- 文档定位是补充官方 README 和 getting_started，重点服务当前 AgiBot G01 真机微调与部署项目。

### 2026-06-11：新增 Agent 修改记录规则

涉及文件：

- `CLAUDE.md`
- `AGENTS.md`（符号链接，指向 `CLAUDE.md`）
- `docs/zh-cn/README.md`
- `docs/zh-cn/change-log-2026-06-11.md`

主要改动：

- 在项目级 agent 指令中新增“每次修改项目代码都必须同步更新修改记录”的强制规则。
- 明确中文修改日志放在 `docs/zh-cn/`。
- 明确新增修改日志时必须在 `docs/zh-cn/README.md` 中添加链接。
- 明确修改记录至少需要包含日期、涉及文件、修改内容、修改原理、实现功能、验证方式和已知限制。

修改原理：

- `AGENTS.md` 是指向 `CLAUDE.md` 的符号链接，因此 agent 规则应写入 `CLAUDE.md`，由 `AGENTS.md` 同步生效。
- 把规则写入 agent 指令文件，可以让后续 Codex/Agent 在进入仓库时自动看到并执行该要求。

实现功能：

- 后续每次代码修改都会留下可追溯的中文工程记录。
- 新聊天窗口迁移时，可以直接阅读中文文档和修改日志理解项目状态。

## 后续维护要求

以后修改项目代码时，必须同步记录：

- 修改日期。
- 涉及文件。
- 修改内容。
- 修改原理。
- 实现的功能。
- 运行或验证方式。
- 已知限制和后续事项。

如果是较小改动，可以追加到本文档；如果是独立主题或跨多天开发，应新建 `docs/zh-cn/change-log-YYYY-MM-DD.md` 并在 `docs/zh-cn/README.md` 中添加链接。
