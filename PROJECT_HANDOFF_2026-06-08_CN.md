# GR00T × AgiBot G01 项目迁移文档

生成时间：2026-06-08  
用途：迁移到新聊天窗口时提供完整上下文。

## 1. 当前目标

使用 NVIDIA Isaac GR00T N1.7 微调并部署 AgiBot G01 真机策略。

当前控制范围：

- 控制：双臂 14 个关节 + 左右夹爪开合
- 不控制：底盘、头部、腰部
- 优先级：真机效果优先，不只看 benchmark 数字

## 2. 服务器环境

服务器项目路径：

```bash
/root/gpufree-data/Isaac-GR00T
```

已知环境：

- GPU：NVIDIA A100-SXM4-80GB
- 系统内存：1 TB
- 数据盘：`/root/gpufree-data`
- Python 环境：项目内 `.venv`

每次新 shell 必须先执行：

```bash
cd /root/gpufree-data/Isaac-GR00T
source /root/gpufree-data/Isaac-GR00T/.venv/bin/activate
export LD_LIBRARY_PATH=/root/gpufree-data/Isaac-GR00T/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

注意：

- 服务器上不要默认用 `uv run` 跑训练或部署。
- 之前 `uv run` 会重新同步依赖，容易遇到 flash-attn 下载源不稳定。
- 直接用激活 venv 后的 `python`。
- `LD_LIBRARY_PATH` 这行不是可选项。之前不设置它时，系统 cuDNN 和 venv 内 cuDNN 冲突，表现为训练进程 `SIGABRT` / exit code `134`，Python 层没有清晰异常；已知症状包括系统 `libcudnn_graph.so.9` 缺少 `cudnnGetLibConfig` 符号。
- 如果命令带 `| tee`，`$?` 可能只表示 `tee` 的退出码；需要看 Python 真实退出码时用 `${PIPESTATUS[0]}`。

另一台机器人/部署服务器路径也在本轮对话中出现过：

```bash
/mnt/10T/Isaac-GR00T
```

这台机器上已知信息：

- GPU：NVIDIA GeForce RTX 4090
- 项目目录：`/mnt/10T/Isaac-GR00T`
- uv cache：`/mnt/4T/uv`
- checkpoint 示例：`/mnt/10T/Isaac-GR00T/checkpoints/checkpoint-40000-opendoor`
- dataset 示例：`/mnt/10T/Isaac-GR00T/datasets/task_2609`
- TensorRT 输出示例：`/mnt/10T/Isaac-GR00T/gr00t_trt_deployment_task2609_40000_single_gpu`

这台机器上不要直接裸跑：

```bash
uv run ...
```

因为它会触发依赖同步，曾经因为 flash-attn wheel 从 GitHub 下载超时失败。已经可用的模式是：

```bash
uv --cache-dir /mnt/4T/uv run --offline --no-sync python ...
```

如果在 `/mnt/10T/Isaac-GR00T/checkpoints` 目录里运行：

```bash
uv run python scripts/deployment/build_trt_pipeline.py ...
```

会报找不到：

```text
/mnt/10T/Isaac-GR00T/checkpoints/scripts/deployment/build_trt_pipeline.py
```

原因是当前工作目录错了。必须先：

```bash
cd /mnt/10T/Isaac-GR00T
```

再运行 repo 内脚本。

TensorRT/ONNX 导出必须只暴露单卡。曾经用多卡环境遇到：

```text
Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0
```

修复方式是只用一个可见 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 ...
```

如果想用物理 GPU1，也应该写：

```bash
CUDA_VISIBLE_DEVICES=1 ...
```

不要在 TensorRT build/export 时暴露 `0,1`。

## 3. 当前训练状态

这次已经把两个同任务数据集一起训练：

```bash
DATASET_A=/root/gpufree-data/Isaac-GR00T/dataset/task_2609
DATASET_B=/root/gpufree-data/Isaac-GR00T/dataset/task_2178
DATASET_PATH="${DATASET_A}:${DATASET_B}"
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${DATASET_A}/meta/config.py" \
  --num-gpus 1 \
  --output-dir /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects \
  --max-steps 60000 \
  --save-steps 20000 \
  --save-total-limit 8 \
  --global-batch-size 32 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 \
  2>&1 | tee train_two_collects.log
```

训练已经跑到 60000 step。日志末尾有：

```text
train_runtime: 44930.082
train_samples_per_second: 42.733
train_steps_per_second: 1.335
train_loss: 0.039473661675055824
```

训练结束时报错：

```text
No space left on device (os error 28)
```

这个不是内存爆了，而是磁盘满了。根据后续检查，`checkpoint-60000` 里的模型本体是完整可用的。

当前可用模型路径：

```bash
/root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000
```

已经验证过的模型文件：

- `model-00001-of-00003.safetensors`：OK，约 4.64 GB
- `model-00002-of-00003.safetensors`：OK，约 4.63 GB
- `model-00003-of-00003.safetensors`：OK，约 2.44 GB
- `model.safetensors.index.json`：存在

之后评估、server、TensorRT、`torch.compile` 都应该使用：

```bash
--model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000
```

不要使用父目录：

```bash
/root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects
```

训练代码还有一个容易复发的点：本地 `gr00t/experiment/experiment.py` 当前调用的是：

```python
trainer.train(resume_from_checkpoint=True)
```

这会让 Trainer 自动从 `output_dir` 里最后一个 checkpoint 续训。如果换了数据、modality 或其他关键配置想重新训练，必须使用新的 `--output-dir`，或先人工确认旧目录里没有会被自动续训的 checkpoint。不要在没确认的情况下复用旧输出目录。

## 4. 磁盘清理建议

当时磁盘状态约为：

```text
/root/gpufree-data: 488G total, 464G used, 24G available
```

建议先释放空间再继续跑评估或 TensorRT。

如果不打算从 `checkpoint-60000` 继续训练，可以删除：

```bash
rm /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000/optimizer.pt
```

`optimizer.pt` 约 13 GB。推理、开环评估、server、TensorRT、`torch.compile` 都不需要它；只有继续训练才需要。

删除其他文件前，先检查是否有根目录残留的不完整模型文件：

```bash
ls -lh /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects | grep safetensors
```

只删除 `checkpoint-60000` 外面的残留 `model-*.safetensors`。不要删除 `checkpoint-60000` 内的模型文件。

## 5. 数据映射和 modality

当前使用自定义 `NEW_EMBODIMENT`。

已确认正确映射：

State：

- `state[28:35]`：左臂 7 关节，弧度
- `state[35:42]`：右臂 7 关节，弧度
- `state[0:1]`：左夹爪
- `state[1:2]`：右夹爪

Action：

- `action[16:23]`：左臂 7 关节目标
- `action[23:30]`：右臂 7 关节目标
- `action[0:1]`：左夹爪目标
- `action[1:2]`：右夹爪目标

重要踩坑：

- 不要把 `state[0:14]` 当手臂关节。
- 不要把 `state[14:28]` 当手臂关节。
- 正确关节 state 是 `state[28:42]`。
- 正确关节 action 是 `action[16:30]`。

这次双数据集训练使用的配置文件：

```bash
/root/gpufree-data/Isaac-GR00T/dataset/task_2609/meta/config.py
```

因为 `task_2609` 和 `task_2178` 是同任务、同配置、分两次采集，所以训练时只传了 `DATASET_A` 的 config。

`embodiment_configs.py` 不需要为了这个自定义机器人手动改库文件。数据集自己的 `meta/config.py` 会通过 `register_modality_config(..., embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)` 在运行时注册；训练和 server 只要正确加载这个 config 即可。

历史上下文里曾经出现过 `task_5093`，并且同样确认过这套 AgiBot G01 的关键切片规律：

- 关节 state：`state[28:42]`
- 关节 action：`action[16:30]`
- 左右夹爪 state：`state[0:2]`
- 左右夹爪 action：`action[0:2]`

但当前这份 handoff 的主线模型是 `task_2609 + task_2178` 训练出的 `checkpoint-60000`。迁移后不要因为旧摘要里的 `task_5093` 路径而误用旧数据或旧 checkpoint。

`task_5093` 作为历史参考数据集，已知元信息如下：

- 本地路径示例：`/Users/johnice/Desktop/nw/Isaac-GR00T/datasets/task_5093`
- 服务器旧路径示例：`/root/gpufree-data/Isaac-GR00T/dataset/task_5093`
- `robot_type`：`a2d`
- `total_episodes`：863
- `total_frames`：272715
- `fps`：30
- task 文本：`Fixed-point Non-generalized Door Opening`
- 三路相机：`top_head` 为 `800x1280`，`hand_left`/`hand_right` 为 `480x848`
- 视频编码：HEVC，`yuv420p`

`task_5093` 诊断时还确认过几个动作特征，可作为 sanity check：

- 左臂关节几乎不动，右臂关节运动明显。
- 左夹爪 action 基本恒为 0。
- 右夹爪 action 在开门过程中变化。
- 第一帧 `state[28:42]` 和 `action[16:30]` 数值完全一致，这是当时确认关节索引正确的关键证据。

注意：`task_5093/meta/config.py` 当时的 action `delta_indices=list(range(0, 16))`，这只是旧配置记录；当前 2609/2178 主线不要据此把 action horizon 写死为 16。

## 6. 多数据集训练注意事项

本地 repo 的 `gr00t/experiment/launch_finetune.py` 已支持用冒号拼接多个 dataset：

```python
dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]
```

并传入：

```python
"dataset_paths": dataset_paths
```

如果服务器报错，试图读取：

```text
task_2609:task_2178/meta/info.json
```

说明服务器上的 `launch_finetune.py` 没有正确拆分 `--dataset-path`，需要按本地版本修补。

Action horizon 注意事项：

- 早期摘要和部分脚本里出现过 `action_horizon=16`。
- 最近 `checkpoint-40000-opendoor` 的 benchmark 明确打印过 `Action Horizon: 40`。
- N1.7 默认模型配置里也常见 `action_horizon=40`。
- 迁移后不要在评估脚本或 client 里写死 16/32/40，应优先读取模型或 server 返回 action 的真实 shape。
- `test_0526_n17_safe_v7.py` / `v8` 当前会动态读取 `action["joint_position"].shape[1]`，并用 `--execute-horizon` 控制每次实际执行前多少个模型步。

## 7. TensorRT 和 torch.compile 结果

之前在 A100 上 benchmark，4 denoising steps：

| 模式 | E2E | 频率 | Backbone | Action Head |
|---|---:|---:|---:|---:|
| PyTorch eager | 240 ms | 4.2 Hz | 52 ms | 90 ms |
| torch.compile | 175 ms | 5.7 Hz | 53 ms | 23 ms |
| TensorRT full pipeline | 210 ms | 4.8 Hz | 76 ms | 21 ms |

TensorRT 校验结果：

```text
cosine=1.000000 PASS
```

A100 当时建议：

- 默认真机 server 路线：`torch.compile`
- TensorRT full pipeline 可用，但在当前 benchmark 里 E2E 比 `torch.compile` 慢

上面这个建议只适用于当时的 A100 benchmark。后来在另一台 RTX 4090 服务器上重新 build TensorRT full pipeline，结果完全不同，4 denoising steps：

| 设备 | 模式 | Data Processing | Backbone | Action Head | E2E | 频率 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 4090 | PyTorch eager | 29 ms | 65 ms | 150 ms | 244 ms | 4.1 Hz |
| RTX 4090 | torch.compile | 29 ms | 65 ms | 22 ms | 116 ms | 8.6 Hz |
| RTX 4090 | TensorRT full pipeline | 29 ms | 19 ms | 19 ms | 67 ms | 14.9 Hz |

RTX 4090 上的 TensorRT 校验结果：

```text
verify cosine=1.000000 PASS
```

所以部署路线需要按机器区分：

- A100 那次 benchmark：`torch.compile` E2E 更快。
- RTX 4090 那次 benchmark：`TensorRT n17_full_pipeline` 明显最快，建议优先用于真机部署。
- 不要把某一台机器的 benchmark 结论直接套到另一台机器。

RTX 4090 上已经跑通的 TensorRT build 命令示例：

```bash
cd /mnt/10T/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 \
uv --cache-dir /mnt/4T/uv run --offline --no-sync python scripts/deployment/build_trt_pipeline.py \
  --model-path /mnt/10T/Isaac-GR00T/checkpoints/checkpoint-40000-opendoor \
  --dataset-path /mnt/10T/Isaac-GR00T/datasets/task_2609 \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /mnt/10T/Isaac-GR00T/gr00t_trt_deployment_task2609_40000_single_gpu
```

RTX 4090 上对应的 TensorRT server 命令示例：

```bash
cd /mnt/10T/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_CACHE="/home/nwrobot/.cache/modelscope/hub/models" \
uv --cache-dir /mnt/4T/uv run --offline --no-sync python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /mnt/10T/Isaac-GR00T/checkpoints/checkpoint-40000-opendoor \
  --inference-mode tensorrt \
  --trt-engine-path /mnt/10T/Isaac-GR00T/gr00t_trt_deployment_task2609_40000_single_gpu/engines \
  --trt-mode n17_full_pipeline \
  --host 0.0.0.0 \
  --port 5555
```

RTX 4090 上的 `torch.compile` server 命令示例：

```bash
cd /mnt/10T/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_CACHE="/home/nwrobot/.cache/modelscope/hub/models" \
uv --cache-dir /mnt/4T/uv run --offline --no-sync python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /mnt/10T/Isaac-GR00T/checkpoints/checkpoint-40000-opendoor \
  --inference-mode torch_compile \
  --host 0.0.0.0 \
  --port 5555
```

TensorRT 产物不在 checkpoint 目录，而是在：

```bash
/root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/
```

主要目录：

```bash
/root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/onnx
/root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/engines
```

`torch.compile` 是运行时内存编译，不会在 checkpoint 目录产生新模型文件。

重要澄清：

- `build_trt_pipeline.py` 会导出 ONNX 并构建 TensorRT engine，但不会修改 checkpoint 目录。
- TensorRT 产物在 `--output-dir` 下，通常是 `onnx/`、`engines/`、`pipeline.log`。
- `torch.compile` 不会“转换模型文件”，只是在 server 运行时编译 PyTorch graph。
- `torch.compile` 不是量化；TensorRT 这里默认是 `bf16`，也不是 INT8 量化。

A100 上最初跑 TensorRT build 时遇到过：

```text
Repo id must be in the form 'repo_name' or 'namespace/repo_name': 'checkpoint/task_2609/checkpoint-30000'
```

这个错误不是 Hugging Face 上真的有这个 repo，而是本地 `--model-path` 没有被识别成存在的目录，代码退回按 Hub repo id 解析。排查顺序：

1. 确认当前工作目录是 repo 根目录，例如 `/root/gpufree-data/Isaac-GR00T`。
2. 确认路径拼写是 `checkpoints` 还是 `checkpoint`，不要混用。
3. 优先传绝对路径，例如：

```bash
--model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609/checkpoint-30000
```

4. build/export 前先 `ls` 这个目录，确认里面有 `model.safetensors.index.json` 和 `model-*.safetensors`。

关于“加速后真机感觉精度下降”的排查结论：

- 如果 TensorRT 和 `torch.compile` 都变差，优先怀疑 client 执行策略、图像预处理、动作时序、夹爪映射，而不是直接归因到量化。
- 需要先做同一条 observation 下的离线动作对齐：PyTorch eager vs `torch.compile` vs TensorRT，比较每个关节的 MAE/MSE/最大误差，以及第一帧动作误差。
- 如果离线输出很接近，但真机表现不同，问题基本在闭环执行和机器人控制侧。
- TensorRT `cosine=1.000000 PASS` 只能说明 benchmark 验证样本输出接近，不等价于真机成功。
- GR00T action head 每次推理会从随机噪声开始生成 action chunk；如果 client 频繁重规划且没有 chunk 融合/限幅，真机会表现为抖动或不稳定。

如果确实怀疑 TensorRT BF16 数值影响，可以另建一版 FP32 TensorRT 做对照：

```bash
cd /mnt/10T/Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 \
uv --cache-dir /mnt/4T/uv run --offline --no-sync python scripts/deployment/build_trt_pipeline.py \
  --model-path /mnt/10T/Isaac-GR00T/checkpoints/checkpoint-40000-opendoor \
  --dataset-path /mnt/10T/Isaac-GR00T/datasets/task_2609 \
  --embodiment-tag NEW_EMBODIMENT \
  --precision fp32 \
  --output-dir /mnt/10T/Isaac-GR00T/gr00t_trt_deployment_task2609_40000_fp32
```

也可以用同一套 engine 分段定位：

- `--trt-mode n17_full_pipeline`：ViT + LLM + action head 全 TensorRT
- `--trt-mode action_head`：只替换 action head，backbone 保持 PyTorch
- `--trt-mode vit_llm_only`：只替换 ViT + LLM，action head 保持 PyTorch
- `--trt-mode dit_only`：只替换 DiT

判断方式：

- `action_head` 好、`n17_full_pipeline` 差：更可能是 ViT/LLM TRT 或图像路径差异。
- `vit_llm_only` 好、`action_head` 差：更可能是 action head TRT 或 action chunk 执行问题。
- `torch.compile` 也差：更可能是 client 动作执行、随机采样、时序、图像或夹爪映射问题。

提高动作质量的另一个可能方向是增加 denoising steps，例如 `4 -> 6 -> 8`。当前 `run_gr00t_server.py` 本地版本尚未明确记录已暴露 `--denoising-steps` CLI；如果要测试，需要确认服务器脚本是否已经支持，或在 server 初始化后设置：

```python
policy.model.action_head.num_inference_timesteps = denoising_steps
```

## 8. Server 脚本现状

本地已修改文件：

```bash
/Users/johnice/Desktop/nw/Isaac-GR00T/gr00t/eval/run_gr00t_server.py
```

本地 server 已支持：

- `--inference-mode pytorch`
- `--inference-mode torch_compile`
- `--inference-mode tensorrt`
- `--compile-mode`
- `--trt-engine-path`
- `--trt-mode`

支持的 TensorRT mode：

- `n17_full_pipeline`
- `vit_llm_only`
- `action_head`
- `dit_only`

本地实现细节：

- `torch_compile` 当前编译的是 `policy.model.action_head.model.forward`，不是把 checkpoint 导出成新文件。
- `tensorrt` 会从 `scripts/deployment/trt_model_forward.py` 导入 `setup_tensorrt_engines()`，并要求 `--trt-engine-path` 指向已经存在的 `engines` 目录。
- `--inference-mode` 只适用于 `Gr00tPolicy`。如果 server 用 `--dataset-path` 启动 `ReplayPolicy`，当前本地脚本会拒绝 `torch_compile` / `tensorrt`，只能走 PyTorch/replay 路线。
- 本地 `ServerConfig` 默认 `--inference-mode pytorch`；要用加速必须显式传 `--inference-mode torch_compile` 或 `--inference-mode tensorrt`。

推荐 server 命令，使用新的 60000 step 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
  --model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000 \
  --embodiment-tag new_embodiment \
  --host 0.0.0.0 \
  --port 5555 \
  --inference-mode torch_compile
```

可选 TensorRT server：

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

服务器上需要检查：

```bash
python -m py_compile gr00t/eval/run_gr00t_server.py
```

注意：不要默认假设本地修改已经同步到了服务器。

## 9. 真机 Client 脚本现状

本地文件：

```bash
/Users/johnice/Desktop/nw/Isaac-GR00T/test_0526_n17_safe_v7.py
/Users/johnice/Desktop/nw/Isaac-GR00T/test_0526_n17_safe_v8.py
```

当前状态：

- `v7` 和 `v8` 当前内容一致。
- 两个文件都是 git 未跟踪文件。
- client 使用 repo 官方 `PolicyClient`：

```python
from gr00t.policy.server_client import PolicyClient
```

这意味着 client 运行环境必须能 import 当前 repo 的 `gr00t` 包。当前 v7/v8 没有保留手写 ZMQ/msgpack fallback；如果机器人侧不是在 Isaac-GR00T repo 环境里运行，需要后续再补一个 fallback client，或先把 repo/venv 环境配置好。

设计原则：

- TensorRT / `torch.compile` 在 server 侧选择，client 不直接加载加速模型。
- client 只负责连接 server、发送 observation、接收 action、执行真机控制。
- client 能兼容以下 action 返回格式：
  - `left_arm_joint_position` + `right_arm_joint_position`
  - 或合并后的 `joint_position`
  - 可带或不带 `action.` 前缀

client 安全默认值：

- 默认不启用控制，必须传 `--enable-control`。
- 默认需要控制确认，`--confirm-control` 开启。
- 首次请求 timeout 更长，因为 `torch.compile` 第一次可能 JIT。
- 包含 NaN/Inf 检查。
- 包含轨迹平滑、blend、关节变化限制。

本轮排查后，client 侧已经补充/确认的关键点：

- 默认 `task` 已改为 `"Fixed-point Non-generalized Door Opening"`。
- 默认 `client_resize=False`，也就是发送原始相机帧，让 GR00T processor 使用训练时保存的 image transform。
- 原因：旧逻辑会把三路图像强行压成 `640x480`，容易破坏训练时 `top_head 800x1280`、手部相机 `480x848` 的视野比例。
- 默认 `model_fps=30.0`，用于保持 action chunk 的时间尺度和训练数据一致。
- 默认 `execute_horizon=8`，即模型返回 40 步时只执行前 8 个模型步就重新观测/推理，避免把完整 40 步开环走完。
- `--interp-factor` 仍保留为 legacy 参数，但当前时序主要由 `--model-fps` 和 `--control-freq` 决定。
- 默认 `ema_alpha=1.0`，避免 EMA 滞后导致轨迹偏移；如果要额外低通，可以手动调小。
- 加入关节限幅：`--max-step-delta 0.2618`、`--max-joint-speed 3.0`、`--max-joint-accel 6.0`。
- 加入夹爪映射参数：`--gripper-scale`、`--gripper-offset`、`--gripper-min`、`--gripper-max`。
- 首次运行会打印 observation 图像 shape、action horizon、关节范围、左右夹爪范围，用于确认图像尺寸和夹爪单位。

重要：如果首次 action 打印显示夹爪范围是 `0.000~1.000`，但机器人 `/wbc/*_ee_command` 实际需要 `0~120`，需要在 client 侧加：

```bash
--gripper-scale 120
```

如果 action 本身已经是 `0~120`，不要加这个 scale。

当前 client 默认任务文本：

```text
Fixed-point Non-generalized Door Opening
```

注意：之前上下文里也出现过 `Partial Discharge Detection`。真机部署前必须确认 `task_2609 + task_2178` 训练数据里的真实 task 文本，不要盲目沿用旧默认值。

推荐 client 命令：

```bash
python test_0526_n17_safe_v7.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "Fixed-point Non-generalized Door Opening" \
  --enable-control \
  --control-freq 60 \
  --model-fps 30 \
  --execute-horizon 8 \
  --no-client-resize \
  --ema-alpha 1.0
```

无控制连通性测试：

```bash
python test_0526_n17_safe_v7.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "Fixed-point Non-generalized Door Opening" \
  --no-enable-control
```

如果真实 task 文本不同，替换 `--task`。

## 10. ROS Topic

状态：

- 手臂状态：`/hal/arm_joint_state`
- 左夹爪状态：`/hal/left_ee_data`
- 右夹爪状态：`/hal/right_ee_data`

控制：

- 手臂控制：`/wbc/arm_command`
- 左夹爪控制：`/wbc/left_ee_command`
- 右夹爪控制：`/wbc/right_ee_command`

相机：

- 头部相机：`/camera/head_color`
- 左手相机：`/camera/hand_left_color`
- 右手相机：`/camera/hand_right_color`

真机手臂关节顺序：

- `joint_state.position[0:7]`：左臂
- `joint_state.position[7:14]`：右臂

这个顺序和训练数据切片后的语义一致，但训练数据原始 state 里的索引不是从 0 开始。

## 11. 真机安全限制

来自 GDK 记录的硬约束：

- 单帧关节变化小于 `0.2618 rad`，即 15 度
- 角速度小于 `3 rad/s`
- 角加速度小于 `6 rad/s^2`

所以：

- 不要直接把模型输出的稀疏 action chunk 当成逐帧命令发给机器人。
- 需要插值、平滑、限幅后再发 `/wbc/arm_command`。
- 先做无控制测试，再做短 horizon、低风险真机测试。

## 12. 推荐下一步

1. 先清理 `/root/gpufree-data` 磁盘空间。
2. 对 `checkpoint-60000` 做开环评估。
3. 确认 `task_2609` 和 `task_2178` 里的真实 task 文本。
4. 启动 server：
   - A100 按已有 benchmark 优先试 `--inference-mode torch_compile`。
   - RTX 4090 按已有 benchmark 优先试 `--inference-mode tensorrt --trt-mode n17_full_pipeline`。
5. client 先跑 `--no-enable-control`，确认：
   - server ping 成功
   - modality config 检查通过
   - 首次 `get_action` 返回 14 维关节 + 左右夹爪
   - action 没有 NaN/Inf
6. 再做低风险真机控制测试：
   - 短 horizon
   - 开启人工确认
   - 观察关节变化、速度、夹爪映射
7. 如果需要，再对比 `torch.compile` 和 TensorRT full pipeline 的真机闭环稳定性。

## 13. 本地 repo 状态

当前本地 git 状态：

```text
 M gr00t/eval/run_gr00t_server.py
?? PROJECT_HANDOFF_2026-06-08.md
?? PROJECT_HANDOFF_2026-06-08_CN.md
?? datasets/
?? test.py
?? test_0526_n17.py
?? test_0526_n17_safe_v7.py
?? test_0526_n17_safe_v8.py
```

不要假设这些修改已经同步到服务器。

本地已通过的 client 静态检查：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m py_compile test_0526_n17_safe_v7.py
```

服务器侧还应检查：

```bash
python -m py_compile gr00t/eval/run_gr00t_server.py
```

本地 macOS 的系统 `python3` 是 Python 3.9 时，不能可靠检查含有 `str | None` 这类 Python 3.10 语法的 server 文件。server 语法检查应在服务器的 Python 3.10 venv 中做。

## 14. 本地开发约定

来自当前 repo 的 AGENTS/CLAUDE 说明：

- 项目语言主线：Python 3.10；Thor/DGX Spark 等部署目录可能使用 Python 3.12。
- 包管理器：`uv`。
- 构建系统：setuptools，见 `pyproject.toml`。
- 格式化/ lint：`ruff format`，`ruff check` 规则主要是 `E/F/I`，忽略 `E501`，行宽 100。
- 提交前推荐跑：`pre-commit run --all-files`。
- CPU 测试：`python -m pytest tests/ -m "not gpu" -v --timeout=300`。
- GPU 测试：`python -m pytest tests/ -m gpu -v --timeout=300`。
- 构建包：`uv build`。
- 锁文件检查：`uv lock --locked`。

这些是本地代码开发/提交约定；服务器真机训练和部署仍优先遵守前面写的“激活 venv 后直接用 python，不默认 uv run”的实战限制。

## 15. 迁移后优先确认的问题

- `task_2609 + task_2178` 的真实 task 文本到底是什么。
- 服务器上的 `run_gr00t_server.py` 是否已经包含 `--inference-mode` 支持。
- 服务器上的 `launch_finetune.py` 是否已经支持冒号分隔的多 dataset。
- 当前机器人夹爪命令范围到底是 `0..120`、`0..1`，还是 GDK 当前配置的其他范围。
- 是否还需要从 `checkpoint-60000` 继续训练；如果不需要，可以删除 `optimizer.pt` 释放 13 GB。
