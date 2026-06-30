# 训练与推理原理

本项目的训练和推理共享同一个 processor/模态契约。训练阶段学到的是“在某个 embodiment 的数据分布下如何从观测生成动作”；推理阶段必须用相同的 config、统计量和动作解释方式把机器人观测送入模型。

## 微调入口

最常用入口是：

```bash
bash examples/finetune.sh \
  --base-model-path <checkpoint> \
  --dataset-path <dataset> \
  --embodiment-tag <tag> \
  --output-dir <output>
```

脚本最终调用：

```text
gr00t/experiment/launch_finetune.py
```

`launch_finetune.py` 做几件事：

1. 解析 `FinetuneConfig`。
2. 解析 `EmbodimentTag`。
3. 如提供 `modality_config_path`，导入自定义模态配置。
4. 用 `get_default_config()` 创建完整配置。
5. 设置 dataset path、mix ratio、embodiment tag。
6. 覆盖模型微调开关、数据增强、batch、学习率、checkpoint 等参数。
7. 强制 N1.7 微调使用 `nvidia/Cosmos-Reason2-2B`、`use_relative_action=True` 等设置。
8. 调用 `gr00t.experiment.experiment.run(config)`。

## 训练主函数

`experiment.run()` 是训练编排中心：

```text
run(config)
  -> warn_configs()
  -> 初始化 distributed
  -> set_seed(config.data.seed)
  -> config.validate()
  -> 创建 output_dir
  -> 保存 config 和 wandb_config
  -> MODEL_REGISTRY 创建 pipeline
  -> pipeline.setup()
  -> 创建 TrainingArguments
  -> 创建 Gr00tTrainer
  -> 添加 checkpoint 回调
  -> trainer.train(resume_from_checkpoint=True)
  -> trainer.save_model()
```

`pipeline.setup()` 内部会：

1. 创建 `Gr00tN1d7` 模型。
2. 创建 `Gr00tN1d7Processor`。
3. 通过 `DatasetFactory` 创建 `ShardedMixtureDataset`。
4. 合并并保存 dataset statistics。
5. 返回 collator。

## 自定义 Trainer 的原因

`Gr00tTrainer` 继承 HuggingFace `Trainer`，但对训练 dataloader 做了定制。

主要原因：

- 训练数据是 `IterableDataset`，不是普通 map-style dataset。
- 数据以 shard 为单位加载，适合后台缓存。
- resume 时不希望 HuggingFace 按样本跳过已训练数据，而是重置 dataset seed。
- 日志中隐藏 epoch，因为 IterableDataset 的 epoch 语义不直观。

分布式训练时，`ShardedMixtureDataset` 的 schedule 依赖一致 seed，再由 rank/worker 分片。这里的 seed 逻辑是训练正确性的关键。

## Checkpoint 和 Processor

训练输出不仅是模型权重，还必须包含 processor 相关文件。推理加载时：

```python
model = AutoModel.from_pretrained(model_dir)
processor = AutoProcessor.from_pretrained(model_dir or model_dir / "processor")
```

processor 保存了：

- modality configs。
- statistics。
- 图像处理参数。
- state/action normalization 参数。
- embodiment id mapping。

如果只拷贝模型权重不拷贝 processor，推理侧就不知道怎么解释输入输出。

## 推理入口

最直接的推理方式是创建 `Gr00tPolicy`：

```python
policy = Gr00tPolicy(
    embodiment_tag="LIBERO_PANDA",
    model_path="checkpoints/GR00T-N1.7-LIBERO/libero_10",
    device="cuda",
)

action, info = policy.get_action(observation)
```

也可以启动服务：

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path <checkpoint> \
  --embodiment-tag <tag> \
  --device cuda \
  --port 5555
```

客户端通过 `PolicyClient` 调用 `get_action`、`reset`、`ping` 等 endpoint。

## `Gr00tPolicy` 推理流程

`Gr00tPolicy._get_action()` 的内部流程：

1. 将 batch observation 拆成单样本。
2. 每个样本转成 `VLAStepData`。
3. 调用 processor 得到模型输入。
4. collator 拼 batch。
5. 转成 bfloat16。
6. `model.get_action(**collated_inputs)`。
7. 取出 `action_pred`。
8. processor `decode_action()`，反归一化并将 relative action 转绝对动作。
9. 校验并返回 float32 动作。

`strict=True` 时，policy 会在推理前后检查 observation/action 结构、dtype、shape 和 horizon。

## 推理输入格式

标准输入是嵌套 dict：

```python
observation = {
    "video": {
        "image": np.zeros((B, T, H, W, 3), dtype=np.uint8),
    },
    "state": {
        "joint_position": np.zeros((B, T, D), dtype=np.float32),
    },
    "language": {
        "annotation.human.task": [["pick up the cup"]],
    },
}
```

注意：

- video 必须是 `uint8`，shape 是 `(B, T, H, W, C)`。
- state 必须是 `float32`，shape 是 `(B, T, D)`。
- language 是 batch x temporal 的 list，并且当前实现一般要求每个 batch item 内只有一个字符串。
- key 名必须和 checkpoint 中的 modality config 匹配。

仿真环境有时使用 flat key，例如 `video.image`、`state.joint_position`。这时可用 `Gr00tSimPolicyWrapper` 做兼容转换。

## Open-loop 评估

`open_loop_eval.py` 用数据集中的真实轨迹验证预测动作：

1. 从某个 trajectory 的固定 step 取观测。
2. policy 预测一个 action chunk。
3. 每隔 `action_horizon` 重复推理。
4. 拼接预测动作。
5. 与 ground truth action 计算 MSE/MAE。
6. 保存轨迹曲线图。

open-loop 的价值是快速发现数据处理、统计量、动作维度和 checkpoint 加载问题。它不等价于真实任务成功率，因为没有环境状态反馈。

## Closed-loop 评估

`rollout_policy.py` 创建 Gymnasium 环境并执行闭环控制。支持的主要环境来自：

- LIBERO
- SimplerEnv
- RoboCasa

流程：

```text
create_eval_env()
  -> 根据 env_name 注册具体环境
  -> VideoRecordingWrapper 可选录制视频
  -> MultiStepWrapper 堆叠观测并执行 action chunk
  -> policy 与环境交互直到成功、失败或超时
```

closed-loop 才能评估真实控制效果，但依赖仿真环境、渲染、机器人 wrapper 和动作执行策略，调试成本更高。

## 训练/推理不一致的常见来源

checkpoint 与 tag 不匹配：

- base model 不一定支持 posttrain tag。
- finetuned checkpoint 可能只保存一个 embodiment。

processor 丢失或旧版本：

- 推理会找 `processor_config.json` 或 `processor/`。
- 缺失统计量会导致动作尺度错误。

模态 key 不一致：

- 训练数据列名和推理 observation key 必须一致。

动作 horizon 不一致：

- `action.delta_indices`、policy 执行 horizon、评估脚本的 `action_horizon` 应该协调。

状态参考错误：

- relative action 需要当前 state 转回绝对动作。
- state key 或维度错，会导致动作方向和尺度都错。

