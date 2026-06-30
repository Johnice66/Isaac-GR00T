# 数据与模态原理

GR00T N1.7 的数据层围绕三件事展开：

- LeRobot 格式的数据组织。
- embodiment tag 与模态配置。
- 状态/动作统计量、归一化和相对动作转换。

## LeRobot 数据假设

训练数据根目录通常包含：

```text
dataset/
  data/*/*.parquet
  meta/info.json
  meta/stats.json
  meta/relative_stats.json
  meta/tasks.jsonl
  meta/episodes.jsonl
```

`data/*/*.parquet` 存储 episode step。低维状态/动作列直接放在 parquet 中；视频列通常通过 LeRobot 元数据指向视频帧或可解码资源。

代码里约定列名前缀：

- `video.<key>`
- `state.<key>`
- `action.<key>`
- language 列通常是 `annotation...`

`LeRobotEpisodeLoader` 负责按 episode 返回 DataFrame；`ShardedSingleStepDataset` 再从 episode 的某个 `step_index` 采样出训练所需的多模态窗口。

## `ModalityConfig`

`ModalityConfig` 描述某种模态取哪些列、取哪些时间偏移。

例如一个常见配置可以理解为：

```python
{
    "video": ModalityConfig(delta_indices=[0], modality_keys=["image", "wrist_image"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_position"]),
    "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=["joint_position"]),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task"]),
}
```

含义：

- video 取当前帧两路相机。
- state 取当前状态。
- action 取未来 16 个控制步。
- language 取当前指令。

`extract_step_data()` 会用 `step_index + delta_index` 取样。如果 `allow_padding=True`，越界索引会被截断到 episode 有效范围，否则需要 episode 长度足够。

## Embodiment Tag 的作用

`EmbodimentTag` 同时绑定以下内容：

- 数据中应有哪些 video/state/action/language 列。
- 动作表示是绝对还是相对。
- EEF 动作格式，例如 XYZ + ROT6D。
- processor 中使用哪个 embodiment id。
- 模型中 category-specific MLP 选择哪组权重。
- checkpoint 是否包含该 embodiment 的 processor config 和统计量。

因此，推理时报 tag 不只是命名问题。如果 checkpoint 里没有对应 tag 的 modality config 或统计量，`Gr00tPolicy` 会拒绝加载或在处理输入时失败。

tag 分三类：

- `PRETRAIN_TAGS`：基础模型内置，可直接用于 base checkpoint。
- `POSTTRAIN_TAGS`：已注册但通常需要对应 finetuned checkpoint。
- `FINETUNE_ONLY_TAGS`：用于新机器人或自定义任务微调。

## 状态处理

状态处理由 `StateActionProcessor.apply_state()` 完成。

主要策略：

- 默认用 min/max 归一化到 `[-1, 1]`。
- 某些 key 可使用 mean/std。
- 某些关节状态可使用 sin/cos 编码，适合周期角度。
- 可通过 `exclude_state` 或 `state_dropout_prob` 做状态消融/增强。

处理后，不同 state group 会 concat 到一个向量，并 padding 到 `max_state_dim`。

## 动作处理

动作处理由 `StateActionProcessor.apply_action()` 完成，顺序是：

1. 如果某个 action group 配置为 `ActionRepresentation.RELATIVE`，先基于当前 state 把绝对动作转成相对动作。
2. 按统计量归一化。
3. 裁剪到 `[-1, 1]`。

推理输出后，`decode_action()` 和 `unapply_action()` 做反向过程：

1. 按 action group 拆分模型输出。
2. 反归一化。
3. 如果是 relative action，基于当前 state 转回绝对动作。

## 相对动作为什么重要

N1.7 强调 relative EEF action space。相对动作不是直接预测世界坐标或关节绝对目标，而是预测相对于当前状态的增量。

好处：

- 跨机器人、跨场景更容易共享动作先验。
- 同一个“向前移动一点”“闭合夹爪”等动作在不同初始位姿下更一致。
- 人类视频预训练和机器人控制之间更容易对齐。

代价：

- 必须有正确的当前 state 作为参考。
- 必须生成 `relative_stats.json`，否则 relative 动作无法正确归一化。
- 数据列维度、action format 和 state reference key 必须严格匹配。

## 统计量生成与合并

`generate_stats()` 会根据 `meta/info.json` 找出 float 特征，为每个低维列计算：

- `mean`
- `std`
- `min`
- `max`
- `q01`
- `q99`

`generate_rel_stats()` 会对配置为 relative 的 action group 额外计算相对动作统计量。

多数据集训练时，`ShardedMixtureDataset.merge_statistics()` 会按 embodiment 分组，并按采样权重合并统计量。合并后写入 processor，用于训练时归一化；训练输出也会保存 `dataset_statistics.json` 供推理复现。

## Sharding 与混合采样

`ShardedSingleStepDataset` 先把每个 episode 的有效 timestep 打散，再按 `shard_size` 组织成 shard。

`ShardedMixtureDataset` 再做跨数据集混合：

- 根据每个数据集的 `mix_ratio` 和平均 shard 大小生成采样概率。
- 生成 `(dataset_index, shard_index)` 的 schedule。
- 在分布式/多 worker 下按 index 分片，避免重复处理。
- 后台预取 shard，减少训练等待。

一个关键约束：分布式训练中所有 rank 的 seed 必须一致。每个 rank 生成相同 schedule，再按 rank/worker index 切分。如果 seed 按 rank 改变，就会出现重复样本和遗漏 shard。

## 常见问题排查

数据列缺失：

- 检查 parquet 列名是否与 `MODALITY_CONFIGS` 一致。
- 检查 language key 是否和数据集实际列一致。

动作维度不匹配：

- 检查 `action.modality_keys` 的顺序和每列维度。
- 检查 `max_action_dim` 是否足够大。
- 检查 EEF action format 是否和 state/action 数据一致。

推理输出尺度异常：

- 检查 checkpoint 是否携带正确 `processor_config.json` 和 `statistics.json`。
- 检查 `override_pretraining_statistics` 是否符合预期。
- relative action 场景下检查 `relative_stats.json` 是否生成。

