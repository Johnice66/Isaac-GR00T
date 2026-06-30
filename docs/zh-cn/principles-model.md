# 模型原理

GR00T N1.7 可以看成两段式模型：

```text
视觉 + 语言 -> VLM backbone -> backbone_features
状态 + noisy action + timestep + embodiment_id + backbone_features -> 动作头 -> action velocity
```

第一段理解任务语义和视觉场景，第二段把这些条件转成连续机器人动作。

## Backbone：Qwen3-VL / Cosmos-Reason2

`Qwen3Backbone` 封装 `Qwen3VLForConditionalGeneration`。输入由 `Qwen3VLProcessor` 生成，包括：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`

backbone 前向时输出最后一层 hidden states，并额外返回：

- `backbone_features`：语言和视觉 token 的特征序列。
- `backbone_attention_mask`：有效 token mask。
- `image_mask`：哪些 token 来自图像。

配置项里有几个重要开关：

- `select_layer`：加载后裁剪语言模型层数，减少计算。
- `tune_llm`：是否训练语言模型部分。
- `tune_visual`：是否训练视觉部分。
- `tune_top_llm_layers`：只训练顶部若干 LLM 层。
- `use_flash_attention`：优先使用 Flash Attention。
- `load_bf16` 和 `backbone_trainable_params_fp32`：控制加载精度和可训练参数精度。

## Processor 如何组织 VLM 输入

`Gr00tN1d7Processor` 会把多帧、多视角图像按顺序转成 PIL image 列表，再构造一个包含多张图片和语言指令的 chat conversation。

简化后类似：

```python
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_0},
            {"type": "image", "image": img_1},
            {"type": "text", "text": instruction},
        ],
    }
]
```

然后调用 Qwen3-VL processor 的 chat template 和 tokenizer/image processor，把它变成模型输入。

## 动作头输入

动作头接收两类输入：

来自 backbone：

- `backbone_features`
- `backbone_attention_mask`
- `image_mask`

来自动作侧：

- `state`：归一化并 padding 后的状态历史。
- `action`：训练时的 ground truth action chunk。
- `action_mask`：有效动作维度和有效 horizon。
- `embodiment_id`：选择 embodiment-specific 参数。

状态会经过 `CategorySpecificMLP` 编码。动作会经过 `MultiEmbodimentActionEncoder` 编码，编码时同时注入 timestep 的 sinusoidal encoding。

## Embodiment-specific MLP

多 embodiment 的关键实现是 `CategorySpecificLinear`。

普通线性层只有一组权重：

```text
y = x W + b
```

这里每个 embodiment/category 有自己的权重：

```text
y_i = x_i W[embodiment_id_i] + b[embodiment_id_i]
```

因此同一个模型可以共享 backbone 和 DiT 主体，同时为不同机器人保留不同的状态编码、动作编码和动作解码投影。

相关模块：

- `CategorySpecificMLP`：状态编码器和动作解码器。
- `MultiEmbodimentActionEncoder`：动作 + timestep 编码器。

## Flow Matching 训练目标

动作头训练时不是直接回归 action，而是学习从噪声到真实动作的 velocity field。

给定真实动作 `actions` 和随机噪声 `noise`，采样时间 `t`：

```text
noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise
```

模型输入 noisy trajectory、timestep、状态和 VLM 特征，输出预测 velocity。损失是：

```text
MSE(pred_velocity, actions - noise) * action_mask
```

这样推理时可以从随机噪声开始，沿模型预测的 velocity 积分，逐步走向动作分布。

## 推理时的 denoising

`get_action_with_features()` 的推理过程：

1. 从标准正态采样初始 `actions`。
2. 设定 `dt = 1 / num_inference_timesteps`。
3. 每一步把当前 actions 编码成 action features。
4. 和 state features 拼接为 `sa_embs`。
5. DiT 在 VLM 条件下预测 `pred_velocity`。
6. Euler 更新：

```text
actions = actions + dt * pred_velocity
```

默认 `num_inference_timesteps=4`，这是速度和质量之间的折中。

## DiT 与 AlternateVLDiT

`DiT` 是带 timestep 条件的 transformer。每层主要包含：

- AdaLayerNorm：用 timestep embedding 调制归一化。
- attention：可在 action/state token 内自注意力，也可 cross-attend 到 VLM token。
- feed-forward。

`AlternateVLDiT` 是 N1.7 默认使用的变体。它更细粒度地区分图像 token 和文本 token，并通过 `attend_text_every_n_blocks` 控制文本条件注入频率。这样做的目的，是让视觉条件成为动作生成的高频条件，同时保持语言指令对动作计划的调制。

## Action Mask

模型的 `max_action_dim` 和 `action_horizon` 是上限，不同 embodiment 的真实动作维度和 horizon 可能更小。

`action_mask` 用来屏蔽：

- padding 出来的时间步。
- padding 出来的动作维度。

训练 loss 只在有效位置计算。推理 decode 时也只取 modality config 定义的真实 action horizon 和真实维度。

## RTC / Action Inpainting

`get_action_with_features()` 中如果 `action_input` 包含已有 action，会进入 RTC 风格的 overlap/inpainting 逻辑：

- 前一段动作可以作为新 action chunk 的开头。
- `rtc_frozen_steps` 范围内 velocity strength 设为 0，保持不变。
- overlap 的后续部分用指数 ramp 逐渐放开。

这用于减小连续 action chunk 切换时的抖动，尤其适合真实机器人控制中的延迟补偿。

## 训练开关如何影响参数

`Gr00tN1d7ActionHead.set_trainable_parameters()` 根据配置冻结模块：

- `tune_projector=False`：冻结 state encoder、action encoder、action decoder 和位置编码。
- `tune_diffusion_model=False`：冻结 DiT。
- `tune_vlln=False`：冻结 VLM LayerNorm 和 VL self-attention。

backbone 侧由 `Qwen3Backbone.set_trainable_parameters()` 控制：

- `tune_llm=False`：冻结 language model。
- `tune_visual=False`：冻结 visual model。
- `tune_top_llm_layers>0`：重新打开顶部若干 LLM 层。

微调时常见做法是冻结大部分 backbone，主要训练动作头和少量投影层，以降低显存和过拟合风险。

