# 自定义 Embodiment 微调指南

本文件对应英文版 [finetune_new_embodiment.md](finetune_new_embodiment.md)。适用于你要把 GR00T 微调到自己的机器人上，通常使用 `NEW_EMBODIMENT`。

## 1. 先准备数据

见 [data_preparation.zh-CN.md](data_preparation.zh-CN.md)。

## 2. 准备模态配置

见 [data_config.zh-CN.md](data_config.zh-CN.md)。

## 3. 启动训练

单卡示例：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ./demo_data/cube_to_bowl_5 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/SO100/so100_config.py \
  --num-gpus 1 \
  --output-dir /tmp/so100 \
  --save-steps 2000 \
  --max-steps 2000 \
  --global-batch-size 32
```

## 4. Open-loop 验证

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path ./demo_data/cube_to_bowl_5 \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/so100/checkpoint-2000 \
  --traj-ids 0 \
  --action-horizon 16
```

## 如何判断微调是否正常

重点看三件事：

1. 训练集上的动作曲线是否逐渐贴近 GT
2. 中间 checkpoint 的 MSE / MAE 是否随训练下降
3. 换配置或换数据后，是否相对你自己的历史基线更好

## 当前 AgiBot G01 项目补充

- 远端训练推荐直接用激活后的 `python`，不要默认 `uv run`
- 每次新 shell 要设置 venv 内 cuDNN 的 `LD_LIBRARY_PATH`
- 如果 output dir 里有旧 checkpoint，当前训练代码可能会自动 resume
- 如果训练用了 `config_subtask.py`，后续 client 要发 `sub_task`
