# GR00T 部署与推理加速指南

本文件对应英文版 [README.md](README.md)。重点说明 PyTorch、`torch.compile`、ONNX、TensorRT 几条部署路径。

## 快速开始：PyTorch 推理

```bash
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
  --dataset-path demo_data/libero_demo \
  --embodiment-tag LIBERO_PANDA \
  --traj-ids 0 1 2 \
  --inference-mode pytorch
```

## TensorRT 全流程

统一流水线脚本：

```bash
uv run python scripts/deployment/build_trt_pipeline.py \
  --model-path /path/to/checkpoint \
  --dataset-path /path/to/dataset \
  --embodiment-tag NEW_EMBODIMENT
```

这会依次做：

1. 导出 ONNX
2. 构建 TensorRT engine
3. 验证 PyTorch 与 TRT 输出一致性
4. 做 benchmark

## 关于 `torch.compile`

`torch.compile` 是零额外 engine 构建成本的中间路线：

- 通常比 eager 明显快
- 不需要维护 TensorRT engine 目录
- 真机项目里经常是默认首选

## 当前 AgiBot G01 项目经验

你自己的 benchmark 已经给出了很明确的结论：

- `torch.compile`：约 `175 ms / 5.7 Hz`
- TensorRT full pipeline：约 `210 ms / 4.8 Hz`

所以对你当前模型：

- 默认部署路线优先 `torch.compile`
- TensorRT 保留为可选实验路线
