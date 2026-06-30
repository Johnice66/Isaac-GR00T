# 部署与加速原理

GR00T N1.7 支持三条主要推理路径：

- PyTorch eager：最稳、最容易调试。
- `torch.compile`：主要加速动作头。
- TensorRT：把多个组件导出为 ONNX 并构建 engine，追求端到端低延迟。

## 推理模式

`run_gr00t_server.py` 中的 `--inference-mode` 支持：

- `pytorch`：默认模式。
- `torch_compile`：编译 `policy.model.action_head.model.forward`。
- `tensorrt`：加载 TensorRT engine。

TensorRT 子模式通过 `--trt-mode` 控制：

- `n17_full_pipeline`
- `vit_llm_only`
- `action_head`
- `dit_only`

部署脚本中的命名有一点历史包袱：`trt_full_pipeline`、`n17_full_pipeline` 和 `full_pipeline` 在不同脚本中出现，实际指向 N1.7 的全 pipeline engine 集合。

## TensorRT Full Pipeline 包含什么

`scripts/deployment/README.md` 中的 full pipeline 会加速：

- Qwen3-VL ViT。
- Qwen3-VL LLM。
- VL self-attention。
- state encoder。
- action encoder。
- DiT / AlternateVLDiT。
- action decoder。

仍保留在 PyTorch 中的通常是轻量 glue ops，例如：

- `embed_tokens`
- `masked_scatter`
- `get_rope_index`
- VLLN

## 一体化构建流程

推荐入口：

```bash
python scripts/deployment/build_trt_pipeline.py \
  --model-path <checkpoint> \
  --dataset-path <dataset> \
  --embodiment-tag <tag>
```

这个脚本按步骤调用：

```text
export  -> export_onnx_n1d7.py
build   -> build_tensorrt_engine.py
verify  -> verify_n1d7_trt.py
benchmark -> benchmark_inference.py
```

输出目录默认是：

```text
gr00t_trt_deployment/
  onnx/
  engines/
  pipeline.log
```

## ONNX 导出模式

`export_onnx_n1d7.py` 支持：

- `dit_only`：只导出 DiT，兼容较早部署路径。
- `action_head`：导出 action head 相关组件，backbone 留在 PyTorch。
- `full_pipeline`：导出 ViT、LLM 和 action head 组件，N1.7 推荐。

导出脚本会用真实 dataset/checkpoint 跑一次 inference，捕获各组件的实际输入 shape，再导出 ONNX。这意味着：

- dataset path 不只是占位参数，它提供样例输入。
- batch size 会固化到 ONNX/TRT engine。
- 更换相机数量、图像 shape、batch size 或模型结构后需要重新导出。

## 静态 Batch Size 约束

`build_trt_pipeline.py` 明确说明 `batch_size` 会被 bake 进 ONNX/TRT 模型。运行时 batch 必须与构建时一致。

生产控制通常使用 batch size 1；如果要做批量评测，必须保证构建和运行的 batch 一致。

## 精度

部署配置支持：

- `bf16`
- `fp16`
- `fp32`

默认偏向 `bf16`，因为 N1.7 训练和推理大量使用 bfloat16。不同平台、TensorRT 版本和 GPU 架构对精度支持不同，出现 engine 构建失败时，应先确认平台依赖和 TensorRT 版本。

## 平台差异

项目按平台提供安装脚本：

- dGPU：`scripts/deployment/dgpu/install_deps.sh`
- Jetson Orin：`scripts/deployment/orin/install_deps.sh`
- Jetson Thor：`scripts/deployment/thor/install_deps.sh`
- DGX Spark：`scripts/deployment/spark/install_deps.sh`

不同平台的差异主要在：

- Python 版本。
- CUDA 版本。
- PyTorch/Triton/TensorRT wheel 来源。
- `torchcodec` 和 FFmpeg 版本。
- 可用的 TensorRT engine 覆盖范围。

Orin 等平台可能只支持 DiT-only 或部分加速路径，具体以 `scripts/deployment/README.md` 和平台测试为准。

## 服务端接入 TensorRT

服务端使用 TensorRT 时：

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path <checkpoint> \
  --embodiment-tag <tag> \
  --inference-mode tensorrt \
  --trt-engine-path ./gr00t_trt_deployment/engines \
  --trt-mode n17_full_pipeline
```

内部会调用 `trt_model_forward.setup_tensorrt_engines()`，把 engine 接到 policy 的模型 forward 路径中。

## 验证与 Benchmark

构建后至少做两类验证：

1. 数值验证：PyTorch 和 TensorRT 输出 cosine similarity/MSE 是否在可接受范围内。
2. 端到端验证：在真实 observation pipeline 下测 latency 和 action 输出。

`build_trt_pipeline.py --steps export,build,verify,benchmark` 会自动执行这些步骤。

如果只改了部署参数，可以运行：

```bash
python scripts/deployment/build_trt_pipeline.py \
  --model-path <checkpoint> \
  --dataset-path <dataset> \
  --embodiment-tag <tag> \
  --steps verify,benchmark
```

前提是 ONNX 和 engines 已经存在且与当前 checkpoint/input shape 匹配。

## 常见问题

Engine 和 GPU 架构不匹配：

- TensorRT engine 通常和 GPU 架构绑定，换 GPU 要重建。

Batch size 不匹配：

- 重新用目标 batch size 导出和构建。

Checkpoint/processor 不匹配：

- TensorRT 导出依赖真实模型结构和 processor，不能混用不同 checkpoint 的 engine。

ONNX 外部数据文件散落：

- 导出脚本包含 consolidation 逻辑，会尝试把 external data 合并为单个 `.data` 文件。

`torch.compile` 首次运行慢：

- 首次编译会有额外开销，benchmark 需要区分 warmup 和稳定迭代。

