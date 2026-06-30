# 硬件建议

本文件对应英文版 [hardware_recommendation.md](hardware_recommendation.md)。GR00T N1.7 的硬件需求可以分成两类：训练和部署。

## 推理部署

最低建议：

- 1 张 16 GB 以上显存的 GPU
- CUDA 12.6+

经验上：

- 30 Hz 以上：H100、RTX Pro 6000 这类配合 TensorRT
- 10 Hz 左右：Thor、Spark 或大部分 dGPU 配合 `torch.compile`
- 5 Hz 以下：Orin 这类较慢边缘设备，只适合低响应任务

## 微调训练

最低建议：

- 1 张 40 GB 以上显存的 GPU

## 当前项目实况

你当前训练服务器环境已经验证过：

- A100-SXM4-80GB
- 1 TB 内存
- `/root/gpufree-data` 数据盘
