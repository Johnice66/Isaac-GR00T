# DROID 示例

本文件对应英文版 [README.md](README.md)。

## 说明

GR00T N1.7 基础模型可以直接用 DROID 预训练 tag 做零样本推理：

- `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`

也有现成微调模型：

- `nvidia/GR00T-N1.7-DROID`

## 示例流程

1. 用 `scripts/download_droid_sample.py` 下载小样本
2. 用 `standalone_inference_script.py` 做直接推理
3. 用 `run_gr00t_server.py` 起推理服务
4. 如需真机部署，再接机器人侧控制脚本
