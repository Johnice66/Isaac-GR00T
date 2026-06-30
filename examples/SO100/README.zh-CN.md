# SO100 微调示例

本文件对应英文版 [README.md](README.md)。

## 基本流程

1. 先把 LeRobot v3 转成 v2
2. 把 `modality.json` 放到数据集根目录
3. 用 `examples/SO100/so100_config.py` 做 `NEW_EMBODIMENT` 微调
4. 用 `open_loop_eval.py` 做开环验证
5. 参考 `gr00t/eval/real_robot/SO100/eval_so100.py` 做真机 client
