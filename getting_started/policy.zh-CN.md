# GR00T Policy API 与推理服务说明

本文件对应英文版 [policy.md](policy.md)。它说明如何加载 `Gr00tPolicy`、如何启动 server，以及 client 需要发送什么 observation。

## 两种使用方式

1. 直接本地加载 `Gr00tPolicy`
2. 使用 `run_gr00t_server.py`

## 基本 server 命令

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path /path/to/checkpoint \
  --embodiment-tag NEW_EMBODIMENT
```

## observation 结构

```python
{
    "video": {...},
    "state": {...},
    "language": {...},
}
```

## action 结构

server 返回的 action 通常是：

```python
{
    "action_name": np.ndarray
}
```

## 当前 AgiBot G01 项目重点

- 如果 checkpoint 训练时用了 `config_subtask.py`，语言 key 必须是 `sub_task`
- client 不需要自己区分 `torch.compile` 还是 TensorRT；这是 server 的启动参数
- 真机控制输出仍要在 client 侧做插值和平滑，然后发 ROS topic
