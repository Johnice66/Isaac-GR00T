# 模态配置指南

本文件对应英文版 [data_config.md](data_config.md)。模态配置负责把数据集里的物理字段映射到模型训练与推理使用的逻辑字段。

## 模态配置解决什么问题

`meta/modality.json` 只描述“索引切片是什么”，而 Python 侧的 `config.py` / `config_subtask.py` 还需要进一步描述：

- 读哪些视频、状态、动作、语言 key
- 每种模态的时间采样窗口
- action 是相对量还是绝对量
- action 属于 EEF 还是 joint space

## 基本结构

一个模态配置通常包含四段：

```python
{
    "video": ModalityConfig(...),
    "state": ModalityConfig(...),
    "action": ModalityConfig(...),
    "language": ModalityConfig(...),
}
```

最后用 `register_modality_config(...)` 注册到某个 `EmbodimentTag`。

## `delta_indices`

这是最重要的时间维配置：

- 视频、状态通常用 `[0]`
- 动作通常用 `list(range(0, N))`

如果你改了 action horizon，必须重新生成统计量：

```bash
python gr00t/data/stats.py --dataset-path <dataset_path> --embodiment-tag <embodiment_tag>
```

## `action_configs`

对 action 模态，`modality_keys` 里的每一项都要有一个一一对应的 `ActionConfig`。

关键参数：

- `rep`：`RELATIVE` 或 `ABSOLUTE`
- `type`：`EEF` 或 `NON_EEF`
- `format`：`DEFAULT`、`XYZ_ROT6D`、`XYZ_ROTVEC`
- `state_key`：当 action key 和参考 state key 不同名时显式指定

## 当前项目额外提醒

- 如果训练用的是 `config_subtask.py`，language key 应该是 `sub_task`
- 真机 AgiBot G01 当前主线控制是双臂 14 关节 + 左右夹爪

更完整的工程背景见：

- [../docs/zh-cn/principles-data-and-modality.md](../docs/zh-cn/principles-data-and-modality.md)
- [../PROJECT_HANDOFF_2026-06-08_CN.md](../PROJECT_HANDOFF_2026-06-08_CN.md)
