# 机器人数据准备指南

本文件对应英文版 [data_preparation.md](data_preparation.md)。目标是把你的机器人数据整理成 GR00T 可直接读取的 LeRobot v2 变体格式。

## 核心结论

如果你已经有 LeRobot v2 数据集，最关键的补充就是：

- 在 `meta/` 下增加 `modality.json`

如果你是 LeRobot v3 数据集，先用仓库脚本转回 v2：

- [../scripts/lerobot_conversion/convert_v3_to_v2.py](../scripts/lerobot_conversion/convert_v3_to_v2.py)

## 目录结构

最小可用结构：

```text
dataset_root/
├── meta/
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── info.json
│   └── modality.json
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
└── videos/
    └── chunk-000/
        └── observation.images.xxx/
            └── episode_000000.mp4
```

## parquet 必要字段

每个 episode parquet 至少应包含：

- `observation.state`
- `action`
- `timestamp`
- `task_index`
- `episode_index`
- `index`
- `next.reward`
- `next.done`

如果有语言或其他标注，使用 `annotation.*` 前缀。

## `meta/tasks.jsonl`

这里保存文本任务内容，parquet 里通常只存索引。

## `meta/episodes.jsonl`

这里保存 episode 级元数据，比如 `episode_index`、`tasks`、`length`。

## `meta/modality.json`

这是 GR00T 相比标准 LeRobot 额外要求的核心文件。它负责说明：

- `observation.state` 里每一段索引代表什么
- `action` 里每一段索引代表什么
- 视频 key 如何映射
- annotation key 如何映射

基本结构：

```json
{
  "state": {"<state_key>": {"start": 0, "end": 7}},
  "action": {"<action_key>": {"start": 0, "end": 7}},
  "video": {"<new_key>": {"original_key": "observation.images.xxx"}},
  "annotation": {"<annotation_key>": {}}
}
```

## 当前项目额外提醒

在 AgiBot G01 项目里，`modality.json` 的索引正确性非常关键。不要凭数值猜测，应以 `meta/info.json` 或已确认的配置为准。相关背景见：

- [../PROJECT_HANDOFF_2026-06-08_CN.md](../PROJECT_HANDOFF_2026-06-08_CN.md)
- [../docs/zh-cn/principles-data-and-modality.md](../docs/zh-cn/principles-data-and-modality.md)
