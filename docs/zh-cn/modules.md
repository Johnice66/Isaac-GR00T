# 重要模块说明

本文档按代码目录解释主要模块职责、关键文件和常见修改点。

## `gr00t/configs`

配置模块把模型、数据和训练参数统一成 dataclass。

关键文件：

- `gr00t/configs/base_config.py`：顶层 `Config`，包含 `model`、`data`、`training` 三块；负责加载/保存 YAML、校验数据集和 DeepSpeed 配置。
- `gr00t/configs/model/gr00t_n1d7.py`：`Gr00tN1d7Config`，定义 backbone、图像处理、动作头、flow matching、训练开关和多 embodiment 参数。
- `gr00t/configs/data/data_config.py`：`DataConfig`、`SingleDatasetConfig`，定义多数据集混合、shard、视频 backend、统计量覆盖策略。
- `gr00t/configs/data/embodiment_configs.py`：内置 embodiment 的 `MODALITY_CONFIGS`。
- `gr00t/configs/training/training_config.py`：训练 batch、学习率、checkpoint、wandb、DeepSpeed、精度等参数。

常见修改：

- 新增机器人时，通常新增或注册一个 modality config。
- 修改动作 horizon 时，要同步关注 `action.delta_indices`、模型 `action_horizon` 和推理侧执行 horizon。
- 修改状态/动作维度时，要确认不超过 `max_state_dim` 和 `max_action_dim`。

## `gr00t/data`

数据模块负责从 LeRobot 格式数据集中抽取 step、计算统计量、归一化状态动作，并为训练提供可混合的 IterableDataset。

关键文件：

- `gr00t/data/embodiment_tags.py`：`EmbodimentTag` 枚举和 tag 分类。
- `gr00t/data/dataset/lerobot_episode_loader.py`：按 episode 读取 LeRobot 数据和视频。
- `gr00t/data/dataset/sharded_single_step_dataset.py`：把 episode 拆成 timestep 样本，并组织为 shard。
- `gr00t/data/dataset/sharded_mixture_dataset.py`：按权重混合多个 sharded dataset，合并统计量，支持分布式 shard 切分。
- `gr00t/data/dataset/factory.py`：根据 `Config` 构造训练数据集。
- `gr00t/data/stats.py`：生成 `meta/stats.json` 和 `meta/relative_stats.json`。
- `gr00t/data/state_action/state_action_processor.py`：状态/动作归一化、sin/cos 编码、绝对/相对动作转换。
- `gr00t/data/collator/collators.py`：batch 拼装。

常见修改：

- 数据列名不匹配时，优先检查 `MODALITY_CONFIGS` 中的 `modality_keys`。
- 动作归一化异常时，检查 `meta/stats.json`、`meta/relative_stats.json` 和 `override_pretraining_statistics`。
- 分布式训练重复样本或漏样本时，重点看 `ShardedMixtureDataset` 的 seed 一致性。

## `gr00t/model`

模型模块实现 GR00T N1.7 的 backbone、动作头和 HuggingFace 注册。

关键文件：

- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`：主模型 `Gr00tN1d7` 和动作头 `Gr00tN1d7ActionHead`。
- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py`：processor 和 data collator。
- `gr00t/model/modules/qwen3_backbone.py`：Qwen3-VL/Cosmos backbone 封装。
- `gr00t/model/modules/dit.py`：DiT、AlternateVLDiT、timestep 条件 transformer。
- `gr00t/model/modules/embodiment_conditioned_mlp.py`：按 embodiment 选择权重的 MLP 和动作编码器。
- `gr00t/model/base/model_pipeline.py`：训练 pipeline 基类和 `BasicPipeline`。
- `gr00t/model/registry.py`：模型注册表。

常见修改：

- 冻结/解冻 backbone 或动作头时，修改 `tune_llm`、`tune_visual`、`tune_projector`、`tune_diffusion_model`、`tune_vlln`。
- 更换 VLM backbone 时，`get_backbone_cls()`、processor 和 config 需要一起适配。
- 扩展动作维度时，除了 config，也要关注 category-specific encoder/decoder 的权重维度。

## `gr00t/experiment`

训练模块把配置、模型、数据和 HuggingFace Trainer 串起来。

关键文件：

- `gr00t/experiment/launch_finetune.py`：单节点微调 CLI，负责把 `FinetuneConfig` 转换为完整 `Config`。
- `gr00t/experiment/launch_train.py`：更通用的训练入口。
- `gr00t/experiment/experiment.py`：训练主函数，初始化分布式、保存配置、创建 pipeline、构造 Trainer、保存最终模型。
- `gr00t/experiment/trainer.py`：`Gr00tTrainer`，对 IterableDataset 的 dataloader 和 resume seed 逻辑做了定制。
- `gr00t/experiment/utils.py`：checkpoint 格式和 best metric 回调。

训练输出通常包含：

- `experiment_cfg/config.yaml` 和 `conf.yaml`。
- `experiment_cfg/dataset_statistics.json`。
- `processor/`。
- HuggingFace checkpoint 文件。

## `gr00t/policy`

策略模块是推理侧的核心抽象。

关键文件：

- `gr00t/policy/policy.py`：`BasePolicy` 接口，定义 `check_observation()`、`_get_action()`、`check_action()`、`reset()`。
- `gr00t/policy/gr00t_policy.py`：`Gr00tPolicy`，加载模型和 processor，完成端到端推理。
- `gr00t/policy/server_client.py`：ZeroMQ `PolicyServer`/`PolicyClient`，使用 msgpack_numpy 序列化请求和响应。
- `gr00t/policy/replay_policy.py`：从数据集回放动作，适合调试接口。

观察输入的标准结构：

```python
{
    "video": {"camera_key": np.ndarray},      # (B, T, H, W, C), uint8
    "state": {"state_key": np.ndarray},       # (B, T, D), float32
    "language": {"language_key": [[text]]},   # batch x horizon
}
```

输出动作结构：

```python
{
    "action_key": np.ndarray,  # (B, T_action, D), float32
}
```

## `gr00t/eval`

评估模块包含数据集 open-loop 评估和仿真 closed-loop rollout。

关键文件：

- `gr00t/eval/open_loop_eval.py`：在 LeRobot 轨迹上按固定 action horizon 做预测，和 ground truth action 计算 MSE/MAE 并画图。
- `gr00t/eval/run_gr00t_server.py`：启动模型推理服务或 ReplayPolicy 服务。
- `gr00t/eval/rollout_policy.py`：创建 Gymnasium 仿真环境并执行 closed-loop rollout。
- `gr00t/eval/sim/`：LIBERO、SimplerEnv、RoboCasa 的环境注册和 wrapper。
- `gr00t/eval/real_robot/SO100/`：SO100 真实机器人评估入口。

open-loop 可以快速验证模型输出是否在数据分布附近；closed-loop 才能验证机器人/仿真任务是否成功。

## `scripts/deployment`

部署模块用于把 PyTorch checkpoint 转成可加速的推理 pipeline。

关键文件：

- `scripts/deployment/standalone_inference_script.py`：独立推理验证脚本，支持 PyTorch/TRT 模式。
- `scripts/deployment/export_onnx_n1d7.py`：导出 DiT-only、action-head 或 full-pipeline ONNX。
- `scripts/deployment/build_tensorrt_engine.py`：从 ONNX 构建 TRT engine。
- `scripts/deployment/build_trt_pipeline.py`：一体化执行 export、build、verify、benchmark。
- `scripts/deployment/trt_model_forward.py`：把 TRT engine 接入 `Gr00tPolicy` 的运行时 forward。
- `scripts/deployment/benchmark_inference.py`：性能测试。

部署模式：

- `pytorch`：最简单，便于调试。
- `torch_compile`：主要加速动作头。
- `tensorrt` / `n17_full_pipeline`：导出并加速 ViT、LLM、VL self-attention、state/action encoder、DiT、action decoder。

## `examples` 与 `getting_started`

`getting_started/` 偏教程，解释数据准备、policy API、新 embodiment 微调、真实部署等。

`examples/` 偏可运行样例，覆盖：

- `SO100`
- `DROID`
- `LIBERO`
- `SimplerEnv`
- `robocasa`
- `mask-guided-background-suppression`

开发新任务时，通常从相近 example 复制 modality config 和 README 流程，再改数据列和 embodiment tag。

## `tests`

测试覆盖项目主要边界：

- `tests/gr00t/data/`：数据集、统计量、embodiment tag、sharded dataset。
- `tests/gr00t/model/`：forward、processor、action head、可变图像尺寸。
- `tests/gr00t/policy/`：policy、GPU policy、policy service。
- `tests/scripts/deployment/`：TensorRT/ONNX/standalone inference 部署逻辑。
- `tests/getting_started/`：用户文档里的命令和示例是否仍有效。

常用测试命令：

```bash
python -m pytest tests/ -m "not gpu" -v --timeout=300
python -m pytest tests/ -m gpu -v --timeout=300
```

