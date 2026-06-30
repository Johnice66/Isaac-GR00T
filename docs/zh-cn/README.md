# Isaac GR00T N1.7 中文项目文档

这组文档面向准备阅读、改造、微调或部署本仓库的开发者。英文 `README.md` 更偏快速上手和官方使用说明；这里补充的是中文的工程总览、模块边界和关键机制原理。

## 阅读路径

建议按下面顺序阅读：

1. [项目总览](project-overview.md)：先建立全局地图，理解项目目标、目录结构、主流程和常用入口。
2. [模块说明](modules.md)：逐个认识配置、数据、模型、训练、推理、评估、部署模块。
3. [数据与模态原理](principles-data-and-modality.md)：理解 LeRobot 数据、embodiment tag、模态配置、统计量和相对动作。
4. [模型原理](principles-model.md)：理解 Qwen3-VL/Cosmos backbone、DiT 动作头和 flow matching。
5. [训练与推理原理](principles-training-inference.md)：理解微调、推理服务、open-loop/closed-loop 评估如何串起来。
6. [部署与加速原理](principles-deployment.md)：理解 PyTorch、`torch.compile`、ONNX、TensorRT 的部署路径。
7. [近期项目修改日志](change-log-2026-06-11.md)：记录近期针对 AgiBot G01 真机训练、subtask、server 加速和 client 的项目修改。
8. [GitHub 发布整理日志](change-log-2026-06-30.md)：记录数据集排除规则和项目分支发布范围。

## 文档定位

这些文档基于当前仓库代码编写，重点覆盖：

- `gr00t/` 主包的核心实现。
- `examples/`、`getting_started/` 和 `scripts/deployment/` 中和主流程直接相关的入口。
- `tests/` 中反映项目边界和回归保障的测试分类。

外部依赖目录 `external_dependencies/` 体量很大，主要作为仿真/评估依赖存在，本文档只在评估部分说明其角色，不逐文件展开。

## 维护规则

每次修改项目代码后，都需要同步更新修改记录。小改动可以追加到 [近期项目修改日志](change-log-2026-06-11.md)，较大的独立主题可以新增 `change-log-YYYY-MM-DD.md` 并在本页链接。
