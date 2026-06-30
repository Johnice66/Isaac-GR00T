# 真机部署指南

本文件对应英文版 [real_world_deployment.md](real_world_deployment.md)。它讲的是从数据采集到闭环部署的一整条工程路径。

## 推荐工作流

1. 准备硬件和传感器
2. 采集示教数据
3. 做时间同步、清洗和格式转换
4. 微调 GR00T
5. 做 open-loop 验证
6. 启动 server/client 闭环控制
7. 做真机安全测试和参数调优

## 部署架构

仓库推荐两种方式：

1. 同机推理
2. ZMQ server/client

## 当前 AgiBot G01 项目重点

- 只控制双臂 14 关节和左右夹爪
- 不控制底盘、头部、腰部
- 真机关节增量、速度、加速度都有硬约束
- 模型输出 action chunk 后，必须插值成更细粒度轨迹再发控制命令
- 如果是 subtask-conditioned checkpoint，语言 key 用 `sub_task`
