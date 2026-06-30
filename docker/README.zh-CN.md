# Docker 使用说明

本文件对应英文版 [README.md](README.md)。目标是在容器里快速得到一个带完整依赖的 GR00T 运行环境。

## 适用范围

- 顶层 `docker/Dockerfile` 同时支持 `x86_64` 和 `aarch64`
- 主要面向 dGPU 环境
- Jetson Thor、DGX Spark、Jetson Orin 还有各自的平台 Dockerfile

## 构建镜像

在仓库根目录执行：

```bash
bash docker/build.sh
```

默认基于 `nvidia/cuda:12.8.0-devel-ubuntu22.04` 构建，并把依赖装到 `/opt/gr00t-venv`。

## 运行容器

推荐先启动容器，再在容器内拉代码：

```bash
docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  gr00t
```

容器内执行：

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T /workspace/Isaac-GR00T
cd /workspace/Isaac-GR00T
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import gr00t; print('GR00T ready')"
```

## 重要约束

- 镜像里的全局 venv 默认已经启用，路径是 `/opt/gr00t-venv`
- 不要随手裸跑 `uv sync`，否则可能更新全局镜像环境
- 如果你需要为当前 checkout 建独立环境，用 `UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync`

更细的平台推理和部署说明见 [../scripts/deployment/README.zh-CN.md](../scripts/deployment/README.zh-CN.md)。
