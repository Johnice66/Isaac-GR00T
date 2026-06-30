# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555
TRT_MODES = {"n17_full_pipeline", "vit_llm_only", "action_head", "dit_only"}


def _normalize_inference_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("-", "_")
    aliases = {
        "pytorch": "pytorch",
        "eager": "pytorch",
        "torch_compile": "torch_compile",
        "torch.compile": "torch_compile",
        "compile": "torch_compile",
        "tensorrt": "tensorrt",
        "trt": "tensorrt",
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported inference mode: {mode}. "
            "Use one of: pytorch, torch_compile, tensorrt."
        )
    return aliases[normalized]


def _setup_gr00t_policy_acceleration(policy: Gr00tPolicy, config: "ServerConfig") -> None:
    inference_mode = _normalize_inference_mode(config.inference_mode)
    if inference_mode == "pytorch":
        print("  Inference mode: PyTorch eager")
        return

    if inference_mode == "torch_compile":
        import torch

        action_model = getattr(policy.model.action_head, "model", None)
        if action_model is None:
            raise ValueError("torch_compile requires the PyTorch action_head.model to be present.")
        action_model.forward = torch.compile(action_model.forward, mode=config.compile_mode)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print(f"  Inference mode: torch.compile ({config.compile_mode})")
        print("  Compiled: policy.model.action_head.model.forward")
        return

    trt_mode = config.trt_mode.strip().lower().replace("-", "_")
    if trt_mode not in TRT_MODES:
        raise ValueError(f"Unsupported TRT mode: {config.trt_mode}. Use one of: {sorted(TRT_MODES)}")
    if config.trt_engine_path is None:
        raise ValueError("--trt-engine-path is required when --inference-mode=tensorrt")

    trt_engine_path = Path(config.trt_engine_path).expanduser()
    if not trt_engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine path does not exist: {trt_engine_path}")

    deploy_dir = Path(__file__).resolve().parents[2] / "scripts" / "deployment"
    if str(deploy_dir) not in sys.path:
        sys.path.insert(0, str(deploy_dir))
    from trt_model_forward import setup_tensorrt_engines

    setup_tensorrt_engines(policy, str(trt_engine_path), mode=trt_mode)
    print(f"  Inference mode: TensorRT ({trt_mode})")
    print(f"  TRT engines: {trt_engine_path}")


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    inference_mode: str = "pytorch"
    """Inference mode for Gr00tPolicy: pytorch, torch_compile, or tensorrt"""

    compile_mode: str = "max-autotune"
    """torch.compile mode used when --inference-mode=torch_compile"""

    trt_engine_path: str | None = None
    """Path to TensorRT engine directory when --inference-mode=tensorrt"""

    trt_mode: str = "n17_full_pipeline"
    """TensorRT mode: n17_full_pipeline, vit_llm_only, action_head, or dit_only"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Requested inference mode: {config.inference_mode}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        _setup_gr00t_policy_acceleration(policy, config)
    elif config.dataset_path is not None:
        if _normalize_inference_mode(config.inference_mode) != "pytorch":
            raise ValueError(
                "--inference-mode only applies to Gr00tPolicy. "
                "ReplayPolicy supports PyTorch/replay mode only."
            )
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                with open(config.modality_config_path, "r") as f:
                    raw = json.load(f)
                # ReplayPolicy expects ModalityConfig instances, not raw dicts.
                modality_configs = {k: ModalityConfig(**v) for k, v in raw.items()}
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
