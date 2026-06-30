"""
AGIBOT G1 / G01 GR00T N1.7 real-robot client.

This client talks to ``gr00t/eval/run_gr00t_server.py`` over the GR00T
PolicyServer protocol. Acceleration is selected on the server side; the client
only needs the host/port and sends observations in the trained modality format.

Torch compile server:

    CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
      --model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609/checkpoint-30000 \
      --embodiment-tag new_embodiment \
      --host 0.0.0.0 \
      --port 5555 \
      --inference-mode torch_compile

TensorRT full-pipeline server:

    CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
      --model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609/checkpoint-30000 \
      --embodiment-tag new_embodiment \
      --host 0.0.0.0 \
      --port 5555 \
      --inference-mode tensorrt \
      --trt-engine-path /root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/engines \
      --trt-mode n17_full_pipeline

Recommended client:

    python test_0526_n17_safe_v7.py \
      --policy-host 127.0.0.1 \
      --policy-port 5555 \
      --task "Fixed-point Non-generalized Door Opening" \
      --language-key auto \
      --enable-control \
      --control-freq 60 \
      --execute-horizon 8
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

try:
    from genie_msgs.msg import EndState
except ImportError:
    EndState = None


REQUIRED_ACTION_KEYS = ("joint_position", "left_effector_position", "right_effector_position")
EXPECTED_STATE_KEYS = {
    "left_arm_joint_position",
    "right_arm_joint_position",
    "left_effector_position",
    "right_effector_position",
}
EXPECTED_SPLIT_ACTION_KEYS = EXPECTED_STATE_KEYS
EXPECTED_COMBINED_ACTION_KEYS = {
    "joint_position",
    "left_effector_position",
    "right_effector_position",
}
DEFAULT_LANGUAGE_KEY = "annotation.human.task_description"


@dataclass
class ClientConfig:
    policy_host: str = "127.0.0.1"
    policy_port: int = 5555
    request_timeout_ms: int = 20000
    first_request_timeout_ms: int = 120000
    task: str = "Fixed-point Non-generalized Door Opening"
    language_key: str = "auto"

    enable_control: bool = False
    confirm_control: bool = True
    max_cycles: int = 0

    control_freq: int = 60
    model_fps: float = 30.0
    execute_horizon: int = 8
    interp_factor: int = 3
    infer_wait_timeout: float = 20.0
    sensor_ready_timeout: float = 30.0
    wait_position_timeout: float = 15.0
    ros_startup_sleep: float = 1.5

    pos_tolerance: float = 0.03
    final_hold_steps: int = 5
    blend_steps: int = 14
    ema_alpha: float = 1.0
    max_step_delta: float = 0.2618
    max_joint_speed: float = 3.0
    max_joint_accel: float = 6.0
    gripper_scale: float = 1.0
    gripper_offset: float = 0.0
    gripper_min: float = 0.0
    gripper_max: float = 120.0

    client_resize: bool = False
    image_width: int = 640
    image_height: int = 480
    log_every_n_cycles: int = 30
    qos_depth: int = 1
    check_modality_config: bool = True

    head_camera_topic: str = "/camera/head_color"
    left_hand_camera_topic: str = "/camera/hand_left_color"
    right_hand_camera_topic: str = "/camera/hand_right_color"
    arm_state_topic: str = "/hal/arm_joint_state"
    left_ee_state_topic: str = "/hal/left_ee_data"
    right_ee_state_topic: str = "/hal/right_ee_data"
    arm_command_topic: str = "/wbc/arm_command"
    left_ee_command_topic: str = "/wbc/left_ee_command"
    right_ee_command_topic: str = "/wbc/right_ee_command"
    left_gripper_joint_name: str = "left_gripper_joint1"
    right_gripper_joint_name: str = "right_gripper_joint1"

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.image_width, self.image_height)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ClientConfig()
    parser = argparse.ArgumentParser(
        description="AgiBot G01 real-robot client for a GR00T N1.7 PolicyServer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    policy = parser.add_argument_group("Policy server")
    policy.add_argument("--policy-host", default=defaults.policy_host)
    policy.add_argument("--policy-port", type=int, default=defaults.policy_port)
    policy.add_argument("--request-timeout-ms", type=int, default=defaults.request_timeout_ms)
    policy.add_argument(
        "--first-request-timeout-ms",
        type=int,
        default=defaults.first_request_timeout_ms,
        help="Longer timeout for the first get_action call; torch.compile may JIT here.",
    )
    policy.add_argument(
        "--skip-modality-check",
        dest="check_modality_config",
        action="store_false",
        help="Skip get_modality_config sanity checks against the AgiBot state/action keys.",
    )

    task = parser.add_argument_group("Task and images")
    task.add_argument("--task", default=defaults.task)
    task.add_argument(
        "--subtask",
        dest="task",
        help="Alias for --task when the checkpoint was trained with language key 'sub_task'.",
    )
    task.add_argument(
        "--language-key",
        default=defaults.language_key,
        help=(
            "Language key to send. Use 'auto' to read the server modality config. "
            "Subtask-trained checkpoints usually expect 'sub_task'."
        ),
    )
    task.add_argument("--image-width", type=int, default=defaults.image_width)
    task.add_argument("--image-height", type=int, default=defaults.image_height)

    control = parser.add_argument_group("Control")
    control.set_defaults(
        enable_control=defaults.enable_control,
        confirm_control=defaults.confirm_control,
        check_modality_config=defaults.check_modality_config,
        client_resize=defaults.client_resize,
    )
    control.add_argument("--enable-control", dest="enable_control", action="store_true")
    control.add_argument("--no-enable-control", dest="enable_control", action="store_false")
    control.add_argument("--confirm-control", dest="confirm_control", action="store_true")
    control.add_argument("--no-confirm-control", dest="confirm_control", action="store_false")
    control.add_argument("--max-cycles", type=int, default=defaults.max_cycles)
    control.add_argument("--control-freq", type=int, default=defaults.control_freq)
    control.add_argument(
        "--model-fps",
        type=float,
        default=defaults.model_fps,
        help="FPS used by the training dataset. Used to keep action chunk timing correct.",
    )
    control.add_argument(
        "--execute-horizon",
        type=int,
        default=defaults.execute_horizon,
        help="Number of model action steps to execute before replanning. Use 0 for full horizon.",
    )
    control.add_argument(
        "--interp-factor",
        type=int,
        default=defaults.interp_factor,
        help="Legacy compatibility argument. Timing now follows --model-fps and --control-freq.",
    )
    control.add_argument("--infer-wait-timeout", type=float, default=defaults.infer_wait_timeout)
    control.add_argument("--sensor-ready-timeout", type=float, default=defaults.sensor_ready_timeout)
    control.add_argument("--wait-position-timeout", type=float, default=defaults.wait_position_timeout)
    control.add_argument("--ros-startup-sleep", type=float, default=defaults.ros_startup_sleep)
    control.add_argument("--pos-tolerance", type=float, default=defaults.pos_tolerance)
    control.add_argument("--final-hold-steps", type=int, default=defaults.final_hold_steps)
    control.add_argument("--blend-steps", type=int, default=defaults.blend_steps)
    control.add_argument("--ema-alpha", type=float, default=defaults.ema_alpha)
    control.add_argument("--max-step-delta", type=float, default=defaults.max_step_delta)
    control.add_argument("--max-joint-speed", type=float, default=defaults.max_joint_speed)
    control.add_argument("--max-joint-accel", type=float, default=defaults.max_joint_accel)
    control.add_argument("--gripper-scale", type=float, default=defaults.gripper_scale)
    control.add_argument("--gripper-offset", type=float, default=defaults.gripper_offset)
    control.add_argument("--gripper-min", type=float, default=defaults.gripper_min)
    control.add_argument("--gripper-max", type=float, default=defaults.gripper_max)
    control.add_argument("--log-every-n-cycles", type=int, default=defaults.log_every_n_cycles)

    ros = parser.add_argument_group("ROS topics")
    ros.add_argument("--qos-depth", type=int, default=defaults.qos_depth)
    ros.add_argument(
        "--client-resize",
        dest="client_resize",
        action="store_true",
        help="Resize camera images in this client before sending to the policy server.",
    )
    ros.add_argument(
        "--no-client-resize",
        dest="client_resize",
        action="store_false",
        help="Send native camera frames and let the GR00T processor apply saved transforms.",
    )
    ros.add_argument("--head-camera-topic", default=defaults.head_camera_topic)
    ros.add_argument("--left-hand-camera-topic", default=defaults.left_hand_camera_topic)
    ros.add_argument("--right-hand-camera-topic", default=defaults.right_hand_camera_topic)
    ros.add_argument("--arm-state-topic", default=defaults.arm_state_topic)
    ros.add_argument("--left-ee-state-topic", default=defaults.left_ee_state_topic)
    ros.add_argument("--right-ee-state-topic", default=defaults.right_ee_state_topic)
    ros.add_argument("--arm-command-topic", default=defaults.arm_command_topic)
    ros.add_argument("--left-ee-command-topic", default=defaults.left_ee_command_topic)
    ros.add_argument("--right-ee-command-topic", default=defaults.right_ee_command_topic)
    ros.add_argument("--left-gripper-joint-name", default=defaults.left_gripper_joint_name)
    ros.add_argument("--right-gripper-joint-name", default=defaults.right_gripper_joint_name)
    return parser


def validate_config(config: ClientConfig) -> None:
    positive_ints = {
        "policy_port": config.policy_port,
        "request_timeout_ms": config.request_timeout_ms,
        "first_request_timeout_ms": config.first_request_timeout_ms,
        "control_freq": config.control_freq,
        "interp_factor": config.interp_factor,
        "final_hold_steps": config.final_hold_steps,
        "image_width": config.image_width,
        "image_height": config.image_height,
        "qos_depth": config.qos_depth,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive; got {value}")

    non_negative_ints = {
        "blend_steps": config.blend_steps,
        "log_every_n_cycles": config.log_every_n_cycles,
        "max_cycles": config.max_cycles,
        "execute_horizon": config.execute_horizon,
    }
    for name, value in non_negative_ints.items():
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative; got {value}")

    positive_floats = {
        "infer_wait_timeout": config.infer_wait_timeout,
        "sensor_ready_timeout": config.sensor_ready_timeout,
        "wait_position_timeout": config.wait_position_timeout,
        "pos_tolerance": config.pos_tolerance,
        "model_fps": config.model_fps,
        "max_step_delta": config.max_step_delta,
        "max_joint_speed": config.max_joint_speed,
        "max_joint_accel": config.max_joint_accel,
    }
    for name, value in positive_floats.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive; got {value}")

    if config.ros_startup_sleep < 0:
        raise ValueError(f"--ros-startup-sleep must be non-negative; got {config.ros_startup_sleep}")
    if not 0.0 < config.ema_alpha <= 1.0:
        raise ValueError(f"--ema-alpha must be in (0, 1]; got {config.ema_alpha}")
    if config.gripper_min > config.gripper_max:
        raise ValueError("--gripper-min must be <= --gripper-max")
    if not config.language_key.strip():
        raise ValueError("--language-key must not be empty")
    config.language_key = config.language_key.strip()


def parse_args(argv: Optional[Sequence[str]] = None) -> ClientConfig:
    parser = build_arg_parser()
    config = ClientConfig(**vars(parser.parse_args(argv)))
    validate_config(config)
    return config


def key_tail(key: Any) -> str:
    return str(key).split(".")[-1].lower()


def canonical_key(key: Any) -> str:
    lk = str(key).lower()
    tail = key_tail(lk)
    if tail == "joint_position" or lk.endswith(".joint_position"):
        return "joint_position"
    if tail == "left_effector_position" or lk.endswith(".left_effector_position"):
        return "left_effector_position"
    if tail == "right_effector_position" or lk.endswith(".right_effector_position"):
        return "right_effector_position"
    return str(key)


def normalize_action(action: Dict[str, Any]) -> Dict[str, np.ndarray]:
    norm = {}
    for key, value in action.items():
        raw_key = str(key)
        stripped_key = raw_key[len("action.") :] if raw_key.startswith("action.") else raw_key
        norm[canonical_key(stripped_key)] = np.asarray(value, dtype=np.float32)

    if "joint_position" not in norm:
        if "left_arm_joint_position" in norm and "right_arm_joint_position" in norm:
            norm["joint_position"] = np.concatenate(
                [norm["left_arm_joint_position"], norm["right_arm_joint_position"]],
                axis=-1,
            )
        else:
            raise KeyError(f"缺少 joint_position, keys: {list(norm.keys())}")

    if norm["joint_position"].shape[-1] != 14:
        raise ValueError(f"joint_position 需要 14 维, 实际 {norm['joint_position'].shape}")

    for key in REQUIRED_ACTION_KEYS:
        if key not in norm:
            raise KeyError(f"缺少 {key}")
        if not np.all(np.isfinite(norm[key])):
            raise ValueError(f"{key} 包含 NaN/Inf")
    return norm


def bezier_chunk(
    chunk: Dict[str, np.ndarray],
    keys: Sequence[str],
    substeps: int,
    ei: float = 0.3,
    eo: float = 0.3,
) -> Dict[str, np.ndarray]:
    out = {}
    for key in keys:
        if key not in chunk:
            continue
        arr = np.asarray(chunk[key][0], dtype=np.float32)
        horizon, dim = arr.shape
        total = horizon * substeps
        res = np.zeros((total, dim), dtype=np.float32)
        for i in range(horizon - 1):
            p0, p1 = arr[i], arr[i + 1]
            delta = p1 - p0
            c0 = p0 + delta * ei
            c1 = p1 - delta * eo
            idx = i * substeps
            for j in range(substeps):
                t = j / substeps
                u = 1 - t
                res[idx + j] = (
                    u * u * u * p0
                    + 3 * u * u * t * c0
                    + 3 * u * t * t * c1
                    + t * t * t * p1
                )
        res[(horizon - 1) * substeps :] = arr[-1]
        out[key] = res[np.newaxis, ...]
    return out


def resample_sequence_by_rate(
    seq: np.ndarray,
    model_fps: float,
    control_freq: int,
) -> np.ndarray:
    """Resample a model-rate action sequence to the controller publish rate."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected sequence shape (T, D), got {seq.shape}")
    if seq.shape[0] <= 1:
        return seq.copy()

    output_steps = max(1, int(round(seq.shape[0] * float(control_freq) / float(model_fps))))
    if output_steps == seq.shape[0]:
        return seq.copy()

    src_t = np.arange(seq.shape[0], dtype=np.float32)
    dst_t = np.linspace(0, seq.shape[0] - 1, output_steps, dtype=np.float32)
    out = np.empty((output_steps, seq.shape[1]), dtype=np.float32)
    for dim in range(seq.shape[1]):
        out[:, dim] = np.interp(dst_t, src_t, seq[:, dim]).astype(np.float32)
    return out


def resample_action_chunk(
    chunk: Dict[str, np.ndarray],
    keys: Sequence[str],
    execute_horizon: int,
    model_fps: float,
    control_freq: int,
) -> Dict[str, np.ndarray]:
    out = {}
    for key in keys:
        if key not in chunk:
            continue
        arr = np.asarray(chunk[key][0], dtype=np.float32)
        horizon = arr.shape[0]
        prefix = horizon if execute_horizon <= 0 else min(execute_horizon, horizon)
        out[key] = resample_sequence_by_rate(
            arr[:prefix],
            model_fps=model_fps,
            control_freq=control_freq,
        )[np.newaxis, ...]
    return out


class EMAFilter:
    def __init__(self, dim: int, alpha: float = 0.3):
        self.alpha = alpha
        self.v = np.zeros(dim, dtype=np.float32)
        self.ok = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if not self.ok:
            self.v = x.copy()
            self.ok = True
            return self.v
        self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

    def reset(self, x: Optional[np.ndarray] = None) -> None:
        if x is None:
            self.v[:] = 0
            self.ok = False
        else:
            self.v = np.asarray(x, dtype=np.float32).copy()
            self.ok = True


class JointSafetyLimiter:
    def __init__(
        self,
        dim: int,
        dt: float,
        max_step_delta: float,
        max_speed: float,
        max_accel: float,
    ):
        self.dt = float(dt)
        self.max_delta = min(float(max_step_delta), float(max_speed) * self.dt)
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.last_pos = np.zeros(dim, dtype=np.float32)
        self.last_vel = np.zeros(dim, dtype=np.float32)
        self.ok = False

    def reset(self, pos: np.ndarray) -> None:
        self.last_pos = np.asarray(pos, dtype=np.float32).copy()
        self.last_vel = np.zeros_like(self.last_pos)
        self.ok = True

    def __call__(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        if not self.ok:
            self.reset(target)
            return self.last_pos.copy()

        desired_delta = np.clip(target - self.last_pos, -self.max_delta, self.max_delta)
        desired_vel = np.clip(desired_delta / self.dt, -self.max_speed, self.max_speed)
        dv = np.clip(
            desired_vel - self.last_vel,
            -self.max_accel * self.dt,
            self.max_accel * self.dt,
        )
        vel = np.clip(self.last_vel + dv, -self.max_speed, self.max_speed)
        delta = np.clip(vel * self.dt, -self.max_delta, self.max_delta)
        cmd = self.last_pos + delta

        self.last_pos = cmd.astype(np.float32)
        self.last_vel = (delta / self.dt).astype(np.float32)
        return self.last_pos.copy()


class _ClosedSocket:
    def close(self, *args: Any, **kwargs: Any) -> None:
        return None


class _ClosedContext:
    def term(self) -> None:
        return None


class NormalizedPolicyClient:
    """Small adapter around the repo PolicyClient that normalizes action names."""

    def __init__(self, host: str, port: int, timeout_ms: int):
        self._client = Gr00tPolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)

    def ping(self) -> bool:
        return self._client.ping()

    def get_modality_config(self) -> Dict[str, Any]:
        return self._client.get_modality_config()

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        action, _info = self._client.get_action(obs)
        return normalize_action(action)

    def close(self) -> None:
        socket = getattr(self._client, "socket", None)
        context = getattr(self._client, "context", None)
        try:
            if socket is not None:
                socket.close(0)
        except Exception:
            pass
        try:
            if context is not None:
                context.term()
        except Exception:
            pass
        self._client.socket = _ClosedSocket()
        self._client.context = _ClosedContext()


def modality_keys(modality_config: Dict[str, Any], section: str) -> set:
    return set(modality_key_list(modality_config, section))


def modality_key_list(modality_config: Dict[str, Any], section: str) -> list:
    section_config = modality_config.get(section, None)
    if section_config is None:
        return []
    if isinstance(section_config, dict):
        return list(section_config.get("modality_keys", []))
    return list(getattr(section_config, "modality_keys", []) or [])


def resolve_language_key(modality_config: Dict[str, Any], requested_key: str) -> str:
    language_keys = modality_key_list(modality_config, "language")
    if requested_key != "auto":
        if language_keys and requested_key not in language_keys:
            raise ValueError(
                f"--language-key={requested_key!r} 不在 server language keys 中: {language_keys}. "
                "如果不确定，请使用 --language-key auto。"
            )
        return requested_key

    if "sub_task" in language_keys:
        return "sub_task"
    if DEFAULT_LANGUAGE_KEY in language_keys:
        return DEFAULT_LANGUAGE_KEY
    if language_keys:
        return language_keys[0]
    return DEFAULT_LANGUAGE_KEY


def maybe_check_modality_config(client: NormalizedPolicyClient, config: ClientConfig) -> None:
    if not config.check_modality_config:
        print("  Modality check: skipped")
        if config.language_key == "auto":
            config.language_key = DEFAULT_LANGUAGE_KEY
            print(f"  Language key:   {config.language_key} (fallback)")
        return
    try:
        modality_config = client.get_modality_config()
    except Exception as exc:
        print(f"  Modality check: warning, get_modality_config failed: {exc}")
        if config.language_key == "auto":
            config.language_key = DEFAULT_LANGUAGE_KEY
            print(f"  Language key:   {config.language_key} (fallback)")
        return

    state_keys = modality_keys(modality_config, "state")
    action_keys = modality_keys(modality_config, "action")
    language_keys = modality_key_list(modality_config, "language")
    config.language_key = resolve_language_key(modality_config, config.language_key)
    state_ok = EXPECTED_STATE_KEYS.issubset(state_keys)
    action_ok = (
        EXPECTED_SPLIT_ACTION_KEYS.issubset(action_keys)
        or EXPECTED_COMBINED_ACTION_KEYS.issubset(action_keys)
    )

    if state_ok and action_ok:
        print("  Modality check: OK")
        print(f"    state keys:  {sorted(state_keys)}")
        print(f"    action keys: {sorted(action_keys)}")
        print(f"    language keys: {language_keys}")
        print(f"  Language key:   {config.language_key}")
        return

    print("  Modality check: warning, server modality does not look like AgiBot dual-arm")
    print(f"    state keys:  {sorted(state_keys)}")
    print(f"    action keys: {sorted(action_keys)}")
    print(f"    language keys: {language_keys}")
    print(f"  Language key:   {config.language_key}")
    if not state_ok:
        print(f"    missing state keys: {sorted(EXPECTED_STATE_KEYS - state_keys)}")
    if not action_ok:
        split_missing = EXPECTED_SPLIT_ACTION_KEYS - action_keys
        combined_missing = EXPECTED_COMBINED_ACTION_KEYS - action_keys
        print(f"    missing split action keys: {sorted(split_missing)}")
        print(f"    missing combined action keys: {sorted(combined_missing)}")


class RobotNode(Node):
    def __init__(self, config: ClientConfig):
        super().__init__("agibot_g01_gr00t_n17_client")
        self.config = config
        self.bridge = CvBridge()
        self.enable_control = config.enable_control
        self.latest_images = {}
        self.latest_joint_state = np.zeros(14, dtype=np.float32)
        self.latest_left_gripper = 0.0
        self.latest_right_gripper = 0.0
        self.last_joint_pos = None
        self.last_left_gripper = None
        self.last_right_gripper = None
        self._ready = {"head": False, "hand_left": False, "hand_right": False, "joint": False}

        qos = config.qos_depth
        self.create_subscription(Image, config.head_camera_topic, self._cb_head, qos)
        self.create_subscription(Image, config.left_hand_camera_topic, self._cb_left, qos)
        self.create_subscription(Image, config.right_hand_camera_topic, self._cb_right, qos)
        self.create_subscription(JointState, config.arm_state_topic, self._cb_joint, qos)

        if EndState is not None:
            self.create_subscription(EndState, config.left_ee_state_topic, self._cb_gl, qos)
            self.create_subscription(EndState, config.right_ee_state_topic, self._cb_gr, qos)
        else:
            self.get_logger().warning("genie_msgs.msg.EndState not available; gripper states stay 0.0")

        if self.enable_control:
            self.arm_pub = self.create_publisher(JointState, config.arm_command_topic, qos)
            self.lg_pub = self.create_publisher(JointState, config.left_ee_command_topic, qos)
            self.rg_pub = self.create_publisher(JointState, config.right_ee_command_topic, qos)
            self._arm_msg = JointState()
            self._lg_msg = JointState()
            self._rg_msg = JointState()
            self._lg_msg.name = [config.left_gripper_joint_name]
            self._rg_msg.name = [config.right_gripper_joint_name]

        if config.ros_startup_sleep > 0:
            time.sleep(config.ros_startup_sleep)

    def _decode_img(self, msg: Image) -> Optional[np.ndarray]:
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception:
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception:
                return None

    def _cb_head(self, msg: Image) -> None:
        img = self._decode_img(msg)
        if img is not None:
            self.latest_images["top_head"] = img
            self._ready["head"] = True

    def _cb_left(self, msg: Image) -> None:
        img = self._decode_img(msg)
        if img is not None:
            self.latest_images["hand_left"] = img
            self._ready["hand_left"] = True

    def _cb_right(self, msg: Image) -> None:
        img = self._decode_img(msg)
        if img is not None:
            self.latest_images["hand_right"] = img
            self._ready["hand_right"] = True

    def _cb_joint(self, msg: JointState) -> None:
        if len(msg.position) >= 14:
            self.latest_joint_state = np.array(msg.position[:14], dtype=np.float32)
            self._ready["joint"] = True

    def _cb_gl(self, msg: Any) -> None:
        if len(msg.end_state) > 0:
            self.latest_left_gripper = float(msg.end_state[0].position)

    def _cb_gr(self, msg: Any) -> None:
        if len(msg.end_state) > 0:
            self.latest_right_gripper = float(msg.end_state[0].position)

    @property
    def ready(self) -> bool:
        return all(self._ready.values())

    def get_obs(self) -> Optional[Dict[str, Any]]:
        if not self.ready:
            return None

        video = {}
        for key in ("top_head", "hand_left", "hand_right"):
            if key not in self.latest_images:
                return None
            img = self.latest_images[key]
            if self.config.client_resize:
                img = cv2.resize(
                    img,
                    self.config.image_size,
                    interpolation=cv2.INTER_LINEAR,
                )
            video[f"observation.images.{key}"] = img[np.newaxis, np.newaxis, ...]

        joints = self.latest_joint_state
        state = {
            "left_arm_joint_position": joints[:7][np.newaxis, np.newaxis, :],
            "right_arm_joint_position": joints[7:14][np.newaxis, np.newaxis, :],
            "left_effector_position": np.array(
                [[[self.latest_left_gripper]]], dtype=np.float32
            ),
            "right_effector_position": np.array(
                [[[self.latest_right_gripper]]], dtype=np.float32
            ),
        }
        language = {self.config.language_key: [[self.config.task]]}
        return {"video": video, "state": state, "language": language}

    def execute_step(self, joint_pos: np.ndarray, left_gripper: float, right_gripper: float) -> None:
        if not self.enable_control:
            return

        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        if joint_pos.shape != (14,) or not np.all(np.isfinite(joint_pos)):
            self.get_logger().error(
                f"非法关节命令: shape={joint_pos.shape}, finite={np.all(np.isfinite(joint_pos))}"
            )
            return

        if not (math.isfinite(float(left_gripper)) and math.isfinite(float(right_gripper))):
            self.get_logger().error(
                f"非法夹爪命令: lg={left_gripper}, rg={right_gripper}"
            )
            return

        stamp = self.get_clock().now().to_msg()
        self._arm_msg.header.stamp = stamp
        self._arm_msg.position = joint_pos.tolist()
        self.arm_pub.publish(self._arm_msg)

        self._lg_msg.header.stamp = stamp
        self._lg_msg.position = [float(left_gripper)]
        self.lg_pub.publish(self._lg_msg)

        self._rg_msg.header.stamp = stamp
        self._rg_msg.position = [float(right_gripper)]
        self.rg_pub.publish(self._rg_msg)


_INFER_CLIENT = None


def init_infer_client(host: str, port: int, timeout_ms: int) -> None:
    # ZMQ REQ sockets are not shared across threads.
    global _INFER_CLIENT
    _INFER_CLIENT = NormalizedPolicyClient(host=host, port=port, timeout_ms=timeout_ms)


def infer_job(obs: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    global _INFER_CLIENT
    try:
        if _INFER_CLIENT is None:
            raise RuntimeError("background PolicyClient is not initialized")
        return _INFER_CLIENT.get_action(obs)
    except Exception as exc:
        print(f"  [推理失败] {exc}")
        return None


def close_infer_client() -> None:
    global _INFER_CLIENT
    if _INFER_CLIENT is not None:
        _INFER_CLIENT.close()
        _INFER_CLIENT = None


def print_config(config: ClientConfig) -> None:
    print("=" * 72)
    print("  GR00T N1.7 AgiBot real-robot client")
    print("  wait到位后拍观测 -> 异步推理 -> blend执行")
    print("=" * 72)
    print(f"  Policy server: {config.policy_host}:{config.policy_port}")
    print("  Server mode:    selected on run_gr00t_server.py")
    print(f"  Task:           {config.task}")
    print(f"  Language key:   {config.language_key}")
    print(f"  Control:        {'ENABLED' if config.enable_control else 'disabled'}")
    print(f"  Control freq:   {config.control_freq} Hz")
    print(f"  Model FPS:      {config.model_fps:g}")
    print(
        "  Execute chunk:  "
        f"{'full horizon' if config.execute_horizon <= 0 else str(config.execute_horizon) + ' model steps'}"
    )
    print(
        "  Image resize:   "
        + (
            f"{config.image_width}x{config.image_height}"
            if config.client_resize
            else "native camera frames"
        )
    )
    print(
        "  Joint limits:   "
        f"step<={config.max_step_delta:g} rad, "
        f"vel<={config.max_joint_speed:g} rad/s, accel<={config.max_joint_accel:g} rad/s^2"
    )
    print(f"  Arm state:      {config.arm_state_topic}")
    print(f"  Arm command:    {config.arm_command_topic}")


def print_action_summary(action: Dict[str, np.ndarray], prefix: str = "Action") -> None:
    horizon = action["joint_position"].shape[1]
    joints = action["joint_position"]
    lg = action["left_effector_position"]
    rg = action["right_effector_position"]
    print(f"  {prefix}: horizon={horizon}, joint_shape={action['joint_position'].shape}")
    print(
        "  "
        f"joint_range=[{float(np.min(joints)):+.3f}, {float(np.max(joints)):+.3f}], "
        f"left_gripper=[{float(np.min(lg)):.3f}, {float(np.max(lg)):.3f}], "
        f"right_gripper=[{float(np.min(rg)):.3f}, {float(np.max(rg)):.3f}]"
    )


def print_observation_summary(obs: Dict[str, Any]) -> None:
    print("  观测图像:")
    for key, value in obs.get("video", {}).items():
        arr = np.asarray(value)
        print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
    for key, value in obs.get("language", {}).items():
        print(f"  language.{key}: {value[0][0]!r}")


def map_gripper(value: float, config: ClientConfig) -> float:
    value = float(value) * config.gripper_scale + config.gripper_offset
    return float(np.clip(value, config.gripper_min, config.gripper_max))


def connect_and_check_server(config: ClientConfig) -> NormalizedPolicyClient:
    client = NormalizedPolicyClient(
        host=config.policy_host,
        port=config.policy_port,
        timeout_ms=config.request_timeout_ms,
    )
    if not client.ping():
        client.close()
        raise RuntimeError(
            f"Policy server ping timeout at {config.policy_host}:{config.policy_port}. "
            "确认 run_gr00t_server.py 已启动，并且 host/port 对得上。"
        )
    maybe_check_modality_config(client, config)
    return client


def first_get_action(config: ClientConfig, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    client = NormalizedPolicyClient(
        host=config.policy_host,
        port=config.policy_port,
        timeout_ms=config.first_request_timeout_ms,
    )
    try:
        return client.get_action(obs)
    except Exception as exc:
        raise RuntimeError(
            f"{exc}\n"
            "首次推理失败。如果 server 使用 --inference-mode torch_compile，"
            "首次 get_action 可能在做 JIT；可以增大 --first-request-timeout-ms。"
        ) from exc
    finally:
        client.close()


def wait_for_sensors(node: RobotNode, timeout: float) -> bool:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.ready:
            print(f"  传感器就绪 ({time.perf_counter() - start:.1f}s)")
            return True
    print(f"  传感器超时，ready={node._ready}")
    return False


def run_control_loop(
    config: ClientConfig,
    node: RobotNode,
    current_action: Dict[str, np.ndarray],
) -> int:
    executor = ThreadPoolExecutor(
        max_workers=1,
        initializer=init_infer_client,
        initargs=(config.policy_host, config.policy_port, config.request_timeout_ms),
    )
    ema = EMAFilter(14, alpha=config.ema_alpha)
    dt = 1.0 / config.control_freq
    limiter = JointSafetyLimiter(
        dim=14,
        dt=dt,
        max_step_delta=config.max_step_delta,
        max_speed=config.max_joint_speed,
        max_accel=config.max_joint_accel,
    )
    limiter.reset(node.latest_joint_state)
    ema.reset(node.latest_joint_state)
    action_keys = ["joint_position", "left_effector_position", "right_effector_position"]
    infer_future = None
    cycle = 1
    cycles_completed = 0

    print("\n" + "=" * 72)
    print("  主循环: blend+执行 | wait到位后拍观测+异步推理")
    print("=" * 72)

    try:
        while rclpy.ok():
            cycle_start = time.perf_counter()

            if infer_future is not None:
                wait_start = time.perf_counter()
                while not infer_future.done() and rclpy.ok():
                    if node.last_joint_pos is not None:
                        node.execute_step(
                            node.last_joint_pos,
                            node.last_left_gripper,
                            node.last_right_gripper,
                        )
                    rclpy.spin_once(node, timeout_sec=0.02)
                    time.sleep(0.02)
                    if time.perf_counter() - wait_start > config.infer_wait_timeout:
                        print("  [错误] 后台推理超时，停止控制，避免重复执行旧轨迹")
                        return cycles_completed
                try:
                    result = infer_future.result()
                except Exception as exc:
                    print(f"  [错误] 获取后台推理结果失败: {exc}")
                    return cycles_completed
                finally:
                    infer_future = None
                if result is None:
                    print("  [错误] 后台推理返回 None，停止控制，避免重复执行旧轨迹")
                    return cycles_completed
                current_action = result

            action = current_action
            if not all(key in action for key in action_keys):
                print(f"  [错误] action 缺少必要字段: {list(action.keys())}")
                return cycles_completed
            for key in action_keys:
                if not np.all(np.isfinite(action[key])):
                    print(f"  [错误] action {key} 包含 NaN/Inf，停止控制")
                    return cycles_completed

            horizon = action["joint_position"].shape[1]
            if horizon <= 0:
                print("  [错误] action horizon <= 0，停止控制")
                return cycles_completed

            interp = resample_action_chunk(
                action,
                action_keys,
                execute_horizon=config.execute_horizon,
                model_fps=config.model_fps,
                control_freq=config.control_freq,
            )
            traj = interp["joint_position"][0]
            total = traj.shape[0]
            if total <= 0:
                print("  [错误] 重采样后轨迹为空，停止控制")
                return cycles_completed

            last_cmd = limiter.last_pos.copy()
            last_lg = map_gripper(interp["left_effector_position"][0, 0, 0], config)
            last_rg = map_gripper(interp["right_effector_position"][0, 0, 0], config)

            if node.last_joint_pos is not None and config.blend_steps > 0:
                t_vals = np.linspace(0, 1, config.blend_steps + 2)[1:-1]
                smooth = t_vals * t_vals * (3 - 2 * t_vals)
                jp0 = node.last_joint_pos
                jp1 = traj[0]
                for i in range(config.blend_steps):
                    blend = (1 - smooth[i]) * jp0 + smooth[i] * jp1
                    last_cmd = limiter(ema(blend))
                    last_lg = node.last_left_gripper
                    last_rg = node.last_right_gripper
                    node.execute_step(last_cmd, last_lg, last_rg)
                    time.sleep(dt)
                    if i % 4 == 0:
                        rclpy.spin_once(node, timeout_sec=0.001)

            seq_start = time.perf_counter()
            for step_idx in range(total):
                last_lg = map_gripper(interp["left_effector_position"][0, step_idx, 0], config)
                last_rg = map_gripper(interp["right_effector_position"][0, step_idx, 0], config)
                last_cmd = limiter(ema(traj[step_idx]))
                node.execute_step(last_cmd, last_lg, last_rg)
                wait_until = seq_start + (step_idx + 1) * dt
                remaining = wait_until - time.perf_counter()
                if remaining > 0.002:
                    time.sleep(remaining)
                if step_idx % 20 == 0:
                    rclpy.spin_once(node, timeout_sec=0.001)

            target_pos = traj[-1].astype(np.float32).copy()
            target_lg = map_gripper(interp["left_effector_position"][0, -1, 0], config)
            target_rg = map_gripper(interp["right_effector_position"][0, -1, 0], config)
            for _ in range(config.final_hold_steps):
                last_cmd = limiter(ema(target_pos))
                last_lg = target_lg
                last_rg = target_rg
                node.execute_step(last_cmd, last_lg, last_rg)
                time.sleep(dt)
                rclpy.spin_once(node, timeout_sec=0.001)

            wait_target = last_cmd.copy()
            node.last_joint_pos = last_cmd.copy()
            node.last_left_gripper = last_lg
            node.last_right_gripper = last_rg

            wait_ok = False
            err = float("inf")
            wait_deadline = time.perf_counter() + config.wait_position_timeout
            while time.perf_counter() < wait_deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
                joints = node.latest_joint_state
                if joints is not None:
                    err = float(np.max(np.abs(joints - wait_target)))
                    if err < config.pos_tolerance:
                        wait_ok = True
                        break
                time.sleep(0.02)
            if not wait_ok:
                print(f"  [警告] wait到位超时，max_err={err:.4f}，仍将使用当前真实观测继续")

            next_obs = node.get_obs()
            if next_obs is not None:
                infer_future = executor.submit(infer_job, next_obs)
            else:
                print("  [错误] wait后观测失败，停止控制")
                return cycles_completed

            cycle_time = time.perf_counter() - cycle_start
            joints = node.latest_joint_state
            should_log = cycle <= 3 or (
                config.log_every_n_cycles > 0 and cycle % config.log_every_n_cycles == 0
            )
            if should_log:
                print(
                    f"  #{cycle:03d}: {cycle_time * 1000:.0f}ms | "
                    f"exec={min(config.execute_horizon or horizon, horizon)}/{horizon}, "
                    f"ctrl={total} | "
                    f"J0={joints[0]:+.3f} J7={joints[7]:+.3f}"
                )

            cycles_completed += 1
            if config.max_cycles > 0 and cycles_completed >= config.max_cycles:
                print(f"  达到 --max-cycles={config.max_cycles}，停止")
                return cycles_completed
            cycle += 1
    finally:
        if infer_future is not None:
            infer_future.cancel()
        try:
            executor.submit(close_infer_client).result(timeout=2.0)
        except Exception:
            pass
        executor.shutdown(wait=False)


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    print_config(config)

    if config.enable_control and config.confirm_control:
        print("\n[!] 控制已启用，将向真机发布 /wbc 命令。")
        if input("  继续? (yes/no): ").strip().lower() != "yes":
            return

    ros_started = False
    node = None
    health_client = None
    cycles_completed = 0
    try:
        print("\n[1] ROS2...")
        rclpy.init()
        ros_started = True
        node = RobotNode(config)

        print("[2] 推理服务器...")
        health_client = connect_and_check_server(config)
        print("  Policy server: OK")
        health_client.close()
        health_client = None

        print("[3] 传感器...")
        if not wait_for_sensors(node, config.sensor_ready_timeout):
            return

        obs = node.get_obs()
        if obs is None:
            print("  观测失败")
            return
        print_observation_summary(obs)

        joints = node.latest_joint_state
        print(f"  当前: J0={joints[0]:+.3f} J7={joints[7]:+.3f}")

        print("[4] 首次推理...")
        first_start = time.perf_counter()
        current_action = first_get_action(config, obs)
        print(f"  首次推理: {(time.perf_counter() - first_start) * 1000:.0f}ms")
        print_action_summary(current_action, prefix="首次 action")

        if not config.enable_control:
            print("  控制未启用：已完成 ROS 观测、server ping、首次 get_action，退出。")
            return

        cycles_completed = run_control_loop(config, node, current_action)
    except KeyboardInterrupt:
        print("\n  中断")
    except Exception as exc:
        print(f"\n  [错误] {exc}")
    finally:
        if health_client is not None:
            health_client.close()
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if ros_started and rclpy.ok():
            rclpy.shutdown()
        if config.enable_control:
            print(f"  共 {cycles_completed} 个推理周期")


if __name__ == "__main__":
    main()
