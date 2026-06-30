# GR00T x AgiBot G01 Project Handoff

Generated: 2026-06-08  
Purpose: paste or provide this document when migrating to a new chat window.

## 1. Current Goal

Use NVIDIA Isaac GR00T N1.7 to fine-tune and deploy an AgiBot G01 real-robot policy.

Current deployment scope:

- Control: dual arms, 14 joints total, plus left/right gripper opening.
- No control needed: base, head, waist.
- Real-robot priority: choose the route that gives the best stable behavior on hardware, not only the best benchmark number.

## 2. Server Environment

Server workspace:

```bash
/root/gpufree-data/Isaac-GR00T
```

Hardware and runtime known from previous runs:

- GPU: NVIDIA A100-SXM4-80GB
- System memory: 1 TB
- CUDA/driver environment has cuDNN conflict risk if library path is not set.
- Storage: `/root/gpufree-data`, about 488 GB total.

Every new shell on the server should use:

```bash
cd /root/gpufree-data/Isaac-GR00T
source /root/gpufree-data/Isaac-GR00T/.venv/bin/activate
export LD_LIBRARY_PATH=/root/gpufree-data/Isaac-GR00T/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

Important:

- Do not use `uv run` on the server for training/deployment commands unless intentionally re-syncing dependencies.
- Previous `uv run` caused dependency churn and can hit unstable flash-attn download sources.
- Use the activated venv `python` directly.
- If using `tee`, remember it can mask the Python process exit code. Use `${PIPESTATUS[0]}` if checking failures.

## 3. Dataset and Training State

Two same-task datasets were trained together:

```bash
DATASET_A=/root/gpufree-data/Isaac-GR00T/dataset/task_2609
DATASET_B=/root/gpufree-data/Isaac-GR00T/dataset/task_2178
DATASET_PATH="${DATASET_A}:${DATASET_B}"
```

The multi-dataset training command used:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${DATASET_A}/meta/config.py" \
  --num-gpus 1 \
  --output-dir /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects \
  --max-steps 60000 \
  --save-steps 20000 \
  --save-total-limit 8 \
  --global-batch-size 32 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4 \
  2>&1 | tee train_two_collects.log
```

Training reached 60000 steps. Final log included:

```text
train_runtime: 44930.082
train_samples_per_second: 42.733
train_steps_per_second: 1.335
train_loss: 0.039473661675055824
```

The run ended with:

```text
No space left on device (os error 28)
```

This was a disk-full error, not a memory error. The actual `checkpoint-60000` model shards were verified complete with `safetensors.safe_open`.

Current usable model checkpoint:

```bash
/root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000
```

Verified files in `checkpoint-60000`:

- `model-00001-of-00003.safetensors`: OK, about 4.64 GB
- `model-00002-of-00003.safetensors`: OK, about 4.63 GB
- `model-00003-of-00003.safetensors`: OK, about 2.44 GB
- `model.safetensors.index.json`: present

Use this path for evaluation, server, TensorRT export/build, and deployment:

```bash
--model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000
```

Do not use the parent directory as the model path:

```bash
/root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects
```

## 4. Disk Cleanup Guidance

The filesystem was at about 96% usage:

```text
/root/gpufree-data: 488G total, 464G used, 24G available
```

Before more evaluation or TensorRT builds, free space should be increased.

Safe cleanup if not continuing training:

```bash
rm /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000/optimizer.pt
```

`optimizer.pt` is about 13 GB and is not needed for inference, open-loop eval, `torch.compile`, or TensorRT. It is only needed if resuming training from this checkpoint.

Before deleting anything else, inspect root-level leftovers:

```bash
ls -lh /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects | grep safetensors
```

Only delete root-level partial `model-*.safetensors` if they are outside `checkpoint-60000`.

## 5. Modality and Action Mapping

The project uses a custom AgiBot/G01 `NEW_EMBODIMENT` modality config.

Known correct mapping from the dataset metadata and prior validation:

State:

- `state[28:35]`: left arm joint position, 7 joints, radians
- `state[35:42]`: right arm joint position, 7 joints, radians
- `state[0:1]`: left effector/gripper position
- `state[1:2]`: right effector/gripper position

Action:

- `action[16:23]`: left arm joint target, 7 joints
- `action[23:30]`: right arm joint target, 7 joints
- `action[0:1]`: left effector/gripper target
- `action[1:2]`: right effector/gripper target

Important lesson from earlier debugging:

- Do not use `state[0:14]` as arm joints.
- Do not use `state[14:28]` as arm joints.
- Correct arm joint state is `state[28:42]`.
- Correct arm joint action is `action[16:30]`.

The config file used for the two-dataset training was:

```bash
/root/gpufree-data/Isaac-GR00T/dataset/task_2609/meta/config.py
```

Because `task_2609` and `task_2178` are assumed to have the same schema/config, only `DATASET_A`'s config was passed to training.

## 6. Multi-Dataset Training Implementation Note

The local repo's `gr00t/experiment/launch_finetune.py` supports colon-separated dataset paths using `os.pathsep`:

```python
dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]
```

and passes:

```python
"dataset_paths": dataset_paths
```

If a server copy fails by looking for a path like:

```text
task_2609:task_2178/meta/info.json
```

then that server copy is not splitting `--dataset-path` correctly and must be patched to match the local repo.

## 7. Acceleration Benchmark Results

Benchmark was run on A100 with 4 denoising steps.

Results:

| Mode | E2E | Frequency | Backbone | Action Head |
|---|---:|---:|---:|---:|
| PyTorch eager | 240 ms | 4.2 Hz | 52 ms | 90 ms |
| torch.compile | 175 ms | 5.7 Hz | 53 ms | 23 ms |
| TensorRT full pipeline | 210 ms | 4.8 Hz | 76 ms | 21 ms |

TensorRT verification passed:

```text
cosine=1.000000 PASS
```

Current recommendation:

- Default deployment route: `torch.compile`
- Keep TensorRT full pipeline as an optional route because it is valid, but in this benchmark it was slower E2E than `torch.compile`.

TensorRT artifacts are not stored in checkpoint directories. They are generated under:

```bash
/root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/
```

Expected subdirectories:

```bash
/root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/onnx
/root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/engines
```

`torch.compile` does not write a converted model to disk. It compiles at runtime in memory.

## 8. Server Script Current Local Status

Local file modified:

```bash
/Users/johnice/Desktop/nw/Isaac-GR00T/gr00t/eval/run_gr00t_server.py
```

Current local server supports:

- `--inference-mode pytorch`
- `--inference-mode torch_compile`
- `--inference-mode tensorrt`
- `--compile-mode`
- `--trt-engine-path`
- `--trt-mode`

Supported TensorRT modes in local server:

- `n17_full_pipeline`
- `vit_llm_only`
- `action_head`
- `dit_only`

Recommended server command for the new 60000-step model:

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
  --model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000 \
  --embodiment-tag new_embodiment \
  --host 0.0.0.0 \
  --port 5555 \
  --inference-mode torch_compile
```

Optional TensorRT server command:

```bash
CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
  --model-path /root/gpufree-data/Isaac-GR00T/checkpoints/task_2609_two_collects/checkpoint-60000 \
  --embodiment-tag new_embodiment \
  --host 0.0.0.0 \
  --port 5555 \
  --inference-mode tensorrt \
  --trt-engine-path /root/gpufree-data/Isaac-GR00T/gr00t_trt_deployment/engines \
  --trt-mode n17_full_pipeline
```

Server-side compile check to run on the server:

```bash
python -m py_compile gr00t/eval/run_gr00t_server.py
```

## 9. Real-Robot Client Current Local Status

Local files:

```bash
/Users/johnice/Desktop/nw/Isaac-GR00T/test_0526_n17_safe_v7.py
/Users/johnice/Desktop/nw/Isaac-GR00T/test_0526_n17_safe_v8.py
```

Current status:

- `v7` and `v8` are currently identical locally.
- They are untracked files in git.
- Client uses the official repo client:

```python
from gr00t.policy.server_client import PolicyClient
```

Client design:

- Acceleration route is selected on the server, not in the client.
- Client sends observation to `run_gr00t_server.py`.
- Client receives normalized actions and executes robot control.
- Client supports both split action keys and merged action keys:
  - `left_arm_joint_position` + `right_arm_joint_position`
  - or `joint_position`
  - optional `action.` prefix

Client default control safety:

- `--enable-control` is off by default.
- `--confirm-control` is on by default.
- First request timeout is longer because `torch.compile` first call may JIT.
- NaN/Inf action checks are included.
- Action execution includes trajectory blending/smoothing and joint limits.

Current client default task string in the local file:

```text
Fixed-point Non-generalized Door Opening
```

Important: earlier notes also mentioned `Partial Discharge Detection` as a task string. Before real-robot deployment, confirm the exact task text in the trained dataset metadata and pass it explicitly with `--task`. Do not rely on an old default if the task changed.

Recommended client command shape:

```bash
python test_0526_n17_safe_v7.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "Fixed-point Non-generalized Door Opening" \
  --enable-control \
  --control-freq 60 \
  --execute-horizon 8
```

For no-control connectivity testing:

```bash
python test_0526_n17_safe_v7.py \
  --policy-host 127.0.0.1 \
  --policy-port 5555 \
  --task "Fixed-point Non-generalized Door Opening" \
  --no-enable-control
```

If the true dataset task string is different, replace the `--task` value.

## 10. ROS Interfaces

Known G01/GDK topic mapping:

State:

- Arm state: `/hal/arm_joint_state`
- Left gripper state: `/hal/left_ee_data`
- Right gripper state: `/hal/right_ee_data`

Control:

- Arm command: `/wbc/arm_command`
- Left gripper command: `/wbc/left_ee_command`
- Right gripper command: `/wbc/right_ee_command`

Cameras:

- Head camera: `/camera/head_color`
- Left hand camera: `/camera/hand_left_color`
- Right hand camera: `/camera/hand_right_color`

Real robot arm state ordering:

- `joint_state.position[0:7]`: left arm
- `joint_state.position[7:14]`: right arm

This matches the semantic order of the dataset arm state after slicing, but the raw dataset indices are different.

## 11. Real-Robot Safety Constraints

Hard constraints from prior GDK notes:

- Single-frame joint change: less than `0.2618 rad` / 15 degrees
- Joint speed: less than `3 rad/s`
- Joint acceleration: less than `6 rad/s^2`

Implication:

- Do not send raw model chunks directly as sparse large jumps.
- Interpolate/smooth actions before publishing to `/wbc/arm_command`.
- Start with no-control and short/low-risk tests before enabling full real-robot control.

## 12. Recommended Next Steps

1. Free disk space on `/root/gpufree-data`.
2. Run open-loop evaluation for `checkpoint-60000`.
3. Confirm the exact task language string in dataset metadata for `task_2609` and `task_2178`.
4. Start server with `--inference-mode torch_compile`.
5. Run client with `--no-enable-control` first and verify:
   - server ping succeeds
   - modality config check succeeds
   - first `get_action` returns 14 arm joints plus two grippers
   - no NaN/Inf
6. Run real-robot low-risk control test:
   - short horizon
   - operator confirmation enabled
   - watch joint deltas, speed, and gripper mapping
7. Compare real-robot stability between:
   - `torch_compile`
   - TensorRT `n17_full_pipeline`

## 13. Local Repo Status

Current local git status at handoff time:

```text
 M gr00t/eval/run_gr00t_server.py
?? datasets/
?? test.py
?? test_0526_n17.py
?? test_0526_n17_safe_v7.py
?? test_0526_n17_safe_v8.py
```

Do not assume these local modifications already exist on the server. Before deployment, copy or patch the server files as needed.

Known local validation:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache python3 -m py_compile test_0526_n17_safe_v7.py
```

This passed locally.

The server file should be validated in the server Python 3.10 environment:

```bash
python -m py_compile gr00t/eval/run_gr00t_server.py
```

## 14. Open Questions To Verify Before Real Robot

- Exact task string to pass to the client for the `task_2609 + task_2178` trained model.
- Whether the server copy has the updated `run_gr00t_server.py` with acceleration flags.
- Whether the server copy has the multi-dataset `launch_finetune.py` fix.
- Whether gripper command values should be `0..120`, `0..1`, or another GDK-specific range on the current robot setup.
- Whether `checkpoint-60000/optimizer.pt` should be kept for possible resume or deleted to reclaim space.

