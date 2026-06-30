from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

task_5093_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "observation.images.top_head",
            "observation.images.hand_left",
            "observation.images.hand_right",
        ],
    ),

    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm_joint_position",
            "right_arm_joint_position",
            "left_effector_position",
            "right_effector_position",
        ],
    ),

    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=[
            "left_arm_joint_position",
            "right_arm_joint_position",
            "left_effector_position",
            "right_effector_position",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm_joint_position",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm_joint_position",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["sub_task"],
    ),
}

register_modality_config(
    task_5093_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
