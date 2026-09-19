"""ActiveArena Astribot data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class AstribotHeadDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.camera_head"]
    state_keys = ["state.astribot"]
    action_keys = ["action.astribot"]
    action_key_dims = {"action.astribot": 18}
    state_key_dims = {"state.astribot": 18}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.astribot": "min_max"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={"action.astribot": "min_max"},
            ),
        ])


class AstribotHeadData50Config(AstribotHeadDataConfig):
    action_indices = list(range(50))


ROBOT_TYPE_CONFIG_MAP = {
    "activearena_astribot": AstribotHeadDataConfig(),
    "activearena_astribot50": AstribotHeadData50Config(),
    # Compatibility aliases for datasets prepared with pre-release examples.
    # New configs should use the ActiveArena names above.
    "robotwin_astribot": AstribotHeadDataConfig(),
    "robotwin_astribot50": AstribotHeadData50Config(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

ASTRIBOT_TASKS = [
    "beat_block_hammer_rotate_view",
    "blocks_ranking_rgb_fan_double",
    "blocks_ranking_rgb_rotate_view",
    "blocks_ranking_size_fan_double",
    "blocks_ranking_size_rotate_view",
    "check_block_color",
    "check_cola_color",
    "check_cola_date",
    "click_bell_rotate_view",
    "count_color_kinds_press_button",
    "count_random_object_press_button",
    "count_target_press_button",
    "match_backside_two_blocks",
    "move_pillbottle_pad_rotate_view",
    "move_stapler_pad_rotate_view",
    "place_a2b_left_rotate_view",
    "place_a2b_right_rotate_view",
    "place_cans_plasticbox_rotate_view",
    "place_container_plate_rotate_view",
    "place_empty_cup_rotate_view",
    "place_fan_rotate_view",
    "place_mouse_pad_rotate_view",
    "place_object_basket_fan_double",
    "place_object_scale_rotate_view",
    "place_object_stand_rotate_view",
    "place_shoe_rotate_view",
    "press_stapler_rotate_view",
    "put_block_on_upper_easy",
    "put_block_on_upper_hard",
    "rank_backside_rgb_blocks",
    "shake_bottle_horizontally_rotate_view",
    "shake_bottle_rotate_view",
    "stack_blocks_two_rotate_view",
    "stamp_seal_rotate_view",
    "turn_switch_rotate_view",
]


DISABLED_ASTRIBOT_TASKS = set()


def _astribot_task_weight(task: str) -> float:
    return 0.0 if task in DISABLED_ASTRIBOT_TASKS else 1.0


DATASET_NAMED_MIXTURES = {
    "activearena_astribot": [(task, _astribot_task_weight(task), "activearena_astribot") for task in ASTRIBOT_TASKS],
    "activearena_astribot_all": [(task, _astribot_task_weight(task), "activearena_astribot") for task in ASTRIBOT_TASKS],
    "activearena_astribot_all_50": [(task, _astribot_task_weight(task), "activearena_astribot50") for task in ASTRIBOT_TASKS],
    "activearena_astribot_task1": [("beat_block_hammer_rotate_view", 1.0, "activearena_astribot")],
    "activearena_astribot_task2": [
        ("place_a2b_left_rotate_view", 1.0, "activearena_astribot"),
        ("place_a2b_right_rotate_view", 1.0, "activearena_astribot"),
    ],
    "activearena_astribot_long": [("rank_backside_rgb_blocks", 1.0, "activearena_astribot")],
}

# Compatibility aliases. Keep these until old checkpoint/config bundles have
# migrated; they intentionally resolve to the frozen ActiveArena registry.
DATASET_NAMED_MIXTURES.update({
    "robotwin_astribot": [(task, weight, "robotwin_astribot") for task, weight, _ in DATASET_NAMED_MIXTURES["activearena_astribot"]],
    "robotwin_astribot_all": [(task, weight, "robotwin_astribot") for task, weight, _ in DATASET_NAMED_MIXTURES["activearena_astribot_all"]],
    "robotwin_astribot_all_50": [(task, weight, "robotwin_astribot50") for task, weight, _ in DATASET_NAMED_MIXTURES["activearena_astribot_all_50"]],
    "robotwin_astribot_task1": [(task, weight, "robotwin_astribot") for task, weight, _ in DATASET_NAMED_MIXTURES["activearena_astribot_task1"]],
    "robotwin_astribot_task2": [(task, weight, "robotwin_astribot") for task, weight, _ in DATASET_NAMED_MIXTURES["activearena_astribot_task2"]],
    "robotwin_astribot_long": [(task, weight, "robotwin_astribot") for task, weight, _ in DATASET_NAMED_MIXTURES["activearena_astribot_long"]],
})
