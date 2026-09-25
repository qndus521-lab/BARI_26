from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import DiscreteAction, RobotState
from bari2d.utils.config import EnvironmentConfig, GapConfig
from bari2d.env.field import GapGenerator


def test_constrained_bank_anchor_requires_a_nearby_cliff_edge() -> None:
    config = EnvironmentConfig()
    config.robot.count = 3
    config.contact.anchor_requires_edge_or_contact = True
    config.contact.anchor_edge_band = 1.2
    env = BridgeEnv(config)
    env.reset(seed=4)
    left_boundary, _ = env.field.boundaries(0.0)
    edge = env.field.center + env.field.normal * left_boundary
    env.set_robot_states(
        [
            RobotState(0, edge - env.field.normal * 0.25, env.field.orientation),
            RobotState(1, edge - env.field.normal * 3.0 + env.field.tangent * 3.0, env.field.orientation),
            RobotState(2, edge - env.field.normal * 3.0 - env.field.tangent * 3.0, env.field.orientation),
        ]
    )
    assert env.action_masks()[0, DiscreteAction.ANCHOR]

    env.robots[0].position = edge - env.field.normal * 1.8
    assert not env.action_masks()[0, DiscreteAction.ANCHOR]


def test_idle_penalty_applies_only_after_beacon_grace_period() -> None:
    config = EnvironmentConfig()
    config.robot.count = 1
    config.max_steps = 10
    config.beacon.enabled = True
    config.reward.idle_penalty = 0.1
    config.reward.idle_grace_steps = 1
    env = BridgeEnv(config)
    env.reset(seed=4)
    env._beacon_source_id = 0
    env._beacon_features[0, 0] = 1.0
    actions = np.array([int(DiscreteAction.IDLE)])

    _, _, _, _, first_info = env.step(actions)
    _, _, _, _, second_info = env.step(actions)

    assert first_info["reward_components"]["idle"] == 0.0
    assert second_info["reward_components"]["idle"] == -0.1


def test_curriculum_gap_widths_are_selected_by_stage() -> None:
    generator = GapGenerator(GapConfig(curriculum_widths=[1.2, 1.7, 2.3, 3.0]))
    rng = np.random.default_rng(3)

    assert generator.generate(rng, stage=1).gap_width == 1.2
    assert generator.generate(rng, stage=2).gap_width == 1.7
    assert generator.generate(rng, stage=4).gap_width == 3.0
