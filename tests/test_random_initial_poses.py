from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.contact_model import oriented_boxes_overlap
from bari2d.env.field import LEFT_BANK


def test_initial_robots_are_scattered_on_left_bank(environment_config) -> None:
    environment_config.robot.count = 20
    env = BridgeEnv(environment_config)
    env.reset(seed=13)

    positions = np.stack([robot.position for robot in env.robots])
    headings = np.array([robot.theta for robot in env.robots])
    assert np.unique(positions, axis=0).shape[0] == environment_config.robot.count
    assert np.unique(np.round(headings, decimals=6)).size > 1
    for robot in env.robots:
        assert all(env.field.bank_at(corner) == LEFT_BANK for corner in robot.corners(env.config.robot))
    for robot_index, robot in enumerate(env.robots):
        assert not any(
            oriented_boxes_overlap(robot, other, env.config.robot)
            for other in env.robots[robot_index + 1 :]
        )


def test_initial_poses_change_with_reset_seed(environment_config) -> None:
    env = BridgeEnv(environment_config)
    env.reset(seed=13)
    first_positions = np.stack([robot.position for robot in env.robots])
    env.reset(seed=14)
    second_positions = np.stack([robot.position for robot in env.robots])

    assert not np.allclose(first_positions, second_positions)
