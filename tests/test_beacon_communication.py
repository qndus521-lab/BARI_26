from __future__ import annotations

import numpy as np

from bari2d.env.bridge_env import BridgeEnv
from bari2d.env.robot import RobotState


def test_first_cliff_detector_relays_a_local_beacon(environment_config) -> None:
    environment_config.robot.count = 4
    environment_config.beacon.enabled = True
    environment_config.beacon.communication_range = 2.1
    environment_config.beacon.max_hops = 3
    env = BridgeEnv(environment_config)
    env.reset(seed=7)
    env.set_robot_states(
        [
            RobotState(0, np.array([7.0, 5.0]), 0.0),
            RobotState(1, np.array([5.0, 5.0]), 0.0),
            RobotState(2, np.array([3.0, 5.0]), 0.0),
            RobotState(3, np.array([1.0, 5.0]), 0.0),
        ]
    )
    env._beacon_source_id = None
    env._beacon_features.fill(0.0)

    activated = env._update_beacon()
    features = env.observations()[:, env.layout.beacon]

    assert activated
    assert env._beacon_source_id == 0
    assert np.all(features[:, 0] == 1.0)
    assert features[0, 5] == 1.0
    assert np.all(features[1:, 5] == 0.0)
    assert features[1, 1] > 0.9
    assert features[1, 4] < features[2, 4] < features[3, 4]


def test_beacon_is_absent_from_legacy_observations(environment_config) -> None:
    env = BridgeEnv(environment_config)
    observations, _ = env.reset(seed=7)

    assert env.layout.beacon.stop == env.layout.beacon.start
    assert observations.shape[1] == 39
