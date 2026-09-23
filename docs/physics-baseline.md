# Physics baseline

BARI_26 imports the physical environment from `qndus521-lab/BARI` commit `198a04b23aac03f2284aeeaeca346c014c7747d0` without modification.

The protected physics scope is the complete `bari2d/env/` package:

- `robot.py`: rectangular robot state, kinematics, climbing layer state
- `field.py`: randomized banks, gap geometry, and terrain queries
- `sensors.py`: planar and downward IR sensing
- `contact_model.py`: contact, friction, anchoring, breakage, and graph construction
- `load_evaluator.py`: fast and incremental structural load tests
- `bridge_env.py`: simulator lifecycle, local observations, reward, and terminal evaluation

The BARI_26 change set is confined to the actor architecture, training defaults, configurations, documentation, and tests. Before the initial release, SHA-256 hashes of every file in `bari2d/env/` were checked against the source snapshot and matched exactly.
