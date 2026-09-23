# BARI_26: Structured MAPPO

BARI_26 keeps the BARI 2D/2.5D physics, contact, anchoring, climbing, load evaluation, rewards, and curriculum unchanged. It makes the shared recurrent policy in `configs/structured_mappo.yaml` the default learning method.

The actor is decentralized at execution time: it never receives a map, absolute robot positions, the global contact graph, bridge capacity, or force distribution. A centralized graph critic uses the global simulator state only while training.

## Default learning architecture

```text
IR history ──► temporal encoder ──► traffic latent ───────┐
      └────► temporal encoder ──► connectivity latent ────┤
strain history ─► temporal encoder ──► mechanical latent ─┤──► GRU ─► shared policy
target load ────► goal encoder ──────► required-strength latent ┘
                         └── FiLM-modulates traffic and mechanical latents
```

The policy outputs one masked discrete action per robot: forward/backward motion and steering, climb, anchor, release, or idle. Traffic, connectivity, contact persistence, and force-trend heads are training-only auxiliary predictions; they are not exposed as global actor input.

## Installation

Create the isolated `bari_26` environment:

```bash
conda env create -f environment.yml
conda activate bari_26
python -m pip install -e '.[dev]'
pytest
```

All commands in this project should run inside that environment.

## Quick start

Run the deterministic simulator verification:

```bash
conda run -n bari_26 python scripts/verify_scripted.py
```

Train the default structured policy:

```bash
conda run -n bari_26 python scripts/train.py \
  --config configs/structured_mappo.yaml \
  --updates 1000
```

The legacy ablations remain available, including the earlier `bio_film` configuration:

```bash
conda run -n bari_26 python scripts/train.py \
  --config configs/bio_film.yaml \
  --updates 1000
```

Evaluate a checkpoint on stage-4 randomized gaps:

```bash
conda run -n bari_26 python scripts/evaluate.py \
  runs/structured_mappo/checkpoint_001000.pt \
  --config configs/structured_mappo.yaml --stage 4 --episodes 100
```

Render a manually assembled load-bearing fixture:

```bash
conda run -n bari_26 python scripts/visualize_episode.py \
  --rows 2 --output bridge.png
```

Manually select and control one robot per step:

```bash
conda run -n bari_26 python scripts/manual.py
```

## Simulator

`BridgeEnv` exposes a Gym-style `reset`/`step` interface without requiring Gymnasium. Each robot is an oriented rectangle with position, heading, scalar forward velocity, steering, head-lift state, anchor state, local strain, previous action, layer, and optional persistent latent. Planar motion uses bounded kinematics. Same-layer overlap produces a normal positional response, frictional velocity damping, and a strain-producing contact force. A climb command moves a robot one layer above a nearby same-layer robot. An elevated robot remains there only while its footprint overlaps a robot on the layer directly below; otherwise it falls to one layer above the highest overlapping lower robot, or to layer 0 when none overlaps. This support rule does not use moments or exempt anchored robots. Robots unsupported over the gap fall out of the active mechanical structure.

The field uses coordinates relative to a randomized gap normal. It supports gap width, orientation, sinusoidal boundary irregularity, bank shape, random initial poses, friction, anchor strength, mass, sensor noise, and actuator noise. Generated widths are capped to remain feasible for the configured field and robot population.

Anchors are persistent robot-robot or robot-bank edges. Configurable tension, compression, shear, rotational stiffness, and stochastic failure values determine their dynamic behavior. Excessive extension or shear breaks an anchor. Contact and bank edges remain transient.

### Local observation

Every continuous value is normalized. One actor observation contains:

- IR history from configurable front, back, left, right, and downward rays;
- local strain history;
- previous-action history;
- anchor state, head-lift state, velocity, steering, height layer, and fallen state;
- normalized target load;
- optional episode-persistent Gaussian latent.

The four planar IR rays ray-march against robot oriented boxes, circular obstacles, field limits, and bank/gap substrate transitions. The downward IR ray returns the vertical distance to a bank under the robot or to the highest lower-layer robot whose oriented footprint has positive-area overlap; maximum range means there is no supporting surface below (a cliff).

The centralized state and graph are separate methods used only by critics and diagnostics. Tests and model construction preserve this actor/critic boundary.

### Discrete action space

The baseline uses the permitted fully discrete fallback:

1. forward
2. backward
3. forward-left
4. forward-right
5. backward-left
6. backward-right
7. climb
8. anchor
9. release
10. idle

Masks remove only physically impossible actions: climb without nearby support, anchor without a mechanical partner, release while unanchored, motion while anchored, and all non-idle actions after falling. Masks do not encode behavioral or structural rules.

### Contact graph and capacity

The mechanical graph contains robot, left-bank, and right-bank nodes. Edges represent active contact, anchors, or bank support. Breadth-first connectivity supplies the binary span indicator. Continuous progress is the furthest left-connected robot measured across the local gap width.

Two replaceable evaluators implement structural scoring:

- `FastLoadEvaluator` uses graph maximum flow and path compliance on every construction step.
- `IncrementalLoadEvaluator` freezes the morphology and raises external load in configurable increments until contact loss, anchor failure, robot slip, excessive displacement/structural collapse, or loss of connectivity.

Both use a modular load-protocol strategy. `center_point` applies demand at the robot nearest the geometric center of the central spanning component. The default `uniform` protocol distributes demand across all independent spanning components and combines their capacities and stiffnesses. The interface is ready for moving, random-position, or multi-point load implementations.

This is a research proxy, not a validated structural-engineering solver. Edge capacity and compliance are meaningful and monotonic under added parallel load paths, but they do not reproduce continuum mechanics.

### Reward

Reward contains only functional terms:

```text
r_span = alpha * (progress[t+1] - progress[t])
r_mech = beta * (min(F_cap[t+1] / F_target, 1) - min(F_cap[t] / F_target, 1))
r_eff  = -time - movement_energy - newly_used_robots - new_anchors - collapse
```

Terminal success requires both bank-to-bank connectivity and an accurate terminal capacity at least as large as the sampled target. A connected weak structure and a strong disconnected structure both fail. No term rewards triangles, trusses, angles, paths, or named bridge forms.

## Policy and critics

The full actor has independent temporal traffic, connectivity, and mechanical branches plus a target-load encoder. Their latent outputs and robot internal state feed a compact GRU. Optional FiLM maps the target embedding to scale and bias traffic and mechanical latents. Connectivity, traffic, contact-persistence, and force-trend heads learn simulator targets during training only. Those targets are not actor inputs and are not needed for inference.

Set `model.architecture` to select an ablation:

- `mlp`
- `gru`
- `gru_traffic`
- `gru_traffic_connectivity`
- `bio`
- `bio_film`
- `bio_heterogeneity`

`bio_heterogeneity` consumes the per-episode latent; its standard deviation is configured by `environment.latent_sigma` and defaults to zero.

The baseline centralized critic is an MLP over flattened simulator state. The graph critic performs dependency-free message passing over training-only robot nodes and mechanical edges. Both produce one cooperative team value.

## MAPPO

The trainer collects team rewards with per-agent actions and log probabilities, computes GAE, and applies clipped PPO updates. Agent sequences remain time-major during updates so the shared GRU receives backpropagation through time; reset masks clear hidden state at episode boundaries. The centralized value loss and branch-specific auxiliary losses are optimized separately from decentralized action selection.

Curriculum stage advances when a rolling success window reaches the configured threshold:

1. straight, narrow gap and low load, with randomized non-overlapping robot poses on the starting bank;
2. randomized gap width and target load;
3. irregular and rotated boundaries with the full width range;
4. friction, mass, anchor strength, sensor noise, and actuator noise randomization.

## Outputs

`runs/<name>/episodes.jsonl` stores success, gap parameters, target and measured capacity, capacity ratio, construction time, used and anchored robot counts, energy proxy, anchor failures, fallen count, maximum progress, final contact graph, final morphology, action distribution, and available branch-latent statistics. Checkpoints include actor, critic, optimizers, configuration, and update number.

Visualization shows terrain, gap, robot rectangles and orientation, layer color, anchored edges, contact graph, planar and downward IR markers, target load, capacity, progress, and any load-bearing path returned by the terminal evaluator. `record_episode` can save a GIF for any callable policy.

## Verification

```bash
conda run -n bari2d pytest
```

Tests cover gap generation, movement and steering, IR detection, contact and climbing, anchoring and failure, graph connectivity, bank-to-bank spanning, fast and terminal load evaluation, monotonic strength of parallel structures, reward invariants, every actor ablation, both critics, and a full MAPPO update/checkpoint.

## Assumptions and current limits

- Height is a discrete layer, not full 3D rigid-body dynamics.
- Contact response is positional and quasi-static; it has no angular impulse solver.
- Structural capacity is a graph-flow/compliance approximation with incremental failure checks.
- One anchor is owned per robot. An anchor may attach to a bank or another robot.
- Robots share the target-load scalar as mission context; all other deployment observations are local.
- Evaluation generalization means held-out random seeds and stage-4 parameter distributions, not arbitrary out-of-distribution terrain.
- Learned morphology quality depends on training budget and hyperparameters. This repository provides a functioning research baseline, not pretrained evidence of convergence.

The structural evaluator, load protocol, dynamics, actor, and critic are separate modules so higher-fidelity physics or alternative learning algorithms can replace them without exposing global state to the decentralized actor.
