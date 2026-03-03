# IsaacLab OpenPI Dense Reward Guide

This document describes the dense reward design used by
`IsaaclabStackCubeEnv` for OpenPI RL training.

## Goal

Task prompt:

`Pick up the red cube and place it on top of the blue cube, then pick up the green cube and place it on top of the red cube.`

Dense reward is used to provide stage-level learning signals before full task success.

## Config Keys

Dense reward is configured in:

- [`examples/embodiment/config/env/isaaclab_stack_cube.yaml`](../examples/embodiment/config/env/isaaclab_stack_cube.yaml)

Relevant fields:

```yaml
reward_coef: 1.0
reward_cfg:
  enable_dense_reward: True
  enable_stage_gating: True
  drop_height_threshold: -0.05
  rewards:
    grasp_red: 0.10
    stack_red_blue: 0.25
    grasp_green: 0.10
    success: 1.00
    fail_drop: -0.30
```

## Reward Terms

Per environment step, reward is:

`step_reward = reward_coef * (stage_rewards + terminal_rewards)`

with stage rewards:

- `grasp_red`: one-time reward when `subtask_terms.grasp_1` becomes true.
- `stack_red_blue`: one-time reward when `subtask_terms.stack_1` becomes true.
- `grasp_green`: one-time reward when `subtask_terms.grasp_2` becomes true.

terminal rewards:

- `success`: one-time reward on success termination.
- `fail_drop`: one-time penalty when termination is triggered by cube dropping.

## Success vs Failure

The IsaacLab environment has multiple termination reasons. In RLinf stack-cube wrapper:

- **Success termination**: `terminations == True` and no cube is below `drop_height_threshold`.
- **Drop failure termination**: `terminations == True` and at least one cube is below `drop_height_threshold`.

This avoids treating every termination as success.

## Stage Gating

If `enable_stage_gating=True`, rewards are unlocked in sequence:

1. `grasp_red`
2. `stack_red_blue`
3. `grasp_green`
4. terminal `success`

Each stage is edge-triggered and paid at most once per episode.

If `enable_stage_gating=False`, each stage can trigger independently (still one-time per episode).

## Logged Metrics

In `infos["episode"]`, the following reward components are logged per env:

- `reward/grasp_red`
- `reward/stack_red_blue`
- `reward/grasp_green`
- `reward/success`
- `reward/fail_drop`

These are visible in TensorBoard/W&B through RLinf metric logging.

## Sparse Fallback

Set `reward_cfg.enable_dense_reward=False` for sparse training.

Sparse mode now rewards **only true success** (not all terminations).

## Recommended 2-GPU Profile Note

For the current senior profile:

- `env.train.total_num_envs = 48`
- `env.train.max_steps_per_rollout_epoch = 440`
- `actor.model.num_action_chunks = 10`
- `algorithm.rollout_epoch = 10`
- `actor.global_batch_size = 10560`

`rollout_size = 48 * (440 / 10) * 10 = 21120`, exactly divisible by `10560`.
