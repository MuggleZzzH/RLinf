# OpenPI + RLinf 完整数据流转换文档

## 📋 目录

1. [概览](#概览)
2. [SFT训练流程](#sft训练流程)
3. [RL训练流程](#rl训练流程)
4. [PPO训练机制](#ppo训练机制)
5. [IsaacLab接入指南](#isaaclab接入指南)
6. [常见问题](#常见问题)

---

## 🎯 概览

### 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    【阶段1：SFT训练】                          │
│  Libero数据集 → 数据转换 → 模型训练 → 保存模型                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    【阶段2：RL训练】                           │
│  Rollout → 计算Advantage → PPO训练 → 模型更新                │
└─────────────────────────────────────────────────────────────┘
```

### 关键文件映射

| 功能 | 文件路径 |
|------|---------|
| 数据配置 | `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py` |
| Policy转换 | `rlinf/models/embodiment/openpi/policies/libero_policy.py` |
| 模型主体 | `rlinf/models/embodiment/openpi_action_model.py` |
| 环境包装 | `rlinf/envs/libero/libero_env.py` |
| PPO算法 | `rlinf/algorithms/losses.py`, `advantages.py` |
| Actor训练 | `rlinf/workers/actor/fsdp_actor_worker.py` |

---

## 📚 SFT训练流程

### 1. 数据加载

**文件**: `rlinf/workers/sft/fsdp_sft_worker.py` (第68-88行)

```python
def build_dataloader(self):
    config = get_openpi_config(
        self.cfg.actor.model.openpi.config_name,  # "pi0_libero"
        model_path=self.cfg.actor.model.model_path,
        batch_size=self.cfg.actor.micro_batch_size * self._world_size,
    )
    
    data_loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=True
    )
    return data_loader, data_loader.data_config()
```

**数据格式**:
```python
{
    "image": np.ndarray,              # [H, W, 3] uint8
    "wrist_image": np.ndarray,        # [H, W, 3] uint8
    "state": np.ndarray,              # [8] float32
    "actions": np.ndarray,            # [action_horizon, 7] float32
    "prompt": str,                    # "pick up the red block"
}
```

### 2. Repack Transform

**文件**: `libero_dataconfig.py` (第32-48行)

```python
repack_transform = _transforms.RepackTransform({
    "observation/image": "image",              # 数据集键 → 统一键
    "observation/wrist_image": "wrist_image",
    "observation/state": "state",
    "actions": "actions",
    "prompt": "prompt",
})
```

**作用**: 将数据集的键名统一为推理时使用的格式

### 3. Data Transform

**文件**: `libero_policy.py` (第38-82行)

```python
def __call__(self, data: dict) -> dict:
    base_image = _parse_image(data["observation/image"])
    wrist_image = _parse_image(data["observation/wrist_image"])
    
    inputs = {
        "state": data["observation/state"],  # [8] float
        "image": {
            "base_0_rgb": base_image,              # [H, W, 3] uint8
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": np.zeros_like(base_image),  # padding
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,  # 标记padding
        },
        "actions": data["actions"],  # [action_horizon, 7]
        "prompt": data["prompt"],
    }
    return inputs
```

### 4. 模型训练

**文件**: `fsdp_sft_worker.py` (第90-157行)

```python
def run_training(self):
    observation, actions = next(self.data_iter)
    
    observation = jax.tree.map(
        lambda x: torch.as_tensor(x, device=self.device).contiguous(),
        observation,
    )
    
    with self.amp_context:
        losses = self.model(
            forward_type="sft_forward",
            data={"observation": observation, "actions": actions},
        )
        loss = losses.mean()
    
    loss.backward()
    self.optimizer_step()
```

---

## 🎮 RL训练流程

### 阶段1: Rollout（环境交互）

#### 1.1 环境观测获取

**文件**: `libero_env.py` (第344-368行)

```python
def _wrap_obs(self, obs_list):
    # 提取图像和状态
    for obs in obs_list:
        images_and_states = {
            "full_image": get_libero_image(obs),      # [H, W, 3]
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate([
                obs["robot0_eef_pos"],                # [3]
                quat2axisangle(obs["robot0_eef_quat"]),  # [3]
                obs["robot0_gripper_qpos"],           # [1]
            ])  # [7]
        }
    
    # 转换为tensor并堆叠
    obs = {
        "main_images": torch.stack([...]),      # [num_envs, H, W, 3]
        "wrist_images": torch.stack([...]),     # [num_envs, H, W, 3]
        "states": torch.stack([...]),           # [num_envs, 7]
        "task_descriptions": List[str],         # [num_envs]
    }
    return obs
```

#### 1.2 观测转换（环境 → OpenPI）

**文件**: `openpi_action_model.py` (第240-260行)

```python
def obs_processor(self, env_obs):
    processed_obs = {
        "observation/image": env_obs["main_images"],
        "prompt": env_obs["task_descriptions"],
    }
    
    # 处理状态（强制float32）
    state = env_obs["states"]
    if torch.is_tensor(state):
        state = state.to(dtype=torch.float32)
    processed_obs["observation/state"] = state
    
    # 添加手腕图像
    if env_obs["wrist_images"] is not None:
        processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
    
    return processed_obs
```

#### 1.3 输入转换（OpenPI → 模型）

**文件**: `openpi_action_model.py` (第177-213行)

```python
def input_transform(self, obs: dict, transpose=True):
    # Tensor → Numpy
    inputs = jax.tree.map(self._tensor_to_numpy, obs)
    
    # 批量处理
    batch_size = next(v.shape[0] for v in inputs.values())
    transformed_samples = []
    
    for i in range(batch_size):
        sample = jax.tree.map(lambda x: x[i], inputs)
        
        # 图像格式转换 [C,H,W] → [H,W,C]
        if transpose:
            sample = jax.tree.map(
                lambda x: x.transpose(1, 2, 0) if len(x.shape) == 3 else x,
                sample,
            )
        
        # 应用LiberoInputs transform
        transformed_sample = self._input_transform(sample)
        transformed_samples.append(transformed_sample)
    
    # 重新组合
    inputs = jax.tree.map(
        lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
        *transformed_samples,
    )
    return inputs
```

#### 1.4 模型推理（Flow Matching采样）

**文件**: `openpi_action_model.py` (第320-413行)

```python
@torch.no_grad()
def sample_actions(self, observation, mode="train"):
    bsize = observation.state.shape[0]
    num_steps = self.config.num_steps  # 10
    
    # 初始化噪声
    noise = self.sample_noise((bsize, action_horizon, action_dim), device)
    
    # 编码观测（VLM）
    images, img_masks, lang_tokens, lang_masks, state = (
        self._preprocess_observation(observation, train=False)
    )
    
    # 计算KV cache
    (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    
    # Flow matching去噪
    x_t = noise
    chains = [x_t]
    log_probs = []
    values = []
    
    for idx in range(num_steps):
        # 预测均值、方差、价值
        x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
            x_t, idx, state, prefix_pad_masks, past_key_values,
            mode, num_steps, compute_values=True
        )
        
        # Euler步骤（SDE采样）
        x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
        
        # 计算log概率
        log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
        
        chains.append(x_t)
        log_probs.append(log_prob)
        values.append(value_t)
    
    return {
        "actions": x_t,                    # [bs, action_horizon, action_dim]
        "chains": torch.stack(chains, dim=1),
        "prev_logprobs": torch.stack(log_probs, dim=1).mean(dim=1),
        "prev_values": torch.stack(values, dim=1).mean(dim=-1, keepdim=True),
        "denoise_inds": denoise_inds,
    }
```

#### 1.5 输出转换（模型 → 环境）

**文件**: `openpi_action_model.py` (第215-232行)

```python
def output_transform(self, outputs):
    batch_size = outputs["actions"].shape[0]
    transformed_samples = []
    
    for i in range(batch_size):
        sample = jax.tree.map(lambda x: self._tensor_to_numpy_single(x, i), outputs)
        sample = self._output_transform(sample)  # LiberoOutputs
        transformed_samples.append(sample)
    
    outputs = jax.tree.map(
        lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
        *transformed_samples,
    )
    
    # 截取action_chunk
    outputs["actions"] = outputs["actions"][:, :self.config.action_chunk]
    return outputs
```

**LiberoOutputs**:
```python
def __call__(self, data: dict) -> dict:
    return {"actions": np.asarray(data["actions"][:, :7])}
    # [action_horizon, 32] → [action_horizon, 7]
```

#### 1.6 环境执行

**文件**: `libero_env.py` (第410-445行)

```python
def step(self, actions):
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    
    raw_obs, _reward, terminations, info_lists = self.env.step(actions)
    obs = self._wrap_obs(raw_obs)
    step_reward = self._calc_step_reward(terminations)
    
    return obs, step_reward, terminations, truncations, infos
```

---

### 阶段2: 计算Advantages（GAE）

**文件**: `advantages.py` (第20-77行)

```python
@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,      # [T, bsz, action_chunk]
    values: torch.Tensor,       # [T+1, bsz, 1]
    dones: torch.Tensor,        # [T+1, bsz, action_chunk]
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0
    
    # 反向遍历
    for step in reversed(range(T)):
        # TD误差
        delta = (
            rewards[step]
            + gamma * values[step + 1] * (~dones[step + 1])
            - values[step]
        )
        
        # 累积GAE
        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae + values[step]
    
    advantages = returns - values[:-1]
    advantages = safe_normalize(advantages, loss_mask)
    
    return advantages, returns
```

---

### 阶段3: PPO训练

#### 3.1 接收Rollout数据

**文件**: `fsdp_actor_worker.py` (第1088-1115行)

```python
def recv_rollout_batch(self, input_channel: Channel):
    recv_list = []
    for _ in range(split_num):
        recv_list.append(input_channel.get())
    
    self.rollout_batch = cat_list_of_dict_tensor(recv_list, dim=1)
    self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
```

**Rollout数据格式**:
```python
{
    "forward_inputs": {
        "chains": torch.Tensor,              # [n_steps, bsz, action_horizon, action_dim]
        "denoise_inds": torch.Tensor,
        "observation/image": torch.Tensor,
        "observation/wrist_image": torch.Tensor,
        "observation/state": torch.Tensor,
        "tokenized_prompt": torch.Tensor,
    },
    "prev_logprobs": torch.Tensor,  # [n_steps, bsz, action_chunk, action_dim]
    "prev_values": torch.Tensor,    # [n_steps, bsz, 1]
    "rewards": torch.Tensor,        # [n_steps, bsz, action_chunk]
    "dones": torch.Tensor,          # [n_steps, bsz, action_chunk]
}
```

#### 3.2 PPO训练循环

**文件**: `fsdp_actor_worker.py` (第1204-1329行)

```python
def run_training(self):
    # 打乱数据
    shuffle_id = torch.randperm(rollout_size)
    self.rollout_batch = process_nested_dict_for_train(
        self.rollout_batch, shuffle_id
    )
    
    # 多轮更新
    update_epoch = 4  # PPO关键：对同一批数据更新4轮
    for epoch in range(update_epoch):
        rollout_dataloader_iter = get_iterator_k_split(
            self.rollout_batch,
            rollout_size // batch_size_per_rank,
        )
        
        for train_global_batch in rollout_dataloader_iter:
            train_micro_batch = get_iterator_k_split(
                train_global_batch,
                train_global_batch_size // self.cfg.actor.micro_batch_size,
            )
            
            self.optimizer.zero_grad()
            for idx, data in enumerate(train_micro_batch):
                # 重新前向传播
                output_dict = self.model(
                    data=data,
                    compute_logprobs=True,
                    compute_values=True,
                )
                
                # 计算PPO loss
                loss, metrics = policy_loss(
                    logprobs=output_dict["logprobs"],      # 新
                    old_logprobs=data["prev_logprobs"],    # 旧
                    advantages=data["advantages"],
                    returns=data["returns"],
                    values=output_dict["values"],
                    prev_values=data["prev_values"],
                )
                
                loss.backward()
            
            self.optimizer_step()
```

#### 3.3 PPO Loss计算

**文件**: `losses.py` (第20-120行)

```python
def compute_ppo_actor_loss(
    logprobs, old_logprobs, advantages, clip_ratio=0.2
):
    # 重要性采样比率
    ratio = torch.exp(logprobs - old_logprobs)
    
    # 裁剪
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    
    # 两种loss
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio
    
    # 取最大值（保守更新）
    policy_loss = torch.max(policy_loss1, policy_loss2)
    policy_loss = masked_mean(policy_loss, loss_mask)
    
    return policy_loss
```

---

## 🔄 PPO训练机制详解

### 更新策略对比

| 算法 | 收集策略 | 更新策略 | 数据重用 |
|------|---------|---------|---------|
| REINFORCE | 在线 | 每步更新1次 | 1次 |
| PPO | 批量收集 | 同一批数据更新4轮 | 4次 |

### 完整训练流程

```python
# 1. Rollout阶段（只推理，不更新）
rollout_batch = []
for step in range(100):  # 推理100次
    actions, result = model.predict_action_batch(env_obs, mode="train")
    next_obs, rewards, dones = env.step(actions)
    rollout_batch.append({...})
# 此时：推理100次，更新0次

# 2. 计算Advantages
advantages, returns = compute_gae(rollout_batch)

# 3. PPO训练（多轮更新）
update_epoch = 4
for epoch in range(update_epoch):  # 4轮
    for mini_batch in split(rollout_batch):  # 10个mini-batch
        output = model(mini_batch)
        loss = compute_ppo_loss(output, mini_batch)
        loss.backward()
        optimizer.step()
# 此时：更新 4 * 10 = 40次

# 总结：
# - 推理次数：100次
# - 更新次数：40次
# - 数据重用率：4倍
```

### 为什么PPO这样设计？

1. **样本效率高**: 同一批数据更新4次，充分利用数据
2. **训练稳定**: 重要性采样+裁剪机制防止策略变化过大
3. **易于实现**: 不需要复杂的replay buffer

---

## 🎯 IsaacLab接入指南

### 关键差异对比

| 维度 | Libero | IsaacLab |
|------|--------|----------|
| 主相机 | `agentview_image` (numpy) | `table_cam` (torch.Tensor CUDA) |
| 手腕相机 | `robot0_eye_in_hand_image` | `wrist_cam` (torch.Tensor CUDA) |
| 状态维度 | 7维 | 7维 ✅ |
| 旋转表示 | 四元数→轴角 | 四元数→轴角 ✅ |
| 数据类型 | numpy CPU | torch.Tensor CUDA ⚠️ |
| 图像旋转 | 需要180° | 不需要 ⚠️ |
| 动作空间 | delta | delta ✅ |

### 需要创建的文件

1. **`rlinf/models/embodiment/openpi/dataconfig/isaaclab_dataconfig.py`**
2. **`rlinf/models/embodiment/openpi/policies/isaaclab_policy.py`**
3. **`examples/embodiment/config/isaaclab_ppo_openpi.yaml`**

### IsaacLab Policy示例

```python
# isaaclab_policy.py
@dataclasses.dataclass(frozen=True)
class IsaacLabInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    
    def __call__(self, data: dict) -> dict:
        table_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        
        inputs = {
            "state": data["observation/state"],  # [7]
            "image": {
                "base_0_rgb": table_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(table_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs
```

---

## ❓ 常见问题

### Q1: 是否需要State Normalization？

**答**: ❌ 不需要

- OpenPI的VLM有足够的表征能力
- State尺度已经在合理范围内
- 如果特殊环境需要，可以在policy的input transform中添加

### Q2: PPO是每推理一次就更新一次吗？

**答**: ❌ 不是

- Rollout阶段：推理N次，收集数据（不更新）
- 训练阶段：对同一批数据更新4轮
- 数据重用率：4倍

### Q3: 为什么Flow Matching能计算log prob？

**答**: 使用SDE采样（有噪声）

- train模式：`x_t = x_t_mean + noise * x_t_std`（随机）
- eval模式：`x_t = x_t_mean`（确定性）
- 噪声提供了可计算的高斯分布log概率

### Q4: Repack Transform的方向是什么？

**答**: 数据集键名 → 统一键名

```python
"observation/image": "image"  # image → observation/image
```

目的是让训练数据和推理数据格式一致

---

## 📝 总结

### 完整数据流

```
环境观测 → obs_processor → input_transform → 模型推理 → output_transform → 环境动作
   ↓                                                                    ↓
存储rollout数据                                                    执行并收集奖励
   ↓
计算GAE advantages
   ↓
PPO训练（4轮更新）
```

### 关键要点

1. **数据转换**: 环境格式 → OpenPI格式 → 模型格式
2. **Flow Matching**: SDE采样提供探索和log概率
3. **PPO机制**: 批量收集 + 多轮更新 = 高样本效率
4. **IsaacLab**: 主要差异是CUDA tensor和图像不需要旋转

---

**文档版本**: v1.0  
**创建日期**: 2025-01-29  
**适用版本**: RLinf + OpenPI (Pi0)
