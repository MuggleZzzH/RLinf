# RECAP X1 (Turtle2) Robot 适配自查表 (Self-Checklist)

该自查表用于追踪 RECAP 流程在 XSquare X1 (Turtle2) 机器人上的适配进度，确保数据流、模型逻辑和部署逻辑的完整性。

## 1. 数据层适配 (Data Layer)

- [x] **LeRobot 字段映射 (Step 1-4)**:
    - 检查 `rlinf/data/datasets/recap/value_model.py` 中的 `_REPACK_KEYS["turtle2"]`。
    - [x] 确认 `face_view`, `left_wrist_view`, `right_wrist_view` 是否与原始数据集中的 key 一致。
    - [x] 确认 `state` 和 `actions` (或 `action`) 的映射是否正确。
- [x] **数据变换逻辑 (Data Transforms)**:
    - 检查 `ValueDataset._build_transform` 中的 `turtle2` 分支。
    - [x] 确认是否正确调用了 `repack_to_nested_images` 将图像放入 `images` 字典。
    - [x] 确认 `x2robot_policy.X2RobotInputs` 是否被实例化，且 `action_dim=14`。
- [x] **状态空间对齐**:
    - [x] 确认 X1 的 14 自由度（7+7）状态在 `X2RobotInputs` 中没有被错误截断。

## 2. 价值模型训练 (Step 2: Value Model SFT)

- [x] **多视角输入支持**:
    - [x] 确认 `ValueImageProcessor` 是否支持同时处理 3 个视角。
    - [x] 确认 `ValueProcessor` 中的 `image_keys` 是否包含了所有 3 个视角。
- [x] **训练指标 (Metrics)**:
    - [x] 确认 `train/actor/loss` 是否正常下降。
    - [x] 确认 `eval/spearman_correlation` 是否被正确计算（衡量价值预测的排序能力）。
- [x] **Prompt 注入**:
    - [x] 确认任务描述（Language Instruction）是否正确加载并传递给 Gemma3。

## 3. 优势估计 (Step 3: Compute Advantages)

- [x] **奖励函数逻辑**:
    - [x] 确认 `compute_returns.py` 是否能正确识别成功/失败轨迹。
    - [x] 确认 `failure_reward` (如 -300) 是否已应用。
- [x] **优势分布验证**:
    - [x] 运行 `visualize_advantage_dataset.py`。
    - [x] 检查正样本（Positive）在轨迹中的分布是否合理（不应只在最后一步）。
    - [x] 确认 `positive_quantile` (默认 0.3) 是否能筛选出足够且高质量的样本。

## 4. 策略训练 (Step 4: CFG Training)

- [ ] **优势标签加载**:
    - [ ] 确认 `turtle2_cfg_openpi.yaml` 中的 `advantage_tag` 与 Step 3 生成的 tag 一致。
    - [ ] 确认 `FSDPCfgWorker` 是否正确读取了 `meta/advantages_{tag}.parquet`。
- [ ] **CFG 训练指标**:
    - [ ] 监控 `conditional_count` 和 `unconditional_count` 比例是否符合预期（受 `unconditional_prob` 影响）。
    - [ ] 监控 `positive_conditional_loss_sum` 和 `negative_unconditional_loss_sum`。
- [ ] **Prompt 构造**:
    - [ ] 确认 `Advantage: positive` 和 `Advantage: negative` 提示词是否正确拼接。

## 5. 真机部署与推理 (Inference & Deployment)

- [ ] **环境端映射 (Env Side)**:
    - 检查 `RealWorldEnv` 配置。
    - [ ] 确认 `main_images` (face) 和 `extra_view_images` (wrists) 的分配逻辑。
- [ ] **模型端映射 (Model Side)**:
    - 检查 `OpenPi0ForCFGActionPrediction.obs_processor`。
    - [ ] 确认 `left_wrist_view`, `face_view`, `right_wrist_view` 的提取逻辑与环境端**严格镜像对称**。
- [ ] **推理逻辑**:
    - [ ] 确认 `cfgrl_guidance_scale` 在部署脚本中可调且生效。
    - [ ] 确认模型在推理时使用的是 `guidance_type: "positive"`。

## 6. 待办事项 (TODOs)

- [ ] 运行 Step 1 脚本生成 X1 数据集的回报 sidecar。
- [ ] 运行 Step 2 训练价值模型并观察 Spearman 相关系数。
- [ ] 运行 Step 3 计算优势并可视化分布。
- [ ] 运行 Step 4 进行 CFG 训练。
- [ ] 真机验证。
