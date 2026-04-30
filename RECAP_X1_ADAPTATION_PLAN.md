# RECAP X1 (Turtle2) Robot 适配计划

该文档记录了将 RECAP 离线策略优化流程适配到 XSquare X1 (Turtle2) 机器人的技术细节与执行步骤。

## 1. 统一命名与标识 (Naming Convention)

为了保持代码的一致性，在 RECAP 流程涉及的所有配置与逻辑中：
*   **`robot_type`**: 统一使用 **`turtle2`**。
*   **`model_type`**: 对于 OpenPI 策略适配层，使用 **`x2robot`** (对应 `x2robot_policy.py`)。

## 2. 数据层适配 (Data Layer Adaptation)

### 2.1 字段映射 (Repack Keys)
在 `rlinf/data/datasets/recap/value_model.py` 的 `_REPACK_KEYS` 中注册 `turtle2` 的 LeRobot 字段映射：
*   `observation/image` -> `observation.images.face_view`
*   `observation/left_wrist_image` -> `observation.images.left_wrist_view`
*   `observation/right_wrist_image` -> `observation.images.right_wrist_view`
*   `observation/state` -> `observation.state`
*   `actions` -> `action`

### 2.2 数据变换 (Data Transforms)
修改 `ValueDataset._build_transform`：
*   当 `robot_type == "turtle2"` 时，导入并实例化 `rlinf.models.embodiment.openpi.policies.x2robot_policy.X2RobotInputs`。
*   确保 14 自由度状态空间（双臂 7+7）能被正确填充和转换。

## 3. 价值模型训练 (Value Model SFT)

### 3.1 视觉输入适配
*   **视角对齐**：确认 `ValueImageProcessor` 支持 `face_view` 和双腕视角的输入。
*   **处理器配置**：在 `ValueProcessor` 初始化时，确保 `image_keys` 与 X1 机器人的相机名称一致。

### 3.2 状态归一化
*   从 `x2robot_dataconfig.py` 提取统计数据，并确保 `stats.json` 正确反映了实机采集数据的范围。

## 4. 奖励与优势计算 (Reward & Advantage Estimation)

### 4.1 奖励函数定义 (Step 1)
*   **基本策略 (第一阶段采用)**：遵循标准 RECAP 逻辑。成功轨迹每步奖励 `-1`，终止步 `0`；失败轨迹终止步 `failure_reward` (建议值 `-300`)。暂时忽略 `intervene_flag`。
*   **进阶讨论 (Optional / Future Work)**：
    *   探讨如何利用 `intervene_flag` 进行更细粒度的奖励分配。
    *   方案：将人类介入后的轨迹段权重调高，或视为“修正动作”；给介入前的机器尝试段更低的奖励。

### 4.2 优势估计 (Step 3)
*   **前瞻步数**：针对实机任务的复杂度，调试 `advantage_lookahead_step` (默认 10)。
*   **正样本筛选**：调试 `positive_quantile`，筛选出真正的“优势”动作。

## 5. 策略训练与推理 (CFG Training & Inference)

### 5.1 条件文本构造 (Training)
*   **Prompt 注入**：验证 `TokenizePromptWithGuidance` 能够正确生成 `positive` 和 `negative` 的引导后缀。
*   **Batch 动态混合**：确保在 `FSDPCfgWorker` 中，conditional 和 unconditional 的样本比例符合预期。

### 5.2 CFG 推理逻辑 (Inference)
*   **双路前向**：模型在推理时需同时计算 $v_{uncond}$ 和 $v_{cond}$。
*   **外推参数**：在真机测试脚本中，暴露 `cfgrl_guidance_scale` 参数用于调节策略的“进取度”。

## 6. 真机部署适配与融合方案 (Finalized Deployment Strategy)

经过分析，我们决定复用 RLinf 现有的真机评估框架 `run_realworld_eval.sh`，通过新增专门的配置文件来实现 RECAP 策略的真机部署。

### 6.1 部署流程
1.  **环境启动**: 在各个节点启动 Ray，并正确设置 `RLINF_NODE_RANK`。
2.  **执行命令**:
    ```bash
    bash examples/embodiment/run_realworld_eval.sh realworld_x1_recap_cfg_eval \
      actor.model.model_path=/path/to/your/cfg_checkpoint
    ```

### 6.2 关键适配点
*   **观测值包装 (Env Side)**: `RealWorldEnv` 会根据 `main_image_key: wrist_2` 自动将 `Turtle2Env` 输出的 `wrist_2` 映射为 `main_images`，将 `wrist_1` 和 `wrist_3` 映射入 `extra_view_images`。
*   **观测值处理器 (Model Side)**: `OpenPi0ForCFGActionPrediction` 的 `obs_processor` 会识别 `robot_type: turtle2`，并进行如下映射：
    *   `face_view` = `main_images` (即 `wrist_2`)
    *   `left_wrist_view` = `extra_view_images[:, 0]` (即 `wrist_1`)
    *   `right_wrist_view` = `extra_view_images[:, 1]` (即 `wrist_3`)
*   **推理逻辑**: `cfg_model` 在推理时会自动注入 `Advantage: positive` 提示词，并根据 `cfgrl_guidance_scale` (默认 1.5) 进行无条件分支和有条件分支的加权融合。
*   **配置管理**: 通过 `realworld_x1_recap_cfg_eval.yaml` 统一管理 `robot_type: turtle2`、`model_type: cfg_model` 以及 CFG 专有参数。

---

## 7. 执行检查清单 (Task Checklist)

- [x] **定义机器人与策略命名规范**: `robot_type: turtle2`, `model_type: x2robot` (for OpenPI adaptation).
- [x] **制定 RECAP 适配计划**: 完成 `RECAP_X1_ADAPTATION_PLAN.md` 初始版本。
- [x] **代码重构 (Robustness)**: 在 `OpenPi0Config` 中增加 `robot_type` 字段，替代硬编码的 config_name 判断。
- [x] **实现部署侧观测值融合**: 在 `openpi_cfg_action_model.py` 中支持 `turtle2` 的三摄像头映射。
- [x] **修复部署侧 Repack 丢失问题**: 在 `openpi_cfg/__init__.py` 中正确加载 `repack_transforms`。
- [x] **创建训练配置文件**: `examples/recap/cfg/config/turtle2_cfg_openpi.yaml`.
- [x] **创建评估配置文件**: `examples/embodiment/config/realworld_x1_recap_cfg_eval.yaml`.
- [ ] **真机验证**: 使用训练好的 CFG 模型在 X1 机器人上进行实际任务测试。
