# XSquare X1 (Turtle2) 机器人集成技术变更草稿

该文档汇总了 `review/turtle2-dagger-clean` 分支相对于上游 `main` 分支在支持 XSquare X1 机器人及其配套 DAgger 流程方面的主要修改。

## 1. 硬件控制与环境实现 (Core Environment)
*   **核心环境**: 新增 `rlinf/envs/realworld/xsquare/turtle2_env.py`，实现了针对 X1 双臂机器人的 Gym 接口。
*   **平滑控制器**: 新增并增强 `turtle2_smooth_controller.py`，负责处理双臂同步、速度平滑以及底层硬件实时通信。
*   **部署逻辑**: 新增 `rlinf/envs/realworld/xsquare/tasks/deploy_env.py`，专门用于真实世界环境中的模型推理与任务执行。

## 2. HG-DAgger (Human-Gated DAgger) 流程
*   **接管介入**: 新增 `master_takeover_intervention.py` 包装器。支持人类操作员通过主臂随时介入并接管机器人，自动标记介入数据段。
*   **数据采集**: 增强 `collect_episode.py`，使其能够识别并持久化存储带有介入标签的轨迹数据，用于后续的在线微调。
*   **通信协议**: 新增 `x2robot_protocol.py`，定义了主从设备在介入切换过程中的握手、数据对齐与状态同步协议。

## 3. 模型适配与推理 (Model Adaptation)
*   **策略定制**:
    *   新增 `rlinf/models/embodiment/openpi/policies/x2robot_policy.py`，针对 X1 机器人的 14 自由度双臂配置进行了 OpenPI 策略适配。
    *   新增 `x2robot_dataconfig.py`，定义了该硬件特有的观测空间归一化参数。
*   **CFG 引导**: 在 `openpi_action_model.py` 中集成了 Classifier-Free Guidance (CFG) 推理逻辑，支持 RECAP 等优势条件引导算法。

## 4. 权重同步系统优化 (System Optimization)
*   **Patch Syncer**: 实现了 `patch_syncer.py`。针对 VLA 大模型，在分布式训练过程中仅同步增量补丁（Diff），显著降低了弱网环境下的带宽占用。
*   **实时压缩**: 新增 `compressor.py`，支持在权重传输前进行实时压缩。

## 5. 配置、测试与文档
*   **任务配置**: 新增数十项 `.yaml` 配置，包括 `realworld_turtle2_dagger_takeover_collect_openpi.yaml` 等。
*   **自动化测试**: 增加了针对介入逻辑、位姿转换和平滑控制器的单元测试：
    *   `test_master_takeover_intervention.py`
    *   `test_dual_pose_action_wrappers.py`
*   **专项文档**:
    *   `xsquare_turtle2.rst`: 硬件配置指南。
    *   `turtle2_dagger_takeover.rst`: DAgger 流程操作手册。

---
*注：代码中 `Turtle2` 为 `XSquare X1` 的内部开发代号。*
