# OpenPI + IsaacLab 全流程实现总结（RLinf）

## 1. 目标与范围

本次改造目标是把 RLinf 中 OpenPI + IsaacLab 路径打通到可重复执行的完整流程：

- SFT 训练入口可直接运行
- Eval 入口可直接运行并稳定产出日志/视频
- RL 训练入口可直接运行
- 支持多 checkpoint（如 25000、30000）批量评测/批量训练
- 尽量保持对主仓库兼容，减少侵入性改动

时间范围（代码演进）从本分支的基线提交 `66db1e6` 到当前 `d2670a6`。

---

## 2. 总体变更规模

`git diff --stat 66db1e6..HEAD` 显示：

- 24 个文件变更
- 1865 行新增，8 行删除

主要集中在：

- `examples/` 脚本和配置
- `rlinf/models/embodiment/openpi/` 数据适配与 transform 对齐
- `rlinf/workers/env/env_worker.py` 评估步骤健壮性
- `rlinf/envs/wrappers/record_video.py` 视频落盘稳定性
- `toolkits/` 辅助工具脚本

---

## 3. 关键提交时间线（从原始到当前）

| 提交 | 主题 | 关键结果 |
|---|---|---|
| `fc3e1c5` | 增加 OpenPI IsaacLab SFT/Eval/RL 工作流 | 初版脚本与配置落地 |
| `e1aaf2a` | 对齐本地 IsaacLab 数据布局 | 解决本地数据目录与配置不一致 |
| `4e60277` | OpenPI selector alias 支持 | 处理 transform selector 键不匹配 |
| `00b1f87` | 增加 state/action 平铺别名 | 缓解 `observation.state` 类键缺失 |
| `e723dd3` | 过滤 IsaacLab 非策略 norm 键 | 避免 `episode_index` 等元数据参与策略 transform |
| `178e5e8` | 自动化环境初始化脚本 | 减少每次手工 source 操作 |
| `5923e35` | source 时放宽 nounset | 解决 `ZSH_VERSION unbound variable` |
| `52c5883` | env worker 支持非 list chunk 输出 | 修复 `extracted_obs` 未绑定等崩溃 |
| `b3f4c51` | 视频 flush 稳定化 | 降低评估结束时 mp4 丢失概率 |
| `95a2634` | 去掉 `async_save` override | 避免 Hydra 配置结构报错 |
| `c9a0be2` | 按 step 记录帧 + 结果目录分离 | 视频时序更完整，SFT/Eval/RL 结果隔离 |
| `ae8ad3a` | Eval 参数向 GR00T 基准对齐 | 统一 benchmark 口径 |
| `6cfb406` | SFT 配置更新 | FSDP 与 batch/训练步数等参数落地 |
| `b0d7708` | max steps 可整除 action chunk 校验 | 避免运行期 assertion 崩溃 |
| `4d4e8a6` | 新增 pt->safetensors 工具 | 简化 checkpoint 转换 |
| `430c330` | 新增 eval/rl 分离 multi 脚本 | 支持多 ckpt 批处理 |
| `d2670a6` | multi 脚本只切换 ckpt 路径 | 去除额外参数覆盖，保持与单跑一致 |

---

## 4. 主要问题与修复细节

## 4.1 SFT 数据路径与 LeRobot 结构不一致

### 现象

- 运行 SFT 报错找不到：
  - `.../RLinf/generated_simdata_full/meta/info.json`
- 随后回退到 HF repo 分支导致 401（本地私有路径/认证未命中）

### 根因

- OpenPI/LeRobot 读取期望的是 `HF_LEROBOT_HOME/<repo_id>/meta/info.json`。
- 初始传入的是叶子目录，导致拼接路径错误。

### 修复

- `examples/sft/run_isaaclab_sft_openpi.sh` 增加目录解析：
  - 自动识别 `DATASET_PATH` 是叶子还是父目录
  - 统一传入 `data.train_data_paths=<LEROBOT_HOME>`
  - 增加 `DATASET_REPO_ID`（默认 `generated_simdata_full`）
  - 启动前强校验 `meta/info.json` 是否存在

---

## 4.2 Eval 初期 Hydra 组合失败

### 现象

- `Overriding hydra.searchpath is only supported from the primary config`

### 根因

- eval 配置缺 `_self_`，导致默认组合顺序不符合 Hydra 预期。

### 修复

- `examples/embodiment/config/isaaclab_stack_cube_openpi_eval.yaml` 的 defaults 增加 `_self_`。

---

## 4.3 OpenPI transform selector 键不匹配

### 现象

- `Selector key observation.state not found in tree`
- 后续又出现 `Selector key episode_index not found in tree`

### 根因

- 运行时 sample 键名与 norm_stats/selectors 键名风格存在差异。
- norm_stats 中含元数据键（`episode_index` 等），不该参与策略归一化。

### 修复

1. 在 `rlinf/models/embodiment/openpi/openpi_action_model.py` 中补充 alias：
- `state -> observation.state / observation/state`
- `actions -> action`

2. 在 `rlinf/models/embodiment/openpi/__init__.py` 中对 `pi0_isaaclab` 做 norm stats 过滤：
- 仅保留策略相关键：`observation.state`, `observation/state`, `action`, `actions`
- 过滤掉 `episode_index`、`frame_index`、`timestamp`、`index`、`task_index` 等元信息

最终避免 selector 缺键与错误匹配问题。

---

## 4.4 Env worker 评估阶段崩溃

### 现象

- `UnboundLocalError: cannot access local variable 'extracted_obs'`

### 根因

- `chunk_step` 返回形态在不同 env 实现下可能是 list 或非 list，原逻辑未覆盖。

### 修复

- `rlinf/workers/env/env_worker.py` 对 train/eval 两条路径增加稳健解析：
  - `obs_list` 支持 list/tuple 与单对象
  - `infos_list` 支持 list/tuple 与单 dict
  - 非 dict 情况兜底为空 dict

---

## 4.5 视频未保存或视频时长异常

### 现象

- 评估结束后没有 mp4 或 mp4 不完整
- 报 `IMAGEIO FFMPEG_WRITER WARNING` 尺寸重采样

### 根因

- 评估完成后异步写盘未等待完成
- chunk 帧记录逻辑对 final obs/reset obs 处理不够完整

### 修复

- `rlinf/envs/wrappers/record_video.py`
  - 增加后台写盘任务管理与等待
  - 改进 chunk_step 帧抽取（包含 final obs / reset obs 处理）
- `rlinf/workers/env/env_worker.py`
  - eval finish 时显式 `flush_video()` 并 `wait_pending_saves()`

说明：`imageio` 的 macro block warning 属编码兼容提醒，不是失败。

---

## 4.6 环境初始化不稳定（手工步骤多）

### 现象

- 每次新 shell 需要多条 source/activate
- `setup_conda_env.sh` 在 `set -u` 下出现 `ZSH_VERSION` 未定义错误

### 修复

- 新增 `examples/common/setup_isaaclab_runtime.sh`
  - 自动 source IsaacLab conda/runtime 脚本
  - 自动激活 `.venv`
  - 支持 `AUTO_SETUP_ENV=0` 手动禁用
  - source 外部脚本时暂时关闭 nounset，兼容 `ZSH_VERSION` 场景

- SFT/Eval/RL 启动脚本统一接入该 runtime setup。

---

## 4.7 Eval/RL 参数一致性与安全校验

### 关键改动

- `OPENPI_MODEL_DIR` 默认改为 `auto`：优先从 `CKPT_INPUT` 推断模型目录
- 启动时打印检测到的 `norm_stats.json` 路径
- `MAX_EPISODE_STEPS % ACTION_CHUNK == 0` 预检查，提前失败并给出建议
- `result` 目录按任务拆分：
  - `result/isaaclab_openpi/sft`
  - `result/isaaclab_openpi/eval`
  - `result/isaaclab_openpi/rl`

这些改动的目标是：减少“权重来自 SFT、norm 却来自 base”这种隐性错配。

---

## 5. multi 脚本行为（最终版）

最终保留两个分离脚本：

- `examples/embodiment/run_isaaclab_openpi_eval_multi_ckpt.sh`
- `examples/embodiment/run_isaaclab_openpi_rl_multi_ckpt.sh`

最终行为已经收敛为：

- 只循环 `CKPT_STEPS`（例如 `25000 30000`）
- 每次只注入 `CKPT_INPUT=<对应step目录>`
- 不覆盖其它参数（不改 epoch、不改 env 数、不改 logger）
- 其它全部继承单跑脚本默认值或外部环境变量

这是当前与单独运行“行为一致”的核心保证。

---

## 6. pt -> safetensors 工具

新增脚本：

- `toolkits/convert_pt_to_safetensors.py`

能力：

- 输入 `.pt/.pth` 输出 `safetensors`
- 支持单文件或分片输出
- 支持 `state_dict` 自动识别或指定 key path
- 支持 key 前缀剥离、dtype 转换

用于把 `full_weights.pt` 转为可直接作为 OpenPI `model_path` 的权重目录组成部分。

---

## 7. 当前版本的运行建议（稳定路径）

## 7.1 Eval 单跑

推荐方式：

- `OPENPI_MODEL_DIR` 指向包含 `safetensors + norm_stats` 的 SFT 导出目录
- `CKPT_INPUT=null`（避免重复覆盖）

或者：

- 只传 `CKPT_INPUT=<model_state_dict目录>`
- 让脚本自动推断 `OPENPI_MODEL_DIR`

## 7.2 Eval 批量（25000/30000）

```bash
SFT_CKPT_ROOT=/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft_2048/checkpoints \
CKPT_STEPS="25000 30000" \
bash examples/embodiment/run_isaaclab_openpi_eval_multi_ckpt.sh
```

## 7.3 RL 批量（25000/30000）

```bash
SFT_CKPT_ROOT=/mnt/project_rlinf_hs/Jiahao/results/isaaclab_sft/isaaclab_stack_cube_sft_2048/checkpoints \
CKPT_STEPS="25000 30000" \
bash examples/embodiment/run_isaaclab_openpi_rl_multi_ckpt.sh
```

---

## 8. 与原始仓库相比的“核心可用性提升”

- 从“需要大量手工步骤 + 易踩数据/transform坑”变为“脚本可直接跑”。
- 从“eval 常见中途崩溃/视频不稳定”变为“可完整跑通并稳定落盘”。
- 从“多 ckpt 对比操作繁琐”变为“批处理自动串行执行”。
- 从“pt/safetensors 手工处理成本高”变为“内置转换工具可复用”。

---

## 9. 已知限制与后续建议

- 当前 OpenPI 的 IsaacLab RL 配置是可训练 baseline，不是极限吞吐配置。
- 若追求与 GR00T 基准完全等口径，需要继续对齐更高并发/更长 horizon/更大 batch 的组合，并重新做稳定性回归。
- 为保证可复现，后续建议在 eval 脚本增加显式 seed 参数并默认开启。

---

## 10. 附：本次主要改动文件清单

- `examples/common/setup_isaaclab_runtime.sh`
- `examples/sft/config/isaaclab_sft_openpi.yaml`
- `examples/sft/run_isaaclab_sft_openpi.sh`
- `examples/embodiment/config/isaaclab_stack_cube_openpi_eval.yaml`
- `examples/embodiment/config/isaaclab_stack_cube_ppo_openpi.yaml`
- `examples/embodiment/run_isaaclab_openpi_eval.sh`
- `examples/embodiment/run_isaaclab_openpi_rl.sh`
- `examples/embodiment/run_isaaclab_openpi_eval_multi_ckpt.sh`
- `examples/embodiment/run_isaaclab_openpi_rl_multi_ckpt.sh`
- `rlinf/models/embodiment/openpi/__init__.py`
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py`
- `rlinf/models/embodiment/openpi/dataconfig/isaaclab_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/isaaclab_policy.py`
- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/envs/wrappers/record_video.py`
- `rlinf/envs/isaaclab/isaaclab_env.py`
- `tests/unit_tests/test_openpi_isaaclab.py`
- `toolkits/convert_pt_to_safetensors.py`

