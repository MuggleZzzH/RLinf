# IsaacLab 数据集结构详细说明

## 📂 目录结构概览

```
/mnt/qiyuan/zhy/isaaclab_data/
├── dataset.hdf5                    # 空文件 (0 GB)
├── generated_dataset.hdf5          # 原始HDF5数据集 (4.17 GB)
├── generated_simdata_full/         # LeRobot格式数据集
├── generated_simdata_full_old/     # LeRobot格式数据集（旧版本）
└── hdf5_to_lerobot.py             # 格式转换脚本
```

---

## 🗂️ HDF5 数据集

### 文件信息
- **文件名**: `generated_dataset.hdf5`
- **大小**: 4.17 GB
- **格式**: HDF5 (Hierarchical Data Format)

### 数据结构
HDF5文件包含原始的机器人演示数据，组织结构如下：

```
generated_dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── actions          # 机器人动作 [T, 7]
    │   └── obs/
    │       ├── eef_pos      # 末端执行器位置 [T, 3]
    │       ├── eef_quat     # 末端执行器四元数 [T, 4] (wxyz格式)
    │       ├── gripper_pos  # 夹爪位置 [T, 2]
    │       ├── table_cam    # 桌面相机图像 [T, H, W, 3]
    │       └── wrist_cam    # 腕部相机图像 [T, H, W, 3]
    ├── demo_1/
    │   └── ...
    └── demo_N/
        └── ...
```

**注意**: 
- T = 时间步数（每个episode不同）
- 四元数格式为 wxyz，转换时需要重新排列为 xyzw
- 第一帧数据在转换时被丢弃

---

## 🤖 LeRobot 格式数据集

### 数据集信息

#### 基本统计
- **Episodes 数量**: 1,000
- **总帧数**: 241,838
- **平均每个episode**: ~242 帧
- **FPS**: 20
- **机器人类型**: Franka
- **任务数量**: 1
- **版本**: v2.1

#### 任务描述
```
Pick up the red cube and place it on top of the blue cube, 
then pick up the green cube and place it on top of the red cube.
```

---

### 📁 目录结构

```
generated_simdata_full/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # 每个episode一个文件
│       ├── episode_000001.parquet
│       └── ...                     # 共1000个文件
│
├── videos/
│   └── chunk-000/
│       ├── observation.images.table/
│       │   ├── episode_000000.mp4  # 桌面视角视频
│       │   └── ...                 # 共1000个视频
│       └── observation.images.wrist/
│           ├── episode_000000.mp4  # 腕部视角视频
│           └── ...                 # 共1000个视频
│
└── meta/
    ├── info.json                   # 数据集元信息
    ├── tasks.jsonl                 # 任务描述
    ├── episodes.jsonl              # Episode元数据
    ├── episodes_stats.jsonl        # 每个episode的统计信息
    ├── modality.json               # 数据模态定义
    └── stats.json                  # 全局统计信息
```

---

### 📊 数据特征 (Features)

#### 1. **observation.state** (观测状态)
- **类型**: float32
- **维度**: [8]
- **内容**:
  - `[0]` x: 末端执行器X坐标
  - `[1]` y: 末端执行器Y坐标
  - `[2]` z: 末端执行器Z坐标
  - `[3]` roll: 翻滚角
  - `[4]` pitch: 俯仰角
  - `[5]` yaw: 偏航角
  - `[6:8]` gripper: 夹爪状态 (2维)
- **统计**:
  - 均值: [0.499, -0.003, 0.141, ...]

#### 2. **action** (动作)
- **类型**: float32
- **维度**: [7]
- **内容**:
  - `[0]` x: X方向动作
  - `[1]` y: Y方向动作
  - `[2]` z: Z方向动作
  - `[3]` roll: 翻滚角动作
  - `[4]` pitch: 俯仰角动作
  - `[5]` yaw: 偏航角动作
  - `[6]` gripper: 夹爪动作 (1维)
- **统计**:
  - 均值: [0.002, 0.001, -0.005, ...]

#### 3. **observation.images.table** (桌面相机)
- **类型**: video
- **分辨率**: 84 × 84 × 3 (RGB)
- **编码**: AV1
- **像素格式**: yuv420p
- **FPS**: 20
- **文件格式**: MP4
- **平均大小**: ~0.05 MB/视频

#### 4. **observation.images.wrist** (腕部相机)
- **类型**: video
- **分辨率**: 84 × 84 × 3 (RGB)
- **编码**: AV1
- **像素格式**: yuv420p
- **FPS**: 20
- **文件格式**: MP4
- **平均大小**: ~0.04 MB/视频

#### 5. **元数据字段**
- **episode_index**: int64, episode索引 (0-999)
- **frame_index**: int64, 帧索引 (0-T)
- **timestamp**: float64, 时间戳 (秒)
- **index**: int64, 全局帧索引 (0-241837)
- **task_index**: int64, 任务索引 (固定为0)
- **next.done**: bool, episode结束标志

---

### 📈 Episode 统计示例

前5个episodes的长度：
- Episode 0: 199 帧 (~10秒)
- Episode 1: 229 帧 (~11.5秒)
- Episode 2: 250 帧 (~12.5秒)
- Episode 3: 238 帧 (~11.9秒)
- Episode 4: 229 帧 (~11.5秒)

**平均episode长度**: ~242 帧 (~12.1秒)

---

### 📄 Parquet 文件结构

每个 `episode_XXXXXX.parquet` 文件包含以下列：

| 列名 | 类型 | 描述 |
|------|------|------|
| observation.state | list[float32] | 机器人状态 (8维) |
| action | list[float32] | 动作 (7维) |
| episode_index | int64 | Episode索引 |
| frame_index | int64 | 帧索引 |
| timestamp | float64 | 时间戳 |
| index | int64 | 全局索引 |
| task_index | int64 | 任务索引 |

**文件大小**: 平均 ~30 KB/episode

---

### 🎯 数据模态映射 (modality.json)

#### State 模态
```json
{
  "x": {"start": 0, "end": 1},
  "y": {"start": 1, "end": 2},
  "z": {"start": 2, "end": 3},
  "roll": {"start": 3, "end": 4},
  "pitch": {"start": 4, "end": 5},
  "yaw": {"start": 5, "end": 6},
  "gripper": {"start": 6, "end": 8}
}
```

#### Action 模态
```json
{
  "x": {"start": 0, "end": 1},
  "y": {"start": 1, "end": 2},
  "z": {"start": 2, "end": 3},
  "roll": {"start": 3, "end": 4},
  "pitch": {"start": 4, "end": 5},
  "yaw": {"start": 5, "end": 6},
  "gripper": {"start": 6, "end": 7}
}
```

#### Video 模态
```json
{
  "image": {"original_key": "observation.images.table"},
  "wrist_image": {"original_key": "observation.images.wrist"}
}
```

---

## 🔄 数据转换流程

HDF5 → LeRobot 格式的转换过程：

1. **读取HDF5数据**
   - 遍历所有demo
   - 跳过第一帧数据

2. **坐标转换**
   - 四元数从 wxyz 转换为 xyzw
   - 使用 `quat2axisangle_np()` 转换为轴角表示

3. **状态组合**
   - 拼接: [eef_pos, angles, gripper_pos] → observation.state (8维)

4. **视频编码**
   - 使用 imageio 将图像序列编码为 MP4
   - 编码器: AV1
   - FPS: 20

5. **Parquet 写入**
   - 使用 PyArrow 写入表格数据
   - 每个episode一个文件

6. **元数据生成**
   - 计算统计信息 (min, max, mean, std)
   - 生成 info.json, tasks.jsonl, episodes.jsonl 等

---

## 🛠️ 使用工具

### 查看数据集结构
```bash
python3 inspect_dataset.py
```

### 转换HDF5到LeRobot格式
```bash
python3 hdf5_to_lerobot.py \
  --hdf5_path generated_dataset.hdf5 \
  --output_path generated_simdata_full \
  --prompts "Pick up the red cube..." \
  --fps 20
```

---

## 📦 依赖库

查看完整数据需要以下Python库：
- `h5py`: 读取HDF5文件
- `numpy`: 数值计算
- `pyarrow`: 读取Parquet文件
- `pandas`: 数据处理
- `imageio`: 视频处理

---

## 📝 注意事项

1. **数据对齐**: HDF5转换时第一帧被丢弃，确保动作和观测对齐
2. **四元数格式**: IsaacLab使用wxyz，LeRobot使用xyzw
3. **角度表示**: 使用轴角表示而非四元数
4. **视频压缩**: 使用AV1编码以节省空间
5. **Chunk组织**: 数据按chunk-000组织，支持大规模数据集

---

## 🔍 数据质量

- ✅ 1000个完整episodes
- ✅ 241,838总帧数
- ✅ 双视角视频 (桌面+腕部)
- ✅ 完整的状态和动作数据
- ✅ 详细的统计信息和元数据

---

**生成时间**: 2026-01-20  
**工具版本**: inspect_dataset.py v1.0


