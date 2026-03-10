# IsaacLab-Stack-Cube-Data 数据集总结

## 📂 目录结构概览

```

/mnt/project_rlinf_hs/Jiahao/IsaacLab-Stack-Cube-Data/

├── data/

│   └── chunk-000/

│       ├── episode_000000.parquet  # 147个episode数据文件

│       └── ... (共147个)

│

├── videos/

│   └── chunk-000/

│       ├── observation.images.front/   # 正面视角视频 (147个)

│       │   └── episode_XXXXXX.mp4

│       └── observation.images.wrist/   # 腕部视角视频 (147个)

│           └── episode_XXXXXX.mp4

│

└── meta/

    ├── info.json               # 数据集元信息

    ├── tasks.jsonl              # 任务描述

    ├── episodes.jsonl           # Episode元数据 (含长度)

    ├── episodes_stats.jsonl     # Episode统计信息

    ├── modality.json            # 数据模态定义

    └── stats.json               # 全局统计信息

```

---

## 📊 基本统计

| 指标 | 值 |

|------|-----|

| **Episodes 数量** | 147 |

| **总帧数** | 53,265 |

| **平均每 episode** | ~362 帧 (~18秒) |

| **FPS** | 20 |

| **机器人类型** | Panda (Franka) |

| **任务数量** | 1 |

| **版本** | v2.1 |

| **视频总数** | 294 (正面+腕部) |

---

## 🎯 任务描述

```

Stack the red block on the blue block, then stack the green block on the red block

```

将红色方块堆叠在蓝色方块上，然后将绿色方块堆叠在红色方块上。

---

## 📊 数据特征 (Features)

### 1. **observation.state** (观测状态)

-**类型**: float32

-**维度**: [8]

-**内容**:

-`[0]` x: 末端执行器X坐标 (范围: 0.339 ~ 0.670, 均值: 0.493)

-`[1]` y: 末端执行器Y坐标 (范围: -0.303 ~ 0.219, 均值: -0.006)

-`[2]` z: 末端执行器Z坐标 (范围: 0.006 ~ 0.288, 均值: 0.110)

-`[3]` roll: 翻滚角 (范围: -π ~ π, 均值: 0.567 rad)

-`[4]` pitch: 俯仰角 (范围: -0.070 ~ 0.236, 均值: 0.053 rad)

-`[5]` yaw: 偏航角 (范围: -π ~ π, 均值: -2.159 rad)

-`[6]` gripper: 夹爪状态1 (范围: 0 ~ 0.04, 均值: 0.032)

-`[7]` gripper: 夹爪状态2 (范围: -0.04 ~ 0, 均值: -0.032)

### 2. **action** (动作)

-**类型**: float32

-**维度**: [7]

-**内容**:

-`[0]` x: X方向动作 (范围: ±0.1, 均值: 0.001)

-`[1]` y: Y方向动作 (范围: ±0.1, 均值: 0.001)

-`[2]` z: Z方向动作 (范围: ±0.1, 均值: -0.001)

-`[3]` roll: 翻滚角动作 (固定为0)

-`[4]` pitch: 俯仰角动作 (固定为0)

-`[5]` yaw: 偏航角动作 (范围: ±0.1, 均值: ~0)

-`[6]` gripper: 夹爪动作 (范围: -1 ~ 1, 均值: 0.060)

### 3. **observation.images.front** (正面相机)

-**类型**: video

-**分辨率**: 256 × 256 × 3 (RGB)

-**编码**: AV1

-**像素格式**: yuv420p

-**FPS**: 20

-**文件格式**: MP4

### 4. **observation.images.wrist** (腕部相机)

-**类型**: video

-**分辨率**: 256 × 256 × 3 (RGB)

-**编码**: AV1

-**像素格式**: yuv420p

-**FPS**: 20

-**文件格式**: MP4

---

## 📈 Episode 统计示例

前10个episode的长度：

- Episode 0: 312 帧 (~15.6秒)
- Episode 1: 422 帧 (~21.1 2: 秒)
- Episode268 帧 (~13.4秒)
- Episode 3: 261 帧 (~13.1秒)
- Episode 4: 240 帧 (~12.0秒)
- Episode 5: 345 帧 (~17.3秒)
- Episode 6: 292 帧 (~14.6秒)
- Episode 7: 588 帧 (~29.4秒)
- Episode 8: 410 帧 (~20.5秒)
- Episode 9: 392 帧 (~19.6秒)

**最大episode长度**: 588 帧 (~29.4秒)

**最小episode长度**: 240 帧 (~12.0秒)

**平均episode长度**: ~362 帧 (~18秒)

---

## 📄 Parquet 文件结构

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

---

## 🗺️ 数据模态映射 (modality.json)

### State 模态

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

### Action 模态

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

### Video 模态

```json

{

"image": {"original_key": "observation.images.front"},

"wrist_image": {"original_key": "observation.images.wrist"}

}

```

---

## 🔄 与参考数据集对比

| 指标 | 参考数据集 | 当前数据集 |

|------|------------|-------------|

| Episodes | 1,000 | 147 |

| 总帧数 | 241,838 | 53,265 |

| 平均每episode | ~242帧 | ~362帧 |

| 相机视角 | table + wrist | front + wrist |

| 图像分辨率 | 84×84 | 256×256 |

| 机器人 | Franka | Panda |

**主要区别**：

1. 当前数据集规模较小 (147 vs 1000 episodes)
2. 图像分辨率更高 (256×256 vs 84×84)
3. 使用正面相机(front)代替桌面相机(table)
4. 任务更复杂（两步堆叠）

---

## 📝 注意事项

1.**数据对齐**: 每个episode的action比state少一帧

2.**姿态表示**: 使用轴角表示 (roll, pitch, yaw) 而非四元数

3.**动作空间**: roll和pitch动作始终为0，机器人主要在XY平面移动

4.**视频编码**: 使用AV1编码以节省空间

5.**Chunk组织**: 数据按chunk-000组织，支持大规模数据集扩展

---

## ✅ 数据质量

- ✅ 147个完整episodes
- ✅ 53,265总帧数
- ✅ 双视角视频 (正面+腕部)
- ✅ 完整的状态和动作数据
- ✅ 详细的统计信息和元数据
- ✅ LeRobot v2.1格式兼容

---

**数据集路径**: `/mnt/project_rlinf_hs/Jiahao/IsaacLab-Stack-Cube-Data/`
