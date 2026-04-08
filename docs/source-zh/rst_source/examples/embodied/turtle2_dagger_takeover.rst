Turtle2 三机接管式 DAgger 设计
================================

本文档固定 ``codex/turtle2-dagger-v1`` 分支的 DAgger v1 设计边界，目标是把
当前 ``s2s + x2robot_raw + absolute_pose`` 部署链扩展为 **可接 master 接管**
的 RLinf HG-DAgger 入口，同时不破坏现有 Turtle2 原生部署语义。

总览
----

.. code-block:: text

   server (Ray head)
     ├── actor / rollout / OpenPI
     └── 只负责模型推理与训练

   slave (Ray worker)
     ├── RLinf env
     ├── Turtle2 controller
     └── MasterTakeoverIntervention + TCP takeover adapter

   master (non-Ray)
     ├── 主臂 + takeover 逻辑
     └── 仅作为远程专家动作源

约束固定为：

- ``master`` 不加入 Ray
- ``slave env`` 是机器人唯一执行者
- 旧的 slave 执行脚本不能再直接向机器人下发控制命令
- TCP 双向 takeover 语义要保留；只替换 slave 侧最终执行入口

为什么不是 SpaceMouse/GELLO
-----------------------------

RLinf 现有真机 HG-DAgger 只要求 env-side wrapper 在 ``info["intervene_action"]``
里写入专家动作即可，SpaceMouse/GELLO 都遵循这个模式。

但 Turtle2 的 master/slave 接管不是简单的单向专家输入，它还依赖：

- ``running_mode`` 模式切换通知
- ``slave -> master`` 的 joint snapshot
- master 先对齐 slave 再进入 takeover
- 接管前清空旧 pose
- 接管延迟窗口和 fresh pose gating

因此，Turtle2 的实现不是复用现有 SpaceMouse/GELLO wrapper，而是单独新增
``MasterTakeoverIntervention``。

时序
----

TCP 版 v1 时序固定为：

1. ``slave/env`` 观察到 ``running_mode`` 从 normal 切到 takeover
2. env-side TCP adapter 向 master 发送 ``MSG_MODE``
3. env-side TCP adapter 向 master 发送 ``MSG_JOINT``，内容是当前双臂 joint snapshot
4. adapter 清空旧 pose 缓存，并启动延迟窗口
5. master 完成主臂对齐 slave 后，开始持续发送 ``MSG_POSE``
6. 只有延迟窗口结束后、并且时间戳晚于 barrier 的 pose，才允许转成
   ``intervene_action``
7. ``MasterTakeoverIntervention`` 用该专家动作覆盖 policy action
8. ``RealWorldEnv`` 把它转成 ``intervene_flag``，后续 replay buffer / DAgger 链复用现有实现

接口
----

新增模块分工如下：

- ``x2robot_protocol.py``
  - 负责 TCP frame 编解码
  - 负责 master/slave takeover 状态机
  - 负责 ``MSG_MODE / MSG_JOINT / MSG_POSE`` 的收发
- ``MasterTakeoverIntervention``
  - 正常时放行 policy action
  - takeover 激活时输出双臂 14D absolute pose
  - 将覆盖后的动作写入 ``info["intervene_action"]``
- ``Turtle2Env``
  - 新增只读快照接口：
  - ``get_arm_pose_snapshot()``
  - ``get_joint_snapshot()``

控制权
------

这一版最重要的工程约束是 **控制权单一**：

- RLinf env 运行时，机器人最终执行只能来自 ``Turtle2Env.step()``
- 旧的 ``bi_teleop_slave.py`` / ``socket2ros_async.py`` 不能继续向机器人发控制
- 如果底层仍需要保留旧脚本，它们只能退化成协议桥或状态桥，不能保留执行权

未决问题
--------

当前 RLinf DAgger 更偏向保存“完整专家 chunk”，而 Turtle2 现有 deploy 合同使用
``num_action_chunks=30``。因此 v1 文档先固定两个结论：

- 代码保持现有 ``30-step chunk`` 不变
- 是否允许“半 chunk takeover 立即生效”留作下一阶段议题

如果后续要求接管在 chunk 中途立即生效，则需要单独评估 trajectory /
replay buffer 对 partial-chunk intervention 的支持。
