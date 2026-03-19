Franka 真机使用 ZED 相机与 Robotiq 夹爪
==========================================

本指南介绍如何在 RLinf 的 Franka 真实世界环境中配置和使用 **Stereolabs ZED 相机**
以及 **Robotiq 2F-85/2F-140 夹爪**。本文是基础 :doc:`franka` 文档的扩展，仅涵盖
ZED 和 Robotiq 硬件所需的 **额外** 步骤。

.. note::

   如果你还没有阅读过基础的 Franka 指南，请先参考 :doc:`franka`。
   本页仅涉及 ZED 和 Robotiq 硬件相关的额外配置。


硬件架构概览
-----------------

典型的 ZED + Robotiq 部署使用 **两个节点**：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 节点
     - 角色
     - 硬件
   * - **GPU 服务器** (node 0)
     - Actor、rollout、env worker；相机采集
     - NVIDIA GPU（如 RTX 4090），1-3 个 ZED 相机
   * - **NUC** (node 1)
     - FrankaController、Robotiq 夹爪
     - Franka 机械臂、Robotiq 2F（USB-RS485）

GPU 服务器运行 ZED 相机，因为 ZED SDK 利用 GPU 加速进行深度和图像处理。
Robotiq 夹爪通过 USB 转 RS485 适配器连接到 NUC（或与机械臂物理连接的机器）。


ZED 相机安装
-----------------------

需要在所有进行图像采集的节点（通常为 GPU 服务器）上安装 ZED SDK 及其 Python API。
完整安装说明请参考
`ZED Python API 官方安装指南 <https://www.stereolabs.com/docs/development/python/install>`_。

1. 安装 ZED SDK
^^^^^^^^^^^^^^^^^^^^^^^^

从 `Stereolabs 下载页面 <https://www.stereolabs.com/developers/release>`_
下载 SDK 安装程序，选择与你的操作系统和 CUDA 版本匹配的版本。

.. code-block:: bash

   # 以 CUDA 12.x + Ubuntu 22.04 为例
   chmod +x ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run
   ./ZED_SDK_Ubuntu22_cuda12.1_v4.2.5.zstd.run

按照屏幕上的提示完成安装。当提示
``Do you want to install the Python API (recommended) [Y/n] ?`` 时，
按 **Y** 以自动安装 Python 绑定。

2. 安装 Python API（如果 SDK 安装时未安装）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果在 SDK 安装过程中跳过了 Python API，或需要安装到其他虚拟环境，
可以运行 SDK 自带的辅助脚本：

.. code-block:: bash

   # 先激活你的虚拟环境
   source /path/to/your/venv/bin/activate

   # 运行官方安装脚本
   cd /usr/local/zed/
   python3 get_python_api.py

该脚本会自动检测你的平台、Python 版本和 ZED SDK 版本，
然后下载并安装匹配的 ``pyzed`` wheel。

也可以直接安装 wheel 文件：

.. code-block:: bash

   python -m pip install --ignore-installed /usr/local/zed/pyzed-*.whl

.. note::

   ``pyzed`` wheel 与特定的 Python 版本和 CUDA 版本绑定。请确保将其安装到
   Ray 在该节点上使用的 **同一虚拟环境** 中。如果你正在使用虚拟环境，
   请在运行安装脚本 **之前** 先激活虚拟环境。

3. 验证相机检测
^^^^^^^^^^^^^^^^^^^^^^^^^^^

列出已连接的 ZED 相机并记录 **序列号**：

.. code-block:: bash

   python -c "
   import pyzed.sl as sl
   for dev in sl.Camera.get_device_list():
       print(f'Serial: {dev.serial_number}  Model: {dev.camera_model}')
   "

记录序列号，后续在 YAML 配置中需要使用。


Robotiq 夹爪安装
-----------------------------

Robotiq 夹爪通过 USB 转 RS485 适配器使用 **Modbus RTU** 协议通信。
所需的 Python 依赖 ``pymodbus`` 会在运行 Franka 安装脚本时 **自动安装**：

.. code-block:: bash

   bash requirements/install.sh embodied --env franka

以下步骤用于在控制夹爪的节点（通常为 NUC）上配置串口设备。

1. 配置串口设备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

将 USB 转 RS485 适配器插入 NUC。识别串口设备：

.. code-block:: bash

   ls /dev/ttyUSB*
   # 通常为 /dev/ttyUSB0

赋予权限：

.. code-block:: bash

   sudo chmod 666 /dev/ttyUSB0
   # 或者将用户添加到 dialout 组以获得持久访问权限：
   sudo usermod -aG dialout $USER

3. 验证夹爪
^^^^^^^^^^^^^^^^^^^^^^^

快速验证（在 NUC 上运行）：

.. code-block:: bash

   python -c "
   from rlinf.envs.realworld.common.gripper.robotiq_gripper import RobotiqGripper
   g = RobotiqGripper(port='/dev/ttyUSB0')
   print(f'Position: {g.position:.4f} m, Ready: {g.is_ready}')
   g.open()
   import time; time.sleep(1)
   g.close()
   g.cleanup()
   "


YAML 配置说明
-------------------

与标准配置（RealSense + Franka 夹爪）相比，主要区别在于 ``hardware.configs``
中新增了 ``camera_type``、``gripper_type``、``gripper_connection`` 和
``controller_node_rank`` 字段。

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       actor:
         node_group: gpu
         placement: 0
       env:
         node_group: gpu
         placement: 0
       rollout:
         node_group: gpu
         placement: 0
     node_groups:
       - label: gpu
         node_ranks: 0
       - label: franka
         node_ranks: 0-1
         hardware:
           type: Franka
           configs:
             - robot_ip: <ROBOT_IP>
               node_rank: 0
               camera_serials:
                 - "<ZED_SERIAL_1>"
                 - "<ZED_SERIAL_2>"
                 - "<ZED_SERIAL_3>"
               camera_type: zed            # "realsense" 或 "zed"
               gripper_type: robotiq       # "franka" 或 "robotiq"
               gripper_connection: "/dev/ttyUSB0"
               controller_node_rank: 1     # FrankaController 运行在 NUC 上
               disable_validate: false

.. list-table:: 新增硬件配置字段
   :header-rows: 1
   :widths: 25 15 60

   * - 字段
     - 默认值
     - 说明
   * - ``camera_type``
     - ``"realsense"``
     - 相机后端。设置为 ``"zed"`` 以使用 ZED 相机。
   * - ``gripper_type``
     - ``"franka"``
     - 夹爪后端。设置为 ``"robotiq"`` 以使用 Robotiq 夹爪。
   * - ``gripper_connection``
     - ``null``
     - Robotiq 串口路径（如 ``"/dev/ttyUSB0"``）。当 ``gripper_type``
       为 ``"franka"`` 时忽略。
   * - ``controller_node_rank``
     - ``null``
     - ``FrankaController`` 运行的节点 rank。为 ``null`` 时与 env worker
       共同部署。当机械臂和相机在不同机器上时需要设置。


Dummy 测试配置
-----------------

我们提供了一个现成的 dummy 配置，可以在 **没有** 真实机器人和相机的情况下验证集群设置：

``examples/embodiment/config/realworld_dummy_franka_zed_robotiq.yaml``

运行方式：

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_dummy_franka_zed_robotiq

该配置在 train 和 eval 环境中均设置了 ``is_dummy: True``，因此不会尝试连接真实硬件。
它使用与标准 Franka CNN dummy 配置相同的算法和模型设置，但将 ``camera_type``
设为 ``zed``、``gripper_type`` 设为 ``robotiq``，以验证后端选择的代码路径。


集群配置注意事项
---------------------

集群配置步骤与 :doc:`franka` 中描述的相同，主要区别如下：

- 在 **GPU 服务器** (node 0) 上：确保在运行 ``ray start`` **之前** 已在虚拟环境中
  安装好 ZED SDK 和 ``pyzed``。
- 在 **NUC** (node 1) 上：确保已安装 ``pymodbus``，且 Robotiq 串口设备可访问。

.. warning::

   请记住 Ray 会在 ``ray start`` 时捕获 Python 解释器和环境变量。任何在
   ``ray start`` **之后** 安装的 SDK 或库对 Ray worker 不可见。请务必先安装
   所有依赖，然后再启动 Ray。

关于多节点 Ray 配置的详细信息，请参考 :doc:`franka` 和
:doc:`../../tutorials/advance/hetero`。


故障排查
----------------

**ZED 相机未检测到**

- 确认 USB 3.0 数据线已连接且相机 LED 亮起。
- 运行 ``lsusb`` 查看是否有 ``Stereolabs`` 设备。
- 确保 ZED SDK 版本与 CUDA 版本匹配。

**Robotiq 夹爪无响应**

- 执行 ``ls /dev/ttyUSB*`` 确认串口设备存在。
- 检查权限：``sudo chmod 666 /dev/ttyUSB0``。
- 确保 ``pymodbus`` 版本为 ``>=3.0,<4.0``。
- 如果通信不稳定，尝试降低波特率：
  ``RobotiqGripper(port='/dev/ttyUSB0', baudrate=9600)``。

**控制器节点 rank 不匹配**

- 如果机械臂在 NUC (node 1) 上而相机在 GPU 服务器 (node 0) 上，
  **必须** 在硬件配置中设置 ``controller_node_rank: 1``。否则控制器会尝试
  在 node 0 上启动，该节点没有 ROS 或机械臂连接。
