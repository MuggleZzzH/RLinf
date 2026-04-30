下面这版是按我刚才实际跑通后修正过的流程。关键差异：`bring_up` 用 `bash -ic`，eval 加 `env.eval.keyboard_reward_wrapper=null`。

**1. arm-slave：冷启动底层 ROS**

```bash
ssh arm-slave

docker restart turtle2_release hybrid_robot_loc
sleep 5

docker exec turtle2_release bash -lc '
tmux kill-session -t stable_roscore 2>/dev/null || true
tmux new-session -d -s stable_roscore "source /opt/ros/noetic/setup.bash; while true; do roscore; sleep 1; done"
sleep 3
source /opt/ros/noetic/setup.bash
rosnode list
'

docker exec -d turtle2_release bash -ic \
  "bash /home/arm/prj/turtle2/utilities/bring_up/run.sh S >/tmp/turtle2_bringup.log 2>&1"

sleep 20

docker exec -i turtle2_release bash -i <<'EOS'
source /opt/ros/noetic/setup.bash
source /home/arm/prj/turtle2/modules/devel/setup.bash
rosparam set /running_mode 1

rostopic info /follow_pos_cmd_1
rostopic info /follow_pos_cmd_2
timeout 8 rostopic echo -n 1 /follow1_pos_back
timeout 8 rostopic echo -n 1 /follow2_pos_back
rostopic info /camera1/usb_cam1/image_raw
rostopic info /camera2/usb_cam2/image_raw
rostopic info /camera3/usb_cam3/image_raw
exit
EOS
```

**2. 5090：启动 Ray head**

```bash
ssh 5090

cd /home/user/zjh_projects/RLinf
source .venv/bin/activate

ray stop --force || true

export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=wlp131s0
export PYTHONPATH=/home/user/zjh_projects/RLinf:$PYTHONPATH

ray start --head \
  --node-ip-address=192.168.120.73 \
  --port=6379 \
  --disable-usage-stats

sleep 5
ray status --address=192.168.120.73:6379
```

**3. arm-slave：启动 Ray worker**

```bash
ssh arm-slave

docker exec robo_avatar_slave_v1_dev bash -lc '
cd /home/arm/Jiahao/RLinf
source /opt/ros/noetic/setup.bash
source /home/arm/prj/turtle2/utilities/env/env.sh
source /home/arm/prj/turtle2/modules/devel/setup.bash
source /home/arm/config/dev_setting.sh
source .venv/bin/activate

export PYTHONPATH=/home/arm/Jiahao/RLinf:$PYTHONPATH
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=wlp2s0
export RLINF_KEYBOARD_DEVICE=/dev/input/event1

ray stop --force || true
ray start --address=192.168.120.73:6379 \
  --node-ip-address=192.168.120.106 \
  --disable-usage-stats

rosparam set /running_mode 1
'
```

在 5090 验证双节点：

```bash
ssh 5090
cd /home/user/zjh_projects/RLinf
source .venv/bin/activate
ray status --address=192.168.120.73:6379
```

**4. arm-master：启动 master TCP client**

如果 master 双臂 launch 没开，先开：

```bash
ssh arm-master

source /opt/ros/noetic/setup.bash
source /home/arm/prj/hybrid-robot/rosWorkspace/devel/setup.bash

roslaunch /home/arm/prj/hybrid-robot/rosWorkspace/src/x2robot-master/open_master_moving.launch
```

再开一个终端，启动 master 端 gripper / 手柄输入节点；否则 `/gripper_left` 和 `/gripper_right` 不会随 trigger 更新，master pose 的第 7 维会一直是 0：

```bash
ssh arm-master

source /opt/ros/noetic/setup.bash
source /home/arm/prj/hybrid-robot/rosWorkspace/devel/setup.bash

roslaunch communication communication_MS.launch
```

另开一个终端：

```bash
ssh arm-master

tmux kill-session -t x1_bi_master 2>/dev/null || true
tmux new-session -d -s x1_bi_master '
source /opt/ros/noetic/setup.bash
source /home/arm/prj/hybrid-robot/rosWorkspace/devel/setup.bash
cd /home/arm/prj/hybrid-robot/rosWorkspace/src/x2robot-master/scripts
python3 bi_teleop_master.py --server-host 192.168.120.106 --server-port 8766 --send-rate 100
'

tmux capture-pane -pt x1_bi_master -S -80
```

**5. 5090：启动 eval + 保存数据**

```bash
ssh 5090

cd /home/user/zjh_projects/RLinf
source .venv/bin/activate

export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=wlp131s0
export PYTHONPATH=/home/user/zjh_projects/RLinf:$PYTHONPATH
export RLINF_REALWORLD_DATA_DIR=/home/arm/Jiahao/RLinf/logs/x1_takeover_rollouts_$(date +%Y%m%d_%H%M%S)

tmux kill-session -t x1_eval_takeover 2>/dev/null || true
tmux new-session -d -s x1_eval_takeover "
cd /home/user/zjh_projects/RLinf
source .venv/bin/activate
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=wlp131s0
export PYTHONPATH=/home/user/zjh_projects/RLinf:\$PYTHONPATH
export RLINF_REALWORLD_DATA_DIR=$RLINF_REALWORLD_DATA_DIR

bash examples/embodiment/run_realworld_eval.sh realworld_x1_fold_towel_takeover_collect_openpi \
  env.eval.keyboard_reward_wrapper=null \
  env.eval.data_collection.only_success=False \
  env.eval.data_collection.export_format=pickle \
  env.eval.max_episode_steps=180 \
  env.eval.max_steps_per_rollout_epoch=180 \
  env.eval.override_cfg.max_num_steps=180 \
  2>&1 | tee /tmp/x1_takeover_current.log
"

sleep 15
tail -n 120 /tmp/x1_takeover_current.log
```

**6. arm-slave：5 秒 policy → 5 秒接管 → 恢复**

等 5090 log 里看到 `Master takeover client connected` 后再切：

```bash
ssh arm-slave

docker exec -i turtle2_release bash -i <<'EOS'
source /opt/ros/noetic/setup.bash
source /home/arm/prj/turtle2/modules/devel/setup.bash

rosparam set /running_mode 1
echo "policy start"
sleep 5

echo "takeover start"
rosparam set /running_mode 2
sleep 5

echo "recover policy"
rosparam set /running_mode 1
rosparam get /running_mode
exit
EOS
```

**7. 验证结果**

5090 看 metrics：

```bash
ssh 5090
grep -E "eval/|intervened|episode_len|num_trajectories|Exception|Traceback" -n /tmp/x1_takeover_current.log | tail -n 80
```

arm-slave 找 pkl：

```bash
ssh arm-slave

docker exec robo_avatar_slave_v1_dev bash -lc '
find /home/arm/Jiahao/RLinf/logs -type f -name "*.pkl" -mmin -30 | sort
'
```

解析 intervention：

```bash
ssh arm-slave

docker exec -i robo_avatar_slave_v1_dev /home/arm/Jiahao/RLinf/.venv/bin/python - <<'PY'
import pickle, numpy as np, torch, glob, os

files = sorted(glob.glob("/home/arm/Jiahao/RLinf/logs/**/rank_0_env_0_episode_0_step_*.pkl", recursive=True), key=os.path.getmtime)
path = files[-1]
data = pickle.load(open(path, "rb"))

flags = []
for info in data["infos"]:
    val = info.get("intervene_flag", False) if isinstance(info, dict) else False
    if isinstance(val, torch.Tensor):
        val = bool(val.detach().cpu().numpy().any())
    elif isinstance(val, np.ndarray):
        val = bool(val.any())
    elif isinstance(val, (list, tuple)):
        val = bool(np.asarray(val).any())
    else:
        val = bool(val)
    flags.append(val)

idx = [i for i, x in enumerate(flags) if x]

states = []
for obs in data["observations"]:
    s = obs.get("states") if isinstance(obs, dict) else None
    if isinstance(s, torch.Tensor):
        s = s.detach().cpu().numpy()
    states.append(np.asarray(s, dtype=float).reshape(-1))
arr = np.stack(states)

print("path =", path)
print("success =", data["success"])
print("intervened =", data["intervened"])
print("observations =", len(data["observations"]))
print("actions =", len(data["actions"]))
print("intervene_indices =", idx)
print("state_shape =", arr.shape)
print("state_max_abs_delta =", np.round(np.max(np.abs(arr - arr[0]), axis=0), 5).tolist())
PY
```

**8. 收尾**

```bash
ssh arm-slave

docker exec -i turtle2_release bash -i <<'EOS'
source /opt/ros/noetic/setup.bash
rosparam set /running_mode 1
rosparam get /running_mode
exit
EOS

docker exec robo_avatar_slave_v1_dev bash -lc '
cd /home/arm/Jiahao/RLinf
source .venv/bin/activate
ray stop --force || true
'

docker exec turtle2_release bash -lc '
source /opt/ros/noetic/setup.bash
rosnode cleanup <<EOF
y
EOF
' || true
```

```bash
ssh 5090
cd /home/user/zjh_projects/RLinf
source .venv/bin/activate
ray stop --force || true
tmux kill-session -t x1_eval_takeover 2>/dev/null || true
```

```bash
ssh arm-master
tmux kill-session -t x1_bi_master 2>/dev/null || true
```

验收标准：`eval/intervened_once=1`、`eval/episode_len=180`、有 pkl、pkl 里 `intervened=True` 且 `intervene_indices` 非空。
