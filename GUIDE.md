# GUIDE

## 1. X2 上半身 MuJoCo 数据采集

### 1.1 环境准备
在仓库根目录执行：

```bash
conda activate xr_tabletop
PYTHONNOUSERSITE=1 python3 -m pip install -e .
```

说明：
- `python` 不可用时请使用 `python3`。
- `-e .` 会把当前项目安装为可编辑模式，避免 `ModuleNotFoundError: No module named 'xrobotoolkit_teleop'`。

### 1.2 启动采集

```bash
PYTHONNOUSERSITE=1 python3 scripts/simulation/teleop_x2_upper_body_mujoco.py \
  --enable-log-data \
  --enable-camera-log \
  --camera-names rgbd_head_front_camera \
  --camera-jpg-quality 0 \
  --log-dir logs/x2_upper_body_sim \
  --control-profile fast
```

参数说明：
- `--enable-log-data`：开启日志记录（必须开启，否则不会保存数据）。
- `--enable-camera-log`：开启仿真相机记录。
- `--camera-names`：要记录的相机名，多个相机可用逗号分隔。
- `--camera-jpg-quality 0`：保存原始 `uint8` RGB，后处理更稳。
- `--log-dir`：日志输出目录。

### 1.3 采集按键
运行后使用 XR 手柄控制：
- 按 `B` 一次：开始记录
- 再按一次 `B`：停止并保存 `.pkl`
- 按 `right_axis_click`：停止并丢弃当前记录

### 1.4 日志输出
日志会保存到：
- `logs/x2_upper_body_sim/teleop_log_YYYYMMDD_HHMMSS_N.pkl`

每条记录通常包含：
- `timestamp`
- `qpos`, `qvel`, `ctrl`, `qpos_des`
- `xr`（头显与手柄姿态、按键值）
- `image`（开启相机采集时）

### 1.5 采后检查

```bash
python3 scripts/misc/test_data_log_analysis.py logs/x2_upper_body_sim/<your_log>.pkl
```

### 1.6 数据集转换（可选）

#### 转 Tabletop HDF5

```bash
python3 scripts/misc/convert_sim_log_to_tabletop_hdf5.py \
  logs/x2_upper_body_sim \
  --output-dir datasets/x2_tabletop_hdf5 \
  --camera-name rgbd_head_front_camera \
  --instruction "Teleoperate x2 upper body in simulation"
```

#### 转 LeRobot

```bash
python3 scripts/misc/convert_x2_sim_log_to_lerobot.py \
  /media/xlq/8E34-513F/log \
  --output-dir datasets/x2_upper_body_lerobot
```

## 2. X2 上半身实机数据采集

### 2.1 环境准备
如果只采集 X2 实机数据，可以使用最小 conda 环境。在仓库根目录执行：

```bash
conda env create -f envs/environment.x2_hardware.yml
conda activate x2-hardware
source /opt/ros/humble/setup.bash
```

如果你的机器人驱动消息定义在工作区里，还需要额外 `source` 对应工作区，例如：

```bash
source ros2_ws/install/setup.bash
```

说明：
- `envs/environment.x2_hardware.yml` 只包含实机采集和日志转换需要的最小 Python 依赖，不包含 MuJoCo、Tabletop-Sim、RealSense SDK、UR、Dynamixel、dex_retargeting 等仿真或其他机器人依赖。
- 实机脚本依赖 `rclpy` 和 `aimdk_msgs`，没有 `source` ROS2 环境会直接启动失败。
- 若开启 Ruckig 平滑，还需要确保 `source ros2_ws/install/setup.bash` 后可以导入 `ruckig`。
- 当前脚本默认 `enable_log_data=True`，也就是启动后按 `B` 就会开始真正写日志。

### 2.2 启动实机遥操作采集

每次开始正式采集前，建议先执行一次回零脚本，让机器人回到站立零位：

```bash
PYTHONNOUSERSITE=1 python3 scripts/hardware/reset_x2_upper_body_zero.py \
  --move-duration-s 3.0 \
  --hold-duration-s 0.5
```

默认会将手臂、头部、腰部和腿部所有已接入的关节平滑运动到 `0` 位，并在终点保持一小段时间。  
其中膝关节会自动裁剪到它的最小允许角度，所以实际会停在接近直立站姿的位置。

然后再启动实机遥操作采集：

```bash
PYTHONNOUSERSITE=1 python3 scripts/hardware/teleop_x2_upper_body_hardware.py \
  --log-dir /media/xlq/8E34-513F/log2\
  --control-rate-hz 50
```

默认关闭头部跟随，头部 yaw/pitch 会持续下发 `0` 位保持。

默认会采集三路相机：
- `head_front`：`/aima/hal/sensor/rgbd_head_front/rgb_image`
- `right_wrist`：`/right/rgb/compressed`
- `left_wrist`：`/left/rgb/compressed`

腕部相机当前按 `30Hz` 发布

如果需要显示相机图像，可以传入 `--show-camera-window True`（开启相机显示可能会导致关节控制卡顿，因此默认为 `False`）。

如果需要打印 `IK raw / clip / final` 三层关节目标，可使用：

```bash
PYTHONNOUSERSITE=1 python3 scripts/hardware/teleop_x2_upper_body_hardware.py \
  --log-dir /media/xlq/ESD-USB \
  --control-rate-hz 30 \
  --debug-print-targets \
  --debug-print-hz 2
```

如果想把“实机最终下发的关节命令”实时镜像到 MuJoCo 里做对照，可在另一个终端启动：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
PYTHONNOUSERSITE=1 python3 scripts/simulation/replay_x2_hardware_command_in_mujoco.py
```

如果想把“机器人实机回传的关节状态（state）”实时映射到 MuJoCo，可在另一个终端启动：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
PYTHONNOUSERSITE=1 python3 scripts/simulation/replay_x2_hardware_state_in_mujoco.py
```

该脚本订阅：
- `/aima/hal/joint/arm/command`
- `/aima/hal/joint/head/command`

状态镜像脚本订阅：
- `/aima/hal/joint/arm/state`
- `/aima/hal/joint/head/state`

用途：
- 看到“实机控制链最终发出的目标姿态”在 MuJoCo 中是什么样子。
- 用来区分问题出在 `IK`、实机限幅/限步，还是底层执行器跟踪。

补充说明：
- `replay_x2_hardware_command_in_mujoco.py`：看“命令侧”结果（controller 下发目标）。
- `replay_x2_hardware_state_in_mujoco.py`：看“状态侧”结果（机器人实际回传姿态）。
- 两者对比可快速定位“命令发出正确但执行不到位”或“命令本身已异常”的问题。

常用可选参数：
- `--log-dir`：日志输出目录，默认 `/media/xlq/ESD-USB`。
- `--validate-log-before-save`：保存前是否运行数据健康检查，默认 `True`。
- `--decode-images-on-log-validate`：健康检查时是否解码图像，默认 `True`。
- `--log-freq`：数据记录频率，默认 `15Hz`，与腕部相机发布频率一致。
- `--control-rate-hz`：控制频率，实机启动脚本默认 `30Hz`。
- `--scale-factor`：XR 位移到机器人位移的缩放，默认 `1.2`。
- `--enable-head-tracking`：是否启用头部跟随，默认 `False`；关闭时头部保持 `0` 位。
- `reset_x2_upper_body_zero.py --move-duration-s`：回零时长，默认 `3.0` 秒。
- `reset_x2_upper_body_zero.py --hold-duration-s`：到达零位后的保持时间，默认 `0.5` 秒。
- `reset_x2_upper_body_zero.py --include-lower-body`：是否同时将腰腿回到零位，默认 `True`。
- `--enable-ruckig-smoothing`：是否在发 ROS2 命令前使用 Ruckig 平滑，默认 `True`。
- `--arm-ruckig-max-velocity / --arm-ruckig-max-acceleration / --arm-ruckig-max-jerk`：手臂 Ruckig 约束。
- `--head-ruckig-max-velocity / --head-ruckig-max-acceleration / --head-ruckig-max-jerk`：头部 Ruckig 约束。
- `--software-estop-button`：软件急停按键，默认 `right_menu_button`。
- `--software-estop-hold-s`：软件急停长按触发时间，默认 `0.5` 秒。
- `--arm-state-topic` / `--arm-command-topic`：自定义手臂状态和命令话题。
- `--head-state-topic` / `--head-command-topic`：自定义头部状态和命令话题。
- `--camera-color-topics` / `--camera-depth-topics`：自定义相机话题，格式为 `camera_name=/topic`，多个相机用逗号分隔。
- `--camera-enable-depth`：是否记录深度图，默认 `False`。
- `--debug-print-targets` / `--debug-print-hz 2`：打印 `IK raw / clip / final` 和当前实机目标值。

### 2.3 采集按键
运行后使用 XR 手柄控制：
- 按 `B` 一次：开始记录
- 再按一次 `B`：停止并保存 `.pkl`
- 按 `right_axis_click`：停止并丢弃当前记录
- 长按 `right_menu_button` 约 `0.5` 秒：触发软件急停，机器人保持当前关节位置并退出 teleop

### 2.4 日志输出
日志默认会保存到：
- `/media/xlq/ESD-USB/teleop_log_YYYYMMDD_HHMMSS_N.pkl`

每次按 `B` 停止记录时，程序会先写入临时 `.tmp.pkl`，然后使用数据健康检查脚本检查结构、时间戳、数值字段和图像解码；检查没有 `ERROR` 后才会改名为正式 `.pkl` 文件。如果检查失败，本次日志不会保存为正式文件。

每条记录通常包含：
- `timestamp`
- 机器人当前状态与目标状态
- `xr`（头显与手柄姿态、按键值）
- `image`，默认三路相机只包含 `color`
- 夹爪会记录在 `hand_command` 和 `hand_trigger_raw`

### 2.5 启动前自检
建议先确认这三项：
- `scripts/misc/check_pico_xr_connection.py` 能看到非零 `pose` 和有效 `timestamp_ns`
- ROS2 话题里能收到 `/aima/hal/joint/arm/state`、`/aima/hal/joint/head/state`
- PC Service 已连接 Pico，且实机控制节点已经在线

### 2.6 实机日志检查与坏帧修复

训练前建议先检查 `.pkl` 日志，尤其是 `arm_command` 是否每一帧都有完整关节字段：

```bash
python3 scripts/misc/check_teleop_log_health.py logs/x2_upper_body_hardware
```

检查脚本除了结构和图像检查，还会做业务层面的运动量观察。如果某条日志的 `arm_command` 和 `arm_state` 基本不变化，会在单条报告中标记 `mostly static`，并在最后的 `Mostly static logs` 汇总里列出文件名。这类数据结构上可能是健康的，但通常不适合当作有效操作数据训练。

如果只检查某一条日志：

```bash
python3 scripts/misc/check_teleop_log_health.py \
  logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_1.pkl
```

如果还想验证相机图像能否正常解码，可以加：

```bash
python3 scripts/misc/check_teleop_log_health.py \
  logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_1.pkl \
  --decode-images
```

若检查结果里出现类似 `field 'arm_command' has inconsistent structure`，通常是某些帧的 `arm_command` 为空或字段不完整。可以先 dry-run 查看会删除哪些帧：

```bash
python3 scripts/misc/repair_teleop_log_frames.py logs/x2_upper_body_hardware
```

如果检查结果是 `Ran out of input`、`pickle data was truncated`，说明该 `.pkl` 文件为空或写坏了，无法恢复单帧数据。修复脚本会在 `--write` 模式下直接删除这类坏文件。

确认后原地修复，并默认给被修改的文件生成 `.bak` 备份：

```bash
python3 scripts/misc/repair_teleop_log_frames.py logs/x2_upper_body_hardware --write
```

如果不需要 `.bak` 备份，可以关闭备份；此时坏帧会直接从可读日志中删除，空文件/损坏文件会直接移除：

```bash
python3 scripts/misc/repair_teleop_log_frames.py logs/x2_upper_body_hardware --write --no-backup
```

如果只想删除坏帧、不想删除空文件或损坏文件，可以加：

```bash
python3 scripts/misc/repair_teleop_log_frames.py \
  logs/x2_upper_body_hardware \
  --write \
  --keep-bad-files
```

如果不想动原始日志，可以把修复后的文件写到新目录：

```bash
python3 scripts/misc/repair_teleop_log_frames.py \
  logs/x2_upper_body_hardware \
  --write \
  --output-dir repaired_logs/x2_upper_body_hardware
```

修复后再检查一次：

```bash
python3 scripts/misc/check_teleop_log_health.py repaired_logs/x2_upper_body_hardware
```

如果是原地修复，则检查原目录：

```bash
python3 scripts/misc/check_teleop_log_health.py logs/x2_upper_body_hardware
```

### 2.7 可视化回放验证数据质量

健康检查能发现结构、时间戳、数值和图像解码问题；如果想进一步确认“画面是否正常、动作是否连续、state/action 是否对得上”，可以用回放脚本：

```bash
python3 scripts/misc/play_collected_data.py \
  /media/xlq/8E34-513F/log/teleop_log_20260521_131406_1.pkl
```

回放窗口会显示：
- 多路相机画面拼接，例如 `head_front`、`right_wrist`、`left_wrist`；默认按原始视频像素显示，不缩放、不在图像区域叠字
- 每路相机的分辨率和视频 FPS
- `state` 曲线，用来观察机器人回传状态是否连续
- `action` 曲线，用来观察控制命令是否连续、是否有突跳或长时间不动

播放快捷键：
- 空格：暂停 / 继续
- `n`：暂停时单步前进
- `q` 或 `Esc`：退出

如果日志里有多路相机，也可以只看指定相机：

```bash
python3 scripts/misc/play_collected_data.py \
  /media/xlq/8E34-513F/log/teleop_log_20260521_131406_1.pkl \
  --camera-names head_front,right_wrist,left_wrist
```

如果已经转换成 LeRobot v3 数据集，也可以直接回放数据集目录：

```bash
python3 scripts/misc/play_collected_data.py datasets/x2_hardware_lerobot_v3
```

读取 LeRobot 数据集里的 parquet 文件需要 `pyarrow`。如果出现 `Missing dependency: pyarrow`，在当前环境安装：

```bash
PYTHONNOUSERSITE=1 python3 -m pip install pyarrow
```

在没有显示器或通过 SSH 运行时，可以只做非 GUI 摘要检查：

```bash
python3 scripts/misc/play_collected_data.py datasets/x2_hardware_lerobot_v3 --no-display
```

摘要会打印帧数、FPS、state/action 是否全是有限值、各维运动范围，以及视频文件是否能读到首帧。若某条数据 `moving dims` 很少、`max_range` 很小，通常说明该条数据运动量不足，不适合作为有效训练样本。

如果屏幕放不下原始分辨率画面，可以额外传 `--fit-window --max-width 1280 --max-height 900` 临时缩放显示。

### 2.8 让实机按采集数据回放运动

如果需要让实机按照采集到的 `arm_command` 或 LeRobot 数据集中的 `action` 运动，可以使用：

```bash
python3 scripts/hardware/replay_x2_collected_data_on_hardware.py \
  logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_1.pkl
```

默认是 dry-run，只会检查轨迹并打印帧数、FPS、关节范围和最大单步变化，不会向机器人发布命令。确认无误后，显式加入 `--execute-hardware` 才会真正下发 ROS2 command：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
PYTHONNOUSERSITE=1 python3 scripts/hardware/replay_x2_collected_data_on_hardware.py \
  /media/xlq/8E34-513F/log/teleop_log_20260521_150158_75.pkl\
  --execute-hardware \
  --move-to-start \
  --speed 0.5
```

常用参数：
- `--execute-hardware`：真正驱动实机的开关；不加时永远只 dry-run。
- `--move-to-start`：回放前先从当前姿态平滑移动到第一帧，建议实机回放时开启。
- `--speed 0.5`：半速回放，首次验证建议低速。
- `--source-field arm_command`：使用原始 `.pkl` 中的命令侧数据回放；也可以改为 `arm_state`。
- `--max-step-rad 0.08`：每个发布周期允许的最大关节变化，超出会被裁剪。
- `--start-index / --end-index`：只回放一小段轨迹，首次验证建议先截取短片段。
- `--include-hands / --no-include-hands`：是否回放 `left_hand` / `right_hand` 夹爪命令，默认有夹爪数据就回放。
- `--include-head`：当日志中有头部命令或状态时，同时回放头部。

如果当前机器人姿态和第一帧差距较大，且没有传 `--move-to-start`，脚本会拒绝执行实机回放，避免突然跳到轨迹起点。

转换后的 LeRobot v3 数据集也可以直接回放，默认使用 `action`：

```bash
PYTHONNOUSERSITE=1 python3 scripts/hardware/replay_x2_collected_data_on_hardware.py \
  datasets/x2_hardware_lerobot_v3 \
  --execute-hardware \
  --move-to-start \
  --speed 0.5
```

注意：LeRobot 数据集必须在 `action` 的 `names` 里包含 `left_hand.pos` / `right_hand.pos`，实机回放脚本才能控制夹爪。如果 dry-run 输出 `hand command frames: 0/N`，说明当前数据集没有夹爪目标值；需要用包含 `hand_command` 的原始 `.pkl` 重新转换，并且不要传 `--exclude-hands`。

如果回放 LeRobot 数据集时报 `Missing dependency: pyarrow`，先在当前环境安装：

```bash
PYTHONNOUSERSITE=1 python3 -m pip install pyarrow
```

实机回放前建议先做三步：
- 先运行 `play_collected_data.py --no-display` 或可视化播放，确认数据本身健康。
- 先不加 `--execute-hardware` dry-run，确认最大单步变化和帧数合理。
- 第一次实机执行用低速、短片段，并保持急停可用。

### 2.9 将实机日志转换为 LeRobot v3 数据集

如果你已经采集好一条或多条实机 `.pkl` 日志，可以使用：

[`scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py`](/home/xlq/XRoboToolkit-Teleop-Sample-Python/scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py)

该脚本的默认设计是：
- 默认只导出双臂数据，不导出头部状态和头部命令。
- 默认将 `arm_state` 写入 `observation.state`。
- 默认将 `arm_command` 和 `hand_command` 写入 `action`，顺序为双臂 14 维后追加 `left_hand`、`right_hand`。
- 默认导出 `head_front`、`right_wrist`、`left_wrist` 三路相机，分别写成 `observation.images.<camera_name>` 对应的视频文件。
- 当前 LeRobot 导出只使用 `color` 图像；原始 `.pkl` 里左右腕部的 `depth` 会保留在日志中，但不会写入当前 LeRobot 视频。
- 优先尝试使用官方 `lerobot` SDK。
- 如果本机没有安装 `lerobot`，会自动回退到内置导出逻辑。

#### 直接转换

如果你的日志保存在默认目录：

```bash
python3 scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py \
  /media/xlq/8E34-513F/log\
  --output-dir datasets/x2_hardware_lerobot_v3 \
  --camera-names head_front,right_wrist,left_wrist \
  --include-hands \
  --instruction "Insert the red cylinder into the black holder ."
```

如果只想转换某一条日志，也可以直接传单个 `.pkl` 文件：

```bash
python3 scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py \
  logs/x2_upper_body_hardware/teleop_log_20260428_184946_1.pkl \
  --output-dir datasets/x2_hardware_lerobot_v3 \
  --camera-names head_front,right_wrist,left_wrist \
  --include-hands \
  --instruction "Insert the red cylinder into the black holder using the left hand."
```

参数说明：
- `log_paths`：可以是一个日志目录，也可以是一个或多个 `.pkl` 文件。
- `--output-dir`：导出后的数据集目录。
- `--camera-names`：读取 `entry["image"]` 时使用的相机键名，多个相机用逗号分隔。当前项目默认使用 `head_front,right_wrist,left_wrist`。
- `--camera-name`：兼容旧用法，只导出单路相机。
- `--instruction`：写入数据集的语言指令，例如 `Insert the red cylinder into the black holder using the left hand.`。该参数会覆盖 `--task`。
- `--task`：兼容旧用法，也会作为任务/语言指令写入 metadata 和数据帧。
- `--fps`：可选。默认会根据日志中的 `timestamp` 自动估算；估算失败时回退到 `30`。
- `--include-head`：可选。若传入该参数，会将 `head_state` 和 `head_command` 一并写入数据集。
- `--include-hands / --no-include-hands`：是否将 `left_hand`、`right_hand` 夹爪控制量追加到 `action`，默认开启。兼容旧参数 `--exclude-hands`。
- `--repo-id`：可选。仅在本机安装了官方 `lerobot` SDK 时使用。

转换时终端会打印 `Action layout`。带夹爪的 X2 上半身数据应为 `16 dim`，其中最后两维是 `left_hand`、`right_hand`。如果仍是 `14 dim`，说明转换时关闭了手部导出，或者原始 `.pkl` 中没有 `hand_command` / `hand_trigger_raw`。

#### 导出结果

导出完成后，结果会保存在：

- `datasets/x2_hardware_lerobot_v3/data/chunk-000/file-000.parquet`
- `datasets/x2_hardware_lerobot_v3/videos/head_front/chunk-000/file-000.mp4`
- `datasets/x2_hardware_lerobot_v3/videos/right_wrist/chunk-000/file-000.mp4`
- `datasets/x2_hardware_lerobot_v3/videos/left_wrist/chunk-000/file-000.mp4`
- `datasets/x2_hardware_lerobot_v3/meta/info.json`
- `datasets/x2_hardware_lerobot_v3/meta/stats.json`
- `datasets/x2_hardware_lerobot_v3/meta/tasks.parquet`
- `datasets/x2_hardware_lerobot_v3/meta/episodes/chunk-000/file-000.parquet`

默认数据对齐方式：
- `observation.state` 使用当前日志时刻的双臂状态。
- `action` 使用当前日志时刻的双臂命令和夹爪命令。
- `action` 默认维度为 16：双臂 14 维 + `left_hand` + `right_hand`。
- `task` 每帧写入同一条语言指令。
- 图像使用该日志条目中保存的最近一帧相机图像。

#### 可选：安装官方 LeRobot SDK

如果你希望转换脚本优先使用官方 `lerobot` SDK，可以单独创建一个环境：

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg=7.1.1
python -m pip install -U pip
python -m pip install lerobot
```

说明：
- `ffmpeg 7.1.1` 是推荐版本。
- 如果安装过程中因磁盘空间不足失败，可以先不安装 `lerobot`，直接使用脚本内置的回退导出逻辑。
- 当前项目的数据转换并不强依赖官方 SDK。

#### 导出前建议

建议先用健康检查脚本确认 `.pkl` 内容是否完整：

```bash
python3 scripts/misc/check_teleop_log_health.py \
  logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_1.pkl \
  --decode-images
```

重点确认：
- 是否存在 `arm_state`
- 是否存在 `arm_command`
- 是否存在 `image`
- `image` 中是否有你要导出的相机键名，例如 `head_front`、`right_wrist`、`left_wrist`
- 是否被标记为 `mostly static`

## 3. X2 Omnihands 上半身 MuJoCo 数据采集

### 3.1 先生成 Omnihands 场景 XML（一次性）

在 `XR` 环境先执行（建议每次改 URDF 后都重新执行一次）：

```bash
PYTHONNOUSERSITE=1 python3 scripts/misc/generate_x2_omnihands_scene_xml.py
```

会生成：
- `X2_URDF/x2_ultra_with_omnihands_generated.xml`
- `X2_URDF/scene_upper_body_omnihands_position.xml`

可选验证（确保场景可被 MuJoCo 直接加载）：

```bash
python3 -c "import mujoco; m=mujoco.MjModel.from_xml_path('X2_URDF/scene_upper_body_omnihands_position.xml'); print('nq=', m.nq, 'nu=', m.nu, 'ncam=', m.ncam)"
```

### 3.2 启动采集（新脚本）

```bash
PYTHONNOUSERSITE=1 python3 scripts/simulation/teleop_x2_omnihands_mujoco.py \
  --enable-log-data \
  --enable-camera-log \
  --camera-names rgbd_head_front_camera \
  --camera-jpg-quality 0 \
  --log-dir logs/x2_omnihands_upper_body_sim \
  --control-profile low_latency
```

说明：
- 该脚本仍然是“只控制上半身”逻辑，和 `teleop_x2_upper_body_mujoco.py` 一致。
- 默认使用 `X2_with_omnihands_URDF/x2_ultra_with_omnihands.urdf` 解析灵巧手关节。
- 默认使用 `X2_URDF/scene_upper_body_omnihands_position.xml`（包含手部关节 actuator）。

### 3.3 关键参数

- `--thumb-abad-angle`：
  - 不传：左右拇指 `thumb_abad` 固定在 URDF 最大角度。
  - 传入数值：固定到该角度（自动限幅到关节上下限）。
- `--allow-missing-hand-joints`（默认 `False`）：
  - 默认按完整 omnihands 场景运行，手关节缺失会报错，便于尽早发现模型问题。
  - 若你切回无手关节 XML，可手动设为 `True` 跳过手关节下发。
- `--sim-steps-per-control`：
  - 该脚本支持低延迟配置，显式传入值会直接生效（不会被 profile 强制提高）。

### 3.4 采集按键

与原上半身脚本一致：
- 按 `B` 一次：开始记录
- 再按一次 `B`：停止并保存 `.pkl`
- 按 `right_axis_click`：停止并丢弃当前记录

### 3.5 日志新增字段

在原有 `xr` 基础上，新增 `hand_control` 字段（每侧手一组）：
- `active`：该手是否处于 Grip 激活状态
- `trigger_raw`：该侧 trigger 原始值
- `trigger_when_active`：仅在激活时记录的 trigger 值
- `driver_joint_targets`：手指主动关节目标值（由 trigger 映射）
- `thumb_abad_angle`：当前固定拇指外展角

### 3.6 快速检查

```bash
python3 scripts/misc/test_data_log_analysis.py logs/x2_omnihands_upper_body_sim/<your_log>.pkl
```
