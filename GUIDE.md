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
  logs/x2_upper_body_sim \
  --output-dir datasets/x2_upper_body_lerobot
```

## 2. X2 上半身实机数据采集

### 2.1 环境准备
在仓库根目录执行：

```bash
conda activate xr_tabletop
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 python3 -m pip install -e .
```

如果你的机器人驱动消息定义在工作区里，还需要额外 `source` 对应工作区，例如：

```bash
source ros2_ws/install/setup.bash
```

说明：
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
  --log-dir logs/x2_upper_body_hardware \
  --control-rate-hz 30
```

如果需要显示头部相机图像，需要修改teleop_x2_upper_body_hardware.py中的show_camera_window参数（开启相机显示可能会导致关节控制卡顿，因此默认为False）


如果需要打印 `IK raw / clip / final` 三层关节目标，可使用：

```bash
PYTHONNOUSERSITE=1 python3 scripts/hardware/teleop_x2_upper_body_hardware.py \
  --log-dir logs/x2_upper_body_hardware \
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
- `--log-dir`：日志输出目录。
- `--control-rate-hz`：控制频率，默认 `100`。
- `--scale-factor`：XR 位移到机器人位移的缩放，默认 `1.2`。
- `--enable-head-tracking False`：关闭头部跟随。
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
- `--debug-print-targets` / `--debug-print-hz 2`：打印 `IK raw / clip / final` 和当前实机目标值。


### 2.3 采集按键
运行后使用 XR 手柄控制：
- 按 `B` 一次：开始记录
- 再按一次 `B`：停止并保存 `.pkl`
- 按 `right_axis_click`：停止并丢弃当前记录
- 长按 `right_menu_button` 约 `0.5` 秒：触发软件急停，机器人保持当前关节位置并退出 teleop

### 2.4 日志输出
日志会保存到：
- `logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_N.pkl`

每条记录通常包含：
- `timestamp`
- 机器人当前状态与目标状态
- `xr`（头显与手柄姿态、按键值）
- 若后续启用相机记录，还会包含 `image`

### 2.5 启动前自检
建议先确认这三项：
- `scripts/misc/check_pico_xr_connection.py` 能看到非零 `pose` 和有效 `timestamp_ns`
- ROS2 话题里能收到 `/aima/hal/joint/arm/state`、`/aima/hal/joint/head/state`
- PC Service 已连接 Pico，且实机控制节点已经在线

### 2.6 将实机日志转换为 LeRobot v3 数据集

如果你已经采集好一条或多条实机 `.pkl` 日志，可以使用：

[`scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py`](/home/xlq/XRoboToolkit-Teleop-Sample-Python/scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py)

该脚本的默认设计是：
- 默认只导出双臂数据，不导出头部状态和头部命令。
- 默认将 `arm_state` 写入 `observation.state`。
- 默认将 `arm_command` 写入 `action`。
- 默认将前视相机写成 `observation.images.<camera_name>` 对应的视频文件。
- 优先尝试使用官方 `lerobot` SDK。
- 如果本机没有安装 `lerobot`，会自动回退到内置导出逻辑。

#### 直接转换

如果你的日志保存在默认目录：

```bash
python3 scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py \
  logs/x2_upper_body_hardware \
  --output-dir datasets/x2_hardware_lerobot_v3 \
  --camera-name head_front \
  --task "x2 teleop"
```

如果只想转换某一条日志，也可以直接传单个 `.pkl` 文件：

```bash
python3 scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py \
  logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_1.pkl \
  --output-dir datasets/x2_hardware_lerobot_v3 \
  --camera-name head_front \
  --task "x2 teleop"
```

参数说明：
- `log_paths`：可以是一个日志目录，也可以是一个或多个 `.pkl` 文件。
- `--output-dir`：导出后的数据集目录。
- `--camera-name`：读取 `entry["image"]` 时使用的相机键名。当前项目推荐使用 `head_front`。
- `--task`：写入数据集 metadata 的任务描述。
- `--fps`：可选。默认会根据日志中的 `timestamp` 自动估算；估算失败时回退到 `30`。
- `--include-head`：可选。若传入该参数，会将 `head_state` 和 `head_command` 一并写入数据集。
- `--repo-id`：可选。仅在本机安装了官方 `lerobot` SDK 时使用。

#### 导出结果

导出完成后，结果会保存在：

- `datasets/x2_hardware_lerobot_v3/data/chunk-000/file-000.parquet`
- `datasets/x2_hardware_lerobot_v3/videos/head_front/chunk-000/file-000.mp4`
- `datasets/x2_hardware_lerobot_v3/meta/info.json`
- `datasets/x2_hardware_lerobot_v3/meta/stats.json`
- `datasets/x2_hardware_lerobot_v3/meta/tasks.parquet`
- `datasets/x2_hardware_lerobot_v3/meta/episodes/chunk-000/file-000.parquet`

默认数据对齐方式：
- `observation.state` 使用当前日志时刻的双臂状态。
- `action` 使用当前日志时刻的双臂命令。
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

建议先用以下脚本检查 `.pkl` 内容是否完整：

```bash
python3 scripts/misc/test_data_log_analysis.py \
  logs/x2_upper_body_hardware/teleop_log_YYYYMMDD_HHMMSS_1.pkl
```

重点确认：
- 是否存在 `arm_state`
- 是否存在 `arm_command`
- 是否存在 `image`
- `image` 中是否有你要导出的相机键名，例如 `head_front`

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
