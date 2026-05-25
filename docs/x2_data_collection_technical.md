# X2 实机数据采集项目技术说明

本文档聚焦本仓库中 X2 上半身实机遥操作与数据采集链路，说明代码入口、运行线程、XR 到关节命令的转换逻辑、ROS2 通信、相机记录、日志保存、健康检查、数据转换和回放流程。

## 1. 项目定位

本项目是一个基于 XRoboToolkit 的机器人遥操作与数据采集样例工程。对 X2 机器人而言，核心目标是：

- 使用 Pico/XR 手柄的 6DoF 位姿控制 X2 左右手臂末端。
- 通过 ROS2 订阅实机关节状态，并发布手臂、头部、手爪命令。
- 同步记录关节状态、关节命令、手爪命令、可选头部状态和多路相机图像。
- 将原始 `.pkl` 日志检查、修复、回放，并可转换为 LeRobot v3 风格数据集。

X2 实机采集主入口是：

```bash
python3 scripts/hardware/teleop_x2_upper_body_hardware.py
```

核心实现文件：

- `scripts/hardware/teleop_x2_upper_body_hardware.py`：命令行入口，组装参数并启动控制器。
- `xrobotoolkit_teleop/hardware/x2_ros2_teleop_controller.py`：X2 ROS2 实机控制器。
- `xrobotoolkit_teleop/common/base_teleop_controller.py`：XR 输入、Placo IK、末端目标更新、手爪目标更新的通用逻辑。
- `xrobotoolkit_teleop/common/base_hardware_teleop_controller.py`：硬件遥操作线程、日志、相机生命周期。
- `xrobotoolkit_teleop/common/data_logger.py`：`.pkl` 日志缓存、保存、保存前健康检查。
- `xrobotoolkit_teleop/hardware/interface/ros2_camera.py`：ROS2 图像话题订阅和图像压缩缓存。

## 2. 启动链路

### 2.1 参数入口

`teleop_x2_upper_body_hardware.py` 使用 `tyro.cli(main)` 将 `main()` 的参数暴露为命令行参数。默认配置包括：

- URDF：`X2_URDF/x2_upper_body_no_waist.urdf`
- 控制频率：`control_rate_hz=30`
- 记录频率：`log_freq=30.0`
- 日志目录：`/media/xlq/ESD-USB`
- 默认开启日志：`enable_log_data=True`
- 默认开启相机：`enable_camera=True`
- 默认相机：
  - `head_front=/aima/hal/sensor/rgbd_head_front/rgb_image/compressed`
  - `right_wrist=/right/rgb/image_compressed`
  - `left_wrist=/left/rgb/image_compressed`
- 默认关闭头部跟随：`enable_head_tracking=False`
- 软件急停：长按 `right_menu_button`，默认 `0.5s`

入口函数只负责把这些参数传给 `X2Ros2TeleopController`，然后调用：

```python
controller.run()
```

### 2.2 控制器初始化

`X2Ros2TeleopController.__init__()` 主要完成以下准备：

1. 检查 ROS2 依赖是否可用。必须能 import `rclpy` 和 `aimdk_msgs`。
2. 建立 X2 关节命名、限位和控制增益配置。
3. 建立左右手臂末端配置，即 `DEFAULT_X2_MANIPULATOR_CONFIG`。
4. 如启用 Ruckig，则创建在线轨迹平滑器。
5. 调用父类初始化，创建 `XrClient`、`DataLogger`、Placo 机器人模型和 IK solver。

这里有一个细节：父类初始化期间会调用 `_robot_setup()` 和 `_placo_setup()`。而 `run()` 开始时也会再调用 `_robot_setup()`。X2 控制器通过 `_ros2_setup_complete` 防止 ROS2 节点重复初始化。

## 3. ROS2 接口

X2 实机控制通过 ROS2 话题通信。默认话题如下：

| 功能 | 默认话题 | 消息 |
| --- | --- | --- |
| 手臂状态订阅 | `/aima/hal/joint/arm/state` | `JointStateArray` |
| 手臂命令发布 | `/upper_body/teleop_joint_states` | `JointCommandArray` |
| 头部状态订阅 | `/aima/hal/joint/head/state` | `JointStateArray` |
| 头部命令发布 | `/aima/hal/joint/head/command` | `JointCommandArray` |
| 手爪命令发布 | `/aima/hal/joint/hand/command` | `HandCommandArray` |

`Ros2JointGroupInterface` 封装关节组的状态订阅和命令发布。它内部缓存：

- `_positions`：每个关节最新位置。
- `_velocities`：每个关节最新速度。
- `_state_seq`：状态帧序号，用于可选的 fresh state 检查。
- `timestamp`：最近一次状态回调时间。

发布命令时，每个关节会写入：

- `name`
- `position`
- `velocity`
- `effort=0`
- `stiffness=kp`
- `damping=kd`

关节限位和 PD 增益定义在 `DEFAULT_HARDWARE_COMMAND_SPECS`。手臂 14 个关节、头部 2 个关节都会在命令发布前裁剪到限位范围内。

## 4. XR 输入到手臂 IK 的转换

### 4.1 左右臂配置

`DEFAULT_X2_MANIPULATOR_CONFIG` 定义了两条机械臂：

- `left_arm`
  - 控制 link：`left_wrist_roll_link`
  - 输入源：`left_controller`
  - 激活键：`left_grip`
  - 手爪触发：`left_trigger`
- `right_arm`
  - 控制 link：`right_wrist_roll_link`
  - 输入源：`right_controller`
  - 激活键：`right_grip`
  - 手爪触发：`right_trigger`

每个末端都有输入滤波和限幅参数：

- `input_linear_deadband_m=0.003`
- `input_angular_deadband_rad=0.04`
- `input_position_alpha=0.35`
- `input_rotation_alpha=0.25`
- `max_target_linear_step_m=0.03`
- `max_target_angular_step_rad=0.35`
- `workspace_min_z=0.0`

这意味着手柄微小抖动会被 deadband 吃掉，较大的输入会经过低通滤波，并限制每次 IK 目标变化幅度。

### 4.2 激活与参考系

手臂不是一直跟随手柄。每条臂都需要对应 grip 值超过阈值才激活：

```text
left_grip  -> left_arm active
right_grip -> right_arm active
```

激活瞬间，系统记录两套参考：

- 当前机器人末端控制点位姿：`ref_ee_xyz/ref_ee_quat`
- 当前 XR 手柄位姿：`ref_controller_xyz/ref_controller_quat`

之后每一帧不是把手柄绝对位姿直接映射到机器人，而是计算“手柄相对激活瞬间的增量”，再叠加到机器人激活瞬间的末端位姿上：

```text
目标末端位姿 = 激活时机器人末端位姿 + 手柄相对位姿变化 * scale_factor
```

默认 `scale_factor=1.2`，即手柄移动 10cm，末端目标移动约 12cm。

### 4.3 坐标和姿态处理

XR SDK 返回姿态格式为：

```text
[x, y, z, qx, qy, qz, qw]
```

`BaseTeleopController._process_xr_pose()` 会：

1. 取出手柄位置和四元数。
2. 使用 `R_HEADSET_TO_WORLD` 将 XR/头显坐标转到机器人世界坐标。
3. 对四元数做同样坐标变换。
4. 对位置和旋转做 deadband + 平滑。
5. 计算相对参考手柄的平移增量和旋转增量。

### 4.4 Placo IK

`_placo_setup()` 用 URDF 创建 `placo.RobotWrapper`，并创建 `KinematicsSolver`。对 X2 上半身：

- 固定 floating base。
- 为左右腕部 link 创建 frame task，也就是完整 6DoF 位姿任务。
- 为每个末端添加 manipulability task，避免 IK 过度靠近奇异位形。
- 在 X2 子类中额外添加 `joints_task`，作为低权重关节正则。

每个 IK 周期会：

1. 从 ROS2 状态缓存读取当前手臂关节。
2. 将硬件关节状态写入 Placo 模型的 `q`。
3. 读取 XR 输入并更新左右末端 frame task。
4. 调用 `solver.solve(True)`。
5. 从 Placo `q` 中取出 14 个手臂目标关节，准备发布给实机。

未激活的手臂会把 IK 任务目标锁在当前 wrist link 位姿，避免另一只手臂运动时未激活手臂被 solver 带着漂移。

## 5. 手爪与头部逻辑

### 5.1 手爪

手爪使用 `parallel` 类型配置。触发器数值范围为 `[0, 1]`：

```text
left_trigger  -> left_hand
right_trigger -> right_hand
```

X2 当前配置中：

```text
open_pos = 1.0
close_pos = 0.0
```

因此触发值越大，目标越接近 `0.0`，即越闭合。最后由 `Ros2HandCommandInterface.publish_command()` 发布到 `/aima/hal/joint/hand/command`，左右手类型设置为 `HandType.CLAW`。

### 5.2 头部

默认 `enable_head_tracking=False`。此时控制器持续将头部 yaw/pitch 目标设为 `0`，用于保持头部在中位。

如果开启头部跟随：

1. 读取 `headset` 姿态。
2. 从四元数转欧拉角。
3. 提取 yaw/pitch 并归一化到合理范围。
4. 乘以 `head_yaw_scale/head_pitch_scale`。
5. 裁剪到头部关节限位。
6. 发布到头部命令话题。

如果 `enable_head_state_feedback=False`，头部状态不依赖 ROS2 状态反馈，而是使用本地 `head_target_positions` 作为头部状态估计。

## 6. 命令发布前的安全处理

`_send_command()` 是实机命令出口。它会在每个控制周期执行：

1. 检查软件急停。
2. 从 Placo 取手臂 IK 目标。
3. 将目标裁剪到硬件关节限位。
4. 未激活手臂回到预设 folded/zero 目标。
5. 可选 Ruckig 平滑。
6. 基于上一帧目标做最大关节步长限制。
7. 对未激活手臂单独使用更小的回收步长。
8. 计算命令速度。
9. 发布手臂命令、手爪命令。
10. 更新上一帧命令缓存，用于日志和下一周期限步。

几个关键保护：

- `max_arm_joint_step_rad`：默认 `1.0 rad`，限制单周期手臂关节突变。
- `inactive_arm_return_max_step_rad`：默认 `0.03 rad`，未激活手臂回收更慢。
- `max_head_joint_step_rad`：默认 `0.05 rad`。
- 软件急停触发后，会发布当前关节位置作为 hold target，并设置 stop event 退出遥操作。

## 7. 运行线程模型

`HardwareTeleopController.run()` 启动多个线程：

| 线程 | 是否默认启动 | 作用 |
| --- | --- | --- |
| ROS2 spin 线程 | 是，由 X2 控制器创建 | 驱动 ROS2 订阅回调和发布器节点 |
| `_ik_thread` | 是 | 更新机器人状态、XR 输入、Placo IK |
| `_control_thread` | 是 | 将 IK 目标整理成 ROS2 命令并发布 |
| `_data_logging_thread` | `enable_log_data=True` 时 | 按频率写入日志缓存，监听 B 键保存 |
| `_camera_thread` | 开启相机显示时 | 更新并显示相机窗口 |

相机订阅本身通过 ROS2 回调进入 `Ros2CameraInterface`，不依赖显示线程；即使 `show_camera_window=False`，仍可记录相机帧。

X2 子类在 IK 和控制线程之间使用 `_control_state_lock`，避免 IK 正在写 Placo 状态时控制线程同时读取目标，减少竞态。

## 8. 相机采集

`Ros2CameraInterface` 支持多个 camera name，每个 camera 可以配置 color 和 depth 话题。默认只记录 color。

对 compressed 话题：

- 如果不需要显示且不需要 resize，会直接缓存原始 `CompressedImage.data` 字节。
- 这样日志保存的是 JPEG/压缩图像 bytes，避免重复解码和编码。

对 raw `sensor_msgs/Image` 话题：

- 支持 `bgr8/rgb8/mono8/16UC1/32FC1` 等编码。
- 如果 `enable_compression=True`，color 会压缩为 JPG，depth 会压缩为 PNG。
- 如果配置 raw passthrough，则可保存原始 bytes 和 encoding/width/height/step 元数据。

日志中相机结构通常为：

```python
{
    "image": {
        "head_front": {"color": bytes, "depth": None},
        "right_wrist": {"color": bytes, "depth": None},
        "left_wrist": {"color": bytes, "depth": None},
    }
}
```

## 9. 数据日志结构

日志由 `DataLogger` 保存在内存列表中，只有按键触发保存时才写入磁盘。每条 entry 由 `_log_data()` 构造：

```python
{
    "timestamp": float,
    "arm_state": {...},
    "arm_velocity": {...},
    "arm_command": {...},
    "head_state": {...},
    "head_velocity": {...},
    "head_command": {...},
    "hand_command": {"left_hand": float, "right_hand": float},
    "hand_trigger_raw": {"left_hand": float, "right_hand": float},
    "image": {...},  # 开启相机时存在
}
```

按键逻辑：

- 按 `B` 一次：开始记录。
- 再按 `B`：停止记录，保存 `.pkl`。
- 记录中按 `right_axis_click`：丢弃当前缓存，不保存。

保存逻辑：

1. 先写入临时 `.tmp.pkl` 文件。
2. 如果 `validate_log_before_save=True`，运行健康检查。
3. 没有 `ERROR` 才用 `os.replace()` 提升为正式文件。
4. 正式文件名为 `teleop_log_YYYYMMDD_HHMMSS_N.pkl`。

## 10. 健康检查与数据质量

`scripts/misc/check_teleop_log_health.py` 检查：

- 顶层对象是否为非空 list。
- 每条 entry 是否为 dict。
- 必要字段是否存在，默认需要 `timestamp/arm_state/arm_command/image`。
- 时间戳是否严格递增，平均频率是否合理。
- 数值字段是否有限、是否结构一致、是否超出阈值。
- 关节命令和状态是否存在异常跳变。
- 图像 key 是否一致，图像 bytes 是否为空，可选解码验证。
- `arm_command/arm_state` 是否几乎不动，标记为 mostly static。

保存前健康检查走的是同一套逻辑，只是参数由 `DataLogger._health_check_args()` 固定。

## 11. LeRobot v3 转换逻辑

转换入口：

```bash
python3 scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py \
  <log.pkl 或日志目录> \
  --output-dir datasets/x2_hardware_lerobot_v3 \
  --camera-names head_front,right_wrist,left_wrist \
  --include-hands \
  --instruction "..."
```

默认映射：

- `observation.state`：来自 `arm_state`，默认 14 维。
- `action`：来自 `arm_command` + `hand_command`，默认 16 维。
- `timestamp`：来自日志 entry 的 `timestamp`。
- `observation.images.<camera>`：来自 `entry["image"][camera]["color"]`。
- `task`：每帧写入同一条语言指令。

如果传 `--include-head`，则会把 `head_state` 追加到 observation，把 `head_command` 追加到 action。

脚本会优先尝试官方 `lerobot` SDK。如果不可用或导出失败，则使用内置导出：

- 写 parquet：`data/chunk-000/file-000.parquet`
- 写 H.265 MP4：`videos/<camera>/chunk-000/file-000.mp4`
- 写 metadata：`meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`meta/episodes/chunk-000/file-000.parquet`

## 12. 回放链路

### 12.1 可视化回放

`scripts/misc/play_collected_data.py` 可播放原始 `.pkl` 或 LeRobot 数据集，用于检查：

- 多路相机画面是否正常。
- state/action 曲线是否连续。
- 是否有长时间静止、突跳或图像异常。

### 12.2 实机回放

`scripts/hardware/replay_x2_collected_data_on_hardware.py` 支持把原始日志或 LeRobot 数据集回放到实机。

默认是 dry-run，不会发命令。必须显式传：

```bash
--execute-hardware
```

推荐实机首次回放：

```bash
python3 scripts/hardware/replay_x2_collected_data_on_hardware.py \
  <log.pkl> \
  --execute-hardware \
  --move-to-start \
  --speed 0.5
```

安全逻辑包括：

- 如果当前姿态和第一帧差距太大，且没有 `--move-to-start`，拒绝执行。
- `--move-to-start` 会先平滑移动到第一帧。
- `--max-step-rad` 限制每个发布周期的关节变化。
- 可选择回放 `arm_command`、`arm_state`、LeRobot `action` 或 `observation.state`。

## 13. 推荐采集流程

1. 准备环境：

```bash
conda activate x2-hardware
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
```

2. 确认 XR、ROS2 和实机控制节点在线。

3. 采集前回零：

```bash
python3 scripts/hardware/reset_x2_upper_body_zero.py \
  --move-duration-s 3.0 \
  --hold-duration-s 0.5
```

4. 启动采集：

```bash
python3 scripts/hardware/teleop_x2_upper_body_hardware.py \
  --log-dir /media/xlq/ESD-USB \
  --control-rate-hz 30 \
  --log-freq 30
```

5. 按 `B` 开始记录，完成一条 episode 后再按 `B` 保存。

6. 检查日志：

```bash
python3 scripts/misc/check_teleop_log_health.py /media/xlq/ESD-USB --decode-images
```

7. 可视化回放：

```bash
python3 scripts/misc/play_collected_data.py <log.pkl>
```

8. 转 LeRobot：

```bash
python3 scripts/misc/convert_x2_hardware_log_to_lerobot_v3.py \
  <log.pkl 或目录> \
  --output-dir datasets/x2_hardware_lerobot_v3 \
  --camera-names head_front,right_wrist,left_wrist \
  --include-hands \
  --instruction "..."
```

## 14. 技术要点总结

X2 采集链路的核心不是简单地“读手柄然后发关节”，而是一个闭环遥操作系统：

```text
XR 手柄位姿
  -> 坐标变换、滤波、激活参考系增量
  -> Placo 末端位姿任务
  -> IK 求解 14 维手臂目标
  -> 关节限位、未激活回收、Ruckig 可选平滑、步长限制
  -> ROS2 JointCommandArray 发布到实机
  -> ROS2 JointStateArray 回传当前状态
  -> 日志线程按频率记录 state/action/image
  -> 健康检查、回放、LeRobot 转换
```

从数据集角度看，训练最关键的对应关系是：

- `observation.state` 通常对应实机当前回传的 `arm_state`。
- `action` 通常对应控制器最终下发的 `arm_command`，并追加 `left_hand/right_hand`。
- 图像使用记录时刻缓存到的最近相机帧。
- `timestamp` 来自采集程序本地时间，表示从本次控制器启动开始的相对时间。

因此，如果要排查数据问题，优先看三类信号是否一致：

- 图像是否完整、相机 key 是否稳定。
- `arm_command` 是否有合理动作且无突跳。
- `arm_state` 是否跟随 `arm_command`，二者差异是否在可接受范围内。
