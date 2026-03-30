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

## 2. X2 Omnihands 上半身 MuJoCo 数据采集

### 2.1 先生成 Omnihands 场景 XML（一次性）

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

### 2.2 启动采集（新脚本）

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

### 2.3 关键参数

- `--thumb-abad-angle`：
  - 不传：左右拇指 `thumb_abad` 固定在 URDF 最大角度。
  - 传入数值：固定到该角度（自动限幅到关节上下限）。
- `--allow-missing-hand-joints`（默认 `False`）：
  - 默认按完整 omnihands 场景运行，手关节缺失会报错，便于尽早发现模型问题。
  - 若你切回无手关节 XML，可手动设为 `True` 跳过手关节下发。
- `--sim-steps-per-control`：
  - 该脚本支持低延迟配置，显式传入值会直接生效（不会被 profile 强制提高）。

### 2.4 采集按键

与原上半身脚本一致：
- 按 `B` 一次：开始记录
- 再按一次 `B`：停止并保存 `.pkl`
- 按 `right_axis_click`：停止并丢弃当前记录

### 2.5 日志新增字段

在原有 `xr` 基础上，新增 `hand_control` 字段（每侧手一组）：
- `active`：该手是否处于 Grip 激活状态
- `trigger_raw`：该侧 trigger 原始值
- `trigger_when_active`：仅在激活时记录的 trigger 值
- `driver_joint_targets`：手指主动关节目标值（由 trigger 映射）
- `thumb_abad_angle`：当前固定拇指外展角

### 2.6 快速检查

```bash
python3 scripts/misc/test_data_log_analysis.py logs/x2_omnihands_upper_body_sim/<your_log>.pkl
```
