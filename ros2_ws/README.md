# ROS2 Workspace

This workspace contains ROS2 packages used by the teleoperation project.

## Packages

- `aimdk_msgs`: a minimal compatibility package that only keeps the message types
  used by this project. The message fields match the official joint-control
  definitions needed by the teleop and motor control code.

## Build

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select aimdk_msgs
source install/setup.bash
```

If the repository was moved to a different directory after a previous build, clear the
CMake cache before rebuilding:

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select aimdk_msgs --cmake-clean-cache
```

After sourcing the workspace, `from aimdk_msgs.msg import ...` will be available to
Python nodes launched from this project.

## Note About Non-ASCII Paths

ROS2 interface generation may fail when this repository is built from a path that contains
 non-ASCII characters, such as `下载`. If that happens, use:

```bash
bash ros2_ws/build_aimdk_msgs_ascii.sh
```

That helper copies the minimal `aimdk_msgs` package into `/tmp/aimdk_ws`, builds it there,
and prints the `source` commands for the generated install space.
