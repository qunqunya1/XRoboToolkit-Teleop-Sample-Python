#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_PKG_DIR="${SCRIPT_DIR}/src/aimdk_msgs"
ASCII_WS_ROOT="${ASCII_WS_ROOT:-/tmp/aimdk_ws}"

if [[ ! -d "${SRC_PKG_DIR}" ]]; then
    echo "aimdk_msgs source package not found: ${SRC_PKG_DIR}" >&2
    exit 1
fi

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS2 Humble was not found at /opt/ros/humble/setup.bash" >&2
    exit 1
fi

rm -rf "${ASCII_WS_ROOT}"
mkdir -p "${ASCII_WS_ROOT}/src"
cp -r "${SRC_PKG_DIR}" "${ASCII_WS_ROOT}/src/"

source /opt/ros/humble/setup.bash
cd "${ASCII_WS_ROOT}"
colcon --log-base log build --packages-select aimdk_msgs --build-base build --install-base install

cat <<EOF

Build finished successfully.
Source the generated workspace with:

  source /opt/ros/humble/setup.bash
  source ${ASCII_WS_ROOT}/install/setup.bash

EOF
