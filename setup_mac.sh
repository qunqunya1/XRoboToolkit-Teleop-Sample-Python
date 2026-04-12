#!/bin/bash

set -euo pipefail

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REPO_ROOT}/envs/requirements.mac-sim.txt"
VENV_DIR="${REPO_ROOT}/.venv-mac"

check_platform() {
    if [[ "${OS_NAME}" != "Darwin" ]]; then
        echo "Unsupported operating system: ${OS_NAME}"
        echo "setup_mac.sh only supports macOS (Apple Silicon)."
        exit 1
    fi

    if [[ "${ARCH_NAME}" != "arm64" ]]; then
        echo "Unsupported mac architecture: ${ARCH_NAME}"
        echo "Intel Mac is not covered by setup_mac.sh."
        exit 1
    fi
}

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required but was not found."
        exit 1
    fi

    python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")
PY
}

install_base_deps() {
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install uv

    uv pip install --upgrade pip setuptools wheel
    uv pip install -r "${REQ_FILE}"

    echo "Installing placo (required for X2 simulation IK)..."
    if ! uv pip install placo; then
        echo "Failed to install 'placo'."
        echo "mac simulation support requires placo for the X2 upper-body and omnihands pipelines."
        exit 1
    fi

    uv pip install -e . --no-deps
}

run_validation() {
    python -c "import mujoco, placo, cv2, meshcat, xrobotoolkit_teleop"
    python -c "import os; os.environ['XROBOTOOLKIT_INPUT']='keyboard'; from xrobotoolkit_teleop.common.xr_client import XrClient; XrClient().close()"
}

print_usage_notes() {
    echo
    echo "[INFO] Created or updated virtual environment at ${VENV_DIR}."
    echo
    echo "Supported on this setup:"
    echo "  - MuJoCo simulation"
    echo "  - Built-in keyboard mock input"
    echo "  - Offline log analysis and dataset conversion"
    echo
    echo "Not supported on this setup:"
    echo "  - XRoboToolkit PC Service / xrobotoolkit_sdk"
    echo "  - Unity keyboard XR client workflow"
    echo "  - ROS / hardware / RealSense / dex hand tracking"
    echo
    echo "Usage:"
    echo "  source .venv-mac/bin/activate"
    echo "  export XROBOTOOLKIT_INPUT=keyboard"
    echo "  python scripts/simulation/teleop_x2_upper_body_mujoco.py --visualize-placo False"
    echo "  python scripts/simulation/teleop_x2_omnihands_mujoco.py --visualize-placo False"
    echo "  python scripts/misc/test_data_log_analysis.py logs/<robot_name>/<your_log>.pkl"
    echo
    echo "If XROBOTOOLKIT_INPUT is not set, XrClient will try the SDK first and only fall back if it is unavailable."
}

check_platform
check_python

cd "${REPO_ROOT}"
python3 -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"

install_base_deps
run_validation
print_usage_notes
