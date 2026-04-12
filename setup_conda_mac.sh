#!/bin/bash

set -euo pipefail

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
PYTHON_VERSION="3.10"
DEFAULT_ENV_NAME="xr-mac-sim"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REPO_ROOT}/envs/requirements.mac-sim.txt"

check_platform() {
    if [[ "${OS_NAME}" != "Darwin" ]]; then
        echo "Unsupported operating system: ${OS_NAME}"
        echo "setup_conda_mac.sh only supports macOS (Apple Silicon)."
        exit 1
    fi

    if [[ "${ARCH_NAME}" != "arm64" ]]; then
        echo "Unsupported mac architecture: ${ARCH_NAME}"
        echo "Intel Mac is not covered by setup_conda_mac.sh."
        exit 1
    fi
}

init_conda() {
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Conda initialization script not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
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
    local env_name="$1"
    echo
    echo "[INFO] macOS simulation-only environment is ready."
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
    echo "  conda activate ${env_name}"
    echo "  export XROBOTOOLKIT_INPUT=keyboard"
    echo "  python scripts/simulation/teleop_x2_upper_body_mujoco.py --visualize-placo False"
    echo "  python scripts/simulation/teleop_x2_omnihands_mujoco.py --visualize-placo False"
    echo "  python scripts/misc/test_data_log_analysis.py logs/<robot_name>/<your_log>.pkl"
    echo
    echo "If XROBOTOOLKIT_INPUT is not set, XrClient will try the SDK first and only fall back if it is unavailable."
}

check_platform

if [[ "${1:-}" == "--conda" ]]; then
    ENV_NAME="${2:-$DEFAULT_ENV_NAME}"
    init_conda

    conda deactivate >/dev/null 2>&1 || true
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

    echo "[INFO] Created conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}."
    echo "Next steps:"
    echo "  conda activate ${ENV_NAME}"
    echo "  bash setup_conda_mac.sh --install"
elif [[ "${1:-}" == "--install" ]]; then
    if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
        echo "Error: No conda environment is currently activated."
        echo "Please run: conda activate <env_name>"
        exit 1
    fi

    init_conda
    install_base_deps
    run_validation
    print_usage_notes "${CONDA_DEFAULT_ENV}"
else
    echo "Invalid argument."
    echo "Use:"
    echo "  bash setup_conda_mac.sh --conda [env_name]"
    echo "  conda activate <env_name>"
    echo "  bash setup_conda_mac.sh --install"
    exit 1
fi
