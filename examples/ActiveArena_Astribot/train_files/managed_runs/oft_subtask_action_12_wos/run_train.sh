#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ACTIVEARENA_VLA_ROOT:-${STARVLA_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}}"
cd "${REPO_ROOT}"

CONFIG_YAML="${SCRIPT_DIR}/config.yaml"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
CONDA_ENV_NAME="${ACTIVEARENA_VLA_ENV_NAME:-${CONDA_ENV_NAME:-activearena-vla}}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-0}}"

# Set CUDA_HOME explicitly when nvcc is not on PATH; no cluster-specific
# CUDA installation is assumed by this launcher.
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if [[ -d /sys/class/net/bond0 ]]; then
    NCCL_SOCKET_IFNAME="bond0"
  elif [[ -d /sys/class/net/ib0 ]]; then
    NCCL_SOCKET_IFNAME="ib0"
  fi
fi
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME
else
  unset NCCL_SOCKET_IFNAME || true
fi
# Leave the InfiniBand device unset by default; cluster users can provide
# NCCL_IB_HCA explicitly when their scheduler exposes a known device.
if [[ -n "${NCCL_IB_HCA:-}" ]]; then
  export NCCL_IB_HCA
else
  unset NCCL_IB_HCA || true
fi
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS// /,}"
fi

infer_num_processes() {
  if [[ -n "${NUM_PROCESSES:-}" ]]; then
    printf '%s\n' "${NUM_PROCESSES}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    local devices="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
    IFS=',' read -r -a device_list <<< "${devices}"
    if [[ "${#device_list[@]}" -gt 0 && -n "${device_list[0]}" ]]; then
      printf '%s\n' "${#device_list[@]}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s\n' "${gpu_count}"
      return
    fi
  fi
  printf '1\n'
}
NUM_PROCESSES="$(infer_num_processes)"

EXPLICIT_WANDB_MODE="${WANDB_MODE:-}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -n "${EXPLICIT_WANDB_MODE}" ]]; then
  export WANDB_MODE="${EXPLICIT_WANDB_MODE}"
else
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

if [[ -n "${ACCELERATE_BIN:-}" ]]; then
  ACCELERATE_CMD=("${ACCELERATE_BIN}")
elif [[ -n "${ACTIVEARENA_VLA_PYTHON:-}" && -x "$(dirname "${ACTIVEARENA_VLA_PYTHON}")/accelerate" ]]; then
  # Keep the launcher and worker in the same environment when the caller
  # provides an explicit Python without activating its conda environment.
  ACCELERATE_CMD=("$(dirname "${ACTIVEARENA_VLA_PYTHON}")/accelerate")
elif [[ -n "${STARVLA_PYTHON:-}" && -x "$(dirname "${STARVLA_PYTHON}")/accelerate" ]]; then
  ACCELERATE_CMD=("$(dirname "${STARVLA_PYTHON}")/accelerate")
elif command -v accelerate >/dev/null 2>&1; then
  ACCELERATE_CMD=("accelerate")
else
  CONDA_BIN="${CONDA_EXE:-conda}"
  ACCELERATE_CMD=("${CONDA_BIN}" "run" "--no-capture-output" "-n" "${CONDA_ENV_NAME}" "accelerate")
fi

"${ACCELERATE_CMD[@]}" launch \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml "${CONFIG_YAML}" \
  "$@"
