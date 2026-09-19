#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <run_name> [run_train args...]" >&2
  exit 2
fi

RUN_NAME="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ACTIVEARENA_VLA_ROOT:-${STARVLA_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}}"
RUN_DIR="${REPO_ROOT}/examples/ActiveArena_Astribot/train_files/managed_runs/${RUN_NAME}"
RUN_TRAIN="${RUN_DIR}/run_train.sh"

if [[ ! -f "${RUN_TRAIN}" ]]; then
  echo "run_train.sh not found: ${RUN_TRAIN}" >&2
  exit 1
fi

count_visible_devices() {
  local devices="${1//[[:space:]]/}"
  if [[ -z "${devices}" || "${devices}" == "NoDevFiles" || "${devices}" == "-1" ]]; then
    return 1
  fi
  IFS=',' read -r -a device_list <<< "${devices}"
  echo "${#device_list[@]}"
}

infer_num_processes() {
  if [[ -n "${NUM_PROCESSES:-}" ]]; then
    echo "${NUM_PROCESSES}"
    return
  fi

  if [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ && "${SLURM_GPUS_ON_NODE}" -gt 0 ]]; then
    echo "${SLURM_GPUS_ON_NODE}"
    return
  fi

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if count_visible_devices "${CUDA_VISIBLE_DEVICES}" >/dev/null; then
      count_visible_devices "${CUDA_VISIBLE_DEVICES}"
      return
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${gpu_count}" =~ ^[0-9]+$ && "${gpu_count}" -gt 0 ]]; then
      echo "${gpu_count}"
      return
    fi
  fi

  echo "1"
}

cd "${REPO_ROOT}"

if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

export ACTIVEARENA_VLA_ENV_NAME="${ACTIVEARENA_VLA_ENV_NAME:-${CONDA_ENV_NAME:-activearena-vla}}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-${ACTIVEARENA_VLA_ENV_NAME}}"
# Set ACCELERATE_BIN when accelerate is not on PATH (for example, inside a
# conda environment).  No machine-specific path is assumed here.

export NUM_PROCESSES="$(infer_num_processes)"
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${MAIN_PROCESS_PORT:-}" ]]; then
  if [[ -n "${MASTER_PORT:-}" ]]; then
    export MAIN_PROCESS_PORT="${MASTER_PORT}"
  else
    export MAIN_PROCESS_PORT="0"
  fi
fi

LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STDOUT_LOG="${LOG_DIR}/${RUN_NAME}-start-${TIMESTAMP}.out"
STDERR_LOG="${LOG_DIR}/${RUN_NAME}-start-${TIMESTAMP}.err"

if [[ "${START_TEE_LOGS:-1}" != "0" ]]; then
  exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)
fi

echo "Run name: ${RUN_NAME}"
echo "Repo root: ${REPO_ROOT}"
echo "Run script: ${RUN_TRAIN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "MAIN_PROCESS_PORT: ${MAIN_PROCESS_PORT}"
echo "ACCELERATE_BIN: ${ACCELERATE_BIN:-<run_train fallback>}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"

bash "${RUN_TRAIN}" "$@"
