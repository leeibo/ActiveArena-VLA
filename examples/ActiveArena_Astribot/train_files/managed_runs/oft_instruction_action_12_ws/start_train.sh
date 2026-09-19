#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="$(basename "${SCRIPT_DIR}")"
exec bash "${SCRIPT_DIR}/../start_managed_train.sh" "${RUN_NAME}" "$@"
