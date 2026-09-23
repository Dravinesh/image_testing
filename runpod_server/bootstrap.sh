#!/usr/bin/env bash
# Runs on the RunPod pod (called from the pod's Container Start Command).
# Sets up a persistent venv on /workspace, installs deps only when
# requirements.txt changed, then starts the FastAPI server.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
PORT="${PORT:-8000}"

# Keep model weights on the persistent volume so restarts don't re-download.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "${HF_HOME}"

# --system-site-packages reuses the template's preinstalled PyTorch.
if [ ! -d "${VENV_DIR}" ]; then
    echo "creating venv at ${VENV_DIR}"
    python -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

REQ_FILE="${APP_DIR}/requirements.txt"
HASH_FILE="${VENV_DIR}/.requirements.sha256"
REQ_HASH="$(sha256sum "${REQ_FILE}" | cut -d' ' -f1)"
if [ ! -f "${HASH_FILE}" ] || [ "$(cat "${HASH_FILE}")" != "${REQ_HASH}" ]; then
    echo "installing requirements..."
    pip install --upgrade pip
    pip install -r "${REQ_FILE}"
    echo "${REQ_HASH}" > "${HASH_FILE}"
else
    echo "requirements unchanged, skipping install"
fi

echo "Starting Qwen-Image server"
echo "  model:         ${MODEL_ID:-Qwen/Qwen-Image-2.1}"
echo "  port:          ${PORT}"
echo "  hf cache:      ${HF_HOME}"
echo "  cpu offload:   ${CPU_OFFLOAD:-0}"
echo "  api key auth:  $([ -n "${API_KEY:-}" ] && echo enabled || echo disabled)"

cd "${APP_DIR}"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT}" --workers 1
