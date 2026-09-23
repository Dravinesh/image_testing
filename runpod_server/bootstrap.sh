#!/usr/bin/env bash
# Runs on the RunPod pod (called from the pod's Container Start Command).
# Sets up a persistent, fully isolated venv on /workspace, installs deps
# only when requirements.txt changed, then starts the FastAPI server.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-/workspace/venv}"
PORT="${PORT:-8000}"

# Keep model weights on the persistent volume so restarts don't re-download.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "${HF_HOME}"

# Some pod images export PYTHONPATH pointing at their system dist-packages.
# That silently shadows whatever we pip-install into the venv (e.g. an old
# system torch getting imported instead of the newer one we just installed),
# so make sure it can't leak in here.
unset PYTHONPATH || true

REQ_FILE="${APP_DIR}/requirements.txt"
HASH_FILE="${VENV_DIR}/.requirements.sha256"

create_venv() {
    echo "creating venv at ${VENV_DIR}"
    # Fully isolated (no --system-site-packages): guarantees whatever we
    # pip-install here is what actually gets imported, with no ambiguity
    # about which torch (or anything else) wins.
    python -m venv "${VENV_DIR}"
}

install_requirements() {
    echo "installing requirements..."
    pip install --upgrade pip
    pip install -r "${REQ_FILE}"
    sha256sum "${REQ_FILE}" | cut -d' ' -f1 > "${HASH_FILE}"
}

[ -d "${VENV_DIR}" ] || create_venv
source "${VENV_DIR}/bin/activate"

REQ_HASH="$(sha256sum "${REQ_FILE}" | cut -d' ' -f1)"
if [ ! -f "${HASH_FILE}" ] || [ "$(cat "${HASH_FILE}")" != "${REQ_HASH}" ]; then
    install_requirements
else
    echo "requirements unchanged, skipping install"
fi

# Self-heal: if torch is missing or older than 2.5 in this venv (e.g. a venv
# left over from before this pin, or some other shadowing issue), wipe it and
# reinstall clean instead of crash-looping forever with the same error.
if ! python -c "
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
major, minor = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
sys.exit(0 if (major, minor) >= (2, 5) else 1)
"; then
    echo "torch missing or older than 2.5 in venv (possibly shadowed) — recreating venv from scratch"
    deactivate || true
    rm -rf "${VENV_DIR}"
    create_venv
    source "${VENV_DIR}/bin/activate"
    install_requirements
fi

python -c "import torch; print('using torch', torch.__version__, 'from', torch.__file__)"

echo "Starting Qwen-Image server"
echo "  model:         ${MODEL_ID:-Qwen/Qwen-Image-2.1}"
echo "  port:          ${PORT}"
echo "  hf cache:      ${HF_HOME}"
echo "  cpu offload:   ${CPU_OFFLOAD:-0}"
echo "  api key auth:  $([ -n "${API_KEY:-}" ] && echo enabled || echo disabled)"

cd "${APP_DIR}"
exec uvicorn app:app --host 0.0.0.0 --port "${PORT}" --workers 1
