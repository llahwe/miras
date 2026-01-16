#!/usr/bin/env bash
set -euo pipefail

# Vast "onstart" entrypoint.
# - Installs a few system deps (rclone) if missing
# - Writes Google Drive service account key if provided via env
# - Syncs Python deps via uv (optional, but recommended)

echo "[entrypoint] starting..."

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends ca-certificates curl git jq
  if ! command -v rclone >/dev/null 2>&1; then
    echo "[entrypoint] installing rclone..."
    apt-get install -y --no-install-recommends rclone
  fi
fi

# Workspace assumptions for Vast:
# - code lives in /workspace (typical Vast default)
cd /workspace || true

# If you provide a base64-encoded service account JSON, write it to disk.
# IMPORTANT: do NOT echo this content.
if [[ -n "${GDRIVE_SERVICE_ACCOUNT_JSON_B64:-}" ]]; then
  mkdir -p /workspace/secrets
  echo "${GDRIVE_SERVICE_ACCOUNT_JSON_B64}" | base64 -d > /workspace/secrets/gdrive-sa.json
  chmod 600 /workspace/secrets/gdrive-sa.json
  echo "[entrypoint] wrote /workspace/secrets/gdrive-sa.json"
fi

# Optional: configure rclone remote via env vars.
# If you set:
#   RCLONE_CONFIG_GDRIVE_TYPE=drive
#   RCLONE_CONFIG_GDRIVE_SCOPE=drive
#   RCLONE_CONFIG_GDRIVE_SERVICE_ACCOUNT_FILE=/workspace/secrets/gdrive-sa.json
# then `rclone ls gdrive:` should work without an rclone.conf.

# Optional: install python deps with uv (assumes uv is already installed in the image).
if command -v uv >/dev/null 2>&1; then
  echo "[entrypoint] syncing deps with uv..."
  uv sync --frozen || uv sync
fi

echo "[entrypoint] done."

