#!/usr/bin/env bash
set -euo pipefail

# One-shot RunPod launcher:
# - Clones or updates repository
# - Runs setup + preflight
# - Starts full training pipeline with deterministic RUN_ID
# - Writes a persistent launch log for post-mortem/debugging

RUNPOD_WORKDIR="${RUNPOD_WORKDIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/Lugier/QLORA.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
PROJECT_DIR_NAME="${PROJECT_DIR_NAME:-QLORA}"
PROJECT_DIR="${PROJECT_DIR:-${RUNPOD_WORKDIR%/}/${PROJECT_DIR_NAME}}"
RUN_ID="${RUN_ID:-run_$(date -u +%Y%m%d_%H%M%S)}"
SKIP_SETUP="${SKIP_SETUP:-0}"

mkdir -p "${RUNPOD_WORKDIR}"

if [[ -f "./scripts/runpod_train_full.sh" && -f "./scripts/runpod_setup.sh" ]]; then
  PROJECT_DIR="$(pwd)"
  echo "[deploy] Using current repository at ${PROJECT_DIR}"
else
  if [[ -d "${PROJECT_DIR}/.git" ]]; then
    echo "[deploy] Existing git repo found at ${PROJECT_DIR}, pulling latest ${REPO_BRANCH}..."
    git -C "${PROJECT_DIR}" fetch origin
    git -C "${PROJECT_DIR}" checkout "${REPO_BRANCH}"
    git -C "${PROJECT_DIR}" pull --rebase origin "${REPO_BRANCH}"
  else
    echo "[deploy] Cloning ${REPO_URL} into ${PROJECT_DIR}..."
    git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${PROJECT_DIR}"
  fi
fi

cd "${PROJECT_DIR}"
mkdir -p "run_manifests/${RUN_ID}"
LOG_PATH="run_manifests/${RUN_ID}/deploy_launcher.log"

{
  echo "[deploy] UTC start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[deploy] project_dir=${PROJECT_DIR}"
  echo "[deploy] repo_url=${REPO_URL}"
  echo "[deploy] repo_branch=${REPO_BRANCH}"
  echo "[deploy] run_id=${RUN_ID}"
  echo "[deploy] python=$(python3 --version 2>&1 || true)"
  echo "[deploy] gpu_info_start"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo "[deploy] gpu_info_end"
} | tee -a "${LOG_PATH}"

# Stable runtime defaults for RTX 3090 / 24GB.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export RUN_ID

if [[ "${SKIP_SETUP}" != "1" ]]; then
  echo "[deploy] Running setup..." | tee -a "${LOG_PATH}"
  bash scripts/runpod_setup.sh 2>&1 | tee -a "${LOG_PATH}"
else
  echo "[deploy] SKIP_SETUP=1 -> skipping dependency setup." | tee -a "${LOG_PATH}"
fi

echo "[deploy] Running preflight..." | tee -a "${LOG_PATH}"
bash scripts/runpod_preflight.sh 2>&1 | tee -a "${LOG_PATH}"

echo "[deploy] Starting full training pipeline..." | tee -a "${LOG_PATH}"
bash scripts/runpod_train_full.sh 2>&1 | tee -a "${LOG_PATH}"

echo "[deploy] Completed successfully at $(date -u +"%Y-%m-%dT%H:%M:%SZ")." | tee -a "${LOG_PATH}"
echo "[deploy] Artifacts are under run_manifests/${RUN_ID}/ and model export directories."
