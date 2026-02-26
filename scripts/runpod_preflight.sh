#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MIN_FREE_GB="${MIN_FREE_GB:-120}"
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-23000}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[preflight] ERROR: python3 not found."
  exit 1
fi

python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) != (3, 10):
    raise SystemExit(
        f"[preflight] ERROR: Python {major}.{minor} detected. "
        "This stack is pinned to Python 3.10."
    )
print(f"[preflight] Python OK: {major}.{minor}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[preflight] ERROR: nvidia-smi not found. Use a GPU RunPod template."
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//')"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
if [[ -z "${GPU_MEM_MB}" ]] || ! [[ "${GPU_MEM_MB}" =~ ^[0-9]+$ ]]; then
  echo "[preflight] ERROR: Could not read GPU memory via nvidia-smi."
  exit 1
fi
if [[ "${GPU_MEM_MB}" -lt "${MIN_GPU_MEM_MB}" ]]; then
  echo "[preflight] ERROR: GPU memory too low (${GPU_MEM_MB} MB < ${MIN_GPU_MEM_MB} MB)."
  exit 1
fi
echo "[preflight] GPU OK: ${GPU_NAME} (${GPU_MEM_MB} MB)"

FREE_KB="$(df -Pk "${ROOT_DIR}" | awk 'NR==2 {print $4}')"
FREE_GB=$(( FREE_KB / 1024 / 1024 ))
if [[ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]]; then
  echo "[preflight] ERROR: Not enough free disk space (${FREE_GB} GB < ${MIN_FREE_GB} GB)."
  exit 1
fi
echo "[preflight] Disk OK: ${FREE_GB} GB free (min ${MIN_FREE_GB} GB)"

mkdir -p run_manifests outputs_sft outputs_dpo outputs_orpo outputs_grpo outputs_grpo_replay artifacts
touch run_manifests/.write_test && rm -f run_manifests/.write_test

python3 - <<'PY'
import importlib
import torch
import trl

required = ["transformers", "trl", "datasets", "vllm", "peft", "unsloth"]
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception as exc:
        missing.append((pkg, str(exc)))
if missing:
    raise SystemExit(f"[preflight] ERROR: Missing imports: {missing}")
if not torch.cuda.is_available():
    raise SystemExit("[preflight] ERROR: torch.cuda.is_available() is False.")
if not hasattr(trl, "GRPOTrainer"):
    raise SystemExit("[preflight] ERROR: TRL build has no GRPOTrainer.")
if not hasattr(trl, "ORPOTrainer"):
    raise SystemExit("[preflight] ERROR: TRL build has no ORPOTrainer.")
print("[preflight] Imports + CUDA + TRL trainers OK")
PY

echo "[preflight] RunPod preflight passed."
