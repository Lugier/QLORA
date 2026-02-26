#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[setup] Upgrading pip toolchain..."
python3 -m pip install --upgrade pip setuptools wheel

python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) != (3, 10):
    raise SystemExit(
        f"Unsupported Python version: {major}.{minor}. "
        "Use Python 3.10 on RunPod for this pinned stack."
    )
print(f"[setup] Python version OK: {major}.{minor}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] ERROR: nvidia-smi not found. Run this on a GPU RunPod instance."
  exit 1
fi

echo "[setup] GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
echo "[setup] Detected CUDA runtime: ${CUDA_VERSION:-unknown}"

echo "[setup] Installing CUDA-enabled torch stack (cu121 wheels)..."
python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

echo "[setup] Installing Unsloth + Unsloth Zoo (CUDA 12.1 / Torch 2.4 profile)..."
python3 -m pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
if ! python3 -m pip install --no-build-isolation \
  "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"; then
  echo "[setup] Falling back to Ampere profile for Unsloth extras..."
  python3 -m pip install --no-build-isolation \
    "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
fi

echo "[setup] Installing project dependencies..."
python3 -m pip install -r requirements.txt

echo "[setup] Installing SWE-bench harness tooling..."
if ! python3 -m pip install swebench; then
  echo "[setup] WARN: pip install swebench failed, trying GitHub source..."
  python3 -m pip install git+https://github.com/SWE-bench/SWE-bench.git
fi

echo "[setup] Recommended runtime env for RTX 3090 / 24GB stability:"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"
echo "  export VLLM_WORKER_MULTIPROC_METHOD=spawn"

echo "[setup] Verifying critical imports..."
python3 - <<'PY'
import importlib
import torch
import trl

required = [
    "transformers",
    "trl",
    "datasets",
    "vllm",
    "peft",
    "unsloth",
]
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception as exc:
        missing.append((pkg, str(exc)))

if missing:
    raise SystemExit(f"Missing imports: {missing}")

if not hasattr(trl, "GRPOTrainer"):
    raise SystemExit("Installed TRL build does not provide GRPOTrainer.")

if not torch.cuda.is_available():
    raise SystemExit("Torch installed but CUDA is not available.")

print("[setup] Import and CUDA checks passed.")
print(f"[setup] Torch version: {torch.__version__}")
print(f"[setup] CUDA device: {torch.cuda.get_device_name(0)}")
PY

echo "[setup] RunPod environment setup complete."
