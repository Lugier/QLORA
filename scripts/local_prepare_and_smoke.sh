#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-local_$(date -u +%Y%m%d_%H%M%S)}"
MANIFEST_DIR="run_manifests/${RUN_ID}"
mkdir -p "${MANIFEST_DIR}"
REPORT_PATH="${MANIFEST_DIR}/local_prepare_report.txt"

touch "${REPORT_PATH}"

echo "[local] UTC start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[local] root=${ROOT_DIR}"
echo "[local] run_id=${RUN_ID}"
{
  echo "[local] UTC start: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[local] root=${ROOT_DIR}"
  echo "[local] run_id=${RUN_ID}"
} >> "${REPORT_PATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[local] ERROR: ${PYTHON_BIN} not found."
  exit 1
fi

echo "[local] python=$(${PYTHON_BIN} --version 2>&1)"
echo "[local] python=$(${PYTHON_BIN} --version 2>&1)" >> "${REPORT_PATH}"

GPU_OK=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_OK=1
  echo "[local] gpu_info_start"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo "[local] gpu_info_end"
else
  echo "[local] INFO: nvidia-smi not available."
  echo "[local] INFO: nvidia-smi not available." >> "${REPORT_PATH}"
fi

echo "[local] Running static safety checks..."
bash -n scripts/runpod_setup.sh
bash -n scripts/runpod_preflight.sh
bash -n scripts/runpod_train_full.sh
bash -n scripts/runpod_deploy_and_train.sh

echo "[local] Running CLI drift check..."
"${PYTHON_BIN}" scripts/validate_cli_drift.py

echo "[local] Running unit tests..."
"${PYTHON_BIN}" -m unittest tests.test_verification tests.test_rewards tests.test_prm_tiny tests.test_scientific_gate

echo "[local] Checking runtime deps for benchmark stage..."
if "${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = ["torch", "datasets", "transformers", "trl", "vllm", "peft", "unsloth"]
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception as exc:
        missing.append((pkg, str(exc)))

if missing:
    print("[local] BENCHMARK_BLOCKED: missing deps")
    for name, msg in missing:
        print(f"[local]   - {name}: {msg}")
    sys.exit(2)

print("[local] Dependency check passed.")
PY
then
  DEPS_EXIT=0
else
  DEPS_EXIT=$?
fi

PY_OK=0
if "${PYTHON_BIN}" - <<'PY'
import sys
sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
then
  PY_OK=1
fi

if [[ "${GPU_OK}" -eq 1 && "${DEPS_EXIT}" -eq 0 && "${PY_OK}" -eq 1 ]]; then
  echo "[local] Environment is benchmark-ready. Running Stage-0 baseline mini-eval..."
  "${PYTHON_BIN}" eval_pipeline.py \
    --model-path "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
    --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval \
    --swebench-mode harness \
    --num-samples 10 \
    --pass-k 2 \
    --bootstrap-samples 200 \
    --seed 3407 \
    --case-log-path "${MANIFEST_DIR}/local_baseline_cases.jsonl" \
    --json-output "${MANIFEST_DIR}/local_baseline.json"
  echo "[local] Baseline mini-eval completed."
else
  echo "[local] BENCHMARK_BLOCKED: local hardware/runtime is not yet sufficient."
  echo "[local] Conditions required:"
  echo "[local]  - Python 3.10"
  echo "[local]  - NVIDIA GPU with CUDA (nvidia-smi)"
  echo "[local]  - Runtime deps: torch/datasets/transformers/trl/vllm/peft/unsloth"
  echo "[local] Next action: run 'bash scripts/runpod_deploy_and_train.sh' on RunPod."
fi

echo "[local] UTC end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[local] Report: ${REPORT_PATH}"
{
  echo "[local] UTC end: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[local] Report: ${REPORT_PATH}"
} >> "${REPORT_PATH}"
