#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[train] ERROR: nvidia-smi not found. Run this on a GPU RunPod instance."
  exit 1
fi

if [[ -x "scripts/runpod_preflight.sh" ]]; then
  echo "[train] Running preflight checks..."
  bash scripts/runpod_preflight.sh
fi

BASE_MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
SEED="3407"
RUN_TS="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
MANIFEST_DIR="run_manifests/${RUN_TS}"
mkdir -p "${MANIFEST_DIR}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | sed 's/^ *//;s/ *$//')"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
if [[ -z "${GPU_MEM_MB}" ]] || ! [[ "${GPU_MEM_MB}" =~ ^[0-9]+$ ]]; then
  GPU_MEM_MB=0
fi

# Default profile (24GB-safe baseline).
SFT_MAX_SEQ_LEN=6144
SFT_BATCH=2
SFT_GRAD_ACC=8
P1B_NUM_SAMPLES=16
P1B_GEN_BATCH=24
GRPO_MAX_SEQ_LEN=6144
EVAL_MAX_MODEL_LEN=4096
SWEBENCH_MODE="${SWEBENCH_MODE:-harness}"
AGENT_SEARCH_MODE="${AGENT_SEARCH_MODE:-beam}"
AGENT_BEAM_WIDTH="${AGENT_BEAM_WIDTH:-2}"
PATCH_STRATEGIES="${PATCH_STRATEGIES:-minimal_diff,api_first,test_first,balanced}"
HARD_MINING_CYCLES="${HARD_MINING_CYCLES:-2}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-./artifacts/prm_tiny_v1.json}"
USE_TINY_PRM="${USE_TINY_PRM:-auto}"
REWARD_PROFILE="dense_exec_v1"
PRM_ENABLED=0
STRICT_SWEBENCH_HARNESS="${STRICT_SWEBENCH_HARNESS:-1}"

if [[ "${STRICT_SWEBENCH_HARNESS}" == "1" ]] && [[ "${SWEBENCH_MODE}" != "harness" ]]; then
  echo "[train] ERROR: STRICT_SWEBENCH_HARNESS=1 requires SWEBENCH_MODE=harness (current=${SWEBENCH_MODE})."
  exit 1
fi

# Profile selection keeps defaults conservative on 24GB cards (e.g. RTX 3090).
if [[ "${GPU_NAME}" == *"RTX 3090"* ]] || [[ "${GPU_MEM_MB}" -le 24576 ]]; then
  echo "[train] Applying RTX 3090 / 24GB profile."
  SFT_MAX_SEQ_LEN=6144
  SFT_BATCH=2
  SFT_GRAD_ACC=8
  P1B_NUM_SAMPLES=16
  P1B_GEN_BATCH=24
  GRPO_MAX_SEQ_LEN=6144
  EVAL_MAX_MODEL_LEN=4096
else
  echo "[train] Applying high-VRAM profile."
  SFT_MAX_SEQ_LEN=8192
  SFT_BATCH=2
  SFT_GRAD_ACC=8
  P1B_NUM_SAMPLES=16
  P1B_GEN_BATCH=64
  GRPO_MAX_SEQ_LEN=8192
  EVAL_MAX_MODEL_LEN=6144
fi

# Fragmentation/stability defaults for long runs on consumer GPUs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[train] Logging into Hugging Face with HF_TOKEN..."
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
fi

echo "[train] Stage 0/13: baseline freeze + run manifest"
if [[ -f "${MANIFEST_DIR}/run_manifest.txt" && -f "${MANIFEST_DIR}/baseline_eval_classic.json" && -f "${MANIFEST_DIR}/baseline_eval_agentic.json" && -f "${MANIFEST_DIR}/baseline_eval_classic_cases.jsonl" && -f "${MANIFEST_DIR}/baseline_eval_agentic_cases.jsonl" ]]; then
  echo "[train] Stage 0 artifacts already exist for RUN_ID=${RUN_TS}; skipping baseline rerun."
else
  {
    echo "run_timestamp_utc=${RUN_TS}"
    echo "base_model_name=${BASE_MODEL_NAME}"
    echo "seed=${SEED}"
    echo "gpu_name=${GPU_NAME}"
    echo "gpu_mem_mb=${GPU_MEM_MB}"
    echo "profile_sft_max_seq_len=${SFT_MAX_SEQ_LEN}"
    echo "profile_p1b_generation_batch_size=${P1B_GEN_BATCH}"
  echo "profile_grpo_max_seq_len=${GRPO_MAX_SEQ_LEN}"
  echo "use_tiny_prm=${USE_TINY_PRM}"
  echo "unsloth_ref=${UNSLOTH_REF:-main}"
  echo "unsloth_zoo_ref=${UNSLOTH_ZOO_REF:-main}"
  echo "swebench_ref=${SWEBENCH_REF:-main}"
  echo "python_version=$(python3 --version 2>&1)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
    python3 - <<'PY'
import hashlib
path = "scripts/runpod_train_full.sh"
with open(path, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
print(f"script_sha256={digest}")
PY
    echo "cuda_info_start"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
    echo "cuda_info_end"
    echo "package_versions_start"
    python3 - <<'PY'
import importlib

pkgs = [
    "torch",
    "transformers",
    "trl",
    "datasets",
    "vllm",
    "peft",
    "unsloth",
]
for name in pkgs:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        version = f"missing:{exc}"
    print(f"{name}={version}")
PY
    echo "package_versions_end"
  } > "${MANIFEST_DIR}/run_manifest.txt"

  python3 eval_pipeline.py \
    --model-path "${BASE_MODEL_NAME}" \
    --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval \
    --swebench-mode "${SWEBENCH_MODE}" \
    --patch-strategies "${PATCH_STRATEGIES}" \
    --max-model-len "${EVAL_MAX_MODEL_LEN}" \
    --num-samples 100 \
    --pass-k 4 \
    --bootstrap-samples 1000 \
    --seed "${SEED}" \
    --case-log-path "${MANIFEST_DIR}/baseline_eval_classic_cases.jsonl" \
    --json-output "${MANIFEST_DIR}/baseline_eval_classic.json"

  python3 eval_pipeline.py \
    --model-path "${BASE_MODEL_NAME}" \
    --benchmarks bigcodebench_instruct,livecodebench,mbpp \
    --agentic \
    --search-mode "${AGENT_SEARCH_MODE}" \
    --beam-width "${AGENT_BEAM_WIDTH}" \
    --max-model-len "${EVAL_MAX_MODEL_LEN}" \
    --num-samples 100 \
    --max-rounds 3 \
    --n-candidates 8 \
    --candidate-schedule 8,6,4 \
    --bootstrap-samples 1000 \
    --seed "${SEED}" \
    --case-log-path "${MANIFEST_DIR}/baseline_eval_agentic_cases.jsonl" \
    --json-output "${MANIFEST_DIR}/baseline_eval_agentic.json"
fi

[[ -f "${MANIFEST_DIR}/run_manifest.txt" ]] || { echo "[train] Missing run manifest."; exit 1; }
[[ -f "${MANIFEST_DIR}/baseline_eval_classic.json" ]] || { echo "[train] Missing baseline classic report."; exit 1; }
[[ -f "${MANIFEST_DIR}/baseline_eval_agentic.json" ]] || { echo "[train] Missing baseline agentic report."; exit 1; }
[[ -f "${MANIFEST_DIR}/baseline_eval_classic_cases.jsonl" ]] || { echo "[train] Missing baseline classic case log."; exit 1; }
[[ -f "${MANIFEST_DIR}/baseline_eval_agentic_cases.jsonl" ]] || { echo "[train] Missing baseline agentic case log."; exit 1; }

echo "[train] Stage 1/13: data pipeline"
python3 data_pipeline.py \
  --min-total-samples 20000 \
  --swe-supervised-output-dir ./swe_supervised_dataset \
  --swe-supervised-max-samples 120000 \
  --min-test-coverage 0.08 \
  --min-prompt-coverage 0.99 \
  --min-unique-ratio 0.75 \
  --max-missing-sources 8 \
  --source-weights code:0.55,repo:0.30,reasoning:0.15 \
  --max-samples-per-source 25000 \
  --min-answer-ast-parse-rate 0.98 \
  --min-assert-density-in-tests 0.65 \
  --holdout-policy source_hash_v1 \
  --holdout-fraction 0.10 \
  --val-fraction 0.05 \
  --seed "${SEED}"
[[ -d "sota_slm_coding_dataset" ]] || { echo "[train] Missing dataset output."; exit 1; }
[[ -d "swe_supervised_dataset" ]] || { echo "[train] Missing SWE-supervised dataset output."; exit 1; }
[[ -f "sota_slm_coding_dataset/dataset_manifest.json" ]] || { echo "[train] Missing main dataset manifest."; exit 1; }
[[ -f "swe_supervised_dataset/dataset_manifest.json" ]] || { echo "[train] Missing SWE-supervised dataset manifest."; exit 1; }

echo "[train] Stage 2/13: SFT"
python3 phase1_sft.py \
  --dataset-path ./sota_slm_coding_dataset \
  --extra-dataset-paths ./swe_supervised_dataset \
  --base-model-name "${BASE_MODEL_NAME}" \
  --adapter-dir qwen_sft_lora \
  --merged-model-dir qwen_sft_merged \
  --output-dir outputs_sft \
  --max-seq-length "${SFT_MAX_SEQ_LEN}" \
  --max-steps 3000 \
  --per-device-train-batch-size "${SFT_BATCH}" \
  --gradient-accumulation-steps "${SFT_GRAD_ACC}" \
  --learning-rate 2e-4 \
  --lora-r 48 \
  --lora-alpha 96 \
  --lora-dropout 0.03 \
  --eval-every-steps 250 \
  --checkpoint-every-steps 250 \
  --resume-from-checkpoint auto \
  --seed "${SEED}"
[[ -d "qwen_sft_lora" ]] || { echo "[train] Missing SFT adapter."; exit 1; }
[[ -d "qwen_sft_merged" ]] || { echo "[train] Missing merged SFT model."; exit 1; }

echo "[train] Stage 3/13: best-of-N + DPO pair mining"
python3 phase1b_rejection_sampling.py \
  --model-path qwen_sft_merged \
  --dataset-path ./sota_slm_coding_dataset \
  --output-path ./sota_best_of_n_dataset \
  --dpo-output-path ./sota_dpo_pairs_dataset \
  --num-prompts 2000 \
  --num-samples-per-prompt "${P1B_NUM_SAMPLES}" \
  --temperatures 0.2,0.8 \
  --timeout 1.0 \
  --verifier-rounds 2 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --min-perfect 50 \
  --min-dpo-pairs 50 \
  --dpo-min-score-gap 1.0 \
  --dpo-max-rejected-score 0.6 \
  --dpo-negatives-per-prompt 3 \
  --pair-weighting score_gap \
  --min-evaluated-candidates 4 \
  --generation-batch-size "${P1B_GEN_BATCH}" \
  --max-prompt-chars 7000 \
  --seed "${SEED}"
[[ -d "sota_best_of_n_dataset" ]] || { echo "[train] Missing distillation dataset."; exit 1; }
[[ -d "sota_dpo_pairs_dataset" ]] || { echo "[train] Missing DPO pair dataset."; exit 1; }

echo "[train] Stage 4/13: DPO"
python3 phase1c_dpo.py \
  --dpo-dataset-path ./sota_dpo_pairs_dataset \
  --base-model-name "${BASE_MODEL_NAME}" \
  --sft-adapter-path qwen_sft_lora \
  --output-dir outputs_dpo \
  --output-model-dir qwen_dpo_lora \
  --max-steps 600 \
  --min-score-gap 1.0 \
  --max-rejected-score 0.6 \
  --min-test-assert-count 2 \
  --gap-weighted-sampling \
  --max-pairs-per-prompt 6 \
  --eval-every-steps 200 \
  --checkpoint-every-steps 200 \
  --resume-from-checkpoint auto \
  --seed "${SEED}"
[[ -d "qwen_dpo_lora" ]] || { echo "[train] Missing DPO adapter."; exit 1; }

echo "[train] Stage 5/13: ORPO (reference-free preference refinement)"
python3 phase1d_orpo.py \
  --dpo-dataset-path ./sota_dpo_pairs_dataset \
  --base-model-name "${BASE_MODEL_NAME}" \
  --adapter-path qwen_dpo_lora \
  --output-dir outputs_orpo \
  --output-model-dir qwen_orpo_lora \
  --max-steps 300 \
  --learning-rate 3e-7 \
  --checkpoint-every-steps 100 \
  --resume-from-checkpoint auto \
  --seed "${SEED}"
[[ -d "qwen_orpo_lora" ]] || { echo "[train] Missing ORPO adapter."; exit 1; }

echo "[train] Stage 6/13: GRPO"
python3 phase2_grpo.py \
  --max-seq-length "${GRPO_MAX_SEQ_LEN}" \
  --base-model-name "${BASE_MODEL_NAME}" \
  --max-steps 1500 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --output-model-dir qwen_grpo_final \
  --output-adapter-dir qwen_grpo_lora \
  --drop-low-quality-fraction 0.10 \
  --stage-drop-fractions easy:0.15,mid:0.10,hard:0.05,expert:0.03 \
  --curriculum-mode two_dimensional_v1 \
  --priority-source-boost 1.8 \
  --priority-sources online_hard_mining,tool_trajectory_distill \
  --reward-profile "${REWARD_PROFILE}" \
  --prm-model-path "${PRM_MODEL_PATH}" \
  --hard-replay-dataset ./sota_best_of_n_dataset \
  --hard-replay-steps 180 \
  --checkpoint-every-steps 120 \
  --resume-from-checkpoint auto \
  --checkpoints-root outputs_grpo \
  --min-rl-samples-after-drop 1000 \
  --seed "${SEED}" \
  --dataset-paths ./sota_best_of_n_dataset,./swe_supervised_dataset,./sota_slm_coding_dataset
[[ -d "qwen_grpo_final" ]] || { echo "[train] Missing final GRPO model."; exit 1; }
[[ -d "qwen_grpo_lora" ]] || { echo "[train] Missing GRPO adapter."; exit 1; }

echo "[train] Stage 7/13: pre-replay classic eval"
python3 eval_pipeline.py \
  --model-path qwen_grpo_final \
  --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval,private_holdout \
  --swebench-mode "${SWEBENCH_MODE}" \
  --private-holdout-path ./sota_slm_coding_dataset \
  --patch-strategies "${PATCH_STRATEGIES}" \
  --case-id-filter-path "${MANIFEST_DIR}/baseline_eval_classic_cases.jsonl" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 100 \
  --pass-k 8 \
  --verifier-rounds 2 \
  --bootstrap-samples 2000 \
  --seed "${SEED}" \
  --case-log-path "${MANIFEST_DIR}/pre_replay_eval_classic_cases.jsonl" \
  --json-output "${MANIFEST_DIR}/pre_replay_eval_classic.json"
[[ -f "${MANIFEST_DIR}/pre_replay_eval_classic.json" ]] || { echo "[train] Missing pre-replay classic eval report."; exit 1; }

echo "[train] Stage 8/13: pre-replay agentic eval"
python3 eval_pipeline.py \
  --model-path qwen_grpo_final \
  --benchmarks bigcodebench_instruct,livecodebench,mbpp \
  --agentic \
  --search-mode "${AGENT_SEARCH_MODE}" \
  --beam-width "${AGENT_BEAM_WIDTH}" \
  --case-id-filter-path "${MANIFEST_DIR}/baseline_eval_agentic_cases.jsonl" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 100 \
  --max-rounds 3 \
  --n-candidates 8 \
  --candidate-schedule 8,6,4 \
  --verifier-rounds 2 \
  --bootstrap-samples 1000 \
  --seed "${SEED}" \
  --case-log-path "${MANIFEST_DIR}/pre_replay_eval_agentic_cases.jsonl" \
  --json-output "${MANIFEST_DIR}/pre_replay_eval_agentic.json"
[[ -f "${MANIFEST_DIR}/pre_replay_eval_agentic.json" ]] || { echo "[train] Missing pre-replay agentic eval report."; exit 1; }

echo "[train] Stage 9/13: online hard-example mining + trajectory distillation"
python3 scripts/build_hard_examples.py \
  --case-log-paths "${MANIFEST_DIR}/pre_replay_eval_classic_cases.jsonl,${MANIFEST_DIR}/pre_replay_eval_agentic_cases.jsonl" \
  --output-path ./sota_hard_examples_dataset \
  --min-samples 80 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --max-error-type-share 0.45 \
  --max-benchmark-share 0.65 \
  --merge-with ./sota_best_of_n_dataset \
  --max-merged-samples 25000
[[ -d "sota_hard_examples_dataset" ]] || { echo "[train] Missing hard-example dataset."; exit 1; }

python3 scripts/build_tool_trajectories.py \
  --case-log-paths "${MANIFEST_DIR}/pre_replay_eval_agentic_cases.jsonl" \
  --output-path ./sota_tool_trajectory_dataset \
  --min-samples 30 \
  --min-history-rounds 1 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --max-samples 5000
[[ -d "sota_tool_trajectory_dataset" ]] || { echo "[train] Missing tool trajectory dataset."; exit 1; }

echo "[train] Stage 9.5/13: train tiny PRM from real failures"
case "${USE_TINY_PRM}" in
  1|true|TRUE|yes|YES|on|ON)
    python3 scripts/train_prm_tiny.py \
      --case-log-paths "${MANIFEST_DIR}/pre_replay_eval_classic_cases.jsonl,${MANIFEST_DIR}/pre_replay_eval_agentic_cases.jsonl" \
      --output-path "${PRM_MODEL_PATH}" \
      --buckets 8192 \
      --epochs 4 \
      --learning-rate 0.08 \
      --min-samples 120
    [[ -f "${PRM_MODEL_PATH}" ]] || { echo "[train] Missing trained PRM model artifact."; exit 1; }
    REWARD_PROFILE="prm_outcome_v1"
    PRM_ENABLED=1
    ;;
  0|false|FALSE|no|NO|off|OFF)
    echo "[train] Tiny PRM disabled via USE_TINY_PRM=${USE_TINY_PRM}."
    REWARD_PROFILE="dense_exec_v1"
    PRM_ENABLED=0
    ;;
  auto|AUTO|Auto)
    if python3 scripts/train_prm_tiny.py \
      --case-log-paths "${MANIFEST_DIR}/pre_replay_eval_classic_cases.jsonl,${MANIFEST_DIR}/pre_replay_eval_agentic_cases.jsonl" \
      --output-path "${PRM_MODEL_PATH}" \
      --buckets 8192 \
      --epochs 4 \
      --learning-rate 0.08 \
      --min-samples 120; then
      if [[ -f "${PRM_MODEL_PATH}" ]]; then
        REWARD_PROFILE="prm_outcome_v1"
        PRM_ENABLED=1
      else
        echo "[train] WARN: Tiny PRM training reported success but artifact missing; falling back to dense_exec_v1."
        REWARD_PROFILE="dense_exec_v1"
        PRM_ENABLED=0
      fi
    else
      echo "[train] WARN: Tiny PRM training failed in auto mode; falling back to dense_exec_v1."
      REWARD_PROFILE="dense_exec_v1"
      PRM_ENABLED=0
    fi
    ;;
  *)
    echo "[train] ERROR: Unsupported USE_TINY_PRM='${USE_TINY_PRM}'. Use one of: auto, 1, 0."
    exit 1
    ;;
esac

echo "[train] Reward profile selected: ${REWARD_PROFILE} (PRM_ENABLED=${PRM_ENABLED})"

echo "[train] Stage 10/13: hard-replay GRPO pass"
python3 phase2_grpo.py \
  --max-seq-length "${GRPO_MAX_SEQ_LEN}" \
  --base-model-name "${BASE_MODEL_NAME}" \
  --max-steps 260 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --output-model-dir qwen_grpo_final \
  --output-adapter-dir qwen_grpo_lora \
  --drop-low-quality-fraction 0.05 \
  --stage-drop-fractions easy:0.08,mid:0.06,hard:0.04,expert:0.02 \
  --curriculum-mode two_dimensional_v1 \
  --priority-source-boost 1.6 \
  --priority-sources online_hard_mining,tool_trajectory_distill \
  --reward-profile "${REWARD_PROFILE}" \
  --prm-model-path "${PRM_MODEL_PATH}" \
  --hard-replay-dataset ./sota_hard_examples_dataset \
  --hard-replay-steps 120 \
  --checkpoint-every-steps 80 \
  --resume-from-checkpoint auto \
  --checkpoints-root outputs_grpo_replay \
  --min-rl-samples-after-drop 600 \
  --seed "${SEED}" \
  --dataset-paths ./sota_hard_examples_dataset,./sota_tool_trajectory_dataset,./sota_best_of_n_dataset,./swe_supervised_dataset,./sota_slm_coding_dataset
[[ -d "qwen_grpo_final" ]] || { echo "[train] Missing final GRPO model after replay."; exit 1; }

if [[ "${HARD_MINING_CYCLES}" =~ ^[0-9]+$ ]] && [[ "${HARD_MINING_CYCLES}" -gt 1 ]]; then
  for CYCLE in $(seq 2 "${HARD_MINING_CYCLES}"); do
    echo "[train] Stage 10.${CYCLE}/13: iterative hard-mining cycle ${CYCLE}"

    CLASSIC_CASE_LOG="${MANIFEST_DIR}/cycle_${CYCLE}_classic_cases.jsonl"
    AGENTIC_CASE_LOG="${MANIFEST_DIR}/cycle_${CYCLE}_agentic_cases.jsonl"
    HARD_DS="./sota_hard_examples_dataset_cycle_${CYCLE}"
    TRAJ_DS="./sota_tool_trajectory_dataset_cycle_${CYCLE}"
    PRM_PATH_CYCLE="./artifacts/prm_tiny_cycle_${CYCLE}.json"

    python3 eval_pipeline.py \
      --model-path qwen_grpo_final \
      --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval,private_holdout \
      --swebench-mode "${SWEBENCH_MODE}" \
      --private-holdout-path ./sota_slm_coding_dataset \
      --patch-strategies "${PATCH_STRATEGIES}" \
      --max-model-len "${EVAL_MAX_MODEL_LEN}" \
      --num-samples 80 \
      --pass-k 8 \
      --verifier-rounds 2 \
      --bootstrap-samples 1000 \
      --seed "${SEED}" \
      --case-log-path "${CLASSIC_CASE_LOG}" \
      --json-output "${MANIFEST_DIR}/cycle_${CYCLE}_classic.json"
    python3 eval_pipeline.py \
      --model-path qwen_grpo_final \
      --benchmarks bigcodebench_instruct,livecodebench,mbpp \
      --agentic \
      --search-mode "${AGENT_SEARCH_MODE}" \
      --beam-width "${AGENT_BEAM_WIDTH}" \
      --max-model-len "${EVAL_MAX_MODEL_LEN}" \
      --num-samples 80 \
      --max-rounds 3 \
      --n-candidates 8 \
      --candidate-schedule 8,6,4 \
      --verifier-rounds 2 \
      --bootstrap-samples 800 \
      --seed "${SEED}" \
      --case-log-path "${AGENTIC_CASE_LOG}" \
      --json-output "${MANIFEST_DIR}/cycle_${CYCLE}_agentic.json"

    python3 scripts/build_hard_examples.py \
      --case-log-paths "${CLASSIC_CASE_LOG},${AGENTIC_CASE_LOG}" \
      --output-path "${HARD_DS}" \
      --min-samples 60 \
      --min-test-asserts 2 \
      --min-test-lines 3 \
      --min-test-quality-score 2.5 \
      --max-error-type-share 0.45 \
      --max-benchmark-share 0.65 \
      --merge-with ./sota_hard_examples_dataset \
      --max-merged-samples 30000
    [[ -d "${HARD_DS}" ]] || { echo "[train] Missing hard-example dataset for cycle ${CYCLE}."; exit 1; }

    python3 scripts/build_tool_trajectories.py \
      --case-log-paths "${AGENTIC_CASE_LOG}" \
      --output-path "${TRAJ_DS}" \
      --min-samples 20 \
      --min-history-rounds 1 \
      --min-test-asserts 2 \
      --min-test-lines 3 \
      --min-test-quality-score 2.5 \
      --max-samples 3000
    [[ -d "${TRAJ_DS}" ]] || { echo "[train] Missing trajectory dataset for cycle ${CYCLE}."; exit 1; }

    if [[ "${PRM_ENABLED}" -eq 1 ]]; then
      python3 scripts/train_prm_tiny.py \
        --case-log-paths "${CLASSIC_CASE_LOG},${AGENTIC_CASE_LOG}" \
        --output-path "${PRM_PATH_CYCLE}" \
        --buckets 8192 \
        --epochs 3 \
        --learning-rate 0.08 \
        --min-samples 80
      [[ -f "${PRM_PATH_CYCLE}" ]] || { echo "[train] Missing PRM model for cycle ${CYCLE}."; exit 1; }
    else
      PRM_PATH_CYCLE=""
    fi

    python3 phase2_grpo.py \
      --max-seq-length "${GRPO_MAX_SEQ_LEN}" \
      --base-model-name "${BASE_MODEL_NAME}" \
      --max-steps 180 \
      --min-test-asserts 2 \
      --min-test-lines 3 \
      --min-test-quality-score 2.5 \
      --output-model-dir qwen_grpo_final \
      --output-adapter-dir qwen_grpo_lora \
      --drop-low-quality-fraction 0.04 \
      --stage-drop-fractions easy:0.08,mid:0.06,hard:0.04,expert:0.02 \
      --curriculum-mode two_dimensional_v1 \
      --priority-source-boost 1.8 \
      --priority-sources online_hard_mining,tool_trajectory_distill \
      --reward-profile "${REWARD_PROFILE}" \
      --prm-model-path "${PRM_PATH_CYCLE}" \
      --hard-replay-dataset "${HARD_DS}" \
      --hard-replay-steps 100 \
      --checkpoint-every-steps 80 \
      --resume-from-checkpoint auto \
      --checkpoints-root "outputs_grpo_replay_cycle_${CYCLE}" \
      --min-rl-samples-after-drop 600 \
      --seed "${SEED}" \
      --dataset-paths "${HARD_DS},${TRAJ_DS},./sota_hard_examples_dataset,./sota_tool_trajectory_dataset,./sota_best_of_n_dataset,./swe_supervised_dataset,./sota_slm_coding_dataset"
  done
fi

echo "[train] Stage 11/13: final eval after replay"
python3 eval_pipeline.py \
  --model-path qwen_grpo_final \
  --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval,private_holdout \
  --swebench-mode "${SWEBENCH_MODE}" \
  --private-holdout-path ./sota_slm_coding_dataset \
  --patch-strategies "${PATCH_STRATEGIES}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 100 \
  --pass-k 8 \
  --verifier-rounds 2 \
  --bootstrap-samples 2000 \
  --seed "${SEED}" \
  --case-log-path "${MANIFEST_DIR}/posttrain_eval_classic_cases.jsonl" \
  --json-output "${MANIFEST_DIR}/posttrain_eval_classic.json"
[[ -f "${MANIFEST_DIR}/posttrain_eval_classic.json" ]] || { echo "[train] Missing post-train classic eval report."; exit 1; }

python3 eval_pipeline.py \
  --model-path qwen_grpo_final \
  --benchmarks bigcodebench_instruct,livecodebench,mbpp \
  --agentic \
  --search-mode "${AGENT_SEARCH_MODE}" \
  --beam-width "${AGENT_BEAM_WIDTH}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 100 \
  --max-rounds 3 \
  --n-candidates 8 \
  --candidate-schedule 8,6,4 \
  --verifier-rounds 2 \
  --bootstrap-samples 1000 \
  --seed "${SEED}" \
  --case-log-path "${MANIFEST_DIR}/posttrain_eval_agentic_cases.jsonl" \
  --json-output "${MANIFEST_DIR}/posttrain_eval_agentic.json"
[[ -f "${MANIFEST_DIR}/posttrain_eval_agentic.json" ]] || { echo "[train] Missing post-train agentic eval report."; exit 1; }

echo "[train] Stage 12/13: export"
python3 export.py

echo "[train] Stage 13/13: final validation + acceptance gates"
[[ -d "model_export_gguf" ]] || { echo "[train] Missing GGUF export directory."; exit 1; }
[[ -d "model_export_hf" ]] || { echo "[train] Missing HF export directory."; exit 1; }
python3 scripts/validate_cli_drift.py
python3 -m unittest tests.test_verification tests.test_rewards tests.test_prm_tiny tests.test_scientific_gate
python3 scripts/scientific_gate.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --main-dataset-manifest-path ./sota_slm_coding_dataset/dataset_manifest.json \
  --swe-supervised-manifest-path ./swe_supervised_dataset/dataset_manifest.json \
  --require-relative-improvement 0.15 \
  --require-relative-improvement-lb 0.05 \
  --max-format-error-rate 0.02 \
  --bootstrap-samples 4000 \
  --null-samples 4000 \
  --significance-alpha 0.05 \
  --seed "${SEED}"

echo "[train] Full RunPod training pipeline completed successfully."
