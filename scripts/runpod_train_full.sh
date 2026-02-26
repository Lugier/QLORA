#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[train] ERROR: nvidia-smi not found. Run this on a GPU RunPod instance."
  exit 1
fi

BASE_MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
SEED="3407"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
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

echo "[train] Stage 0/11: baseline freeze + run manifest"
{
  echo "run_timestamp_utc=${RUN_TS}"
  echo "base_model_name=${BASE_MODEL_NAME}"
  echo "seed=${SEED}"
  echo "gpu_name=${GPU_NAME}"
  echo "gpu_mem_mb=${GPU_MEM_MB}"
  echo "profile_sft_max_seq_len=${SFT_MAX_SEQ_LEN}"
  echo "profile_p1b_generation_batch_size=${P1B_GEN_BATCH}"
  echo "profile_grpo_max_seq_len=${GRPO_MAX_SEQ_LEN}"
  echo "python_version=$(python3 --version 2>&1)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
  echo "cuda_info_start"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo "cuda_info_end"
} > "${MANIFEST_DIR}/run_manifest.txt"

python3 eval_pipeline.py \
  --model-path "${BASE_MODEL_NAME}" \
  --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval \
  --swebench-mode "${SWEBENCH_MODE}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 30 \
  --pass-k 4 \
  --bootstrap-samples 500 \
  --json-output "${MANIFEST_DIR}/baseline_eval_classic.json"

python3 eval_pipeline.py \
  --model-path "${BASE_MODEL_NAME}" \
  --benchmarks bigcodebench_instruct,livecodebench,mbpp \
  --agentic \
  --search-mode "${AGENT_SEARCH_MODE}" \
  --beam-width "${AGENT_BEAM_WIDTH}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 30 \
  --max-rounds 3 \
  --n-candidates 8 \
  --candidate-schedule 8,6,4 \
  --bootstrap-samples 300 \
  --json-output "${MANIFEST_DIR}/baseline_eval_agentic.json"

[[ -f "${MANIFEST_DIR}/run_manifest.txt" ]] || { echo "[train] Missing run manifest."; exit 1; }
[[ -f "${MANIFEST_DIR}/baseline_eval_classic.json" ]] || { echo "[train] Missing baseline classic report."; exit 1; }
[[ -f "${MANIFEST_DIR}/baseline_eval_agentic.json" ]] || { echo "[train] Missing baseline agentic report."; exit 1; }

echo "[train] Stage 1/11: data pipeline"
python3 data_pipeline.py \
  --min-total-samples 20000 \
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

echo "[train] Stage 2/11: SFT"
python3 phase1_sft.py \
  --dataset-path ./sota_slm_coding_dataset \
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

echo "[train] Stage 3/11: best-of-N + DPO pair mining"
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

echo "[train] Stage 4/11: DPO"
python3 phase1c_dpo.py \
  --dpo-dataset-path ./sota_dpo_pairs_dataset \
  --base-model-name "${BASE_MODEL_NAME}" \
  --sft-adapter-path qwen_sft_lora \
  --output-dir outputs_dpo \
  --output-model-dir qwen_dpo_lora \
  --max-steps 800 \
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

echo "[train] Stage 5/11: GRPO"
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
  --reward-profile prm_outcome_v1 \
  --hard-replay-dataset ./sota_best_of_n_dataset \
  --hard-replay-steps 180 \
  --checkpoint-every-steps 120 \
  --resume-from-checkpoint auto \
  --checkpoints-root outputs_grpo \
  --min-rl-samples-after-drop 1000 \
  --seed "${SEED}" \
  --dataset-paths ./sota_best_of_n_dataset,./sota_slm_coding_dataset
[[ -d "qwen_grpo_final" ]] || { echo "[train] Missing final GRPO model."; exit 1; }
[[ -d "qwen_grpo_lora" ]] || { echo "[train] Missing GRPO adapter."; exit 1; }

echo "[train] Stage 6/11: pre-replay classic eval"
python3 eval_pipeline.py \
  --model-path qwen_grpo_final \
  --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval \
  --swebench-mode "${SWEBENCH_MODE}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 100 \
  --pass-k 8 \
  --verifier-rounds 2 \
  --bootstrap-samples 2000 \
  --case-log-path "${MANIFEST_DIR}/pre_replay_eval_classic_cases.jsonl" \
  --json-output "${MANIFEST_DIR}/pre_replay_eval_classic.json"
[[ -f "${MANIFEST_DIR}/pre_replay_eval_classic.json" ]] || { echo "[train] Missing pre-replay classic eval report."; exit 1; }

echo "[train] Stage 7/11: pre-replay agentic eval"
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
  --case-log-path "${MANIFEST_DIR}/pre_replay_eval_agentic_cases.jsonl" \
  --json-output "${MANIFEST_DIR}/pre_replay_eval_agentic.json"
[[ -f "${MANIFEST_DIR}/pre_replay_eval_agentic.json" ]] || { echo "[train] Missing pre-replay agentic eval report."; exit 1; }

echo "[train] Stage 8/11: online hard-example mining + trajectory distillation"
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

echo "[train] Stage 9/11: hard-replay GRPO pass"
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
  --priority-source-boost 2.0 \
  --priority-sources online_hard_mining,tool_trajectory_distill \
  --reward-profile prm_outcome_v1 \
  --hard-replay-dataset ./sota_hard_examples_dataset \
  --hard-replay-steps 120 \
  --checkpoint-every-steps 80 \
  --resume-from-checkpoint auto \
  --checkpoints-root outputs_grpo_replay \
  --min-rl-samples-after-drop 600 \
  --seed "${SEED}" \
  --dataset-paths ./sota_hard_examples_dataset,./sota_tool_trajectory_dataset,./sota_best_of_n_dataset,./sota_slm_coding_dataset
[[ -d "qwen_grpo_final" ]] || { echo "[train] Missing final GRPO model after replay."; exit 1; }

echo "[train] Stage 10/11: final eval after replay"
python3 eval_pipeline.py \
  --model-path qwen_grpo_final \
  --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval \
  --swebench-mode "${SWEBENCH_MODE}" \
  --max-model-len "${EVAL_MAX_MODEL_LEN}" \
  --num-samples 100 \
  --pass-k 8 \
  --verifier-rounds 2 \
  --bootstrap-samples 2000 \
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
  --case-log-path "${MANIFEST_DIR}/posttrain_eval_agentic_cases.jsonl" \
  --json-output "${MANIFEST_DIR}/posttrain_eval_agentic.json"
[[ -f "${MANIFEST_DIR}/posttrain_eval_agentic.json" ]] || { echo "[train] Missing post-train agentic eval report."; exit 1; }

echo "[train] Stage 11/11: export"
python3 export.py

echo "[train] Final validation: artifacts + tests + acceptance gates"
[[ -d "model_export_gguf" ]] || { echo "[train] Missing GGUF export directory."; exit 1; }
[[ -d "model_export_hf" ]] || { echo "[train] Missing HF export directory."; exit 1; }
python3 scripts/validate_cli_drift.py
python3 -m unittest tests.test_verification tests.test_rewards

MANIFEST_DIR="${MANIFEST_DIR}" python3 - <<'PY'
import json
import os
import sys

manifest_dir = os.environ["MANIFEST_DIR"]
baseline_classic_path = os.path.join(manifest_dir, "baseline_eval_classic.json")
baseline_agentic_path = os.path.join(manifest_dir, "baseline_eval_agentic.json")
post_classic_path = os.path.join(manifest_dir, "posttrain_eval_classic.json")
post_agentic_path = os.path.join(manifest_dir, "posttrain_eval_agentic.json")

for path in [baseline_classic_path, baseline_agentic_path, post_classic_path, post_agentic_path]:
    if not os.path.exists(path):
        raise SystemExit(f"[train] Missing evaluation report: {path}")

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mean_ci(report, bench_name, metric):
    per_bench = report.get("per_benchmark", {})
    if bench_name not in per_bench:
        raise RuntimeError(f"Benchmark '{bench_name}' missing in report.")
    values = per_bench[bench_name].get(metric)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Metric '{metric}' missing for benchmark '{bench_name}'.")
    return float(values[0])

def global_mean(report, metric):
    global_block = report.get("global", {})
    values = global_block.get(metric)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Global metric '{metric}' missing.")
    return float(values[0])

def primary_score(classic, agentic):
    lcb = mean_ci(classic, "livecodebench", "pass_at_1_ci")
    bcb = mean_ci(classic, "bigcodebench_instruct", "pass_at_1_ci")
    swe = mean_ci(classic, "swebench_verified_subset", "pass_at_1_ci")
    agentic_success = global_mean(agentic, "pass_at_1_ci")
    format_error_rate = global_mean(classic, "format_error_rate_ci")
    return (
        (0.35 * lcb)
        + (0.25 * bcb)
        + (0.20 * swe)
        + (0.10 * agentic_success)
        + (0.10 * (1.0 - format_error_rate))
    ), format_error_rate

def benchmark_mean(report, benchmark, metric):
    block = report.get("per_benchmark", {}).get(benchmark, {})
    values = block.get(metric)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Missing metric '{metric}' for benchmark '{benchmark}'")
    return float(values[0])

baseline_classic = load(baseline_classic_path)
baseline_agentic = load(baseline_agentic_path)
post_classic = load(post_classic_path)
post_agentic = load(post_agentic_path)

baseline_score, baseline_format = primary_score(baseline_classic, baseline_agentic)
post_score, post_format = primary_score(post_classic, post_agentic)
improvement = (post_score - baseline_score) / max(1e-6, baseline_score)

print(
    "[train] Primary objective summary: "
    f"baseline={baseline_score:.6f}, post={post_score:.6f}, improvement={improvement * 100:.2f}%, "
    f"post_format_error_rate={post_format * 100:.2f}%"
)

if post_format >= 0.02:
    raise SystemExit(
        f"[train] Acceptance gate failed: format_error_rate={post_format:.4f} >= 0.0200"
    )
if improvement < 0.15:
    raise SystemExit(
        f"[train] Acceptance gate failed: relative primary score improvement={improvement:.4f} < 0.1500"
    )

# Strict no-regression gates per benchmark (Pass@1 and format error rate).
benchmarks = [
    "livecodebench",
    "bigcodebench_instruct",
    "swebench_verified_subset",
    "mbpp",
    "humaneval",
]
for bench in benchmarks:
    base_p1 = benchmark_mean(baseline_classic, bench, "pass_at_1_ci")
    post_p1 = benchmark_mean(post_classic, bench, "pass_at_1_ci")
    # Small tolerance for bootstrap noise.
    if post_p1 + 0.005 < base_p1:
        raise SystemExit(
            "[train] Acceptance gate failed: pass@1 regression on "
            f"{bench}: baseline={base_p1:.4f}, post={post_p1:.4f}"
        )

    base_fmt = benchmark_mean(baseline_classic, bench, "format_error_rate_ci")
    post_fmt = benchmark_mean(post_classic, bench, "format_error_rate_ci")
    if post_fmt > base_fmt + 0.005:
        raise SystemExit(
            "[train] Acceptance gate failed: format error regression on "
            f"{bench}: baseline={base_fmt:.4f}, post={post_fmt:.4f}"
        )
PY

echo "[train] Full RunPod training pipeline completed successfully."
