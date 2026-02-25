#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[train] Logging into Hugging Face with HF_TOKEN..."
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
fi

echo "[train] Stage 1/8: data pipeline"
python3 data_pipeline.py \
  --min-total-samples 20000 \
  --min-test-coverage 0.08 \
  --min-prompt-coverage 0.99 \
  --min-unique-ratio 0.75 \
  --max-missing-sources 6
[[ -d "sota_slm_coding_dataset" ]] || { echo "[train] Missing dataset output."; exit 1; }

echo "[train] Stage 2/8: SFT"
python3 phase1_sft.py \
  --dataset-path ./sota_slm_coding_dataset \
  --adapter-dir qwen_sft_lora \
  --merged-model-dir qwen_sft_merged \
  --output-dir outputs_sft \
  --max-seq-length 8192 \
  --max-steps 3000
[[ -d "qwen_sft_lora" ]] || { echo "[train] Missing SFT adapter."; exit 1; }
[[ -d "qwen_sft_merged" ]] || { echo "[train] Missing merged SFT model."; exit 1; }

echo "[train] Stage 3/8: best-of-N + DPO pair mining"
python3 phase1b_rejection_sampling.py \
  --model-path qwen_sft_merged \
  --dataset-path ./sota_slm_coding_dataset \
  --output-path ./sota_best_of_n_dataset \
  --dpo-output-path ./sota_dpo_pairs_dataset \
  --num-prompts 2000 \
  --num-samples-per-prompt 16 \
  --timeout 1.0 \
  --verifier-rounds 2 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --min-perfect 50 \
  --min-dpo-pairs 50 \
  --dpo-min-score-gap 1.0 \
  --dpo-max-rejected-score 0.6 \
  --min-evaluated-candidates 4
[[ -d "sota_best_of_n_dataset" ]] || { echo "[train] Missing distillation dataset."; exit 1; }
[[ -d "sota_dpo_pairs_dataset" ]] || { echo "[train] Missing DPO pair dataset."; exit 1; }

echo "[train] Stage 4/8: DPO"
python3 phase1c_dpo.py \
  --dpo-dataset-path ./sota_dpo_pairs_dataset \
  --sft-adapter-path qwen_sft_lora \
  --output-dir outputs_dpo \
  --output-model-dir qwen_dpo_lora \
  --max-steps 800 \
  --min-score-gap 1.0 \
  --max-rejected-score 0.6 \
  --min-test-assert-count 2
[[ -d "qwen_dpo_lora" ]] || { echo "[train] Missing DPO adapter."; exit 1; }

echo "[train] Stage 5/8: GRPO"
python3 phase2_grpo.py \
  --max-seq-length 8192 \
  --max-steps 1500 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --output-model-dir qwen_grpo_final \
  --output-adapter-dir qwen_grpo_lora \
  --dataset-paths ./sota_best_of_n_dataset,./sota_slm_coding_dataset
[[ -d "qwen_grpo_final" ]] || { echo "[train] Missing final GRPO model."; exit 1; }
[[ -d "qwen_grpo_lora" ]] || { echo "[train] Missing GRPO adapter."; exit 1; }

echo "[train] Stage 6/8: classic eval"
python3 eval_pipeline.py --model-path qwen_grpo_final --benchmarks mbpp,humaneval --num-samples 100 --pass-k 8 --verifier-rounds 2

echo "[train] Stage 7/8: agentic eval"
python3 eval_pipeline.py --model-path qwen_grpo_final --benchmarks mbpp,humaneval --agentic --num-samples 100 --max-rounds 3 --n-candidates 8 --verifier-rounds 2

echo "[train] Stage 8/8: export"
python3 export.py

echo "[train] Full RunPod training pipeline completed successfully."
