# SOTA SLM Pipeline (RunPod-Ready, Coder-First 1.5B)

Production stack for a compact coding model with fixed base checkpoint:

- Base model (fixed): `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- No A/B branch
- Spot-safe checkpoint/resume
- Fail-fast artifact gates
- SWE harness-first evaluation

## Implemented Upgrades

1. Real SWE harness as primary eval target (not only proxy).
2. Online hard-example mining from real evaluation failures.
3. Combined PRM + outcome reward profile (`prm_outcome_v1`) in GRPO.
4. Tool-use trajectory distillation from agentic histories.
5. Search-time compute for inference/eval (`greedy|beam|mcts`) with verifier-aware reranking.
6. Two-dimensional curriculum (`core_code` -> `repo_resolution`) with difficulty sub-stages.
7. Priority-source replay upsampling for hard mined + trajectory data.
8. Dedicated `swe_supervised_dataset` path (Issue/Patch/Tests/Trajectory metadata).
9. ORPO reference-free preference stage before GRPO.
10. Tiny trainable PRM (`scripts/train_prm_tiny.py`) integrated into GRPO reward.
11. Iterative hard-mining cycles (`HARD_MINING_CYCLES`) with failure-cluster replay.
12. Private holdout benchmark support (`private_holdout`) with CI-aware acceptance gates.
13. Scientific acceptance gate with paired significance testing on case-level logs.
14. Dataset manifests (`dataset_manifest.json`) with split fingerprints and provenance metadata.
15. Seeded deterministic evaluation sampling for reproducible case-level comparisons.

## RunPod Quickstart

```bash
cd /workspace/sota_slm_pipeline
bash scripts/runpod_setup.sh
```

Optional for gated datasets:

```bash
export HF_TOKEN=hf_xxx
```

Run full production pipeline:

```bash
bash scripts/runpod_train_full.sh
```

Default RunPod profile is tuned for RTX 3090 / 24GB:

- SFT/GRPO `max_seq_length=6144`
- conservative generation batch in Phase 1b
- allocator settings for long-run stability

## Full Flow (13 stages)

1. Stage 0: baseline freeze + run manifest.
2. Stage 1: contamination-safe data build + dedicated SWE supervised build.
3. Stage 2: SFT with coder base + LoRA (`r=48`, `alpha=96`, `dropout=0.03`) using merged dataset paths.
4. Stage 3: best-of-N + hardened DPO pair mining (multi-temp, multi-negative).
5. Stage 4: DPO with gap-weighting, max-pairs guard, resume-safe checkpoints.
6. Stage 5: ORPO reference-free preference refinement.
7. Stage 6: GRPO curriculum with `prm_outcome_v1` reward.
8. Stage 7-8: pre-replay classic/agentic eval (+ case logs).
9. Stage 9: hard-example mining + trajectory distillation.
10. Stage 9.5: train tiny PRM on real run failures.
11. Stage 10: hard-replay GRPO pass (+ optional iterative cycles).
12. Stage 11-12: final eval (incl. private holdout) + export.
13. Stage 13: strict CI-aware acceptance gates.

## Key Scripts

- `scripts/runpod_setup.sh`
- `scripts/runpod_train_full.sh`
- `scripts/build_hard_examples.py`
- `scripts/build_tool_trajectories.py`
- `scripts/train_prm_tiny.py`
- `scripts/validate_cli_drift.py`
- `scripts/scientific_gate.py`

## Core CLI Knobs

### Eval Harness + Search

```bash
python3 eval_pipeline.py \
  --model-path qwen_grpo_final \
  --benchmarks livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval,private_holdout \
  --swebench-mode harness \
  --private-holdout-path ./sota_slm_coding_dataset \
  --patch-strategies minimal_diff,api_first,test_first,balanced \
  --search-mode beam \
  --beam-width 2 \
  --bootstrap-samples 2000
```

### GRPO Reward Profile

```bash
python3 phase2_grpo.py \
  --base-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --curriculum-mode two_dimensional_v1 \
  --priority-source-boost 1.8 \
  --priority-sources online_hard_mining,tool_trajectory_distill \
  --reward-profile prm_outcome_v1 \
  --prm-model-path ./artifacts/prm_tiny_v1.json \
  --stage-drop-fractions easy:0.15,mid:0.10,hard:0.05,expert:0.03 \
  --hard-replay-dataset ./sota_hard_examples_dataset
```

### Distillation Mining

```bash
python3 phase1b_rejection_sampling.py \
  --temperatures 0.2,0.8 \
  --dpo-negatives-per-prompt 3 \
  --pair-weighting score_gap
```

### ORPO

```bash
python3 phase1d_orpo.py \
  --dpo-dataset-path ./sota_dpo_pairs_dataset \
  --adapter-path qwen_dpo_lora \
  --output-model-dir qwen_orpo_lora
```

## Acceptance Gates

`scripts/runpod_train_full.sh` enforces:

- `format_error_rate < 2%`
- `>= +15%` relative primary objective improvement vs baseline
- per-benchmark no-regression on `pass@1` and `format_error_rate` (small CI tolerance)
- mandatory artifacts for every stage (no empty "success")

## Important Artifacts

- `run_manifests/<timestamp>/run_manifest.txt`
- `run_manifests/<timestamp>/baseline_eval_classic.json`
- `run_manifests/<timestamp>/baseline_eval_agentic.json`
- `run_manifests/<timestamp>/pre_replay_eval_*.json`
- `run_manifests/<timestamp>/posttrain_eval_*.json`
- `run_manifests/<timestamp>/*_cases.jsonl`
- `run_manifests/<timestamp>/scientific_acceptance.json`
- `sota_slm_coding_dataset`
- `sota_slm_coding_dataset/dataset_manifest.json`
- `sota_best_of_n_dataset`
- `sota_dpo_pairs_dataset`
- `swe_supervised_dataset`
- `swe_supervised_dataset/dataset_manifest.json`
- `sota_hard_examples_dataset`
- `sota_tool_trajectory_dataset`
- `artifacts/prm_tiny_v1.json`
- `qwen_sft_lora`, `qwen_sft_merged`, `qwen_dpo_lora`, `qwen_orpo_lora`, `qwen_grpo_lora`, `qwen_grpo_final`
- `model_export_gguf`, `model_export_hf`

## Notes

- In `--swebench-mode harness`, eval fails hard if harness execution fails.
- In `--swebench-mode auto`, eval falls back to proxy scoring only if harness fails.
- `STRICT_SWEBENCH_HARNESS=1` enforces harness-only operation for primary runs.
- You can increase iterative failure replay via `HARD_MINING_CYCLES` (default `2`).
- Trajectory distillation uses successful trajectories by default (failed final states are excluded).
- Keep seed fixed (`3407`) for high reproducibility across stages.
- Final eval reuses baseline case IDs (`--case-id-filter-path`) for paired significance tests.
- See scientific methodology and references: `docs/SCIENTIFIC_PROTOCOL.md`.
