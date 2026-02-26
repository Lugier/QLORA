# QLORA Coding Model Pipeline (RunPod Ready)

This repository trains a compact coding model (1.5B) to become a strong software-engineering assistant.

Base model:
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`

## What This Project Does (Non-Technical)

Think of this as a 13-step bootcamp for an AI coder:

1. We test the raw model first (baseline).
2. We prepare high-quality training data.
3. We teach formatting and coding fundamentals.
4. We teach preference between better/worse answers.
5. We strengthen behavior with reinforcement learning.
6. We repeatedly test, collect failures, and retrain on hard cases.
7. We only accept the run if strict quality/statistics gates pass.

Goal:
- Better real coding performance
- Better repo bug-fixing behavior
- Lower format mistakes
- Reproducible, auditable training runs

---

## End-to-End Flow (What Happens, When, and Why)

### Stage 0: Baseline Freeze
- What: Evaluate the start model and save environment + run manifest.
- Why: We need a trustworthy before/after comparison.

### Stage 1: Data Pipeline
- What: Build curated datasets, deduplicate, split into train/validation/holdout.
- Why: Data quality drives model quality.

### Stage 2: SFT (Supervised Fine-Tuning)
- What: Teach clean structure and coding behavior on curated examples.
- Why: Gives stable foundation before preference/RL stages.

### Stage 3: Best-of-N + Pair Mining
- What: Generate many candidates and keep high-quality outcomes + preference pairs.
- Why: Converts raw generation into stronger training signals.

### Stage 4: DPO
- What: Train on chosen vs rejected responses.
- Why: Better alignment with preferred coding style/outcomes.

### Stage 5: ORPO
- What: Additional preference optimization stage.
- Why: Stabilizes policy before RL.

### Stage 6: GRPO (Reinforcement Learning)
- What: Curriculum RL using execution rewards, process rewards, and penalties.
- Why: Improves real solve-rate under tests and constraints.

### Stage 7-8: Pre-Replay Evaluation
- What: Run benchmark evaluations and collect detailed case logs.
- Why: Identify concrete weaknesses before replay training.

### Stage 9: Hard Example + Trajectory Mining
- What: Build hard replay dataset and tool-use trajectory dataset from failures.
- Why: Train directly on what the model currently gets wrong.

### Stage 9.5: Tiny PRM Training
- What: Train a compact process-reward helper model from real logs.
- Why: Better process-level reward shaping.

### Stage 10: Hard Replay RL
- What: Short focused RL pass on hardest cases.
- Why: Push performance on difficult SWE patterns.

### Stage 11-12: Final Evaluation + Export
- What: Final benchmark run and model export.
- Why: Produce deployable artifacts with validated quality.

### Stage 13: Scientific Acceptance Gate
- What: CI/statistical checks, no-regression checks, and quality thresholds.
- Why: Prevent false wins and ensure robust improvements.

---

## Repository Structure

```text
sota_slm_pipeline/
├── pipeline/
│   ├── core/                      # Shared runtime/reward/verification logic
│   │   ├── rewards.py
│   │   ├── verification.py
│   │   ├── runtime_agent.py
│   │   ├── sandbox.py
│   │   └── prm_tiny.py
│   ├── stages/                    # Main training/eval stage implementations
│   │   ├── data_pipeline.py
│   │   ├── phase1_sft.py
│   │   ├── phase1b_rejection_sampling.py
│   │   ├── phase1c_dpo.py
│   │   ├── phase1d_orpo.py
│   │   ├── phase2_grpo.py
│   │   ├── eval_pipeline.py
│   │   └── export.py
│   ├── docs/
│   └── config/
├── scripts/                       # RunPod orchestration + utilities
│   ├── runpod_setup.sh
│   ├── runpod_train_full.sh
│   ├── build_hard_examples.py
│   ├── build_tool_trajectories.py
│   ├── train_prm_tiny.py
│   ├── scientific_gate.py
│   └── validate_cli_drift.py
├── tests/                         # Unit/regression tests
├── docs/
│   └── SCIENTIFIC_PROTOCOL.md
└── *.py                           # Backward-compatible wrappers (legacy entrypoints)
```

Notes:
- Root-level `*.py` files are compatibility wrappers so old commands still work.
- New canonical implementation lives under `pipeline/core` and `pipeline/stages`.

---

## Quick Start (RunPod)

```bash
cd /workspace/sota_slm_pipeline
bash scripts/runpod_setup.sh
bash scripts/runpod_preflight.sh
bash scripts/runpod_train_full.sh
```

Optional:
```bash
export HF_TOKEN=hf_xxx
```

Spot-resume best practice:
```bash
export RUN_ID=run_20260226_a
bash scripts/runpod_train_full.sh
```

If the pod is interrupted, start a new pod with the same volume and run again with the same `RUN_ID`.
Stage 0 baseline artifacts are reused, and training stages continue from checkpoints (`resume_from_checkpoint=auto`).

---

## What “Success” Means

The run is accepted only if strict gates pass, including:
- low format error rate
- significant improvement over baseline
- no major benchmark regressions
- complete artifacts + manifests

Scientific details:
- see `docs/SCIENTIFIC_PROTOCOL.md`

---

## For External Readers

If you are new:
1. Read this README first.
2. Read `docs/SCIENTIFIC_PROTOCOL.md` for evaluation rigor.
3. Run `scripts/runpod_train_full.sh` for the full pipeline.
4. Inspect `run_manifests/<timestamp>/` for reports and audit trail.
