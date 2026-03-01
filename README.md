<div align="center">
  <h1>🧬 QLORA: My First SLM Fine-Tuning Journey & Scientific Report</h1>
  <p><i>From a raw 1.5B parameter generative model to a structured, reasoning AI Agent.</i></p>
</div>

Welcome to my **very first attempt** at fine-tuning a Small Language Model (SLM)! This repository documents my journey of taking a foundational 1.5B parameter coding model and subjecting it to a state-of-the-art multi-stage Reinforcement Learning (RL) pipeline. 

Spoiler alert: It wasn't just a simple `.train()` call. It failed in fascinating ways, taught me crucial lessons about AI behavior, and ultimately yielded an optimized model capable of advanced reasoning and real-world repository bug-fixing.

---

## 📊 The Final Verdict: Was it a success?

Yes and no. The pipeline successfully ran end-to-end, and the model learned to "think" out loud using `<reasoning>` tags. However, I discovered that Reinforcement Learning is a double-edged sword for small models. 

| Benchmark | Baseline (Qwen 1.5B) | Finetuned (GRPO Final) | Difference |
| :--- | :---: | :---: | :--- |
| **HumanEval** (Pass@K) | 69.0% | **81.0%** | 🟢 **+12.0%** (Massive improvement in logic extraction) |
| **SWE-Bench** (Proxy Score) | 0.0%* | **34.8%** | 🟢 SLM successfully locates bugs in complex repos |
| **Custom 50 Hard** (Pass@1) | **90.0%** | 86.0% | 🔴 **-4.0%** (The "Thinking-Tax" penalty) |
| **MBPP** (Agentic Pass@4)| 20.0% | 20.0% | 🟡 **Stagnant** (Limits of SLM self-correction) |

*\*Baseline not evaluated on SWE-Bench due to output format incompatibility.*

---

## 🔬 Scientific Findings & Why Things "Failed"

This project was a massive learning curve. Here is exactly what failed, why it failed, and what scientists and engineers can learn from it:

### 1. The "Thinking-Tax" (Zero-Shot vs. Multi-Shot Reasoning)
* **The Expectation:** RL using GRPO would make the model strictly smarter at everything.
* **The Reality:** On heavily memorized standard algorithmic problems (my `Custom 50 Hard` set), the finetuned model saw a slight dip in immediate Zero-Shot precision (Pass@1 dropped from 90% to 86%). 
* **Why it failed:** GRPO enforces long `<reasoning>` chains. The model over-engineered simple problems in its reasoning step, essentially "talking itself" into a misunderstood approach. However, on complex logic tasks (HumanEval), this forced reasoning increased the Pass@K score from 69% to 81%. 
* **Takeaway:** RL trades absolute first-shot memorization precision for stable, iterative, and deep problem-solving capabilities.

### 2. The Limits of SLM Agentic Self-Correction
* **The Expectation:** If the model gets an execution error from a Python sandbox, it will read the error and rewrite the code perfectly.
* **The Reality:** During Phase 8 Agentic Eval on MBPP, the model was allowed to self-correct up to 3 times. The Pass@1 and Pass@4 scores remained absolutely stagnant at 20%.
* **Why it failed:** A 1.5B parameter model acts like a Junior Developer. It can fix a syntax typo (changing `>` to `>=`), but it completely lacks the architectural macro-perspective to delete a flawed algorithm and rewrite it from scratch. 
* **Takeaway:** SLMs hit a hard "Self-Correction" ceiling. To break this, you need Process Reward Models (PRMs) providing step-by-step logic guidance, rather than just outcome-based execution errors.

### 3. Data Contamination & Role-Playing Strictness
* **The Expectation:** The model obeys the system prompt to output Python inside `<answer>` tags.
* **The Reality:** During evaluation on my private holdout set (competitive programming questions), the baseline model had a **75% format error rate**. It immediately reverted to generating C++ routines with `cin` and `cout`, ignoring the XML tags entirely.
* **Why it failed:** Models inherently default back to their pre-training distributions (Codeforces C++ templates) when the prompt context smells like a competitive programming contest. 
* **Takeaway:** You need heavy negative RL penalization (format rewards) to enforce strict role-playing compliance and break the model out of its pre-training habits.

### 4. SWE-Bench: SLMs as Effective "Scout Agents"
* **The Reality:** On real-world GitHub issues (SWE-Bench Verified), the 1.5B model achieved a **3% perfect patch rate** and an impressive **34.8% proxy score** (successfully locating the correct file and function to edit). 
* **Takeaway:** Massive 80B+ models are not required for repository navigation! A 1.5B SLM is a highly cost-effective "Scout Agent", efficiently scanning repositories to locate bugs before handing the complex patch-writing off to larger, more expensive models.

---

## 🏗️ The 13-Step SOTA Pipeline

Here is the exact pipeline I built to achieve these results (fully reproducible via RunPod):

### Phase 1: Foundation & Alignment
* **Stage 0: Baseline Freeze** - Evaluate the start model and save environments to ensure a trustworthy before/after comparison.
* **Stage 1: Data Pipeline** - Build curated datasets, deduplicate, and split into train/validation/holdout.
* **Stage 2: SFT (Supervised Fine-Tuning)** - Teach clean structure (`<reasoning>` and `<answer>` tags) on curated examples.
* **Stage 3 & 4: Rejection Sampling & DPO** - Generate many candidates, score them, and train the model on chosen vs. rejected responses to align coding style.
* **Stage 5: ORPO** - Additional preference optimization to stabilize the policy before the volatile RL stage.

### Phase 2: Reinforcement Learning & Agentic Growth
* **Stage 6: GRPO (Reinforcement Learning)** - Curriculum RL using execution rewards (sandbox tests), process rewards, and strict formatting penalties.
* **Stage 7 & 8: Pre-Replay Evaluation** - Run benchmark evaluations (Classic & Agentic) and collect detailed case logs to identify concrete weaknesses.

### Phase 3: Targeted Hard-Mining
* **Stage 9: Hard Example Mining** - Extract the exact failures from Stage 7 & 8 to build a dynamic "Hard Replay" dataset.
* **Stage 9.5: Tiny PRM Training** - Train a compact process-reward helper model from real logs.
* **Stage 10: Hard Replay RL** - A short, focused aggressive RL pass exclusively on the hardest cases identified.

### Phase 4: Validation
* **Stage 11-13: Final Evaluation, Export & Scientific Gate** - Final benchmark run, model export to HuggingFace/GGUF, and strict CI/statistical checks to prevent false positive wins.

---

## 💻 Repository Structure & Usage

```text
sota_slm_pipeline/
├── pipeline/
│   ├── core/                      # Shared runtime/reward/sandbox logic
│   └── stages/                    # Main training/eval stage implementations
├── scripts/                       # RunPod orchestration + utilities
│   ├── runpod_deploy_and_train.sh # 🚀 ONE-CLICK LAUNCHER
│   └── runpod_train_full.sh       # Full 13-stage execution
├── tests/                         # Unit/regression tests
├── docs/
│   └── SCIENTIFIC_PROTOCOL.md     # Statistical rigor documentation
├── create_50_hard.py              # Custom 50-Algorithm Dataset generator
└── README.md
```

### Quick Start (Launch on RunPod)

1. **Create the pod**: `RTX 3090` (24GB) or stronger.
2. **Clone & Run**:
```bash
git clone https://github.com/Lugier/QLORA.git
cd QLORA
export RUN_ID=my_first_slm_run
bash scripts/runpod_deploy_and_train.sh
```
This single command installs dependencies, runs preflight checks, and executes the entire 13-stage training/evaluation pipeline automatically. Check the `run_manifests/<RUN_ID>/` folder for your scientific reports!

---
> **License:** MIT | **Base Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
