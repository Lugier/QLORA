# 🚀 SOTA SLM Pipeline: The Architecture of Reasoning

This repository houses a **State-of-the-Art (SOTA) Small Language Model (SLM) Fine-Tuning Pipeline**, engineered in 2026 to distill high-tier algorithmic reasoning into sub-2-Billion parameter models.

Leveraging the architectural paradigms of **OpenAI o1** and **DeepSeek-R1**, this codebase transitions classical instruction-tuning toward autonomous, self-correcting agentic behavior. Designed specifically for execution on a $20 compute budget (e.g., standard RTX 4090 environments), it sets a new baseline for local, high-performance coding SLMs.

---

## 🔬 Scientific Foundation & Paradigms

The prevailing dogma in scaling laws dictates that massive parameter counts are requisite for deep algorithmic capability. This pipeline challenges that assertion by prioritizing **data density**, **multi-turn tool trajectories**, and **hardware-verifiable execution environments (Dense Rewards)** over raw parameter scale.

We employ a dual-phase training strategy:
1.  **Phase 1 (SFT + Distillation):** Imitation learning via High-Quality Reasoning Traces.
2.  **Phase 2 (GRPO - Reinforcement Learning):** Self-Alignment through Execution Feedback.

---

## 🏗 System Architecture

The pipeline is modularly designed, separating data curation, secure execution, reward formulation, and sequential tuning phases.

### 1. Data Curation & Distillation (`data_pipeline.py`)

A model is only as intelligent as its training corpus. We synthesize a highly diversified dataset:

*   **`nvidia/OpenCodeReasoning` (50k):** Injects detailed "Teacher-Rationales" forcing the model to explicitly delineate its thought process within `<reasoning>` tags before generating `<answer>` blocks.
*   **`WizardLM/WizardLM_evol_instruct_V2` (25k):** Scales structural syntax and semantic depth via complex, iteratively-evolved programming challenges.
*   **`princeton-nlp/SWE-bench_Lite` (15k):** Grounds the model in repository-level bug fixing, teaching it to isolate issues from problem statements.
*   **📡 The Multi-Turn Breakthrough: `SWE-agent-trajectories` (5k)**
    *   Moves the model from "Zero-Shot Guesser" to "Agentic Developer".
    *   The model learns to navigate codebases iteratively (e.g., `grep_search`, `view_file`) before writing the final patch.

### 2. Phase 1b: Best-of-N Rejection Sampling (`phase1b_rejection_sampling.py`)

GRPO algorithms frequently suffer from "Reward Collapse" when applied to small models with weak base logic.

*   **The R1 / Distillation Trick:** Before true Reinforcement Learning begins, the model generates $N=16$ divergent solutions per prompt.
*   These generations are evaluated in a physical sandbox.
*   Only the 100% flawless trajectories (perfect logic + executing code) are retained to construct an ultra-clean "Distillation Dataset", ensuring a solid foundation for RL phase.

### 3. The Hardware Validator: Secure AST Sandbox (`sandbox.py`)

To prevent the SLM from learning "Reward Hacking" (e.g., deleting test files or injecting malicious payloads to artificially boost scores), a rigorous security layer is implemented:
*   **Abstract Syntax Tree (AST) Parsing:** Pre-execution verification blocks restricted built-ins (e.g., `eval`, `exec`) and lethal os-level modules.
*   **Isolation:** Code is executed in a fortified `subprocess` with draconian timeouts.

### 4. Dense Rewards Shaping (`rewards.py`)

Sparse rewards (Pass/Fail) provide insufficient gradient density for 1.5B parameter models. We employ **Dense Reward Shaping** based on computational complexity:

| Outcome | Reward Score | Scientific Rationale |
| :--- | :--- | :--- |
| **Flawless $O(1)/O(n)$** | `+2.5` | Perfect logic coupled with optimal algorithmic execution time ($t < 0.05s$). |
| **Operational $O(n^2)$**| `+1.5` | Logic passes asserts, but execution is inefficient ($t > 0.5s$). |
| **Logic Failure** | `+0.5` | Code parses and runs, but violates algorithmic assertions. |
| **Runtime Crash** | `+0.1` | Code is syntaktically parsed but crashes. |
| **Syntax Error** | `0.0` | Malformed Python structure; unreadable AST. |
| **Timeout / Infinite Loop** | `-0.5` | Destructive computational inefficiency. |
| **Reward Hacking** | `-1.0` | Malicious module usage detected via AST. |
| **Format Violation** | `-1.5` | Failure to respect `<reasoning>` and `<answer>` XML constraints. |

### 5. Training Engine (`phase1_sft.py` & `phase2_grpo.py`)

*   **Unsloth Framework:** Enables rapid iteration by quantizing computations and leveraging FlashAttention-2, allowing advanced SFT and GRPO tuning directly on consumer VRAM (RTX 4090).
*   **vLLM Integration:** During GRPO (`PatchFastRL`), vLLM blitzes through the generation of 8 simultaneous response candidates for maximal comparative policy optimization.
*   **Cost Optimization:** The hyperparameters are strictly tuned to maximize validation gains while adhering to a strict $20 RunPod inference budget ceiling (approx. 25-28 hours computation).

---

## 🛫 Quick Start / Execution Flow

Assuming a fresh GPU environment with Python 3.10+:

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Build the Multi-Turn Dataset
python data_pipeline.py

# 3. Supervised Fine-Tuning (SFT) - The Immitation Phase
python phase1_sft.py

# 4. Best-of-N Rejection Sampling - The Clean Foundation
python phase1b_rejection_sampling.py

# 5. Group Relative Policy Optimization (GRPO) - The Intelligence Phase
python phase2_grpo.py

# 6. Export and Deploy (GGUF / HuggingFace Merge)
python export.py
```

---

## 📈 Future Research Vectors

While this pipeline scales the current baseline dramatically, researchers can explore:
1.  **Process Reward Models (PRMs):** Implementing Monte Carlo Tree Search (MCTS) evaluations at the *token level* to reward intermediate logical backtracking.
2.  **Inference-Time Scaling:** Allowing the model $N \rightarrow \infty$ generation budgets during deployment to iteratively brute-force complex MBPP benchmarks prior to returning the final output.
