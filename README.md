# SOTA SLM Pipeline (RunPod-Ready, Fail-Fast)

End-to-end Pipeline fuer ein kleines Coding-LLM auf Basis von `Qwen/Qwen2.5-1.5B-Instruct`:

1. Datenaufbereitung mit harten Qualitaets-Gates
2. SFT (LoRA)
3. Best-of-N Distillation
4. DPO (Preference Alignment)
5. GRPO (Execution-rewarded RL)
6. Eval (Classic + Agentic Self-Debug)
7. Export (GGUF + merged HF)

Alle Stages sind fail-fast ausgelegt (kein stilles Weiterlaufen mit leeren Artefakten).
Die Setup-Stage prueft explizit `Unsloth`, `TRL` und `GRPOTrainer`, damit spaetere RL-Stages nicht erst zur Laufzeit brechen.

## Steps 1-7 (Produktiv eingebaut)

Diese sieben Punkte sind im Code und in den RunPod-Skripten bereits integriert:

1. `pass@k` statt nur `pass@1` in der Classic-Eval (`--pass-k` Default `8`).
2. Strengere Hidden-Test-Qualitaet vor Distillation/RL (`min_asserts`, `min_lines`, `min_quality_score`).
3. DPO-Paare mit Score-Gap-Filter (`min_score_gap`, `max_rejected_score`, Assert-Mindestanzahl).
4. 3-Stage Curriculum in GRPO (`easy -> mid -> hard`) mit robuster Step-Allokation.
5. Retrieval im Agenten mit BM25-Scoring plus Pfad-/Symbol-Heuristiken.
6. Self-Debug mit Fehlerklassen-Policy und Early-Stop (`early_stop_patience`).
7. Multi-Benchmark-Eval (`mbpp` + `humaneval`) fuer realistischere Modellmessung.

## Projektstruktur

- `data_pipeline.py`: Multi-Source Dataset Build, Dedup, Quality Gates
- `phase1_sft.py`: SFT Training
- `phase1b_rejection_sampling.py`: Best-of-N + DPO Pair Mining
- `phase1c_dpo.py`: DPO Training
- `phase2_grpo.py`: GRPO Training
- `runtime_agent.py`: Retrieval + iterative Self-Debug Loop
- `eval_pipeline.py`: Evaluation (Classic oder Agentic)
- `export.py`: Deployment-Artefakte
- `scripts/runpod_setup.sh`: RunPod-Setup (CUDA Torch + Abhaengigkeiten)
- `scripts/runpod_train_full.sh`: Vollstaendige Trainings-Orchestrierung

## RunPod Quickstart (empfohlen)

### 1. GPU-Instanz

- Ubuntu + NVIDIA GPU
- Empfohlen: mindestens 24 GB VRAM
- Python 3.10

### 2. Setup

```bash
cd /workspace/sota_slm_pipeline
bash scripts/runpod_setup.sh
```

Optional fuer gated HuggingFace-Datasets:

```bash
export HF_TOKEN=hf_xxx
```

### 3. Voller Lauf

```bash
bash scripts/runpod_train_full.sh
```

Das Skript stoppt sofort bei Fehlern und prueft nach jeder Stage, ob die erwarteten Artefakte vorhanden sind.

## Manueller Ablauf (falls du pro Stage steuern willst)

```bash
# 1) Dataset Build (harter Quality Gate)
python3 data_pipeline.py \
  --min-total-samples 20000 \
  --min-test-coverage 0.08 \
  --min-prompt-coverage 0.99 \
  --min-unique-ratio 0.75 \
  --max-missing-sources 6

# 2) SFT
python3 phase1_sft.py \
  --dataset-path ./sota_slm_coding_dataset \
  --adapter-dir qwen_sft_lora \
  --merged-model-dir qwen_sft_merged \
  --output-dir outputs_sft \
  --max-seq-length 8192 \
  --max-steps 3000

# 3) Best-of-N + DPO pairs
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

# 4) DPO
python3 phase1c_dpo.py \
  --dpo-dataset-path ./sota_dpo_pairs_dataset \
  --sft-adapter-path qwen_sft_lora \
  --output-dir outputs_dpo \
  --output-model-dir qwen_dpo_lora \
  --max-steps 800 \
  --min-score-gap 1.0 \
  --max-rejected-score 0.6 \
  --min-test-assert-count 2

# 5) GRPO (strict, ohne MBPP-Fallback)
python3 phase2_grpo.py \
  --max-seq-length 8192 \
  --max-steps 1500 \
  --min-test-asserts 2 \
  --min-test-lines 3 \
  --min-test-quality-score 2.5 \
  --output-model-dir qwen_grpo_final \
  --output-adapter-dir qwen_grpo_lora \
  --dataset-paths ./sota_best_of_n_dataset,./sota_slm_coding_dataset

# 6) Eval classic (pass@k + multi-benchmark)
python3 eval_pipeline.py --model-path qwen_grpo_final --benchmarks mbpp,humaneval --num-samples 100 --pass-k 8 --verifier-rounds 2

# 7) Eval agentic (self-debug + retrieval + verifier)
python3 eval_pipeline.py --model-path qwen_grpo_final --benchmarks mbpp,humaneval --agentic --num-samples 100 --max-rounds 3 --n-candidates 8 --verifier-rounds 2

# 8) Export
python3 export.py
```

## Wichtige Outputs

- `sota_slm_coding_dataset`
- `qwen_sft_lora`
- `qwen_sft_merged`
- `sota_best_of_n_dataset`
- `sota_dpo_pairs_dataset`
- `qwen_dpo_lora`
- `qwen_grpo_final`
- `qwen_grpo_lora`
- `model_export_gguf`
- `model_export_hf`

## Design-Entscheidungen fuer Stabilitaet

1. Keine Dummy-/Placeholder-Zeilen in der Datenformatierung.
2. Harte Mindestschwellen in Data-Build und Distillation.
3. Stage-Skripte werfen Fehler statt still `return`.
4. GRPO faellt standardmaessig nicht automatisch auf MBPP zurueck.
5. Agentic Eval nutzt echte Test-Feedback-Schleifen statt nur Single-Shot Generation.

## Flow (was passiert wann und warum)

1. `data_pipeline.py` baut den Trainingskorpus und bricht bei schlechter Datenqualitaet hart ab.
2. `phase1_sft.py` gibt dem Basismodell die Coding-Struktur (Instruction-Following + XML-Ausgabeformat).
3. `phase1b_rejection_sampling.py` generiert mehrere Kandidaten pro Prompt, laesst sie gegen Hidden Tests laufen und behaelt nur starke Trajektorien; daraus werden auch DPO-Paare erzeugt.
4. `phase1c_dpo.py` richtet das Modell mit sauberen `chosen > rejected` Praeferenzpaaren feiner aus.
5. `phase2_grpo.py` optimiert mit Execution-Rewards in einem Curriculum (leichte bis schwere Aufgaben), damit sich Robustheit statt nur Stil verbessert.
6. `eval_pipeline.py` misst klassisch (`pass@1`, `pass@k`) und agentisch (Self-Debug) auf mehreren Benchmarks.
7. `export.py` erstellt finale Deployment-Artefakte fuer Inferenz (GGUF/HF merged).

## Troubleshooting

1. `CUDA is required...`
   - Auf CPU-Instanz gestartet oder Torch ohne CUDA installiert.
   - Loesung: `bash scripts/runpod_setup.sh` auf GPU-RunPod.

2. `Too few DPO pairs`
   - SFT-Checkpoint noch zu schwach oder Sampling zu konservativ.
   - Loesung: `--num-prompts`/`--num-samples-per-prompt` erhoehen.

3. Datenquelle nicht ladbar
   - HuggingFace Auth oder Netzwerkproblem.
   - Loesung: `HF_TOKEN` setzen und Zugriff auf die Datasets pruefen.
