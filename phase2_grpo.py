import torch
import os
import argparse
import inspect
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import TrainerCallback
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
import rewards as rewards_module

# Import der sicheren Reward-Funktionen aus unserem Modul
from rewards import strict_format_reward_func, length_penalty_reward_func, execution_reward_func, self_verification_reward_func
from verification import assess_test_quality, is_test_quality_sufficient

# 1. Wir überschreiben vLLM Flags, um FP8 KV-Cache zu nutzen
# Dies spart massiv VRAM und erlaubt uns, Multi-Turn Chats im RL zu simulieren
os.environ["VLLM_KV_CACHE_DTYPE"] = "fp8"

# ==============================================================================
# Phase 2: GRPO Reinforcement Learning (The Intelligence Phase)
# ==============================================================================

class EMACheckpointCallback(TrainerCallback):
    """
    Production-Grade EMA (Exponential Moving Average) Mechanismus.
    Glättet die Rewards im Late-Game, indem Gewichte über die Zeit als 
    Average gespeichert werden, was vor catastrophic forgetting im RL schützt.
    """
    def __init__(self, model, max_steps, decay=0.999):
        self.ema_model_weights = None
        self.model = model
        self.decay = decay
        self.trainable_param_names = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        
        # Aktivieren erst in den letzten 30% des Trainings (Late-Game Stabilization)
        self.start_ema_step = int(max_steps * 0.7)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.start_ema_step:
            return
            
        with torch.no_grad():
            if self.ema_model_weights is None:
                # Initialisiere EMA Gewichte beim ersten gültigen Step
                state_dict = self.model.state_dict()
                self.ema_model_weights = {
                    name: state_dict[name].clone().detach()
                    for name in self.trainable_param_names
                    if name in state_dict
                }
            else:
                # Fließender Durchschnitts-Update
                state_dict = self.model.state_dict()
                for name in self.ema_model_weights.keys():
                    self.ema_model_weights[name].mul_(self.decay).add_(
                        state_dict[name].detach(), alpha=1 - self.decay
                    )

    def on_train_end(self, args, state, control, **kwargs):
        # Beim Trainingsende laden wir die robusten EMA-Gewichte zurück ins aktive Modell
        if self.ema_model_weights is not None:
            print("\nApplying Exponential Moving Average (EMA) weights for robust production checkpoint...")
            self.model.load_state_dict(self.ema_model_weights, strict=False)

def format_rl_prompt(
    example,
    min_test_asserts=2,
    min_test_lines=3,
    min_test_quality_score=2.5,
):
    """
    Formatiert die Prompts für die RL-Evaluation. 
    Hier zwingen wir das Modell in den Reasoning-Mode für die tiefgreifende Aufgabenlösung.
    """
    system_prompt = (
        "You are an expert python developer. Analyze thoroughly in <reasoning> tags, "
        "then output purely the executable code in <answer> tags."
    )
    user_prompt = example.get("prompt", "") or example.get("text", "")
    # Distillation records already carry a full chat prompt up to assistant turn.
    if "<|im_start|>assistant" in user_prompt and "<|im_start|>system" in user_prompt:
        prompt = user_prompt
    else:
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    tests = example.get("tests", "")
    if not tests:
        test_list = example.get("test_list", [])
        if isinstance(test_list, list):
            tests = "\n".join([t for t in test_list if t])
        else:
            tests = str(test_list or "")

    tests = tests.strip()
    quality = assess_test_quality(tests)
    tests_quality_ok = is_test_quality_sufficient(
        tests,
        min_asserts=min_test_asserts,
        min_nonempty_lines=min_test_lines,
        min_quality_score=min_test_quality_score,
    )
    return {
        "prompt": prompt,
        "answer": tests,
        "has_tests": bool(tests),
        "tests_quality_ok": tests_quality_ok,
        "test_quality_score": quality["quality_score"],
        "test_assert_count": quality["assert_count"],
    }


def _load_distillation_dataset(candidate_paths):
    loaded = []
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        ds = load_from_disk(path)
        ds = ds.filter(
            lambda x: len((x.get("prompt", "") or x.get("text", "")).strip()) > 0
            and len((x.get("tests", "") or "").strip()) > 0
        )
        if len(ds) > 0:
            loaded.append(ds)

    if not loaded:
        return None
    if len(loaded) == 1:
        return loaded[0]
    return concatenate_datasets(loaded)


def _split_curriculum_stages(rl_dataset):
    if len(rl_dataset) == 0:
        return []

    lengths = sorted(rl_dataset["prompt_length"])
    q1 = lengths[len(lengths) // 3]
    q2 = lengths[(2 * len(lengths)) // 3]

    easy = rl_dataset.filter(lambda x: x["prompt_length"] <= q1)
    mid = rl_dataset.filter(lambda x: q1 < x["prompt_length"] <= q2)
    hard = rl_dataset.filter(lambda x: x["prompt_length"] > q2)

    stages = [
        ("easy", easy, 0.25),
        ("mid", mid, 0.35),
        ("hard", hard, 0.40),
    ]
    return [(name, ds, ratio) for name, ds, ratio in stages if len(ds) > 0]


def _build_training_args(stage_steps):
    return GRPOConfig(
        weight_decay=0.1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=5,
        max_steps=stage_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=512,
        max_completion_length=1500,
        fp16=not torch.cuda.is_bfloat16_supported(),
        bf16=torch.cuda.is_bfloat16_supported(),
        optim="adamw_8bit",
        beta=0.05,
    )


def _allocate_stage_steps(stages, total_steps, min_stage_steps=30):
    # Allocates curriculum steps deterministically so total steps remain exactly bounded.
    total_steps = max(1, int(total_steps))
    num_stages = len(stages)
    if num_stages == 0:
        return []

    ratios = [max(0.0, float(ratio)) for _, _, ratio in stages]
    ratio_sum = sum(ratios) if sum(ratios) > 0 else float(num_stages)

    if total_steps < num_stages:
        order = sorted(range(num_stages), key=lambda i: ratios[i], reverse=True)
        alloc = [0] * num_stages
        for idx in range(total_steps):
            alloc[order[idx]] = 1
        return alloc

    effective_min = min_stage_steps if total_steps >= (min_stage_steps * num_stages) else 1
    alloc = [effective_min] * num_stages
    remaining = total_steps - sum(alloc)
    if remaining <= 0:
        return alloc

    raw = [remaining * (ratio / ratio_sum) for ratio in ratios]
    extra = [int(x) for x in raw]
    for idx, steps in enumerate(extra):
        alloc[idx] += steps

    leftover = remaining - sum(extra)
    if leftover > 0:
        order = sorted(range(num_stages), key=lambda i: raw[i] - extra[i], reverse=True)
        for idx in range(leftover):
            alloc[order[idx % num_stages]] += 1
    return alloc


def train_grpo(
    max_seq_length=8192,
    allow_mbpp_fallback=False,
    candidate_dataset_paths=None,
    max_steps=1500,
    min_test_asserts=2,
    min_test_lines=3,
    min_test_quality_score=2.5,
    output_model_dir="qwen_grpo_final",
    output_adapter_dir="qwen_grpo_lora",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GRPO training on this pipeline.")

    if candidate_dataset_paths is None:
        candidate_dataset_paths = [
            "./sota_best_of_n_dataset",
            "./sota_slm_coding_dataset",
        ]

    # Aktivierung des Unsloth FastRL Patches. 
    # vLLM übernimmt in GRPO die Erzeugung der 8 divergierenden Lösungswege (num_generations=8)
    PatchFastRL("GRPO", FastLanguageModel)
    rewards_module.AERO_GLOBAL_STEP = 0
    rewards_module.AERO_MAX_STEPS = max_steps

    # Das Basis-Modell laden
    print("Loading baseline model for GRPO alignment...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = False,
        fast_inference = True, # Essentiell zur Aktivierung des vLLM Backends
    )

    # Injizieren der in Phase 1 gelernten Strukturen (SFT/DPO), wenn vorhanden.
    preferred_adapter_paths = ["qwen_dpo_lora", "qwen_sft_lora"]
    loaded_adapter = None
    for adapter_path in preferred_adapter_paths:
        if os.path.exists(adapter_path):
            print(f"Loading structural adapter from '{adapter_path}'...")
            model.load_adapter(adapter_path)
            loaded_adapter = adapter_path
            break

    if loaded_adapter is None:
        print("WARNUNG: Kein SFT/DPO Adapter gefunden. Starte reines GRPO auf dem Basismodell.")
    
    # Bevorzugt den Distillation-Flow: Best-of-N / SFT-Datasets mit versteckten Tests.
    raw_dataset = _load_distillation_dataset(candidate_dataset_paths)
    if raw_dataset is not None and len(raw_dataset) > 0:
        print(f"Loaded {len(raw_dataset)} RL samples from distillation pipeline datasets.")
    else:
        if allow_mbpp_fallback:
            print("WARNUNG: Kein distillation-tauglicher RL Datensatz mit Tests gefunden. Fallback auf MBPP.")
            raw_dataset = load_dataset("mbpp", "sanitized", split="train[:800]")
        else:
            raise RuntimeError(
                "Kein distillation-tauglicher RL Datensatz mit Tests gefunden. "
                "Starte zuerst data_pipeline.py + phase1_sft.py + phase1b_rejection_sampling.py."
            )
    
    rl_dataset = raw_dataset.map(
        lambda x: format_rl_prompt(
            x,
            min_test_asserts=min_test_asserts,
            min_test_lines=min_test_lines,
            min_test_quality_score=min_test_quality_score,
        ),
        remove_columns=raw_dataset.column_names,
    )
    before_quality_filter = len(rl_dataset)
    rl_dataset = rl_dataset.filter(lambda x: x["has_tests"] and x["tests_quality_ok"])
    print(f"Prepared {len(rl_dataset)} RL samples with hidden execution tests.")
    print(
        f"Quality filter removed {before_quality_filter - len(rl_dataset)} samples "
        f"(min_asserts={min_test_asserts}, min_lines={min_test_lines}, "
        f"min_quality_score={min_test_quality_score})."
    )
    if len(rl_dataset) == 0:
        raise RuntimeError("No RL samples with tests available after filtering.")
    
    # === Curriculum Learning (SOTA Optimization) ===
    # Wir sortieren den Datensatz aufsteigend nach der Länge (Komplexitäts-Proxy) des Prompts.
    # Dadurch sieht das Modell in den frühen GRPO-Schritten leichtere Aufgaben und stabilisiert
    # seine Reward-Map, bevor es an harte SWE-Probleme geht.
    print("Applying Curriculum Learning: Sorting dataset by complexity (prompt length)...")
    rl_dataset = rl_dataset.map(lambda x: {"prompt_length": len(x["prompt"])})
    rl_dataset = rl_dataset.sort("prompt_length")
    
    stages = _split_curriculum_stages(rl_dataset)
    if not stages:
        raise RuntimeError("Curriculum split produced no stages.")

    stage_allocations = _allocate_stage_steps(
        stages=stages,
        total_steps=max_steps,
        min_stage_steps=30,
    )
    stage_plan = [
        (name, stage_ds, stage_steps)
        for (name, stage_ds, _ratio), stage_steps in zip(stages, stage_allocations)
        if stage_steps > 0
    ]
    if not stage_plan:
        raise RuntimeError("Curriculum allocation produced no trainable stage.")

    print("Initiating 3-stage curriculum GRPO training...")
    for idx, (stage_name, stage_ds, stage_steps) in enumerate(stage_plan):
        if len(stage_ds) == 0 or stage_steps <= 0:
            continue
        print(
            f"[Curriculum] Stage {idx + 1}/{len(stage_plan)} "
            f"'{stage_name}' with {len(stage_ds)} samples for {stage_steps} steps."
        )
        training_args = _build_training_args(stage_steps=stage_steps)
        callbacks = [EMACheckpointCallback(model, max_steps=stage_steps)] if idx == len(stage_plan) - 1 else []
        trainer_kwargs = {
            "model": model,
            "reward_funcs": [
                strict_format_reward_func,
                length_penalty_reward_func,
                self_verification_reward_func,
                execution_reward_func,
            ],
            "args": training_args,
            "train_dataset": stage_ds,
            "callbacks": callbacks,
        }
        trainer_sig = inspect.signature(GRPOTrainer.__init__)
        if "processing_class" in trainer_sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_sig.parameters:
            trainer_kwargs["tokenizer"] = tokenizer
        else:
            raise RuntimeError("Unsupported GRPOTrainer API: neither processing_class nor tokenizer argument found.")

        trainer = GRPOTrainer(**trainer_kwargs)
        trainer.train()
    
    # Finale Modell-Sicherung
    print("GRPO Alignment successfully executed.")
    model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    print(f"GRPO adapter stored under '{output_adapter_dir}'.")

    print(f"Exporting merged GRPO checkpoint for vLLM to '{output_model_dir}'...")
    model.save_pretrained_merged(output_model_dir, tokenizer, save_method="merged_16bit")
    print(f"Alignment weights securely stored under '{output_model_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO training stage.")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--allow-mbpp-fallback", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--min-test-asserts", type=int, default=2)
    parser.add_argument("--min-test-lines", type=int, default=3)
    parser.add_argument("--min-test-quality-score", type=float, default=2.5)
    parser.add_argument("--output-model-dir", default="qwen_grpo_final")
    parser.add_argument("--output-adapter-dir", default="qwen_grpo_lora")
    parser.add_argument(
        "--dataset-paths",
        default="./sota_best_of_n_dataset,./sota_slm_coding_dataset",
        help="Comma-separated dataset paths for RL training.",
    )
    args = parser.parse_args()

    dataset_paths = [p.strip() for p in args.dataset_paths.split(",") if p.strip()]

    train_grpo(
        max_seq_length=args.max_seq_length,
        allow_mbpp_fallback=args.allow_mbpp_fallback,
        candidate_dataset_paths=dataset_paths,
        max_steps=args.max_steps,
        min_test_asserts=args.min_test_asserts,
        min_test_lines=args.min_test_lines,
        min_test_quality_score=args.min_test_quality_score,
        output_model_dir=args.output_model_dir,
        output_adapter_dir=args.output_adapter_dir,
    )
