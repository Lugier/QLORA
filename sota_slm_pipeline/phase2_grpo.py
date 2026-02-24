import torch
import os
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, TrainerCallback
import copy
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer

# Import der sicheren Reward-Funktionen aus unserem Modul
from rewards import strict_format_reward_func, length_penalty_reward_func, execution_reward_func, self_verification_reward_func

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
        
        # Aktivieren erst in den letzten 30% des Trainings (Late-Game Stabilization)
        self.start_ema_step = int(max_steps * 0.7)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.start_ema_step:
            return
            
        with torch.no_grad():
            if self.ema_model_weights is None:
                # Initialisiere EMA Gewichte beim ersten gültigen Step
                self.ema_model_weights = {
                    k: v.clone().detach() 
                    for k, v in self.model.state_dict().items() if v.requires_grad
                }
            else:
                # Fließender Durchschnitts-Update
                state_dict = self.model.state_dict()
                for k in self.ema_model_weights.keys():
                    self.ema_model_weights[k].mul_(self.decay).add_(
                        state_dict[k].detach(), alpha=1 - self.decay
                    )

    def on_train_end(self, args, state, control, **kwargs):
        # Beim Trainingsende laden wir die robusten EMA-Gewichte zurück ins aktive Modell
        if self.ema_model_weights is not None:
            print("\nApplying Exponential Moving Average (EMA) weights for robust production checkpoint...")
            self.model.load_state_dict(self.ema_model_weights, strict=False)

def format_rl_prompt(example):
    """
    Formatiert die Prompts für die RL-Evaluation. 
    Hier zwingen wir das Modell in den Reasoning-Mode für die tiefgreifende Aufgabenlösung.
    """
    system_prompt = (
        "You are an expert python developer. Analyze thoroughly in <reasoning> tags, "
        "then output purely the executable code in <answer> tags."
    )
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{example['text']}<|im_end|>\n<|im_start|>assistant\n"
    
    # MBPP enthält die Unittests im Array 'test_list', welche wir verdeckt als `answer` übergeben.
    tests = "\n".join(example.get('test_list', []))
    
    return {"prompt": prompt, "answer": tests}


def train_grpo():
    # Aktivierung des Unsloth FastRL Patches. 
    # vLLM übernimmt in GRPO die Erzeugung der 8 divergierenden Lösungswege (num_generations=8)
    PatchFastRL("GRPO", FastLanguageModel)

    max_seq_length = 8192
    
    # Das Basis-Modell laden
    print("Loading baseline model for GRPO alignment...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = False,
        fast_inference = True, # Essentiell zur Aktivierung des vLLM Backends
    )

    # Injizieren der in Phase 1 gelernten SFT-Strukturen (Format, Reasoning)
    sft_adapter_path = "qwen_sft_lora"
    if not os.path.exists(sft_adapter_path):
         print(f"WARNUNG: Phase 1 SFT Adapter '{sft_adapter_path}' nicht gefunden. Starte reines GRPO Pipeline.")
    else:
         print(f"Loading learned SFT structural formats from '{sft_adapter_path}'...")
         model.load_adapter(sft_adapter_path)
    
    # Für das Execution-Reward-Alignment nutzen wir    # Datensatz laden und formattieren
    dataset_path = "./sota_best_of_n_dataset" # Basiert auf Destillations-Phase
    if not os.path.exists(dataset_path):
        dataset_path = "./sota_slm_coding_dataset" # Fallback auf SFT Dataset
        
    raw_dataset = load_from_disk(dataset_path)
    # Entfernt evtl. leere Instruktionen/Lösungen für RL
    raw_dataset = raw_dataset.filter(lambda x: len(x.get('text', '')) > 50) 
    
    rl_dataset = raw_dataset.map(format_rl_prompt, remove_columns=raw_dataset.column_names)
    
    # === Curriculum Learning (SOTA Optimization) ===
    # Wir sortieren den Datensatz aufsteigend nach der Länge (Komplexitäts-Proxy) des Prompts.
    # Dadurch sieht das Modell in den frühen GRPO-Schritten leichtere Aufgaben und stabilisiert
    # seine Reward-Map, bevor es an harte SWE-Probleme geht.
    print("Applying Curriculum Learning: Sorting dataset by complexity (prompt length)...")
    rl_dataset = rl_dataset.map(lambda x: {"prompt_length": len(x["prompt"])})
    rl_dataset = rl_dataset.sort("prompt_length")
    
    # 3. GRPOTrainer Configuration für Max-Reasoning VRAM-Gefahr: Wir erzeugen simultan num_generations=8 Antworten. 
    # Das kostet extrem viel Speicher. Daher Batch-Size auf 1 oder 2 limitieren.
    training_args = GRPOConfig(
        weight_decay = 0.1,
        learning_rate = 5e-6, # Sehr konservativ, um KL-Divergenz gering zu halten
        lr_scheduler_type = "cosine",
        logging_steps = 5,
        max_steps = 1500, # Massiv erhöht ($20 Target-Budget) für tiefergehende Kausalitäts-Exploration
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_generations = 8, # Gruppengröße: 8 divergierende Lösungswege pro Prompt
        max_prompt_length = 512,
        max_completion_length = 1500, # Bietet Platz für Multi-Turn Reflexion im Antwort-Token-Space
        fp16 = not torch.cuda.is_bfloat16_supported(),
        bf16 = True, # Hardware Optimierung auf RTX 4090 / A100
        optim = "adamw_8bit",
        # GRPO-spezifisch: KL-Strafe (verhindert das Abdriften des Modells in kryptische Ausgaben)
        beta = 0.05, 
    )

    # 4. GRPOTrainer initialisieren (Mit AERO Rewards, Curriculum und EMA Callback)
    trainer = GRPOTrainer(
        model=model,
        processing_class = tokenizer,
        reward_funcs=[
            strict_format_reward_func, 
            length_penalty_reward_func, 
            self_verification_reward_func, # TDD-Proactive Reward
            execution_reward_func
        ],
        args=training_args,
        train_dataset=rl_dataset,
        callbacks=[EMACheckpointCallback(model, max_steps=training_args.max_steps)]
    )

    print("Initiating Multi-Turn Optimized GRPO Deep-Reasoning Simulation...")
    trainer.train()
    
    # Finale Modell-Sicherung
    print("GRPO Alignment successfully executed.")
    model.save_pretrained("qwen_grpo_final")
    tokenizer.save_pretrained("qwen_grpo_final")
    print("Alignment weights securely stored under './qwen_grpo_final'.")

if __name__ == "__main__":
    train_grpo()
