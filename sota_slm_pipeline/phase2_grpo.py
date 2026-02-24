import torch
import os
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer

# Import der sicheren Reward-Funktionen aus unserem Modul
from rewards import strict_format_reward_func, length_penalty_reward_func, execution_reward_func

# ==============================================================================
# Phase 2: Alignment via GRPO (Group Relative Policy Optimization)
# Wendet Reinforcement Learning (RL) an und nutzt vLLM für rapide Generation.
# ==============================================================================

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
    
    # Für das Execution-Reward-Alignment nutzen wir hier stellvertretend The MBPP (sanitized)
    # in der echten Pipeline würde hier ein Mix aus MBPP, HumanEval+ und Leetcode Datensätzen genutzt.
    print("Loading continuous RL environment datasets (MBPP examples)...")
    eval_dataset = load_dataset("mbpp", "sanitized", split="train[:500]")
    rl_dataset = eval_dataset.map(format_rl_prompt)

    # GRPO Konfiguration
    # Achtung VRAM-Gefahr: Wir erzeugen simultan num_generations=8 Antworten. 
    # Das kostet extrem viel Speicher. Daher Batch-Size auf 1 oder 2 limitieren.
    training_args = GRPOConfig(
        output_dir = "outputs_grpo",
        learning_rate = 5e-6, # Sehr konservativ, um KL-Divergenz gering zu halten
        lr_scheduler_type = "cosine",
        logging_steps = 5,
        max_steps = 1500, # Massiv erhöht ($20 Target-Budget) für tiefergehende Kausalitäts-Exploration
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_generations = 8, # Gruppengröße: 8 divergierende Lösungswege pro Prompt
        max_prompt_length = 512,
        max_completion_length = 2048, # Maximale Länge für Reasoning + Code
        bf16 = True, # Hardware Optimierung auf RTX 4090 / A100
        optim = "adamw_8bit",
        # GRPO-spezifisch: KL-Strafe (verhindert das Abdriften des Modells in kryptische Ausgaben)
        beta = 0.05, 
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            strict_format_reward_func, # 1. Schritt: Stimmt das Format? (<reasoning> / <answer>)
            length_penalty_reward_func, # 2. Schritt: Ist es unnatürlich lang? (Spam-Penalty)
            execution_reward_func      # 3. Schritt: Läuft der Code sicher in der Sandbox?
        ],
        args = training_args,
        train_dataset = rl_dataset,
    )

    print("Commencing GRPO Reinforcement Learning Sequence...")
    trainer.train()
    
    # Finale Modell-Sicherung
    print("GRPO Alignment successfully executed.")
    model.save_pretrained("qwen_grpo_final")
    tokenizer.save_pretrained("qwen_grpo_final")
    print("Alignment weights securely stored under './qwen_grpo_final'.")

if __name__ == "__main__":
    train_grpo()
