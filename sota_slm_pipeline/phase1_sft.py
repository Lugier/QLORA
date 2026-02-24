import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_from_disk
import os

# ==============================================================================
# Phase 1: Supervised Fine-Tuning (SFT) via Unsloth LoRA
# Liefert das kognitive Warm-Up ("Format-Lernen") vor dem RLHF.
# ==============================================================================

def train_sft():
    max_seq_length = 8192
    
    # FP8 Support: Falls auf einer H100 trainiert wird, aktiviere 8-bit.
    # Auf Consumer-Karten (RTX 4090) nutzen wir bfloat16 für maximale Reasoning-Fähigkeit
    # und belassen load_in_4bit auf False.
    load_in_4bit = False 

    print("Initializing Qwen 2.5 1.5B Architecture...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-1.5B-Instruct", 
        max_seq_length = max_seq_length,
        dtype = None, # Auto-Detektion (BFloat16 für RTX 4090)
        load_in_4bit = load_in_4bit,
    )

    # PEFT / LoRA Konfiguration
    # Adapter auf allen Layern (o_proj, up_proj) für maximale Expressivität
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha = 64,
        lora_dropout = 0, # Ein Dropout von 0 ermöglicht stark optimierte Hardware-Pfade
        bias = "none",
        # Unsloth's Gradient Checkpointing spart signifikant VRAM, zwingend für 8k Tokens auf 24GB
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    dataset_path = "./sota_slm_coding_dataset"
    if not os.path.exists(dataset_path):
        print(f"Fehler: Datensatz '{dataset_path}' nicht gefunden. Bitte data_pipeline.py zuerst ausführen.")
        return

    print("Loading compiled dataset...")
    dataset = load_from_disk(dataset_path)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 100,
            max_steps = 1500, # Wir trainieren nur 1.500 Schritte als SFT-Warmstart
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bfloat16_supported(),
            bf16 = torch.cuda.is_bfloat16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit", # Reduziert den Speicherbedarf der Optimizer-States drastisch
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs_sft",
            save_strategy="steps",
            save_steps=500,
        ),
    )

    print("Commencing Supervised Fine-Tuning...")
    trainer.train()
    
    # SFT-Gewichte als Basis für die zweite Phase (GRPO) sichern.
    model.save_pretrained("qwen_sft_lora")
    tokenizer.save_pretrained("qwen_sft_lora")
    print("SFT Completed. Adapter weights securely stored under './qwen_sft_lora'")

if __name__ == "__main__":
    train_sft()
