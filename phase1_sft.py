import argparse
import inspect
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

def train_sft(
    dataset_path="./sota_slm_coding_dataset",
    output_dir="outputs_sft",
    adapter_dir="qwen_sft_lora",
    merged_model_dir="qwen_sft_merged",
    max_seq_length=8192,
    max_steps=3000,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFT training on this pipeline.")
    
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

    if not os.path.exists(dataset_path):
        raise RuntimeError(f"Datensatz '{dataset_path}' nicht gefunden. Bitte data_pipeline.py zuerst ausführen.")

    print("Loading compiled dataset...")
    dataset = load_from_disk(dataset_path)
    if len(dataset) == 0:
        raise RuntimeError(f"Datensatz '{dataset_path}' ist leer.")
    dataset_num_proc = max(1, min(8, os.cpu_count() or 1))

    training_args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        max_steps = max_steps, # Skaliert auf 3.000 Schritte (~5 Std auf 4090) für Max-Gain innerhalb 20$ Budget
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bfloat16_supported(),
        bf16 = torch.cuda.is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # Reduziert den Speicherbedarf der Optimizer-States drastisch
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = output_dir,
        save_strategy="steps",
        save_steps=500,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in trainer_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_sig.parameters:
        trainer_kwargs["max_seq_length"] = max_seq_length
    if "dataset_num_proc" in trainer_sig.parameters:
        trainer_kwargs["dataset_num_proc"] = dataset_num_proc

    trainer = SFTTrainer(**trainer_kwargs)

    print("Commencing Supervised Fine-Tuning...")
    trainer.train()
    
    # Adapter für weitere LoRA-Phasen sichern.
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"SFT Completed. Adapter weights securely stored under '{adapter_dir}'")

    # Merged HF checkpoint für vLLM-Rejection-Sampling erzeugen.
    print(f"Exporting merged SFT checkpoint for vLLM to '{merged_model_dir}'...")
    model.save_pretrained_merged(merged_model_dir, tokenizer, save_method="merged_16bit")
    print(f"SFT merged checkpoint stored under '{merged_model_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT stage.")
    parser.add_argument("--dataset-path", default="./sota_slm_coding_dataset")
    parser.add_argument("--output-dir", default="outputs_sft")
    parser.add_argument("--adapter-dir", default="qwen_sft_lora")
    parser.add_argument("--merged-model-dir", default="qwen_sft_merged")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()

    train_sft(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        merged_model_dir=args.merged_model_dir,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
    )
