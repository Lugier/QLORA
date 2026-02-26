import argparse
import inspect
import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

# ==============================================================================
# Phase 1: Supervised Fine-Tuning (SFT) via Unsloth LoRA
# Liefert das kognitive Warm-Up ("Format-Lernen") vor dem RLHF.
# ==============================================================================


def _latest_checkpoint(output_dir):
    # Spot/preemptible resume helper: pick the newest Trainer checkpoint-* directory.
    if not output_dir or not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("checkpoint-")[-1])
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def train_sft(
    dataset_path="./sota_slm_coding_dataset",
    extra_dataset_paths="",
    output_dir="outputs_sft",
    adapter_dir="qwen_sft_lora",
    merged_model_dir="qwen_sft_merged",
    base_model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length=8192,
    max_steps=3000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lora_r=48,
    lora_alpha=96,
    lora_dropout=0.03,
    eval_every_steps=250,
    checkpoint_every_steps=250,
    resume_from_checkpoint="auto",
    seed=3407,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SFT training on this pipeline.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # FP8 Support: Falls auf einer H100 trainiert wird, aktiviere 8-bit.
    # Auf Consumer-Karten (RTX 4090) nutzen wir bfloat16 für maximale Reasoning-Fähigkeit
    # und belassen load_in_4bit auf False.
    load_in_4bit = False 

    print(f"Initializing base model architecture: {base_model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name,
        max_seq_length = max_seq_length,
        dtype = None, # Auto-Detektion (BFloat16 für RTX 4090)
        load_in_4bit = load_in_4bit,
    )

    # PEFT / LoRA Konfiguration
    # Adapter auf allen Layern (o_proj, up_proj) für maximale Expressivität
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        # Unsloth's Gradient Checkpointing spart signifikant VRAM, zwingend für 8k Tokens auf 24GB
        use_gradient_checkpointing = "unsloth", 
        random_state = seed,
    )

    all_paths = [dataset_path] + [p.strip() for p in str(extra_dataset_paths).split(",") if p.strip()]
    all_paths = [p for p in all_paths if p]
    for ds_path in all_paths:
        if not os.path.exists(ds_path):
            raise RuntimeError(f"Datensatz '{ds_path}' nicht gefunden. Bitte data_pipeline.py zuerst ausführen.")

    print(f"Loading compiled datasets: {all_paths}")
    train_splits = []
    eval_splits = []
    for ds_path in all_paths:
        loaded = load_from_disk(ds_path)
        if isinstance(loaded, DatasetDict):
            if "train" not in loaded:
                raise RuntimeError(f"DatasetDict in '{ds_path}' has no 'train' split.")
            train_splits.append(loaded["train"])
            if "val_strict" in loaded and len(loaded["val_strict"]) > 0:
                eval_splits.append(loaded["val_strict"])
        else:
            train_splits.append(loaded)

    if not train_splits:
        raise RuntimeError("No training splits were loaded for SFT.")
    dataset: Dataset = train_splits[0] if len(train_splits) == 1 else concatenate_datasets(train_splits).shuffle(seed=seed)
    eval_dataset = None
    if eval_splits:
        eval_dataset = eval_splits[0] if len(eval_splits) == 1 else concatenate_datasets(eval_splits).shuffle(seed=seed)

    if len(dataset) == 0:
        raise RuntimeError(f"Datensatz-Mix ist leer. Pfade: {all_paths}")
    dataset_num_proc = max(1, min(8, os.cpu_count() or 1))

    use_eval = eval_dataset is not None and len(eval_dataset) > 0 and eval_every_steps > 0
    training_args_kwargs = {
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": 100,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "fp16": not torch.cuda.is_bfloat16_supported(),
        "bf16": torch.cuda.is_bfloat16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "max_grad_norm": 0.3,
        "lr_scheduler_type": "cosine",
        "seed": seed,
        "output_dir": output_dir,
        "save_strategy": "steps",
        "save_steps": checkpoint_every_steps,
        "save_total_limit": 3,
        "report_to": [],
    }
    if use_eval:
        training_args_kwargs.update(
            {
                "evaluation_strategy": "steps",
                "eval_steps": eval_every_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    training_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    if use_eval:
        trainer_kwargs["eval_dataset"] = eval_dataset
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
    resume_path = None
    if resume_from_checkpoint and str(resume_from_checkpoint).lower() != "none":
        if str(resume_from_checkpoint).lower() == "auto":
            # "auto" means: continue from latest local checkpoint if present.
            resume_path = _latest_checkpoint(output_dir)
        else:
            resume_path = resume_from_checkpoint
    if resume_path:
        print(f"Resuming SFT from checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)
    
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
    parser.add_argument("--extra-dataset-paths", default="")
    parser.add_argument("--output-dir", default="outputs_sft")
    parser.add_argument("--adapter-dir", default="qwen_sft_lora")
    parser.add_argument("--merged-model-dir", default="qwen_sft_merged")
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=48)
    parser.add_argument("--lora-alpha", type=int, default=96)
    parser.add_argument("--lora-dropout", type=float, default=0.03)
    parser.add_argument("--eval-every-steps", type=int, default=250)
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    train_sft(
        dataset_path=args.dataset_path,
        extra_dataset_paths=args.extra_dataset_paths,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        merged_model_dir=args.merged_model_dir,
        base_model_name=args.base_model_name,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        eval_every_steps=args.eval_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
    )
