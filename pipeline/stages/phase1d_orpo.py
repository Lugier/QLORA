import argparse
import inspect
import os

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_trl_orpo():
    import trl

    trainer_cls = getattr(trl, "ORPOTrainer", None)
    config_cls = getattr(trl, "ORPOConfig", None)
    if trainer_cls is None or config_cls is None:
        raise RuntimeError(
            "Installed TRL build does not provide ORPOTrainer/ORPOConfig. "
            "Upgrade TRL in runpod_setup.sh or disable phase1d_orpo.py in orchestration."
        )
    return trainer_cls, config_cls


def _extract_split(dataset):
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        split_name = sorted(dataset.keys())[0]
        print(f"Info: DatasetDict without 'train'. Using split '{split_name}'.")
        return dataset[split_name]
    return dataset


def _latest_checkpoint(output_dir):
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


def _prepare_orpo_dataset(ds):
    required = {"prompt", "chosen", "rejected"}
    missing = [col for col in required if col not in ds.column_names]
    if missing:
        raise RuntimeError(f"ORPO dataset missing required columns: {missing}")

    rows = []
    for row in ds:
        prompt = str(row.get("prompt", "") or "").strip()
        chosen = str(row.get("chosen", "") or "").strip()
        rejected = str(row.get("rejected", "") or "").strip()
        if not prompt or not chosen or not rejected:
            continue
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if not rows:
        return Dataset.from_list([])
    return Dataset.from_list(rows)


def _build_orpo_config(config_cls, **kwargs):
    init_sig = inspect.signature(config_cls.__init__)
    supported = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
    dropped = sorted(set(kwargs.keys()) - set(supported.keys()))
    if dropped:
        print(f"Info: Ignoring unsupported ORPOConfig args for this TRL version: {dropped}")
    return config_cls(**supported)


def train_orpo(
    dpo_dataset_path="./sota_dpo_pairs_dataset",
    base_model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    adapter_path="qwen_dpo_lora",
    output_dir="outputs_orpo",
    output_model_dir="qwen_orpo_lora",
    max_steps=300,
    learning_rate=3e-7,
    checkpoint_every_steps=100,
    resume_from_checkpoint="auto",
    seed=3407,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ORPO training on this pipeline.")
    if not os.path.exists(dpo_dataset_path):
        raise RuntimeError(f"ORPO dataset path not found: {dpo_dataset_path}")

    ORPOTrainer, ORPOConfig = _load_trl_orpo()

    raw = _extract_split(load_from_disk(dpo_dataset_path))
    dataset = _prepare_orpo_dataset(raw)
    if len(dataset) == 0:
        raise RuntimeError("No ORPO samples available after filtering.")

    dtype = torch.bfloat16 if torch.cuda.is_bfloat16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    if os.path.exists(adapter_path):
        print(f"Loading adapter before ORPO from '{adapter_path}'...")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print("WARNUNG: Adapter path for ORPO not found. ORPO starts from base model.")

    tokenizer_path = adapter_path if os.path.exists(adapter_path) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    cfg = _build_orpo_config(
        ORPOConfig,
        output_dir=output_dir,
        max_steps=max_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=checkpoint_every_steps,
        save_total_limit=2,
        warmup_steps=20,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_length=2048,
        max_prompt_length=1024,
        max_completion_length=1024,
        fp16=not torch.cuda.is_bfloat16_supported(),
        bf16=torch.cuda.is_bfloat16_supported(),
        beta=0.1,
        remove_unused_columns=False,
        report_to=[],
        seed=seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": cfg,
        "train_dataset": dataset,
    }
    init_sig = inspect.signature(ORPOTrainer.__init__)
    if "processing_class" in init_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = ORPOTrainer(**trainer_kwargs)
    resume_path = None
    if resume_from_checkpoint and str(resume_from_checkpoint).lower() != "none":
        if str(resume_from_checkpoint).lower() == "auto":
            resume_path = _latest_checkpoint(output_dir)
        else:
            resume_path = resume_from_checkpoint
    if resume_path:
        print(f"Resuming ORPO from checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)

    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"ORPO adapter saved to '{output_model_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ORPO reference-free preference alignment.")
    parser.add_argument("--dpo-dataset-path", default="./sota_dpo_pairs_dataset")
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="qwen_dpo_lora")
    parser.add_argument("--output-dir", default="outputs_orpo")
    parser.add_argument("--output-model-dir", default="qwen_orpo_lora")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    train_orpo(
        dpo_dataset_path=args.dpo_dataset_path,
        base_model_name=args.base_model_name,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        output_model_dir=args.output_model_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
    )
