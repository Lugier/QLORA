import argparse
import inspect
import os
import hashlib

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


# ==============================================================================
# Phase 1c: Direct Preference Optimization (DPO)
# Trains on prompt/chosen/rejected tuples emitted by phase1b rejection sampling.
# ==============================================================================


def _load_trainable_model(base_model_name, sft_adapter_path):
    dtype = torch.bfloat16 if torch.cuda.is_bfloat16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
    )

    if os.path.exists(sft_adapter_path):
        print(f"Loading trainable SFT adapter from '{sft_adapter_path}'...")
        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
        return model

    print("WARNUNG: Kein SFT-Adapter gefunden. Erzeuge frische LoRA-Adapter fuer DPO.")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return get_peft_model(model, lora_cfg)


def _prepare_dpo_dataset(
    dataset,
    min_score_gap=1.0,
    max_rejected_score=0.6,
    min_test_assert_count=2,
):
    required_columns = {"prompt", "chosen", "rejected"}
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise RuntimeError(f"DPO dataset missing required columns: {missing}")

    filtered_rows = []
    seen = set()
    for row in dataset:
        prompt = (row.get("prompt", "") or "").strip()
        chosen = (row.get("chosen", "") or "").strip()
        rejected = (row.get("rejected", "") or "").strip()
        if not prompt or not chosen or not rejected:
            continue

        score_gap = float(row.get("score_gap", row.get("chosen_score", 2.0) - row.get("rejected_score", 0.0)))
        rejected_score = float(row.get("rejected_score", 0.0))
        test_assert_count = int(row.get("test_assert_count", 0))
        if score_gap < min_score_gap:
            continue
        if rejected_score > max_rejected_score:
            continue
        if test_assert_count < min_test_assert_count:
            continue

        key = hashlib.sha256(f"{prompt}\n{chosen}\n{rejected}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        filtered_rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "score_gap": score_gap,
                "rejected_score": rejected_score,
                "test_assert_count": test_assert_count,
            }
        )

    if not filtered_rows:
        return Dataset.from_list([])
    return Dataset.from_list(filtered_rows)


def _build_dpo_config(**kwargs):
    init_sig = inspect.signature(DPOConfig.__init__)
    supported = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
    dropped = sorted(set(kwargs.keys()) - set(supported.keys()))
    if dropped:
        print(f"Info: Ignoring unsupported DPOConfig args for this TRL version: {dropped}")
    return DPOConfig(**supported)


def train_dpo(
    dpo_dataset_path="./sota_dpo_pairs_dataset",
    base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    sft_adapter_path="qwen_sft_lora",
    output_dir="outputs_dpo",
    output_model_dir="qwen_dpo_lora",
    max_steps=800,
    min_score_gap=1.0,
    max_rejected_score=0.6,
    min_test_assert_count=2,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO training on this pipeline.")

    if not os.path.exists(dpo_dataset_path):
        raise RuntimeError(
            f"DPO-Datensatz '{dpo_dataset_path}' nicht gefunden. "
            "Bitte zuerst phase1b_rejection_sampling.py ausführen."
        )

    print(f"Loading DPO preference pairs from '{dpo_dataset_path}'...")
    raw_dataset = load_from_disk(dpo_dataset_path)
    train_dataset = _prepare_dpo_dataset(
        raw_dataset,
        min_score_gap=min_score_gap,
        max_rejected_score=max_rejected_score,
        min_test_assert_count=min_test_assert_count,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Keine validen DPO-Paare nach Filterung.")

    print(f"Prepared {len(train_dataset)} DPO pairs.")
    model = _load_trainable_model(base_model_name, sft_adapter_path)

    tokenizer_path = sft_adapter_path if os.path.exists(sft_adapter_path) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = _build_dpo_config(
        output_dir=output_dir,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=200,
        warmup_steps=50,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        beta=0.1,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=not torch.cuda.is_bfloat16_supported(),
        bf16=torch.cuda.is_bfloat16_supported(),
        max_length=2048,
        max_prompt_length=1024,
        max_target_length=1024,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    init_sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in init_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)
    print("Starting DPO preference alignment...")
    trainer.train()

    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"DPO completed. Adapter saved to '{output_model_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO preference alignment.")
    parser.add_argument("--dpo-dataset-path", default="./sota_dpo_pairs_dataset")
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft-adapter-path", default="qwen_sft_lora")
    parser.add_argument("--output-dir", default="outputs_dpo")
    parser.add_argument("--output-model-dir", default="qwen_dpo_lora")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-score-gap", type=float, default=1.0)
    parser.add_argument("--max-rejected-score", type=float, default=0.6)
    parser.add_argument("--min-test-assert-count", type=int, default=2)
    args = parser.parse_args()

    train_dpo(
        dpo_dataset_path=args.dpo_dataset_path,
        base_model_name=args.base_model_name,
        sft_adapter_path=args.sft_adapter_path,
        output_dir=args.output_dir,
        output_model_dir=args.output_model_dir,
        max_steps=args.max_steps,
        min_score_gap=args.min_score_gap,
        max_rejected_score=args.max_rejected_score,
        min_test_assert_count=args.min_test_assert_count,
    )
