import argparse
import hashlib
import inspect
import math
import os

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer


# ==============================================================================
# Phase 1c: Direct Preference Optimization (DPO)
# Trains on prompt/chosen/rejected tuples emitted by phase1b rejection sampling.
# ==============================================================================


class DivergenceGuardCallback(TrainerCallback):
    """
    Stops training early when loss becomes non-finite or clearly divergent.
    """

    def __init__(self, loss_threshold=40.0):
        self.loss_threshold = float(loss_threshold)
        self.diverged = False
        self.reason = ""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is None:
            return
        try:
            loss_value = float(loss)
        except Exception:
            return

        if not math.isfinite(loss_value):
            self.diverged = True
            self.reason = f"Non-finite DPO loss at step {state.global_step}: {loss}"
            control.should_training_stop = True
            return

        if self.loss_threshold > 0 and loss_value > self.loss_threshold:
            self.diverged = True
            self.reason = (
                f"DPO loss divergence detected at step {state.global_step}: "
                f"loss={loss_value:.4f} > threshold={self.loss_threshold:.4f}"
            )
            control.should_training_stop = True


def _latest_checkpoint(output_dir):
    # Spot/preemptible resume helper shared across DPO runs.
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


def _iter_dataset_rows(dataset):
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        split_name = sorted(dataset.keys())[0]
        print(f"Info: DatasetDict without 'train'. Using split '{split_name}'.")
        return dataset[split_name]
    return dataset


def _prepare_dpo_dataset(
    dataset,
    min_score_gap=1.0,
    max_rejected_score=0.6,
    min_test_assert_count=2,
    max_pairs_per_prompt=6,
):
    required_columns = {"prompt", "chosen", "rejected"}
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise RuntimeError(f"DPO dataset missing required columns: {missing}")

    grouped = {}
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

        grouped.setdefault(prompt, []).append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "score_gap": score_gap,
                "rejected_score": rejected_score,
                "test_assert_count": test_assert_count,
                "pair_weight": float(row.get("pair_weight", 1.0)),
            }
        )

    if not grouped:
        return Dataset.from_list([])

    prepared_rows = []
    limit = max(1, int(max_pairs_per_prompt)) if max_pairs_per_prompt > 0 else 0
    for prompt, pairs in grouped.items():
        pairs.sort(
            key=lambda p: (
                -float(p["score_gap"]),
                float(p["rejected_score"]),
                -int(p["test_assert_count"]),
                len(p["rejected"]),
            )
        )
        selected = pairs[:limit] if limit else pairs
        prepared_rows.extend(selected)

    if not prepared_rows:
        return Dataset.from_list([])
    return Dataset.from_list(prepared_rows)


def _apply_gap_weighted_resampling(dataset, enabled=False, max_replication=4):
    if not enabled or len(dataset) == 0:
        return dataset

    weights = []
    for row in dataset:
        base_weight = float(row.get("pair_weight", row.get("score_gap", 1.0)))
        weights.append(max(0.1, base_weight))

    sorted_weights = sorted(weights)
    median = sorted_weights[len(sorted_weights) // 2] if sorted_weights else 1.0
    median = max(0.1, float(median))
    max_replication = max(1, int(max_replication))

    # Replicate high-gap pairs slightly more often to sharpen preference learning.
    expanded_rows = []
    for row, weight in zip(dataset, weights):
        normalized = weight / median
        multiplier = int(round(max(1.0, min(float(max_replication), normalized))))
        for _ in range(multiplier):
            expanded_rows.append(dict(row))

    print(
        "Applied gap-weighted resampling: "
        f"original_pairs={len(dataset)} -> resampled_pairs={len(expanded_rows)}"
    )
    return Dataset.from_list(expanded_rows)


def _split_train_eval(dataset, eval_fraction=0.05, seed=3407):
    if len(dataset) < 200:
        return dataset, None
    eval_fraction = max(0.02, min(0.2, float(eval_fraction)))
    shuffled = dataset.shuffle(seed=seed)
    eval_size = int(len(shuffled) * eval_fraction)
    eval_size = max(50, min(len(shuffled) - 1, eval_size))
    eval_dataset = shuffled.select(range(eval_size))
    train_dataset = shuffled.select(range(eval_size, len(shuffled)))
    return train_dataset, eval_dataset


def _build_dpo_config(**kwargs):
    init_sig = inspect.signature(DPOConfig.__init__)
    supported = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
    dropped = sorted(set(kwargs.keys()) - set(supported.keys()))
    if dropped:
        print(f"Info: Ignoring unsupported DPOConfig args for this TRL version: {dropped}")
    return DPOConfig(**supported)


def train_dpo(
    dpo_dataset_path="./sota_dpo_pairs_dataset",
    base_model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    sft_adapter_path="qwen_sft_lora",
    output_dir="outputs_dpo",
    output_model_dir="qwen_dpo_lora",
    max_steps=800,
    min_score_gap=1.0,
    max_rejected_score=0.6,
    min_test_assert_count=2,
    max_pairs_per_prompt=6,
    gap_weighted_sampling=True,
    eval_every_steps=200,
    divergence_loss_threshold=40.0,
    checkpoint_every_steps=200,
    resume_from_checkpoint="auto",
    seed=3407,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO training on this pipeline.")

    if not os.path.exists(dpo_dataset_path):
        raise RuntimeError(
            f"DPO-Datensatz '{dpo_dataset_path}' nicht gefunden. "
            "Bitte zuerst phase1b_rejection_sampling.py ausführen."
        )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Loading DPO preference pairs from '{dpo_dataset_path}'...")
    raw_dataset = _iter_dataset_rows(load_from_disk(dpo_dataset_path))
    train_dataset = _prepare_dpo_dataset(
        raw_dataset,
        min_score_gap=min_score_gap,
        max_rejected_score=max_rejected_score,
        min_test_assert_count=min_test_assert_count,
        max_pairs_per_prompt=max_pairs_per_prompt,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Keine validen DPO-Paare nach Filterung.")

    train_dataset = _apply_gap_weighted_resampling(
        train_dataset,
        enabled=gap_weighted_sampling,
    )
    train_dataset, eval_dataset = _split_train_eval(train_dataset, eval_fraction=0.05, seed=seed)

    print(f"Prepared {len(train_dataset)} DPO train pairs.")
    if eval_dataset is not None:
        print(f"Prepared {len(eval_dataset)} DPO eval pairs for best-checkpoint selection.")

    model = _load_trainable_model(base_model_name, sft_adapter_path)

    tokenizer_path = sft_adapter_path if os.path.exists(sft_adapter_path) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_eval = eval_dataset is not None and eval_every_steps > 0
    config_kwargs = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "logging_steps": 10,
        "save_steps": checkpoint_every_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "warmup_steps": 50,
        "learning_rate": 5e-7,
        "lr_scheduler_type": "cosine",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "beta": 0.1,
        "optim": "adamw_torch",
        "gradient_checkpointing": True,
        "fp16": not torch.cuda.is_bfloat16_supported(),
        "bf16": torch.cuda.is_bfloat16_supported(),
        "max_length": 2048,
        "max_prompt_length": 1024,
        "max_target_length": 1024,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": seed,
    }
    if use_eval:
        config_kwargs.update(
            {
                "eval_strategy": "steps",
                "evaluation_strategy": "steps",
                "eval_steps": eval_every_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )

    training_args = _build_dpo_config(**config_kwargs)

    divergence_guard = DivergenceGuardCallback(loss_threshold=divergence_loss_threshold)
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if use_eval:
        trainer_kwargs["eval_dataset"] = eval_dataset

    init_sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in init_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "callbacks" in init_sig.parameters:
        trainer_kwargs["callbacks"] = [divergence_guard]

    trainer = DPOTrainer(**trainer_kwargs)
    print("Starting DPO preference alignment...")
    resume_path = None
    if resume_from_checkpoint and str(resume_from_checkpoint).lower() != "none":
        if str(resume_from_checkpoint).lower() == "auto":
            # Auto-resume keeps long spot jobs from restarting from step 0.
            resume_path = _latest_checkpoint(output_dir)
        else:
            resume_path = resume_from_checkpoint
    if resume_path:
        print(f"Resuming DPO from checkpoint: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)

    if divergence_guard.diverged:
        raise RuntimeError(divergence_guard.reason)

    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt:
        print(f"Best DPO checkpoint selected from: {best_ckpt}")

    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"DPO completed. Adapter saved to '{output_model_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPO preference alignment.")
    parser.add_argument("--dpo-dataset-path", default="./sota_dpo_pairs_dataset")
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--sft-adapter-path", default="qwen_sft_lora")
    parser.add_argument("--output-dir", default="outputs_dpo")
    parser.add_argument("--output-model-dir", default="qwen_dpo_lora")
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-score-gap", type=float, default=1.0)
    parser.add_argument("--max-rejected-score", type=float, default=0.6)
    parser.add_argument("--min-test-assert-count", type=int, default=2)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=6)
    parser.add_argument("--gap-weighted-sampling", action="store_true")
    parser.add_argument("--no-gap-weighted-sampling", action="store_true")
    parser.add_argument("--eval-every-steps", type=int, default=200)
    parser.add_argument("--checkpoint-every-steps", type=int, default=200)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--divergence-loss-threshold", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    gap_weighted_sampling = not args.no_gap_weighted_sampling
    if args.gap_weighted_sampling:
        gap_weighted_sampling = True

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
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        gap_weighted_sampling=gap_weighted_sampling,
        eval_every_steps=args.eval_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        divergence_loss_threshold=args.divergence_loss_threshold,
        seed=args.seed,
    )
