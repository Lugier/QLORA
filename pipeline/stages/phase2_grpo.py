import argparse
import inspect
import os

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL

import rewards as rewards_module
from verification import assess_test_quality, is_test_quality_sufficient


# 1. Override vLLM KV cache dtype with GPU-aware defaults.
def _configure_vllm_kv_cache_dtype():
    if not torch.cuda.is_available():
        os.environ.setdefault("VLLM_KV_CACHE_DTYPE", "auto")
        return
    major, _minor = torch.cuda.get_device_capability(0)
    # FP8 KV cache is generally safer on Hopper+; Ampere (e.g. RTX 3090) uses FP16.
    if major >= 9:
        os.environ["VLLM_KV_CACHE_DTYPE"] = "fp8"
    else:
        os.environ["VLLM_KV_CACHE_DTYPE"] = "fp16"


_configure_vllm_kv_cache_dtype()


# ==============================================================================
# Phase 2: GRPO Reinforcement Learning (The Intelligence Phase)
# ==============================================================================


class EMACheckpointCallback(TrainerCallback):
    """
    EMA stabilization in the late phase to reduce RL drift.
    """

    def __init__(self, model, max_steps, decay=0.9995, start_fraction=0.30):
        self.ema_model_weights = None
        self.model = model
        self.decay = decay
        self.trainable_param_names = {
            name for name, param in self.model.named_parameters() if param.requires_grad
        }
        start_fraction = max(0.05, min(0.95, float(start_fraction)))
        self.start_ema_step = int(max_steps * start_fraction)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.start_ema_step:
            return

        with torch.no_grad():
            if self.ema_model_weights is None:
                state_dict = self.model.state_dict()
                self.ema_model_weights = {
                    name: state_dict[name].clone().detach()
                    for name in self.trainable_param_names
                    if name in state_dict
                }
            else:
                state_dict = self.model.state_dict()
                for name in self.ema_model_weights.keys():
                    self.ema_model_weights[name].mul_(self.decay).add_(
                        state_dict[name].detach(), alpha=1 - self.decay
                    )

    def on_train_end(self, args, state, control, **kwargs):
        if self.ema_model_weights is not None:
            print("\nApplying EMA weights for robust production checkpoint...")
            self.model.load_state_dict(self.ema_model_weights, strict=False)


def _latest_checkpoint(output_dir):
    # Finds latest stage checkpoint for spot/preemptible continuation.
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


def _checkpoint_step(path: str) -> int:
    if not path:
        return 0
    name = os.path.basename(str(path).rstrip("/"))
    if not name.startswith("checkpoint-"):
        return 0
    try:
        return max(0, int(name.split("checkpoint-")[-1]))
    except ValueError:
        return 0


def _extract_split(dataset):
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        split_name = sorted(dataset.keys())[0]
        print(f"Info: DatasetDict without 'train'. Using split '{split_name}'.")
        return dataset[split_name]
    return dataset


def _canonicalize_rl_row(row):
    prompt = (row.get("prompt", "") or row.get("text", "") or "").strip()
    text = (row.get("text", "") or prompt).strip()
    tests = (row.get("tests", "") or "").strip()
    source = str(row.get("source", "") or "unknown").strip().lower()
    benchmark = str(row.get("benchmark", "") or "").strip().lower()
    error_type = str(row.get("error_type", "") or "").strip().lower()

    quality = assess_test_quality(tests)
    score = float(row.get("test_quality_score", quality.get("quality_score", 0.0)) or 0.0)
    asserts = int(row.get("test_assert_count", quality.get("assert_count", 0)) or 0)
    return {
        "prompt": prompt,
        "text": text,
        "tests": tests,
        "source": source,
        "benchmark": benchmark,
        "error_type": error_type,
        "test_quality_score": score,
        "test_assert_count": asserts,
    }


def format_rl_prompt(
    example,
    min_test_asserts=2,
    min_test_lines=3,
    min_test_quality_score=2.5,
):
    system_prompt = (
        "You are an expert python developer. Analyze thoroughly in <reasoning> tags, "
        "then output purely the executable code in <answer> tags."
    )
    user_prompt = example.get("prompt", "") or example.get("text", "")
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
    source = str(example.get("source", "") or "").strip().lower() or "unknown"
    benchmark = str(example.get("benchmark", "") or "").strip().lower() or "unknown"
    error_type = str(example.get("error_type", "") or "").strip().lower()
    return {
        "prompt": prompt,
        "answer": tests,
        "source": source,
        "benchmark": benchmark,
        "error_type": error_type,
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
        ds = _extract_split(load_from_disk(path))
        ds = ds.map(
            _canonicalize_rl_row,
            remove_columns=ds.column_names,
        )
        ds = ds.filter(lambda x: len(x["prompt"]) > 0 and len(x["tests"]) > 0)
        if len(ds) > 0:
            loaded.append(ds)

    if not loaded:
        return None
    if len(loaded) == 1:
        return loaded[0]
    return concatenate_datasets(loaded)


def _load_hard_replay_dataset(path, min_test_asserts, min_test_lines, min_test_quality_score):
    if not path or not os.path.exists(path):
        return None
    ds = _extract_split(load_from_disk(path))
    if len(ds) == 0:
        return None
    replay = ds.map(
        lambda x: format_rl_prompt(
            x,
            min_test_asserts=min_test_asserts,
            min_test_lines=min_test_lines,
            min_test_quality_score=min_test_quality_score,
        ),
        remove_columns=ds.column_names,
    )
    replay = replay.filter(lambda x: x["has_tests"] and x["tests_quality_ok"])
    return replay if len(replay) > 0 else None


def _split_curriculum_stages(rl_dataset):
    if len(rl_dataset) == 0:
        return []

    scores = sorted(float(x) for x in rl_dataset["difficulty_score"])
    q1 = scores[len(scores) // 4]
    q2 = scores[len(scores) // 2]
    q3 = scores[(3 * len(scores)) // 4]

    easy = rl_dataset.filter(lambda x: float(x["difficulty_score"]) <= q1)
    mid = rl_dataset.filter(lambda x: q1 < float(x["difficulty_score"]) <= q2)
    hard = rl_dataset.filter(lambda x: q2 < float(x["difficulty_score"]) <= q3)
    expert = rl_dataset.filter(lambda x: float(x["difficulty_score"]) > q3)

    stages = [
        ("easy", easy, 0.20),
        ("mid", mid, 0.30),
        ("hard", hard, 0.30),
        ("expert", expert, 0.20),
    ]
    return [(name, ds, ratio) for name, ds, ratio in stages if len(ds) > 0]


def _macro_bucket(example):
    source = str(example.get("source", "") or "").lower()
    benchmark = str(example.get("benchmark", "") or "").lower()
    prompt_len = int(example.get("prompt_length", 0) or 0)
    error_type = str(example.get("error_type", "") or "").lower()
    repo_like_sources = {"online_hard_mining", "tool_trajectory_distill"}
    repo_like_benchmarks = {"swebench_verified_subset", "bigcodebench_instruct", "livecodebench"}
    if (
        source in repo_like_sources
        or benchmark in repo_like_benchmarks
        or prompt_len >= 1700
        or error_type in {"timeout", "security", "syntax", "format"}
    ):
        return "repo_resolution"
    return "core_code"


def _tier_splits_easy_mid_hard(ds):
    if len(ds) == 0:
        return []
    scores = sorted(float(x) for x in ds["difficulty_score"])
    q1 = scores[len(scores) // 3]
    q2 = scores[(2 * len(scores)) // 3]
    buckets = [
        ("easy", ds.filter(lambda x: float(x["difficulty_score"]) <= q1)),
        ("mid", ds.filter(lambda x: q1 < float(x["difficulty_score"]) <= q2)),
        ("hard", ds.filter(lambda x: float(x["difficulty_score"]) > q2)),
    ]
    return [(name, split) for name, split in buckets if len(split) > 0]


def _split_two_dimensional_curriculum(rl_dataset):
    # Macro phase split: first stabilize core coding behavior, then stress repo/SWE-like resolution.
    core_ds = rl_dataset.filter(lambda x: str(x.get("macro_bucket", "")) == "core_code")
    repo_ds = rl_dataset.filter(lambda x: str(x.get("macro_bucket", "")) == "repo_resolution")

    if len(core_ds) == 0 or len(repo_ds) == 0:
        print(
            "WARNUNG: 2D curriculum fallback to 1D difficulty split because one macro bucket is empty "
            f"(core={len(core_ds)}, repo={len(repo_ds)})."
        )
        return _split_curriculum_stages(rl_dataset)

    stages = []
    core_ratio_total = 0.45
    repo_ratio_total = 0.55
    core_parts = _tier_splits_easy_mid_hard(core_ds)
    repo_parts = _tier_splits_easy_mid_hard(repo_ds)
    if not core_parts or not repo_parts:
        return _split_curriculum_stages(rl_dataset)

    core_part_ratio = core_ratio_total / len(core_parts)
    repo_part_ratio = repo_ratio_total / len(repo_parts)

    for difficulty, split in core_parts:
        stages.append((f"core_{difficulty}", split, core_part_ratio))
    for difficulty, split in repo_parts:
        stages.append((f"repo_{difficulty}", split, repo_part_ratio))
    return stages


def _drop_low_quality_samples(rl_dataset, drop_fraction=0.10, min_samples_after_drop=1000):
    drop_fraction = max(0.0, min(0.8, float(drop_fraction)))
    if drop_fraction <= 0.0 or len(rl_dataset) <= min_samples_after_drop:
        return rl_dataset, 0

    scores = sorted(float(s) for s in rl_dataset["test_quality_score"])
    cutoff_index = int(len(scores) * drop_fraction)
    if cutoff_index <= 0:
        return rl_dataset, 0
    cutoff = scores[min(cutoff_index, len(scores) - 1)]
    filtered = rl_dataset.filter(lambda x: float(x["test_quality_score"]) >= cutoff)
    if len(filtered) < min_samples_after_drop:
        return rl_dataset, 0
    return filtered, len(rl_dataset) - len(filtered)


def _parse_stage_drop_fractions(value):
    defaults = {"easy": 0.15, "mid": 0.10, "hard": 0.05, "expert": 0.03}
    if not value:
        return defaults
    parsed = dict(defaults)
    for part in str(value).split(","):
        if ":" not in part:
            continue
        key, raw = part.split(":", 1)
        key = key.strip().lower()
        try:
            parsed[key] = max(0.0, min(0.8, float(raw.strip())))
        except ValueError:
            continue
    return parsed


def _stage_drop_for_name(stage_name, drop_map):
    key = str(stage_name or "").strip().lower()
    if key in drop_map:
        return float(drop_map[key])
    for base in ("easy", "mid", "hard", "expert"):
        if key.endswith(f"_{base}") or key == base:
            return float(drop_map.get(base, 0.05))
    return float(drop_map.get("hard", 0.05))


def _upsample_priority_sources(rl_dataset, priority_sources, boost_factor=1.0, seed=3407):
    boost_factor = max(1.0, float(boost_factor))
    normalized = {str(x or "").strip().lower() for x in (priority_sources or []) if str(x or "").strip()}
    if boost_factor <= 1.0 or not normalized:
        return rl_dataset, 0

    priority_ds = rl_dataset.filter(lambda x: str(x.get("source", "") or "").strip().lower() in normalized)
    if len(priority_ds) == 0:
        return rl_dataset, 0

    target_extra = int(round((boost_factor - 1.0) * len(priority_ds)))
    if target_extra <= 0:
        return rl_dataset, 0

    extras = []
    remaining = target_extra
    cycle = 0
    while remaining > 0:
        take = min(len(priority_ds), remaining)
        sampled = priority_ds.shuffle(seed=seed + cycle).select(range(take))
        extras.append(sampled)
        remaining -= take
        cycle += 1

    boosted = concatenate_datasets([rl_dataset] + extras).shuffle(seed=seed)
    return boosted, target_extra


def _build_auto_hard_replay_dataset(rl_dataset, fraction=0.15, min_samples=200, max_samples=1500):
    if len(rl_dataset) == 0:
        return None
    rows = [dict(row) for row in rl_dataset]
    rows.sort(
        key=lambda row: (
            int(row.get("prompt_length", 0)),
            float(row.get("test_quality_score", 0.0)),
            int(row.get("test_assert_count", 0)),
        ),
        reverse=True,
    )
    n_target = int(len(rows) * fraction)
    n_target = max(min_samples, n_target)
    n_target = min(max_samples, n_target, len(rows))
    if n_target <= 0:
        return None
    return Dataset.from_list(rows[:n_target])


def _build_grpo_config(**kwargs):
    init_sig = inspect.signature(GRPOConfig.__init__)
    supported = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
    dropped = sorted(set(kwargs.keys()) - set(supported.keys()))
    if dropped:
        print(f"Info: Ignoring unsupported GRPOConfig args for this TRL version: {dropped}")
    return GRPOConfig(**supported)


def _stage_context_lengths(stage_name: str, max_seq_length: int):
    stage_name = (stage_name or "").lower()
    max_seq_length = max(1024, int(max_seq_length))
    long_context = any(token in stage_name for token in ["repo", "hard", "expert", "replay"])

    if long_context:
        prompt_target = 3072
        completion_target = 2048
    else:
        prompt_target = 1536
        completion_target = 1400

    # Keep prompt+completion strictly within available context budget.
    budget = max(768, max_seq_length - 256)
    min_prompt = 256
    min_completion = 256

    prompt_len = min(prompt_target, max(min_prompt, budget - min_completion))
    completion_len = min(completion_target, max(min_completion, budget - prompt_len))

    total = prompt_len + completion_len
    if total > budget:
        overflow = total - budget
        reducible_prompt = max(0, prompt_len - min_prompt)
        reduce_prompt = min(reducible_prompt, overflow)
        prompt_len -= reduce_prompt
        overflow -= reduce_prompt
        if overflow > 0:
            reducible_completion = max(0, completion_len - min_completion)
            reduce_completion = min(reducible_completion, overflow)
            completion_len -= reduce_completion

    return int(max(min_prompt, prompt_len)), int(max(min_completion, completion_len))


def _build_training_args(stage_steps, stage_name, seed, stage_output_dir, checkpoint_every_steps, max_seq_length):
    stage_name = (stage_name or "").lower()
    if stage_name == "hard_replay":
        lr = 3e-6
        num_generations = 6
    elif "easy" in stage_name:
        lr = 6e-6
        num_generations = 8
    elif "mid" in stage_name:
        lr = 5e-6
        num_generations = 8
    elif "hard" in stage_name:
        lr = 4e-6
        num_generations = 8
    elif "expert" in stage_name:
        lr = 3.5e-6
        num_generations = 8
    else:
        lr = 4e-6
        num_generations = 6

    # Smooth transition from core_code -> repo_resolution phases to reduce phase-shock.
    if stage_name.startswith("repo_"):
        lr *= 0.92

    max_prompt_length, max_completion_length = _stage_context_lengths(stage_name, max_seq_length=max_seq_length)

    return _build_grpo_config(
        output_dir=stage_output_dir,
        weight_decay=0.1,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        logging_steps=5,
        max_steps=stage_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        fp16=not torch.cuda.is_bfloat16_supported(),
        bf16=torch.cuda.is_bfloat16_supported(),
        optim="adamw_8bit",
        max_grad_norm=0.3,
        beta=0.05,
        save_strategy="steps",
        save_steps=max(20, int(checkpoint_every_steps)),
        save_total_limit=2,
        warmup_ratio=0.08,
        seed=seed,
        report_to=[],
    )


def _allocate_stage_steps(stages, total_steps, min_stage_steps=30):
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


def _annotate_difficulty(example):
    prompt_length = int(len(example["prompt"]))
    assert_count = float(example.get("test_assert_count", 0))
    quality = float(example.get("test_quality_score", 0.0))
    # Difficulty proxy used for curriculum ordering.
    # Longer prompts + denser/stronger tests typically correlate with harder tasks.
    difficulty_score = (
        (0.55 * min(1.0, prompt_length / 2400.0))
        + (0.25 * min(1.0, assert_count / 10.0))
        + (0.20 * min(1.0, quality / 10.0))
    )
    return {
        "prompt_length": prompt_length,
        "difficulty_score": float(difficulty_score),
    }


def _build_trainer(model, tokenizer, reward_funcs, training_args, train_dataset, callbacks):
    trainer_kwargs = {
        "model": model,
        "reward_funcs": reward_funcs,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    if "callbacks" in trainer_sig.parameters:
        trainer_kwargs["callbacks"] = callbacks
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError("Unsupported GRPOTrainer API: neither processing_class nor tokenizer argument found.")
    return GRPOTrainer(**trainer_kwargs)


def train_grpo(
    max_seq_length=8192,
    base_model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    allow_mbpp_fallback=False,
    candidate_dataset_paths=None,
    max_steps=1500,
    min_test_asserts=2,
    min_test_lines=3,
    min_test_quality_score=2.5,
    output_model_dir="qwen_grpo_final",
    output_adapter_dir="qwen_grpo_lora",
    drop_low_quality_fraction=0.10,
    stage_drop_fractions="easy:0.15,mid:0.10,hard:0.05,expert:0.03",
    curriculum_mode="two_dimensional_v1",
    min_rl_samples_after_drop=1000,
    priority_source_boost=1.8,
    priority_sources="online_hard_mining,tool_trajectory_distill",
    reward_profile="prm_outcome_v1",
    prm_model_path="",
    hard_replay_dataset="",
    hard_replay_steps=180,
    checkpoint_every_steps=120,
    resume_from_checkpoint="auto",
    checkpoints_root="outputs_grpo",
    seed=3407,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GRPO training on this pipeline.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if candidate_dataset_paths is None:
        candidate_dataset_paths = [
            "./sota_best_of_n_dataset",
            "./sota_slm_coding_dataset",
        ]

    PatchFastRL("GRPO", FastLanguageModel)
    rewards_module.AERO_GLOBAL_STEP = 0
    rewards_module.AERO_MAX_STEPS = max_steps
    rewards_module.configure_process_reward_model(prm_model_path)

    print("Loading baseline model for GRPO alignment...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
    )

    preferred_adapter_paths = [output_adapter_dir, "qwen_orpo_lora", "qwen_dpo_lora", "qwen_sft_lora"]
    loaded_adapter = None
    for adapter_path in preferred_adapter_paths:
        if os.path.exists(adapter_path):
            print(f"Loading structural adapter from '{adapter_path}'...")
            model.load_adapter(adapter_path)
            loaded_adapter = adapter_path
            break

    if loaded_adapter is None:
        print("WARNUNG: Kein SFT/DPO Adapter gefunden. Starte reines GRPO auf dem Basismodell.")

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

    rl_dataset, dropped_low_quality = _drop_low_quality_samples(
        rl_dataset,
        drop_fraction=drop_low_quality_fraction,
        min_samples_after_drop=min_rl_samples_after_drop,
    )
    if dropped_low_quality:
        print(
            f"Dropped {dropped_low_quality} low-quality RL samples "
            f"(drop_fraction={drop_low_quality_fraction}, min_samples_after_drop={min_rl_samples_after_drop})."
        )
    print(f"RL dataset size after global quality prioritization: {len(rl_dataset)}")

    parsed_priority_sources = [x.strip().lower() for x in str(priority_sources).split(",") if x.strip()]
    rl_dataset, upsampled = _upsample_priority_sources(
        rl_dataset,
        priority_sources=parsed_priority_sources,
        boost_factor=priority_source_boost,
        seed=seed,
    )
    if upsampled > 0:
        print(
            f"Upsampled priority-source samples by +{upsampled} rows "
            f"(boost={priority_source_boost}, sources={parsed_priority_sources})."
        )

    print("Applying Curriculum Learning: sorting by composite difficulty score...")
    rl_dataset = rl_dataset.map(_annotate_difficulty)
    rl_dataset = rl_dataset.sort("difficulty_score")
    rl_dataset = rl_dataset.map(lambda x: {"macro_bucket": _macro_bucket(x)})

    curriculum_mode = (curriculum_mode or "two_dimensional_v1").strip().lower()
    if curriculum_mode == "two_dimensional_v1":
        stages = _split_two_dimensional_curriculum(rl_dataset)
    else:
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

    reward_funcs = rewards_module.get_reward_functions(reward_profile)
    drop_map = _parse_stage_drop_fractions(stage_drop_fractions)

    # Stage checkpoints live under checkpoints_root so each stage can resume independently.
    os.makedirs(checkpoints_root, exist_ok=True)
    print(f"Initiating {len(stage_plan)}-stage curriculum GRPO training...")
    for idx, (stage_name, stage_ds, stage_steps) in enumerate(stage_plan):
        if len(stage_ds) == 0 or stage_steps <= 0:
            continue

        stage_drop = _stage_drop_for_name(stage_name, drop_map)
        stage_ds, dropped = _drop_low_quality_samples(
            stage_ds,
            drop_fraction=stage_drop,
            min_samples_after_drop=max(160, min_rl_samples_after_drop // max(1, len(stage_plan))),
        )
        if dropped:
            print(
                f"[Curriculum] Stage '{stage_name}' dropped {dropped} low-quality samples "
                f"with stage_drop_fraction={stage_drop:.3f}."
            )

        print(
            f"[Curriculum] Stage {idx + 1}/{len(stage_plan)} "
            f"'{stage_name}' with {len(stage_ds)} samples for {stage_steps} steps."
        )
        stage_output_dir = os.path.join(checkpoints_root, f"stage_{idx+1}_{stage_name}")
        os.makedirs(stage_output_dir, exist_ok=True)
        training_args = _build_training_args(
            stage_steps=stage_steps,
            stage_name=stage_name,
            seed=seed,
            stage_output_dir=stage_output_dir,
            checkpoint_every_steps=checkpoint_every_steps,
            max_seq_length=max_seq_length,
        )
        callbacks = [EMACheckpointCallback(model, max_steps=stage_steps, decay=0.9995, start_fraction=0.30)] if idx == len(stage_plan) - 1 else []
        trainer = _build_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            training_args=training_args,
            train_dataset=stage_ds,
            callbacks=callbacks,
        )
        stage_resume = None
        if resume_from_checkpoint and str(resume_from_checkpoint).lower() != "none":
            if str(resume_from_checkpoint).lower() == "auto":
                # Auto mode resumes within the current stage directory only.
                stage_resume = _latest_checkpoint(stage_output_dir)
            else:
                stage_resume = resume_from_checkpoint
        if stage_resume:
            print(f"[Curriculum] Resuming stage '{stage_name}' from {stage_resume}")
        rewards_module.AERO_MAX_STEPS = max(1, int(stage_steps))
        rewards_module.AERO_GLOBAL_STEP = min(
            rewards_module.AERO_MAX_STEPS,
            _checkpoint_step(stage_resume) if stage_resume else 0,
        )
        trainer.train(resume_from_checkpoint=stage_resume)

    replay_dataset = _load_hard_replay_dataset(
        path=hard_replay_dataset,
        min_test_asserts=min_test_asserts,
        min_test_lines=min_test_lines,
        min_test_quality_score=min_test_quality_score,
    )
    if replay_dataset is None:
        replay_dataset = _build_auto_hard_replay_dataset(rl_dataset)
        if replay_dataset is not None:
            print(f"Built automatic hard-replay set with {len(replay_dataset)} samples.")
    else:
        print(f"Loaded explicit hard-replay dataset with {len(replay_dataset)} samples.")

    if replay_dataset is not None and len(replay_dataset) > 0:
        replay_steps = max(40, min(int(hard_replay_steps), max_steps))
        print(f"Running hard-replay pass for {replay_steps} steps...")
        replay_output_dir = os.path.join(checkpoints_root, "stage_replay_hard")
        os.makedirs(replay_output_dir, exist_ok=True)
        replay_args = _build_training_args(
            stage_steps=replay_steps,
            stage_name="hard_replay",
            seed=seed,
            stage_output_dir=replay_output_dir,
            checkpoint_every_steps=checkpoint_every_steps,
            max_seq_length=max_seq_length,
        )
        replay_trainer = _build_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_funcs=reward_funcs,
            training_args=replay_args,
            train_dataset=replay_dataset,
            callbacks=[],
        )
        replay_resume = None
        if resume_from_checkpoint and str(resume_from_checkpoint).lower() != "none":
            if str(resume_from_checkpoint).lower() == "auto":
                # Replay phase uses its own checkpoint namespace.
                replay_resume = _latest_checkpoint(replay_output_dir)
            else:
                replay_resume = resume_from_checkpoint
        if replay_resume:
            print(f"[Replay] Resuming hard replay from {replay_resume}")
        rewards_module.AERO_MAX_STEPS = max(1, int(replay_steps))
        rewards_module.AERO_GLOBAL_STEP = min(
            rewards_module.AERO_MAX_STEPS,
            _checkpoint_step(replay_resume) if replay_resume else 0,
        )
        replay_trainer.train(resume_from_checkpoint=replay_resume)

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
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--allow-mbpp-fallback", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--min-test-asserts", type=int, default=2)
    parser.add_argument("--min-test-lines", type=int, default=3)
    parser.add_argument("--min-test-quality-score", type=float, default=2.5)
    parser.add_argument("--output-model-dir", default="qwen_grpo_final")
    parser.add_argument("--output-adapter-dir", default="qwen_grpo_lora")
    parser.add_argument("--drop-low-quality-fraction", type=float, default=0.10)
    parser.add_argument("--stage-drop-fractions", default="easy:0.15,mid:0.10,hard:0.05,expert:0.03")
    parser.add_argument("--curriculum-mode", default="two_dimensional_v1", choices=["one_dimensional_v1", "two_dimensional_v1"])
    parser.add_argument("--min-rl-samples-after-drop", type=int, default=1000)
    parser.add_argument("--priority-source-boost", type=float, default=1.8)
    parser.add_argument("--priority-sources", default="online_hard_mining,tool_trajectory_distill")
    parser.add_argument("--reward-profile", default="prm_outcome_v1")
    parser.add_argument("--prm-model-path", default="")
    parser.add_argument("--hard-replay-dataset", default="")
    parser.add_argument("--hard-replay-steps", type=int, default=180)
    parser.add_argument("--checkpoint-every-steps", type=int, default=120)
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--checkpoints-root", default="outputs_grpo")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--dataset-paths",
        default="./sota_best_of_n_dataset,./sota_slm_coding_dataset",
        help="Comma-separated dataset paths for RL training.",
    )
    args = parser.parse_args()

    dataset_paths = [p.strip() for p in args.dataset_paths.split(",") if p.strip()]

    train_grpo(
        max_seq_length=args.max_seq_length,
        base_model_name=args.base_model_name,
        allow_mbpp_fallback=args.allow_mbpp_fallback,
        candidate_dataset_paths=dataset_paths,
        max_steps=args.max_steps,
        min_test_asserts=args.min_test_asserts,
        min_test_lines=args.min_test_lines,
        min_test_quality_score=args.min_test_quality_score,
        output_model_dir=args.output_model_dir,
        output_adapter_dir=args.output_adapter_dir,
        drop_low_quality_fraction=args.drop_low_quality_fraction,
        stage_drop_fractions=args.stage_drop_fractions,
        curriculum_mode=args.curriculum_mode,
        min_rl_samples_after_drop=args.min_rl_samples_after_drop,
        priority_source_boost=args.priority_source_boost,
        priority_sources=args.priority_sources,
        reward_profile=args.reward_profile,
        prm_model_path=args.prm_model_path,
        hard_replay_dataset=args.hard_replay_dataset,
        hard_replay_steps=args.hard_replay_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        checkpoints_root=args.checkpoints_root,
        seed=args.seed,
    )
