import argparse
import os
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from vllm import LLM, SamplingParams
import re
from verification import assess_test_quality, is_test_quality_sufficient, run_test_verifier

# ==============================================================================
# Phase 1b: Best-of-N Rejection Sampling (Distillation)
# Generiert N Lösungen pro Prompt, testet sie in der Sandbox und behält nur
# die 100% perfekten (Reasoning + Code) für ein "flawless" SFT-Distillation.
# ==============================================================================

def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _is_adapter_only_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_adapter = os.path.exists(os.path.join(path, "adapter_config.json"))
    has_model_cfg = os.path.exists(os.path.join(path, "config.json"))
    return has_adapter and not has_model_cfg


def _truncate_prompt(prompt: str, max_prompt_chars: int) -> str:
    prompt = prompt or ""
    if max_prompt_chars <= 0 or len(prompt) <= max_prompt_chars:
        return prompt
    assistant_token = "<|im_start|>assistant\n"
    if prompt.endswith(assistant_token):
        keep = max(1, max_prompt_chars - len(assistant_token))
        return prompt[:keep] + assistant_token
    return prompt[:max_prompt_chars]


def _parse_temperatures(value):
    if isinstance(value, (list, tuple)):
        parsed = [float(v) for v in value]
    else:
        parsed = []
        for part in str(value or "").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(float(part))
            except ValueError:
                continue
    valid = [max(0.0, min(1.5, temp)) for temp in parsed]
    return valid if valid else [0.2, 0.8]


def _compute_pair_weight(pair_weighting: str, score_gap: float) -> float:
    mode = (pair_weighting or "none").strip().lower()
    if mode == "score_gap":
        return max(0.1, min(8.0, float(score_gap)))
    return 1.0


def _ranking_key(candidate):
    return (
        -float(candidate.get("score", 0.0)),
        float(candidate.get("exec_time", 999.0)),
        len(str(candidate.get("text", "") or "")),
    )


def _load_prompt_dataset(dataset_path, num_prompts):
    full_ds = load_from_disk(dataset_path)
    if isinstance(full_ds, DatasetDict):
        if "train" in full_ds:
            base = full_ds["train"]
        else:
            split_name = sorted(full_ds.keys())[0]
            print(f"Info: DatasetDict without 'train'. Using split '{split_name}'.")
            base = full_ds[split_name]
    else:
        base = full_ds
    if len(base) == 0:
        raise RuntimeError(f"Dataset '{dataset_path}' is empty.")
    return base.select(range(min(num_prompts, len(base))))

def generate_and_filter(
    model_path="qwen_sft_merged",
    dataset_path="./sota_slm_coding_dataset",
    output_path="./sota_best_of_n_dataset",
    dpo_output_path="./sota_dpo_pairs_dataset",
    num_prompts=2000,
    num_samples_per_prompt=16,
    timeout=1.0,
    verifier_rounds=2,
    min_test_asserts=2,
    min_test_lines=3,
    min_test_quality_score=2.5,
    min_perfect=50,
    min_dpo_pairs=50,
    dpo_min_score_gap=1.0,
    dpo_max_rejected_score=0.6,
    dpo_negatives_per_prompt=3,
    temperatures="0.2,0.8",
    pair_weighting="score_gap",
    min_evaluated_candidates=4,
    generation_batch_size=64,
    max_prompt_chars=7000,
    seed=3407,
    require_dpo_pairs=True,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rejection sampling on this pipeline.")

    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        raise RuntimeError("Model or dataset not found. Please run phase1_sft.py first.")
    if _is_adapter_only_dir(model_path):
        raise RuntimeError(
            f"'{model_path}' is a LoRA adapter-only directory. "
            "vLLM requires a merged/full HF checkpoint. "
            "Run phase1_sft.py and use --merged-model-dir (default: qwen_sft_merged)."
        )

    print("Loading optimized SFT model with vLLM backend...")
    
    # Nutzt vLLM, da wir massiven Durchsatz brauchen
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        tensor_parallel_size=1,
        enforce_eager=True # wichtig für LoRA
    )
    
    print("Loading prompt dataset...")
    dataset = _load_prompt_dataset(dataset_path, num_prompts=num_prompts)
    
    # N Generationen pro Prompt und Temperatur.
    N = num_samples_per_prompt
    temperatures = _parse_temperatures(temperatures)
    dpo_negatives_per_prompt = max(1, int(dpo_negatives_per_prompt))
    
    def _prompt_only(example):
        prompt = example.get("prompt")
        if prompt:
            return prompt
        text = example.get("text", "")
        split_token = "<|im_start|>assistant\n"
        if split_token in text:
            return text.split(split_token, 1)[0] + split_token
        return text

    prompts = [_truncate_prompt(_prompt_only(example), max_prompt_chars=max_prompt_chars) for example in dataset]
    test_cases = [example.get("tests", "") or "" for example in dataset]
    
    distilled_data = []
    dpo_pairs = []
    skipped_without_tests = 0
    skipped_low_quality_tests = 0
    
    print("Evaluating generations in secure AST sandbox...")
    generation_batch_size = max(1, int(generation_batch_size))
    for start_idx in range(0, len(prompts), generation_batch_size):
        end_idx = min(len(prompts), start_idx + generation_batch_size)
        prompt_chunk = prompts[start_idx:end_idx]
        print(
            f"[Distill] Generating chunk {start_idx}-{end_idx - 1} "
            f"({len(prompt_chunk)} prompts, {N} samples x {len(temperatures)} temps)..."
        )
        chunk_outputs_by_temp = []
        for temp_idx, temp in enumerate(temperatures):
            sampling_params = SamplingParams(
                n=N,
                temperature=temp,
                top_p=0.9,
                max_tokens=1500,
                stop=["<|im_end|>"],
                seed=seed + (temp_idx * 997) + start_idx,
            )
            outputs = llm.generate(prompt_chunk, sampling_params)
            chunk_outputs_by_temp.append((temp, outputs))

        for local_idx in range(len(prompt_chunk)):
            idx = start_idx + local_idx
            tests = (test_cases[idx] or "").strip()
            if not tests:
                skipped_without_tests += 1
                continue
            test_quality = assess_test_quality(tests)
            if not is_test_quality_sufficient(
                tests,
                min_asserts=min_test_asserts,
                min_nonempty_lines=min_test_lines,
                min_quality_score=min_test_quality_score,
            ):
                skipped_low_quality_tests += 1
                continue

            passing_candidates = []
            non_passing_candidates = []
            evaluated_candidates = 0

            for temp, outputs in chunk_outputs_by_temp:
                output = outputs[local_idx]
                for completion in output.outputs:
                    text = completion.text
                    code = extract_xml_content(text, "answer")
                    if not code:
                        continue

                    code = code.replace("```python", "").replace("```", "").strip()
                    verify = run_test_verifier(
                        code=code,
                        tests=tests,
                        timeout=timeout,
                        rounds=verifier_rounds,
                        require_all_pass=True,
                    )
                    score = float(verify["score"])
                    exec_time = float(verify["exec_time"])
                    evaluated_candidates += 1
                    candidate = {
                        "text": text,
                        "score": score,
                        "exec_time": exec_time,
                        "temperature": temp,
                    }
                    if score == 2.0:
                        passing_candidates.append(candidate)
                    else:
                        non_passing_candidates.append(candidate)

            passing_candidates.sort(key=_ranking_key)
            non_passing_candidates.sort(key=_ranking_key)
            best_passing = passing_candidates[0] if passing_candidates else None

            if best_passing:
                distilled_prompt = prompts[idx] + best_passing["text"]
                distilled_data.append(
                    {
                        "text": distilled_prompt,
                        "prompt": prompts[idx],
                        "tests": test_cases[idx],
                        "chosen_score": best_passing["score"],
                        "chosen_temperature": best_passing["temperature"],
                        "test_quality_score": test_quality["quality_score"],
                        "test_assert_count": test_quality["assert_count"],
                    }
                )
                if non_passing_candidates and evaluated_candidates >= min_evaluated_candidates:
                    selected_negatives = non_passing_candidates[:dpo_negatives_per_prompt]
                    for rank_idx, negative in enumerate(selected_negatives, start=1):
                        score_gap = best_passing["score"] - negative["score"]
                        if score_gap < dpo_min_score_gap:
                            continue
                        if negative["score"] > dpo_max_rejected_score:
                            continue
                        dpo_pairs.append(
                            {
                                "prompt": prompts[idx],
                                "chosen": best_passing["text"],
                                "rejected": negative["text"],
                                "tests": test_cases[idx],
                                "chosen_score": best_passing["score"],
                                "rejected_score": negative["score"],
                                "chosen_temperature": best_passing["temperature"],
                                "rejected_temperature": negative["temperature"],
                                "score_gap": score_gap,
                                "pair_weight": _compute_pair_weight(pair_weighting, score_gap),
                                "pair_weighting": pair_weighting,
                                "negative_rank": rank_idx,
                                "test_quality_score": test_quality["quality_score"],
                                "test_assert_count": test_quality["assert_count"],
                                "num_evaluated": evaluated_candidates,
                            }
                        )

        print(
            f"[Distill] Progress: processed {end_idx}/{len(prompts)} prompts | "
            f"perfect={len(distilled_data)} | dpo_pairs={len(dpo_pairs)}"
        )
            
    print(f"Distillation complete. Kept {len(distilled_data)} perfect trajectories out of {len(prompts)}.")
    if skipped_without_tests:
        print(f"Skipped {skipped_without_tests} candidate generations because no hidden tests were available.")
    if skipped_low_quality_tests:
        print(
            "Skipped "
            f"{skipped_low_quality_tests} prompts due to weak test quality "
            f"(min_asserts={min_test_asserts}, min_lines={min_test_lines}, "
            f"min_quality_score={min_test_quality_score})."
        )

    if len(distilled_data) < min_perfect:
        raise RuntimeError(
            f"Too few perfect samples for reliable distillation: {len(distilled_data)} < min_perfect={min_perfect}"
        )
    
    # Speichern des destillierten, fehlerfreien Datensatzes
    if distilled_data:
        distilled_dataset = Dataset.from_list(distilled_data)
        distilled_dataset.save_to_disk(output_path)
        print(f"Saved flawless dataset to {output_path}. You can now run a brief SFT on this before GRPO.")
    else:
        print("No perfect solutions found. Check your model performance or sandbox constraints.")

    if dpo_pairs:
        dpo_dataset = Dataset.from_list(dpo_pairs)
        dpo_dataset.save_to_disk(dpo_output_path)
        print(f"Saved DPO preference pairs to {dpo_output_path}.")
    else:
        print("No usable DPO pairs were produced (missing negatives or no perfect generations).")

    if require_dpo_pairs and len(dpo_pairs) < min_dpo_pairs:
        raise RuntimeError(
            f"Too few DPO pairs: {len(dpo_pairs)} < min_dpo_pairs={min_dpo_pairs}. "
            "Increase N, prompt count, or improve SFT checkpoint quality."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Best-of-N distillation and DPO pair mining.")
    parser.add_argument("--model-path", default="qwen_sft_merged")
    parser.add_argument("--dataset-path", default="./sota_slm_coding_dataset")
    parser.add_argument("--output-path", default="./sota_best_of_n_dataset")
    parser.add_argument("--dpo-output-path", default="./sota_dpo_pairs_dataset")
    parser.add_argument("--num-prompts", type=int, default=2000)
    parser.add_argument("--num-samples-per-prompt", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--verifier-rounds", type=int, default=2)
    parser.add_argument("--min-test-asserts", type=int, default=2)
    parser.add_argument("--min-test-lines", type=int, default=3)
    parser.add_argument("--min-test-quality-score", type=float, default=2.5)
    parser.add_argument("--min-perfect", type=int, default=50)
    parser.add_argument("--min-dpo-pairs", type=int, default=50)
    parser.add_argument("--dpo-min-score-gap", type=float, default=1.0)
    parser.add_argument("--dpo-max-rejected-score", type=float, default=0.6)
    parser.add_argument("--dpo-negatives-per-prompt", type=int, default=3)
    parser.add_argument("--temperatures", default="0.2,0.8")
    parser.add_argument("--pair-weighting", default="score_gap")
    parser.add_argument("--min-evaluated-candidates", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-chars", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--allow-missing-dpo", action="store_true")
    args = parser.parse_args()

    generate_and_filter(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        dpo_output_path=args.dpo_output_path,
        num_prompts=args.num_prompts,
        num_samples_per_prompt=args.num_samples_per_prompt,
        timeout=args.timeout,
        verifier_rounds=args.verifier_rounds,
        min_test_asserts=args.min_test_asserts,
        min_test_lines=args.min_test_lines,
        min_test_quality_score=args.min_test_quality_score,
        min_perfect=args.min_perfect,
        min_dpo_pairs=args.min_dpo_pairs,
        dpo_min_score_gap=args.dpo_min_score_gap,
        dpo_max_rejected_score=args.dpo_max_rejected_score,
        dpo_negatives_per_prompt=args.dpo_negatives_per_prompt,
        temperatures=args.temperatures,
        pair_weighting=args.pair_weighting,
        min_evaluated_candidates=args.min_evaluated_candidates,
        generation_batch_size=args.generation_batch_size,
        max_prompt_chars=args.max_prompt_chars,
        seed=args.seed,
        require_dpo_pairs=not args.allow_missing_dpo,
    )
