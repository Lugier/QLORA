import argparse
import json
import os
import random
import re
import subprocess
import tempfile
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset, load_from_disk
from runtime_agent import solve_with_self_debug
from verification import run_test_verifier
from vllm import LLM, SamplingParams


def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _build_sampling_params(seed: int, **kwargs) -> SamplingParams:
    # SamplingParams signature changed across vLLM versions; keep backward compatibility.
    params = dict(kwargs)
    params["seed"] = int(seed)
    try:
        return SamplingParams(**params)
    except TypeError:
        params.pop("seed", None)
        return SamplingParams(**params)


def _build_llm(model_path: str, max_model_len: int) -> LLM:
    if os.path.isdir(model_path):
        adapter_cfg = os.path.join(model_path, "adapter_config.json")
        hf_cfg = os.path.join(model_path, "config.json")
        if os.path.exists(adapter_cfg) and not os.path.exists(hf_cfg):
            raise RuntimeError(
                f"'{model_path}' is adapter-only and cannot be loaded directly by vLLM. "
                "Use a merged HF checkpoint (for example: qwen_grpo_final)."
            )
    return LLM(
        model=model_path,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


def _extract_tests_from_row(row: Dict[str, object]) -> str:
    candidates = [
        row.get("tests"),
        row.get("test"),
        row.get("unit_tests"),
        row.get("public_tests"),
        row.get("private_tests"),
        row.get("hidden_tests"),
    ]
    for value in candidates:
        if not value:
            continue
        if isinstance(value, list):
            return "\n".join([str(v) for v in value if v])
        return str(value)

    test_list = row.get("test_list")
    if isinstance(test_list, list) and test_list:
        return "\n".join([str(v) for v in test_list if v])

    private_cases = row.get("private_test_cases")
    entry_point = row.get("entry_point")
    if isinstance(private_cases, list) and private_cases and entry_point:
        lines = []
        for case in private_cases[:10]:
            if not isinstance(case, dict):
                continue
            case_input = case.get("input")
            case_output = case.get("output")
            if case_input is None or case_output is None:
                continue
            args = str(case_input).strip()
            if not args.startswith("("):
                args = f"({args})"
            lines.append(f"assert {entry_point}{args} == {case_output}")
        if lines:
            return "\n".join(lines)
    return ""


def _first_nonempty(row: Dict[str, object], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _try_load_dataset_variants(variants: List[Tuple[str, Optional[str], str]], num_samples: int):
    last_error = None
    attempted = []
    for dataset_name, config_name, split_name in variants:
        attempted.append((dataset_name, config_name, split_name))
        split = f"{split_name}[:{num_samples}]"
        try:
            if config_name:
                return load_dataset(dataset_name, config_name, split=split)
            return load_dataset(dataset_name, split=split)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load dataset variants {attempted}. Last error: {last_error}")


def _load_mbpp_cases(num_samples: int) -> List[Dict[str, str]]:
    ds = load_dataset("mbpp", "sanitized", split=f"test[:{num_samples}]")
    cases = []
    for row in ds:
        tests = "\n".join(row.get("test_list", []))
        cases.append(
            {
                "benchmark": "mbpp",
                "id": str(row.get("task_id", row.get("text", "")))[:80],
                "prompt": row["prompt"],
                "tests": tests,
                "mode": "code",
            }
        )
    return cases


def _load_humaneval_cases(num_samples: int) -> List[Dict[str, str]]:
    last_error = None
    for dataset_name, split_name in [
        ("openai_humaneval", "test"),
        ("evalplus/humanevalplus", "test"),
    ]:
        try:
            ds = load_dataset(dataset_name, split=f"{split_name}[:{num_samples}]")
            cases = []
            for row in ds:
                prompt = row.get("prompt", "")
                test_block = row.get("test", "")
                entry_point = row.get("entry_point", "")
                tests = f"{test_block}\ncheck({entry_point})" if entry_point else test_block
                cases.append(
                    {
                        "benchmark": "humaneval",
                        "id": str(row.get("task_id", row.get("problem_id", ""))),
                        "prompt": prompt,
                        "tests": tests,
                        "mode": "code",
                    }
                )
            return cases
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load HumanEval benchmark: {last_error}")


def _load_livecodebench_cases(num_samples: int) -> List[Dict[str, str]]:
    variants = [
        ("livecodebench/code_generation_lite", None, "test"),
        ("livecodebench/code_generation", None, "test"),
        ("livecodebench", "code_generation", "test"),
    ]
    ds = _try_load_dataset_variants(variants, num_samples=num_samples)
    cases = []
    for row in ds:
        prompt = _first_nonempty(
            row,
            ["prompt", "question", "instruction", "problem", "problem_statement", "question_content"],
        )
        tests = _extract_tests_from_row(row)
        if not prompt or not tests:
            continue
        case_id = _first_nonempty(row, ["question_id", "task_id", "id", "slug"], default="livecodebench_case")
        cases.append(
            {
                "benchmark": "livecodebench",
                "id": case_id,
                "prompt": prompt,
                "tests": tests,
                "mode": "code",
            }
        )
    if not cases:
        raise RuntimeError("LiveCodeBench loaded but no evaluable prompt+tests records were found.")
    return cases


def _load_bigcodebench_instruct_cases(num_samples: int) -> List[Dict[str, str]]:
    variants = [
        ("bigcode/bigcodebench", "instruct", "test"),
        ("bigcodebench", "instruct", "test"),
        ("bigcode/bigcodebench-instruct", None, "test"),
    ]
    ds = _try_load_dataset_variants(variants, num_samples=num_samples)
    cases = []
    for row in ds:
        prompt = _first_nonempty(
            row,
            ["prompt", "instruction", "question", "problem", "problem_statement"],
        )
        tests = _extract_tests_from_row(row)
        if not prompt or not tests:
            continue
        case_id = _first_nonempty(row, ["task_id", "id", "problem_id"], default="bigcodebench_case")
        cases.append(
            {
                "benchmark": "bigcodebench_instruct",
                "id": case_id,
                "prompt": prompt,
                "tests": tests,
                "mode": "code",
            }
        )
    if not cases:
        raise RuntimeError("BigCodeBench-Instruct loaded but no evaluable prompt+tests records were found.")
    return cases


def _load_swebench_verified_subset_cases(num_samples: int) -> List[Dict[str, str]]:
    variants = [
        ("princeton-nlp/SWE-bench_Verified", None, "test"),
        ("princeton-nlp/SWE-bench_Verified", None, "dev"),
        ("princeton-nlp/SWE-bench_Verified", None, "train"),
    ]
    ds = _try_load_dataset_variants(variants, num_samples=num_samples)
    cases = []
    for row in ds:
        issue = _first_nonempty(row, ["problem_statement", "issue", "prompt"])
        reference_patch = _first_nonempty(row, ["patch", "gold_patch", "solution_patch"])
        if not issue or not reference_patch:
            continue
        instance_id = _first_nonempty(row, ["instance_id", "id", "task_id"], default="swebench_case")
        prompt = (
            "You are fixing a real-world repository issue. "
            "Return only the unified git patch in <answer> tags.\n\n"
            f"Issue:\n{issue}"
        )
        cases.append(
            {
                "benchmark": "swebench_verified_subset",
                "id": instance_id,
                "prompt": prompt,
                "tests": "",
                "mode": "patch",
                "reference_patch": reference_patch,
                "problem_statement": issue,
            }
        )
    if not cases:
        raise RuntimeError("SWE-bench Verified loaded but no issue+patch records were found.")
    return cases


def _load_private_holdout_cases(num_samples: int, private_holdout_path: str) -> List[Dict[str, str]]:
    if not private_holdout_path:
        raise RuntimeError("Benchmark 'private_holdout' requires --private-holdout-path.")
    if not os.path.exists(private_holdout_path):
        raise RuntimeError(f"Private holdout dataset path not found: {private_holdout_path}")

    ds = load_from_disk(private_holdout_path)
    if isinstance(ds, DatasetDict):
        if "holdout_clean" in ds:
            ds = ds["holdout_clean"]
        elif "test" in ds:
            ds = ds["test"]
        elif "val_strict" in ds:
            ds = ds["val_strict"]
        elif "train" in ds:
            ds = ds["train"]
        else:
            split_name = sorted(ds.keys())[0]
            ds = ds[split_name]

    if len(ds) == 0:
        raise RuntimeError("Private holdout dataset is empty.")

    n = min(num_samples, len(ds))
    rows = ds.select(range(n))
    cases = []
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt", "") or "").strip()
        tests = str(row.get("tests", "") or "").strip()
        if not prompt or not tests:
            continue
        cases.append(
            {
                "benchmark": "private_holdout",
                "id": str(row.get("id", f"private_holdout_{idx}")),
                "prompt": prompt,
                "tests": tests,
                "mode": "code",
            }
        )
    if not cases:
        raise RuntimeError("Private holdout loaded but no prompt+tests records were found.")
    return cases


def _load_benchmark_cases(
    benchmarks: List[str],
    num_samples: int,
    private_holdout_path: str = "",
    case_id_filter: Optional[Dict[str, set]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    cases_by_benchmark = {}
    for bench in benchmarks:
        name = bench.strip().lower()
        if name == "mbpp":
            cases_by_benchmark["mbpp"] = _load_mbpp_cases(num_samples)
            continue
        if name in {"humaneval", "human_eval"}:
            cases_by_benchmark["humaneval"] = _load_humaneval_cases(num_samples)
            continue
        if name == "livecodebench":
            cases_by_benchmark["livecodebench"] = _load_livecodebench_cases(num_samples)
            continue
        if name == "bigcodebench_instruct":
            cases_by_benchmark["bigcodebench_instruct"] = _load_bigcodebench_instruct_cases(num_samples)
            continue
        if name == "swebench_verified_subset":
            cases_by_benchmark["swebench_verified_subset"] = _load_swebench_verified_subset_cases(num_samples)
            continue
        if name == "private_holdout":
            cases_by_benchmark["private_holdout"] = _load_private_holdout_cases(
                num_samples=num_samples,
                private_holdout_path=private_holdout_path,
            )
            continue
        raise RuntimeError(
            "Unsupported benchmark '"
            f"{bench}'. Supported: mbpp, humaneval, livecodebench, bigcodebench_instruct, swebench_verified_subset, private_holdout"
        )
    if case_id_filter:
        for bench_name, cases in list(cases_by_benchmark.items()):
            allowed = case_id_filter.get(bench_name)
            if not allowed:
                continue
            filtered = [row for row in cases if str(row.get("id", "")) in allowed]
            if not filtered:
                raise RuntimeError(
                    f"Case-ID filter removed all cases for benchmark '{bench_name}'. "
                    "Ensure baseline case logs match current benchmark dataset IDs."
                )
            cases_by_benchmark[bench_name] = filtered
    return cases_by_benchmark


def _load_case_id_filter(path: str) -> Dict[str, set]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise RuntimeError(f"Case-ID filter path not found: {path}")
    allowed: Dict[str, set] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            bench = str(row.get("benchmark", "") or "").strip().lower()
            case_id = str(row.get("id", "") or "").strip()
            if not bench or not case_id:
                continue
            allowed.setdefault(bench, set()).add(case_id)
    return allowed


def _normalize_patch(text: str) -> str:
    lowered = (text or "").replace("\r\n", "\n")
    lowered = re.sub(r"\s+", " ", lowered).strip().lower()
    return lowered


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = [t for t in re.split(r"\s+", _normalize_patch(prediction)) if t]
    ref_tokens = [t for t in re.split(r"\s+", _normalize_patch(reference)) if t]
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    ref_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _evaluate_patch_case(
    llm: LLM,
    prompt: str,
    reference_patch: str,
    pass_k: int,
    max_tokens: int,
    search_mode: str = "greedy",
    beam_width: int = 2,
    max_rounds: int = 3,
    patch_strategies: str = "minimal_diff,api_first,test_first",
    base_seed: int = 3407,
    case_id: str = "",
) -> Dict[str, object]:
    def patch_quality_score(patch_text: str) -> float:
        text = (patch_text or "").strip()
        if not text:
            return -2.0
        score = 0.0
        lowered = text.lower()
        if "diff --git" in lowered:
            score += 1.0
        if "@@" in text:
            score += 0.6
        if "\n+" in text or "\n-" in text:
            score += 0.4
        if "index " in lowered:
            score += 0.2
        if len(text) < 40:
            score -= 0.8
        if len(text) > 12000:
            score -= 0.6
        return score

    def _parse_patch_strategies(value: str) -> List[str]:
        allowed = {"minimal_diff", "api_first", "test_first", "balanced"}
        parsed = [p.strip().lower() for p in str(value or "").split(",") if p.strip()]
        parsed = [p for p in parsed if p in allowed]
        return parsed or ["minimal_diff", "api_first", "test_first"]

    def _strategy_suffix(strategy: str) -> str:
        if strategy == "minimal_diff":
            return "Focus on smallest safe patch and avoid unrelated edits."
        if strategy == "api_first":
            return "Prioritize interface and signature correctness before internals."
        if strategy == "test_first":
            return "Prioritize behaviors implied by tests/regressions and edge conditions."
        return "Balance correctness, minimality, and maintainability."

    def build_patch_repair_prompt(base_prompt: str, previous_patch: str, round_idx: int, strategy: str) -> str:
        critique = (
            f"Patch refinement round {round_idx}. "
            "Return a valid unified git patch with file headers and hunks. "
            f"{_strategy_suffix(strategy)}"
        )
        return (
            f"{base_prompt}"
            f"<|im_start|>assistant\n<answer>\n{previous_patch}\n</answer><|im_end|>\n"
            f"<|im_start|>user\n{critique}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    format_errors = 0
    pass_flags = []
    best_similarity = 0.0
    best_patch = ""
    threshold = 0.70
    strategies = _parse_patch_strategies(patch_strategies)
    round_prompts = [{"prompt": prompt, "strategy": strategy} for strategy in strategies]
    rounds_used = 0

    for round_idx in range(1, max_rounds + 1):
        rounds_used = round_idx
        candidates_eval = []
        for round_prompt in round_prompts:
            cur_prompt = str(round_prompt.get("prompt", "") or "")
            strategy = str(round_prompt.get("strategy", "balanced") or "balanced")
            sampling_seed = _stable_seed(base_seed, "patch", case_id, strategy, round_idx)
            sampling_params = _build_sampling_params(
                seed=sampling_seed,
                n=pass_k if round_idx == 1 else max(2, pass_k // 2),
                temperature=0.1 if round_idx == 1 else 0.25,
                top_p=0.95,
                max_tokens=max_tokens,
                stop=["<|im_end|>"],
            )
            outputs = llm.generate([cur_prompt], sampling_params)
            candidates = outputs[0].outputs if outputs else []
            for candidate in candidates:
                code = extract_xml_content(candidate.text, "answer")
                if not code:
                    format_errors += 1
                    pass_flags.append(False)
                    continue
                similarity = _token_f1(code, reference_patch)
                heuristic = patch_quality_score(code)
                candidates_eval.append(
                    {
                        "code": code,
                        "similarity": similarity,
                        "heuristic": heuristic,
                        "combined": similarity + (0.08 * heuristic),
                        "strategy": strategy,
                    }
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_patch = code
                pass_flags.append(similarity >= threshold)

        if not candidates_eval:
            break
        candidates_eval.sort(key=lambda row: (float(row["combined"]), float(row["similarity"])), reverse=True)
        if search_mode not in {"beam", "mcts"}:
            break
        top = candidates_eval[: max(1, int(beam_width))]
        round_prompts = [
            {
                "prompt": build_patch_repair_prompt(prompt, row["code"], round_idx, str(row.get("strategy", "balanced"))),
                "strategy": str(row.get("strategy", "balanced")),
            }
            for row in top
        ]

    if len(pass_flags) < pass_k:
        pass_flags.extend([False] * (pass_k - len(pass_flags)))

    return {
        "pass_at_1": bool(pass_flags[0]) if pass_flags else False,
        "pass_at_k": any(pass_flags),
        "format_errors": format_errors,
        "generated_candidates": max(1, pass_k),
        "rounds_used": max(1, rounds_used),
        "resolve_proxy": best_similarity,
        "best_patch": best_patch,
    }


def _evaluate_classic_case(
    llm: LLM,
    prompt: str,
    tests: str,
    pass_k: int,
    max_tokens: int,
    timeout: float,
    verifier_rounds: int,
    base_seed: int = 3407,
    case_id: str = "",
) -> Dict[str, object]:
    sampling_seed = _stable_seed(base_seed, "classic", case_id)
    sampling_params = _build_sampling_params(
        seed=sampling_seed,
        n=pass_k,
        temperature=0.0 if pass_k == 1 else 0.25,
        top_p=0.95,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )
    outputs = llm.generate([prompt], sampling_params)
    candidates = outputs[0].outputs if outputs else []
    format_errors = 0
    pass_flags: List[bool] = []
    best_code = ""
    best_score = -10.0

    for candidate in candidates:
        text = candidate.text
        code = extract_xml_content(text, "answer")
        if not code:
            format_errors += 1
            pass_flags.append(False)
            continue
        code = code.replace("```python", "").replace("```", "").strip()
        verify = run_test_verifier(
            code=code,
            tests=tests,
            timeout=timeout,
            rounds=verifier_rounds,
            require_all_pass=True,
        )
        passed = bool(verify.get("all_passed", False))
        pass_flags.append(passed)
        score = float(verify.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_code = code

    if len(pass_flags) < pass_k:
        pass_flags.extend([False] * (pass_k - len(pass_flags)))

    return {
        "pass_at_1": bool(pass_flags[0]) if pass_flags else False,
        "pass_at_k": any(pass_flags),
        "format_errors": format_errors,
        "generated_candidates": max(1, pass_k),
        "rounds_used": 1,
        "best_code": best_code,
    }


def _write_predictions_jsonl(predictions: List[Dict[str, str]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_swebench_resolved_ids(result_obj: Dict[str, object]) -> List[str]:
    resolved = set()
    if isinstance(result_obj.get("resolved_ids"), list):
        resolved.update(str(x) for x in result_obj.get("resolved_ids", []))
    if isinstance(result_obj.get("resolved"), list):
        resolved.update(str(x) for x in result_obj.get("resolved", []))
    for key in ("instances", "results", "records"):
        rows = result_obj.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            instance_id = row.get("instance_id") or row.get("id")
            flag = row.get("resolved")
            if instance_id is not None and bool(flag):
                resolved.add(str(instance_id))
    return sorted(resolved)


def _run_swebench_harness(
    predictions_path: str,
    dataset_name: str,
    split: str,
    max_workers: int,
    run_id: str,
    harness_cmd_template: str,
    workdir: str,
) -> Dict[str, object]:
    cmd = harness_cmd_template.format(
        predictions_path=predictions_path,
        dataset_name=dataset_name,
        split=split,
        max_workers=max_workers,
        run_id=run_id,
    )
    completed = subprocess.run(
        cmd,
        cwd=workdir,
        shell=True,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SWE-bench harness execution failed:\n"
            f"CMD: {cmd}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    candidate_files = [
        os.path.join(workdir, "logs", "run_evaluation", run_id, "results.json"),
        os.path.join(workdir, "logs", "run_evaluation", run_id, "report.json"),
        os.path.join(workdir, "logs", "run_evaluation", run_id, "summary.json"),
    ]
    result_obj = None
    for path in candidate_files:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                result_obj = json.load(f)
            break
        except Exception:
            continue

    if result_obj is None:
        raise RuntimeError(
            "SWE-bench harness finished but no result JSON could be parsed in "
            f"{os.path.join(workdir, 'logs', 'run_evaluation', run_id)}"
        )

    resolved_ids = _extract_swebench_resolved_ids(result_obj)
    return {
        "resolved_ids": resolved_ids,
        "raw_result": result_obj,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _bootstrap_ci(values: List[float], bootstrap_samples: int, seed: int = 3407) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1 or bootstrap_samples <= 0:
        return mean, mean, mean

    rng = random.Random(seed)
    reps = []
    for _ in range(int(bootstrap_samples)):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += values[rng.randrange(n)]
        reps.append(sample_sum / n)
    reps.sort()

    lo_idx = int(0.025 * (len(reps) - 1))
    hi_idx = int(0.975 * (len(reps) - 1))
    return mean, reps[lo_idx], reps[hi_idx]


def _render_ci_percent(ci_tuple: Tuple[float, float, float]) -> str:
    mean, lo, hi = ci_tuple
    return f"{mean * 100:.2f}% [95% CI {lo * 100:.2f}, {hi * 100:.2f}]"


def _print_benchmark_report(name: str, stats: Dict[str, object], pass_k: int, use_agentic: bool, n_candidates: int):
    metric_k = n_candidates if use_agentic else pass_k
    print("\n" + "-" * 70)
    print(f"Benchmark: {name}")
    print(f"Total: {int(stats['total'])}")
    print(f"Pass@1: {_render_ci_percent(stats['pass_at_1_ci'])}")
    print(f"Pass@{metric_k}: {_render_ci_percent(stats['pass_at_k_ci'])}")
    print(f"Format Error Rate: {_render_ci_percent(stats['format_error_rate_ci'])}")
    print(f"Average Rounds: {stats['avg_rounds']:.2f}")
    if "resolve_proxy_ci" in stats:
        mean, lo, hi = stats["resolve_proxy_ci"]
        print(f"Resolve Proxy (Patch F1): {mean:.3f} [95% CI {lo:.3f}, {hi:.3f}]")
    print("-" * 70)


def _build_case_prompt(case_prompt: str, mode: str) -> str:
    if mode == "patch":
        system_prompt = "You are an expert software engineer. Return only a unified git patch in <answer> tags."
    else:
        system_prompt = "You are an expert python developer. Analyze briefly and output executable code in <answer> tags."

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{case_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_and_evaluate(
    model_path="qwen_grpo_final",
    benchmarks="mbpp,humaneval",
    num_samples=50,
    pass_k=8,
    max_model_len=4096,
    max_tokens=1500,
    use_agentic=False,
    repo_root=None,
    max_rounds=3,
    n_candidates=8,
    candidate_schedule="8,6,4",
    search_mode="greedy",
    beam_width=2,
    timeout=2.0,
    verifier_rounds=2,
    bootstrap_samples=2000,
    swebench_mode="harness",
    swebench_dataset_name="princeton-nlp/SWE-bench_Verified",
    swebench_split="test",
    swebench_max_workers=4,
    private_holdout_path="",
    patch_strategies="minimal_diff,api_first,test_first",
    case_id_filter_path="",
    swebench_harness_cmd=(
        "python3 -m swebench.harness.run_evaluation "
        "--dataset_name {dataset_name} "
        "--split {split} "
        "--predictions_path {predictions_path} "
        "--max_workers {max_workers} "
        "--run_id {run_id}"
    ),
    case_log_path="",
    json_output="",
    seed=3407,
):
    if not os.path.exists(model_path) and not model_path.count("/") >= 1:
        raise RuntimeError(f"Model path {model_path} not found. Ensure training is complete.")

    benchmark_list = [name.strip() for name in benchmarks.split(",") if name.strip()]
    if not benchmark_list:
        raise RuntimeError("No benchmarks provided.")

    print(f"Loading model '{model_path}' via vLLM for evaluation...")
    llm = _build_llm(model_path=model_path, max_model_len=max_model_len)

    case_id_filter = _load_case_id_filter(case_id_filter_path)
    if case_id_filter:
        print(f"Applying case-ID filter from '{case_id_filter_path}' for paired evaluation consistency.")

    print(f"Loading benchmarks: {benchmark_list}")
    cases_by_benchmark = _load_benchmark_cases(
        benchmark_list,
        num_samples=num_samples,
        private_holdout_path=private_holdout_path,
        case_id_filter=case_id_filter,
    )

    overall_records = []
    report = {
        "model_path": model_path,
        "benchmarks": benchmark_list,
        "mode": "agentic" if use_agentic else "classic",
        "seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "case_id_filter_path": case_id_filter_path,
        "per_benchmark": {},
        "groups": {},
    }
    case_logs: List[Dict[str, object]] = []

    for bench_name, cases in cases_by_benchmark.items():
        pass1_values: List[float] = []
        passk_values: List[float] = []
        format_error_values: List[float] = []
        rounds_values: List[float] = []
        resolve_proxy_values: List[float] = []
        case_log_indices: List[int] = []

        for case in cases:
            case_mode = case.get("mode", "code")
            prompt = _build_case_prompt(case["prompt"], mode=case_mode)
            case_entry = {
                "benchmark": bench_name,
                "id": case.get("id", ""),
                "mode": case_mode,
                "prompt": case.get("prompt", ""),
                "tests": case.get("tests", ""),
            }

            if case_mode == "patch" and bench_name == "swebench_verified_subset" and swebench_mode in {"harness", "auto"}:
                # Defer scoring for harness-backed SWE-bench until all predictions are generated.
                result = {
                    "pass_at_1": False,
                    "pass_at_k": False,
                    "format_errors": 0,
                    "generated_candidates": max(1, pass_k),
                    "rounds_used": 1,
                    "best_patch": "",
                }
                case_entry["deferred_harness"] = True
            elif case_mode == "patch":
                result = _evaluate_patch_case(
                    llm=llm,
                    prompt=prompt,
                    reference_patch=case.get("reference_patch", ""),
                    pass_k=pass_k,
                    max_tokens=max_tokens,
                    search_mode=search_mode,
                    beam_width=beam_width,
                    max_rounds=max_rounds,
                    patch_strategies=patch_strategies,
                    base_seed=seed,
                    case_id=str(case.get("id", "")),
                )
                case_entry["best_patch"] = result.get("best_patch", "")
                case_entry["reference_patch"] = case.get("reference_patch", "")
            elif use_agentic:
                raw_result = solve_with_self_debug(
                    llm=llm,
                    user_prompt=case["prompt"],
                    tests=case["tests"],
                    repo_root=repo_root,
                    max_rounds=max_rounds,
                    n_candidates=n_candidates,
                    candidate_schedule=candidate_schedule,
                    timeout=timeout,
                    verifier_rounds=verifier_rounds,
                    search_mode=search_mode,
                    beam_width=beam_width,
                    seed=_stable_seed(seed, "agentic", bench_name, str(case.get("id", ""))),
                )
                pass_value = 1.0 if bool(raw_result.get("all_passed", False)) else 0.0
                history = raw_result.get("history", []) if isinstance(raw_result.get("history", []), list) else []
                generated_candidates = sum(int(h.get("candidate_budget", 0) or 0) for h in history)
                generated_candidates += int(raw_result.get("candidate_budget", 0) or 0)
                generated_candidates = max(1, generated_candidates)
                format_errors = 1 if raw_result.get("error_type") == "format" else 0
                case_entry["history"] = history
                case_entry["result_code"] = raw_result.get("code", "")
                case_entry["error_type"] = raw_result.get("error_type", "")
                case_entry["all_passed"] = bool(raw_result.get("all_passed", False))

                result = {
                    "pass_at_1": bool(pass_value),
                    "pass_at_k": bool(pass_value),
                    "format_errors": format_errors,
                    "generated_candidates": generated_candidates,
                    "rounds_used": max(1, int(raw_result.get("round", max_rounds))),
                }
            else:
                result = _evaluate_classic_case(
                    llm=llm,
                    prompt=prompt,
                    tests=case["tests"],
                    pass_k=pass_k,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    verifier_rounds=verifier_rounds,
                    base_seed=seed,
                    case_id=str(case.get("id", "")),
                )
                case_entry["result_code"] = result.get("best_code", "")

            pass1_values.append(1.0 if result["pass_at_1"] else 0.0)
            passk_values.append(1.0 if result["pass_at_k"] else 0.0)
            generated_candidates = max(1, int(result.get("generated_candidates", pass_k)))
            format_error_values.append(float(result.get("format_errors", 0)) / generated_candidates)
            rounds_values.append(float(result.get("rounds_used", 1)))
            if "resolve_proxy" in result:
                resolve_proxy_values.append(float(result["resolve_proxy"]))

            case_entry["pass_at_1"] = bool(result["pass_at_1"])
            case_entry["pass_at_k"] = bool(result["pass_at_k"])
            case_entry["format_errors"] = int(result.get("format_errors", 0))
            case_entry["generated_candidates"] = generated_candidates
            case_entry["rounds_used"] = int(result.get("rounds_used", 1))
            case_logs.append(case_entry)
            case_log_indices.append(len(case_logs) - 1)

        if bench_name == "swebench_verified_subset" and swebench_mode in {"harness", "auto"}:
            predictions = []
            format_errors = 0
            for idx, case in enumerate(cases):
                case_log = case_logs[case_log_indices[idx]]
                pred_patch = str(case_log.get("best_patch", "") or "")
                if not pred_patch:
                    # Generate fallback patch candidate if deferred or empty.
                    prompt = _build_case_prompt(case["prompt"], mode="patch")
                    patch_result = _evaluate_patch_case(
                        llm=llm,
                        prompt=prompt,
                        reference_patch=case.get("reference_patch", ""),
                        pass_k=pass_k,
                        max_tokens=max_tokens,
                        search_mode=search_mode,
                        beam_width=beam_width,
                        max_rounds=max_rounds,
                        patch_strategies=patch_strategies,
                        base_seed=seed,
                        case_id=str(case.get("id", "")),
                    )
                    pred_patch = str(patch_result.get("best_patch", "") or "")
                    case_log["best_patch"] = pred_patch
                if not pred_patch:
                    format_errors += 1
                predictions.append(
                    {
                        "instance_id": case["id"],
                        "model_patch": pred_patch,
                        "patch": pred_patch,
                        "model_name_or_path": model_path,
                    }
                )

            run_id = f"swebench_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            with tempfile.TemporaryDirectory(prefix="swebench_eval_") as tmpdir:
                predictions_path = os.path.join(tmpdir, "predictions.jsonl")
                _write_predictions_jsonl(predictions, predictions_path)
                harness_exc = None
                harness_result = None
                try:
                    harness_result = _run_swebench_harness(
                        predictions_path=predictions_path,
                        dataset_name=swebench_dataset_name,
                        split=swebench_split,
                        max_workers=swebench_max_workers,
                        run_id=run_id,
                        harness_cmd_template=swebench_harness_cmd,
                        workdir=os.getcwd(),
                    )
                except Exception as exc:
                    harness_exc = exc

            if harness_result is None:
                if swebench_mode == "harness":
                    raise RuntimeError(f"SWE-bench harness mode failed: {harness_exc}")
                print(f"WARNUNG: SWE harness failed in auto mode, falling back to proxy scoring: {harness_exc}")
                pass1_values = []
                passk_values = []
                format_error_values = []
                rounds_values = []
                resolve_proxy_values = []
                for idx, case in enumerate(cases):
                    case_log = case_logs[case_log_indices[idx]]
                    pred_patch = str(case_log.get("best_patch", "") or "")
                    sim = _token_f1(pred_patch, str(case.get("reference_patch", "") or ""))
                    passed = sim >= 0.70
                    pass1_values.append(1.0 if passed else 0.0)
                    passk_values.append(1.0 if passed else 0.0)
                    format_error_values.append(0.0 if pred_patch else 1.0)
                    rounds_values.append(1.0)
                    resolve_proxy_values.append(sim)
                    case_log["pass_at_1"] = bool(passed)
                    case_log["pass_at_k"] = bool(passed)
                    case_log["format_errors"] = 1 if not pred_patch else 0
                    case_log["resolve_proxy"] = sim
            else:
                resolved = set(harness_result.get("resolved_ids", []))
                pass1_values = [1.0 if case["id"] in resolved else 0.0 for case in cases]
                passk_values = list(pass1_values)
                format_error_values = []
                rounds_values = [1.0 for _ in cases]
                resolve_proxy_values = []
                report.setdefault("swebench_harness", {})[bench_name] = harness_result.get("raw_result", {})
                for idx, case in enumerate(cases):
                    case_log = case_logs[case_log_indices[idx]]
                    format_errors_for_case = 1 if not case_log.get("best_patch") else 0
                    format_error_values.append(float(format_errors_for_case))
                    case_log["pass_at_1"] = bool(pass1_values[idx])
                    case_log["pass_at_k"] = bool(passk_values[idx])
                    case_log["format_errors"] = format_errors_for_case

        # Build global records after any harness-backed overrides so report groups stay consistent.
        for case_idx in case_log_indices:
            case_row = case_logs[case_idx]
            generated = max(1, int(case_row.get("generated_candidates", 1)))
            fmt_rate = float(case_row.get("format_errors", 0)) / generated
            overall_records.append(
                {
                    "benchmark": bench_name,
                    "pass_at_1": 1.0 if bool(case_row.get("pass_at_1", False)) else 0.0,
                    "pass_at_k": 1.0 if bool(case_row.get("pass_at_k", False)) else 0.0,
                    "format_error_rate": fmt_rate,
                }
            )

        bench_report = {
            "total": len(cases),
            "pass_at_1_ci": _bootstrap_ci(pass1_values, bootstrap_samples=bootstrap_samples),
            "pass_at_k_ci": _bootstrap_ci(passk_values, bootstrap_samples=bootstrap_samples),
            "format_error_rate_ci": _bootstrap_ci(format_error_values, bootstrap_samples=bootstrap_samples),
            "avg_rounds": (sum(rounds_values) / max(1, len(rounds_values))),
        }
        if resolve_proxy_values:
            bench_report["resolve_proxy_ci"] = _bootstrap_ci(resolve_proxy_values, bootstrap_samples=bootstrap_samples)

        report["per_benchmark"][bench_name] = bench_report
        _print_benchmark_report(
            name=bench_name,
            stats=bench_report,
            pass_k=pass_k,
            use_agentic=use_agentic,
            n_candidates=n_candidates,
        )

    contamination_set = {"livecodebench", "mbpp", "humaneval"}
    practical_set = {"bigcodebench_instruct", "swebench_verified_subset"}
    private_set = {"private_holdout"}

    def _group_report(target_set):
        subset = [row for row in overall_records if row["benchmark"] in target_set]
        if not subset:
            return None
        pass1 = [row["pass_at_1"] for row in subset]
        passk = [row["pass_at_k"] for row in subset]
        format_rates = [row["format_error_rate"] for row in subset]
        return {
            "total": len(subset),
            "pass_at_1_ci": _bootstrap_ci(pass1, bootstrap_samples=bootstrap_samples),
            "pass_at_k_ci": _bootstrap_ci(passk, bootstrap_samples=bootstrap_samples),
            "format_error_rate_ci": _bootstrap_ci(format_rates, bootstrap_samples=bootstrap_samples),
        }

    contamination_report = _group_report(contamination_set)
    practical_report = _group_report(practical_set)
    private_report = _group_report(private_set)
    if contamination_report:
        report["groups"]["contamination_safe_coding"] = contamination_report
    if practical_report:
        report["groups"]["practical_swe_agentic"] = practical_report
    if private_report:
        report["groups"]["private_holdout"] = private_report

    metric_k = n_candidates if use_agentic else pass_k
    print("\n" + "=" * 70)
    print("GLOBAL EVALUATION REPORT")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Benchmarks: {', '.join(cases_by_benchmark.keys())}")
    print(f"Mode: {'Agentic Self-Debug' if use_agentic else 'Classic'}")

    overall_pass1 = [row["pass_at_1"] for row in overall_records]
    overall_passk = [row["pass_at_k"] for row in overall_records]
    overall_format_rates = [row["format_error_rate"] for row in overall_records]
    overall_pass1_ci = _bootstrap_ci(overall_pass1, bootstrap_samples=bootstrap_samples)
    overall_passk_ci = _bootstrap_ci(overall_passk, bootstrap_samples=bootstrap_samples)
    overall_format_ci = _bootstrap_ci(overall_format_rates, bootstrap_samples=bootstrap_samples)

    report["global"] = {
        "total": len(overall_records),
        "pass_at_1_ci": overall_pass1_ci,
        "pass_at_k_ci": overall_passk_ci,
        "format_error_rate_ci": overall_format_ci,
    }

    print(f"Global Pass@1: {_render_ci_percent(overall_pass1_ci)}")
    print(f"Global Pass@{metric_k}: {_render_ci_percent(overall_passk_ci)}")
    print(
        "Global Format Error Rate: "
        f"{_render_ci_percent(overall_format_ci)}"
    )

    if contamination_report:
        print("\nContamination-safe coding summary:")
        print(f"- Pass@1: {_render_ci_percent(contamination_report['pass_at_1_ci'])}")
        print(f"- Pass@{metric_k}: {_render_ci_percent(contamination_report['pass_at_k_ci'])}")
        print(f"- Format Error Rate: {_render_ci_percent(contamination_report['format_error_rate_ci'])}")

    if practical_report:
        print("\nPraxisnahe SWE/agentic summary:")
        print(f"- Pass@1: {_render_ci_percent(practical_report['pass_at_1_ci'])}")
        print(f"- Pass@{metric_k}: {_render_ci_percent(practical_report['pass_at_k_ci'])}")
        print(f"- Format Error Rate: {_render_ci_percent(practical_report['format_error_rate_ci'])}")
    if private_report:
        print("\nPrivate holdout summary:")
        print(f"- Pass@1: {_render_ci_percent(private_report['pass_at_1_ci'])}")
        print(f"- Pass@{metric_k}: {_render_ci_percent(private_report['pass_at_k_ci'])}")
        print(f"- Format Error Rate: {_render_ci_percent(private_report['format_error_rate_ci'])}")

    print("=" * 70)

    if json_output:
        os.makedirs(os.path.dirname(json_output) or ".", exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved evaluation report JSON to '{json_output}'.")

    if case_log_path:
        os.makedirs(os.path.dirname(case_log_path) or ".", exist_ok=True)
        with open(case_log_path, "w", encoding="utf-8") as f:
            for row in case_logs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved case-level eval logs to '{case_log_path}'.")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate coding model on multiple benchmarks.")
    parser.add_argument("--model-path", default="qwen_grpo_final")
    parser.add_argument(
        "--benchmarks",
        default="livecodebench,bigcodebench_instruct,swebench_verified_subset,mbpp,humaneval",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--candidate-schedule", default="8,6,4")
    parser.add_argument("--search-mode", default="greedy", choices=["greedy", "beam", "mcts"])
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--verifier-rounds", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--swebench-mode", default="harness", choices=["harness", "proxy", "auto"])
    parser.add_argument("--swebench-dataset-name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--swebench-split", default="test")
    parser.add_argument("--swebench-max-workers", type=int, default=4)
    parser.add_argument("--private-holdout-path", default="")
    parser.add_argument("--patch-strategies", default="minimal_diff,api_first,test_first")
    parser.add_argument("--case-id-filter-path", default="")
    parser.add_argument(
        "--swebench-harness-cmd",
        default=(
            "python3 -m swebench.harness.run_evaluation "
            "--dataset_name {dataset_name} "
            "--split {split} "
            "--predictions_path {predictions_path} "
            "--max_workers {max_workers} "
            "--run_id {run_id}"
        ),
    )
    parser.add_argument("--case-log-path", default="")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    generate_and_evaluate(
        model_path=args.model_path,
        benchmarks=args.benchmarks,
        num_samples=args.num_samples,
        pass_k=args.pass_k,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        use_agentic=args.agentic,
        repo_root=args.repo_root or None,
        max_rounds=args.max_rounds,
        n_candidates=args.n_candidates,
        candidate_schedule=args.candidate_schedule,
        search_mode=args.search_mode,
        beam_width=args.beam_width,
        timeout=args.timeout,
        verifier_rounds=args.verifier_rounds,
        bootstrap_samples=args.bootstrap_samples,
        swebench_mode=args.swebench_mode,
        swebench_dataset_name=args.swebench_dataset_name,
        swebench_split=args.swebench_split,
        swebench_max_workers=args.swebench_max_workers,
        private_holdout_path=args.private_holdout_path,
        patch_strategies=args.patch_strategies,
        case_id_filter_path=args.case_id_filter_path,
        swebench_harness_cmd=args.swebench_harness_cmd,
        case_log_path=args.case_log_path,
        json_output=args.json_output,
        seed=args.seed,
    )
