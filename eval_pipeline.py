import argparse
import os
import re
from typing import Dict, List, Tuple

from datasets import load_dataset
from runtime_agent import solve_with_self_debug
from verification import run_test_verifier
from vllm import LLM, SamplingParams


def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


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
                    }
                )
            return cases
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to load HumanEval benchmark: {last_error}")


def _load_benchmark_cases(benchmarks: List[str], num_samples: int) -> Dict[str, List[Dict[str, str]]]:
    cases_by_benchmark = {}
    for bench in benchmarks:
        name = bench.strip().lower()
        if name == "mbpp":
            cases_by_benchmark["mbpp"] = _load_mbpp_cases(num_samples)
            continue
        if name in {"humaneval", "human_eval"}:
            cases_by_benchmark["humaneval"] = _load_humaneval_cases(num_samples)
            continue
        raise RuntimeError(f"Unsupported benchmark '{bench}'. Supported: mbpp, humaneval")
    return cases_by_benchmark


def _evaluate_classic_case(
    llm: LLM,
    prompt: str,
    tests: str,
    pass_k: int,
    max_tokens: int,
    timeout: float,
    verifier_rounds: int,
) -> Dict[str, object]:
    sampling_params = SamplingParams(
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
        pass_flags.append(float(verify["score"]) == 2.0)

    if len(pass_flags) < pass_k:
        pass_flags.extend([False] * (pass_k - len(pass_flags)))

    return {
        "pass_at_1": bool(pass_flags[0]) if pass_flags else False,
        "pass_at_k": any(pass_flags),
        "format_errors": format_errors,
        "rounds_used": 1,
    }


def _print_benchmark_report(name: str, stats: Dict[str, float], pass_k: int, use_agentic: bool, n_candidates: int):
    metric_k = n_candidates if use_agentic else pass_k
    print("\n" + "-" * 60)
    print(f"Benchmark: {name}")
    print(f"Total: {int(stats['total'])}")
    print(f"Pass@1: {stats['pass_at_1']:.2f}%")
    print(f"Pass@{metric_k}: {stats['pass_at_k']:.2f}%")
    print(f"Format Errors: {int(stats['format_errors'])}")
    print(f"Average Rounds: {stats['avg_rounds']:.2f}")
    print("-" * 60)


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
    timeout=2.0,
    verifier_rounds=2,
):
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path {model_path} not found. Ensure training is complete.")

    benchmark_list = [name.strip() for name in benchmarks.split(",") if name.strip()]
    if not benchmark_list:
        raise RuntimeError("No benchmarks provided.")

    print(f"Loading model '{model_path}' via vLLM for evaluation...")
    llm = _build_llm(model_path=model_path, max_model_len=max_model_len)

    print(f"Loading benchmarks: {benchmark_list}")
    cases_by_benchmark = _load_benchmark_cases(benchmark_list, num_samples=num_samples)

    system_prompt = (
        "You are an expert python developer. Analyze briefly and output executable code in <answer> tags."
    )

    overall = {
        "total": 0,
        "pass_at_1": 0,
        "pass_at_k": 0,
        "format_errors": 0,
        "rounds": 0,
    }

    for bench_name, cases in cases_by_benchmark.items():
        stats = {
            "total": 0,
            "pass_at_1": 0,
            "pass_at_k": 0,
            "format_errors": 0,
            "rounds": 0,
        }
        for case in cases:
            tests = case["tests"]
            if use_agentic:
                result = solve_with_self_debug(
                    llm=llm,
                    user_prompt=case["prompt"],
                    tests=tests,
                    repo_root=repo_root,
                    max_rounds=max_rounds,
                    n_candidates=n_candidates,
                    timeout=timeout,
                    verifier_rounds=verifier_rounds,
                )
                passed = float(result.get("score", 0.0)) == 2.0
                stats["pass_at_1"] += int(passed)
                stats["pass_at_k"] += int(passed)
                if result.get("error_type") == "format":
                    stats["format_errors"] += 1
                stats["rounds"] += int(result.get("round", max_rounds))
                stats["total"] += 1
                continue

            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{case['prompt']}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            result = _evaluate_classic_case(
                llm=llm,
                prompt=prompt,
                tests=tests,
                pass_k=pass_k,
                max_tokens=max_tokens,
                timeout=timeout,
                verifier_rounds=verifier_rounds,
            )
            stats["pass_at_1"] += int(result["pass_at_1"])
            stats["pass_at_k"] += int(result["pass_at_k"])
            stats["format_errors"] += int(result["format_errors"])
            stats["rounds"] += int(result["rounds_used"])
            stats["total"] += 1

        total = max(1, stats["total"])
        stats_percent = {
            "total": stats["total"],
            "pass_at_1": (stats["pass_at_1"] / total) * 100.0,
            "pass_at_k": (stats["pass_at_k"] / total) * 100.0,
            "format_errors": stats["format_errors"],
            "avg_rounds": stats["rounds"] / total,
        }
        _print_benchmark_report(
            name=bench_name,
            stats=stats_percent,
            pass_k=pass_k,
            use_agentic=use_agentic,
            n_candidates=n_candidates,
        )

        overall["total"] += stats["total"]
        overall["pass_at_1"] += stats["pass_at_1"]
        overall["pass_at_k"] += stats["pass_at_k"]
        overall["format_errors"] += stats["format_errors"]
        overall["rounds"] += stats["rounds"]

    total = max(1, overall["total"])
    overall_pass1 = (overall["pass_at_1"] / total) * 100.0
    overall_passk = (overall["pass_at_k"] / total) * 100.0
    overall_avg_rounds = overall["rounds"] / total

    metric_k = n_candidates if use_agentic else pass_k
    print("\n" + "=" * 60)
    print("GLOBAL EVALUATION REPORT")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Benchmarks: {', '.join(cases_by_benchmark.keys())}")
    print(f"Mode: {'Agentic Self-Debug' if use_agentic else 'Classic'}")
    print(f"Total Samples: {overall['total']}")
    print(f"Global Pass@1: {overall_pass1:.2f}%")
    print(f"Global Pass@{metric_k}: {overall_passk:.2f}%")
    print(f"Global Format Errors: {overall['format_errors']}")
    print(f"Global Average Rounds: {overall_avg_rounds:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate coding model on multiple benchmarks.")
    parser.add_argument("--model-path", default="qwen_grpo_final")
    parser.add_argument("--benchmarks", default="mbpp,humaneval")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--verifier-rounds", type=int, default=2)
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
        timeout=args.timeout,
        verifier_rounds=args.verifier_rounds,
    )
