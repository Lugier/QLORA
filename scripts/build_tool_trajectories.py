#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from typing import Dict, List

from datasets import Dataset
from verification import assess_test_quality, is_test_quality_sufficient


def _load_case_logs(paths: List[str]) -> List[Dict[str, object]]:
    rows = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _render_history(history: List[Dict[str, object]], max_rounds: int = 6) -> str:
    chunks = []
    for idx, step in enumerate(history[:max_rounds], start=1):
        err = str(step.get("error_type", "") or "")
        score = float(step.get("score", 0.0))
        stderr = str(step.get("stderr", "") or "")[:220]
        chunks.append(
            f"[Round {idx}] error={err} score={score:.3f} stderr={stderr}"
        )
    return "\n".join(chunks)


def _build_prompt(task_prompt: str, history_text: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are an autonomous SWE agent. Use tool feedback loops to repair code.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Task:\n{task_prompt}\n\n"
        "Agent Tool/Test History:\n"
        f"{history_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_text(prompt: str, code: str, history_text: str) -> str:
    code = (code or "").strip()
    reasoning = (
        "I reviewed prior failing attempts, extracted the dominant error signatures, "
        "and produced a corrected implementation consistent with test feedback.\n"
        f"History summary:\n{history_text}"
    )
    return (
        f"{prompt}"
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>\n{code}\n</answer><|im_end|>"
    )


def _dedup_key(prompt: str, tests: str) -> str:
    return hashlib.sha256(f"{prompt}\n---\n{tests}".encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build tool-use trajectory distillation dataset from agentic logs.")
    parser.add_argument("--case-log-paths", required=True, help="Comma-separated JSONL case log paths.")
    parser.add_argument("--output-path", default="./sota_tool_trajectory_dataset")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-history-rounds", type=int, default=1)
    parser.add_argument("--min-test-asserts", type=int, default=2)
    parser.add_argument("--min-test-lines", type=int, default=3)
    parser.add_argument("--min-test-quality-score", type=float, default=2.5)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include trajectories whose final solution did not pass. Off by default to avoid learning flawed end states.",
    )
    args = parser.parse_args()

    case_log_paths = [p.strip() for p in args.case_log_paths.split(",") if p.strip()]
    rows = _load_case_logs(case_log_paths)
    if not rows:
        raise RuntimeError("No case logs found for trajectory distillation.")

    samples = []
    seen = set()
    for row in rows:
        mode = str(row.get("mode", "") or "")
        if mode != "code":
            continue

        final_success = bool(row.get("all_passed", False) or row.get("pass_at_1", False))
        if not final_success and not args.include_failed:
            continue

        history = row.get("history", [])
        if not isinstance(history, list) or len(history) < args.min_history_rounds:
            continue

        prompt_task = str(row.get("prompt", "") or "").strip()
        tests = str(row.get("tests", "") or "").strip()
        code = str(row.get("result_code", "") or "").strip()
        if not prompt_task or not tests or not code:
            continue

        if not is_test_quality_sufficient(
            tests,
            min_asserts=args.min_test_asserts,
            min_nonempty_lines=args.min_test_lines,
            min_quality_score=args.min_test_quality_score,
        ):
            continue

        history_text = _render_history(history)
        prompt = _build_prompt(prompt_task, history_text)
        text = _build_text(prompt, code, history_text)

        key = _dedup_key(prompt, tests)
        if key in seen:
            continue
        seen.add(key)

        quality = assess_test_quality(tests)
        samples.append(
            {
                "text": text,
                "prompt": prompt,
                "tests": tests,
                "source": "tool_trajectory_distill",
                "final_success": final_success,
                "test_quality_score": float(quality.get("quality_score", 0.0)),
                "test_assert_count": int(quality.get("assert_count", 0)),
            }
        )
        if args.max_samples > 0 and len(samples) >= args.max_samples:
            break

    if len(samples) < args.min_samples:
        raise RuntimeError(
            f"Too few trajectory samples built: {len(samples)} < min_samples={args.min_samples}."
        )

    ds = Dataset.from_list(samples)
    ds.save_to_disk(args.output_path)
    print(f"Saved tool trajectory dataset to '{args.output_path}' with {len(ds)} samples.")


if __name__ == "__main__":
    main()
