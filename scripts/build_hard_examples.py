#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from typing import Dict, List

from datasets import Dataset, concatenate_datasets, load_from_disk
from verification import assess_test_quality, is_test_quality_sufficient

_ERROR_SEVERITY = {
    "security": 1.00,
    "timeout": 0.90,
    "format": 0.80,
    "syntax": 0.72,
    "assertion": 0.60,
    "runtime": 0.50,
}


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


def _dedup_key(prompt: str, tests: str) -> str:
    payload = f"{prompt.strip()}\n---\n{tests.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hardness_score(row: Dict[str, object], quality: Dict[str, float]) -> float:
    error_type = str(row.get("error_type", "") or "").strip().lower()
    severity = _ERROR_SEVERITY.get(error_type, 0.45)
    rounds_used = max(1, int(row.get("rounds_used", 1) or 1))
    generated_candidates = max(1, int(row.get("generated_candidates", 1) or 1))
    test_quality = float(quality.get("quality_score", 0.0))
    # Favor failures with costly/debug-heavy dynamics and stronger tests.
    return (
        (1.35 * severity)
        + min(0.60, 0.12 * rounds_used)
        + min(0.55, 0.03 * max(0, generated_candidates - 1))
        + min(0.70, 0.05 * test_quality)
    )


def _rebalance_by_share(rows: List[Dict[str, object]], key_name: str, max_share: float) -> List[Dict[str, object]]:
    if not rows:
        return rows
    max_share = float(max_share)
    if max_share <= 0.0 or max_share >= 1.0:
        return rows

    cap = max(1, int(len(rows) * max_share))
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        key = str(row.get(key_name, "") or "unknown")
        grouped.setdefault(key, []).append(row)

    selected = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda x: float(x.get("hardness_score", 0.0)), reverse=True)
        kept = group_sorted[:cap]
        selected.extend(kept)
        if len(group_sorted) > len(kept):
            print(
                f"[hard-mining] Capped {key_name}='{key}': kept {len(kept)} of {len(group_sorted)} "
                f"(max_share={max_share:.2f})"
            )
    selected.sort(key=lambda x: float(x.get("hardness_score", 0.0)), reverse=True)
    return selected


def _build_hard_examples(
    rows: List[Dict[str, object]],
    min_asserts: int,
    min_lines: int,
    min_quality_score: float,
) -> List[Dict[str, object]]:
    built = []
    seen = set()

    for row in rows:
        if bool(row.get("pass_at_1", False)):
            continue
        mode = str(row.get("mode", "") or "")
        if mode != "code":
            continue

        prompt = str(row.get("prompt", "") or "").strip()
        tests = str(row.get("tests", "") or "").strip()
        if not prompt or not tests:
            continue

        if not is_test_quality_sufficient(
            tests,
            min_asserts=min_asserts,
            min_nonempty_lines=min_lines,
            min_quality_score=min_quality_score,
        ):
            continue

        key = _dedup_key(prompt, tests)
        if key in seen:
            continue
        seen.add(key)

        quality = assess_test_quality(tests)
        error_type = str(row.get("error_type", "") or "").strip().lower() or "unknown"
        benchmark = str(row.get("benchmark", "") or "").strip().lower() or "unknown"
        hardness = _hardness_score(row, quality)
        built.append(
            {
                "prompt": prompt,
                "text": prompt,
                "tests": tests,
                "source": "online_hard_mining",
                "benchmark": benchmark,
                "error_type": error_type,
                "hardness_score": hardness,
                "test_quality_score": float(quality.get("quality_score", 0.0)),
                "test_assert_count": int(quality.get("assert_count", 0)),
            }
        )
    return built


def _merge_with_existing(output_rows: List[Dict[str, object]], merge_path: str, max_merged_samples: int) -> Dataset:
    new_ds = Dataset.from_list(output_rows)
    if not merge_path or not os.path.exists(merge_path):
        return new_ds

    existing = load_from_disk(merge_path)
    if hasattr(existing, "keys") and "train" in existing:
        existing = existing["train"]

    # Normalize schemas so heterogeneous replay sources can be concatenated safely.
    canonical_columns = [
        "prompt",
        "text",
        "tests",
        "source",
        "benchmark",
        "error_type",
        "hardness_score",
        "test_quality_score",
        "test_assert_count",
    ]
    defaults = {
        "prompt": "",
        "text": "",
        "tests": "",
        "source": "legacy_replay",
        "benchmark": "",
        "error_type": "",
        "hardness_score": 0.0,
        "test_quality_score": 0.0,
        "test_assert_count": 0,
    }

    for col in canonical_columns:
        if col not in new_ds.column_names:
            new_ds = new_ds.add_column(col, [defaults[col]] * len(new_ds))
        if col not in existing.column_names:
            existing = existing.add_column(col, [defaults[col]] * len(existing))

    new_extras = [c for c in new_ds.column_names if c not in canonical_columns]
    if new_extras:
        new_ds = new_ds.remove_columns(new_extras)
    existing_extras = [c for c in existing.column_names if c not in canonical_columns]
    if existing_extras:
        existing = existing.remove_columns(existing_extras)

    def _normalize_row(row):
        return {
            "prompt": str(row.get("prompt", "") or "").strip(),
            "text": str(row.get("text", "") or "").strip(),
            "tests": str(row.get("tests", "") or "").strip(),
            "source": str(row.get("source", "") or "legacy_replay").strip().lower(),
            "benchmark": str(row.get("benchmark", "") or "").strip().lower(),
            "error_type": str(row.get("error_type", "") or "").strip().lower(),
            "hardness_score": float(row.get("hardness_score", 0.0) or 0.0),
            "test_quality_score": float(row.get("test_quality_score", 0.0) or 0.0),
            "test_assert_count": int(row.get("test_assert_count", 0) or 0),
        }

    new_ds = new_ds.map(_normalize_row, remove_columns=new_ds.column_names)
    existing = existing.map(_normalize_row, remove_columns=existing.column_names)

    merged = concatenate_datasets([new_ds, existing])
    if max_merged_samples > 0 and len(merged) > max_merged_samples:
        merged = merged.shuffle(seed=3407).select(range(max_merged_samples))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build hard-example replay dataset from eval case logs.")
    parser.add_argument("--case-log-paths", required=True, help="Comma-separated JSONL case log paths.")
    parser.add_argument("--output-path", default="./sota_hard_examples_dataset")
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-test-asserts", type=int, default=2)
    parser.add_argument("--min-test-lines", type=int, default=3)
    parser.add_argument("--min-test-quality-score", type=float, default=2.5)
    parser.add_argument("--max-error-type-share", type=float, default=0.45)
    parser.add_argument("--max-benchmark-share", type=float, default=0.65)
    parser.add_argument("--merge-with", default="")
    parser.add_argument("--max-merged-samples", type=int, default=20000)
    args = parser.parse_args()

    case_log_paths = [p.strip() for p in args.case_log_paths.split(",") if p.strip()]
    rows = _load_case_logs(case_log_paths)
    if not rows:
        raise RuntimeError("No case logs found for hard-example mining.")

    hard_rows = _build_hard_examples(
        rows,
        min_asserts=args.min_test_asserts,
        min_lines=args.min_test_lines,
        min_quality_score=args.min_test_quality_score,
    )
    hard_rows = _rebalance_by_share(hard_rows, key_name="error_type", max_share=args.max_error_type_share)
    hard_rows = _rebalance_by_share(hard_rows, key_name="benchmark", max_share=args.max_benchmark_share)
    if len(hard_rows) < args.min_samples:
        raise RuntimeError(
            f"Too few hard examples mined: {len(hard_rows)} < min_samples={args.min_samples}."
        )

    final_ds = _merge_with_existing(hard_rows, args.merge_with, args.max_merged_samples)
    final_ds.save_to_disk(args.output_path)
    print(
        f"Saved hard-example dataset to '{args.output_path}' "
        f"with {len(final_ds)} samples (new={len(hard_rows)})."
    )


if __name__ == "__main__":
    main()
