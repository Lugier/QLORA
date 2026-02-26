#!/usr/bin/env python3
import argparse
import json
import os
import random
from typing import Dict, List, Tuple


def _load_json(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise RuntimeError(f"Required JSON file is missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise RuntimeError(f"Required JSONL file is missing: {path}")
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"JSONL file is empty: {path}")
    return rows


def _mean_ci(values: List[float], bootstrap_samples: int, seed: int) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1 or bootstrap_samples <= 0:
        return mean, mean, mean
    rng = random.Random(seed)
    reps = []
    for _ in range(int(bootstrap_samples)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        reps.append(sum(sample) / n)
    reps.sort()
    lo = reps[int(0.025 * (len(reps) - 1))]
    hi = reps[int(0.975 * (len(reps) - 1))]
    return mean, lo, hi


def _mean_ci_from_report(report: Dict[str, object], benchmark: str, metric: str) -> Tuple[float, float, float]:
    block = report.get("per_benchmark", {}).get(benchmark, {})
    values = block.get(metric)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Missing metric '{metric}' for benchmark '{benchmark}'.")
    if len(values) >= 3:
        return float(values[0]), float(values[1]), float(values[2])
    value = float(values[0])
    return value, value, value


def _global_ci_from_report(report: Dict[str, object], metric: str) -> Tuple[float, float, float]:
    block = report.get("global", {})
    values = block.get(metric)
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"Missing global metric '{metric}'.")
    if len(values) >= 3:
        return float(values[0]), float(values[1]), float(values[2])
    value = float(values[0])
    return value, value, value


def _primary_score_ci(classic: Dict[str, object], agentic: Dict[str, object]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    lcb = _mean_ci_from_report(classic, "livecodebench", "pass_at_1_ci")
    bcb = _mean_ci_from_report(classic, "bigcodebench_instruct", "pass_at_1_ci")
    swe = _mean_ci_from_report(classic, "swebench_verified_subset", "pass_at_1_ci")
    agent = _global_ci_from_report(agentic, "pass_at_1_ci")
    fmt = _global_ci_from_report(classic, "format_error_rate_ci")

    mean = (0.35 * lcb[0]) + (0.25 * bcb[0]) + (0.20 * swe[0]) + (0.10 * agent[0]) + (0.10 * (1.0 - fmt[0]))
    lo = (0.35 * lcb[1]) + (0.25 * bcb[1]) + (0.20 * swe[1]) + (0.10 * agent[1]) + (0.10 * (1.0 - fmt[2]))
    hi = (0.35 * lcb[2]) + (0.25 * bcb[2]) + (0.20 * swe[2]) + (0.10 * agent[2]) + (0.10 * (1.0 - fmt[1]))
    return (mean, lo, hi), fmt


def _row_key(row: Dict[str, object]) -> str:
    return f"{row.get('benchmark', '')}::{row.get('mode', '')}::{row.get('id', '')}"


def _format_error_rate(row: Dict[str, object]) -> float:
    generated = max(1, int(row.get("generated_candidates", 1) or 1))
    return float(row.get("format_errors", 0) or 0) / generated


def _align_diffs(
    base_rows: List[Dict[str, object]],
    post_rows: List[Dict[str, object]],
    metric: str,
    benchmark: str = "",
) -> List[float]:
    post_map = {_row_key(row): row for row in post_rows}
    diffs: List[float] = []
    for base in base_rows:
        if benchmark and str(base.get("benchmark", "")) != benchmark:
            continue
        key = _row_key(base)
        post = post_map.get(key)
        if post is None:
            continue
        if metric == "pass":
            base_v = 1.0 if bool(base.get("pass_at_1", False)) else 0.0
            post_v = 1.0 if bool(post.get("pass_at_1", False)) else 0.0
        elif metric == "format":
            base_v = _format_error_rate(base)
            post_v = _format_error_rate(post)
        else:
            raise RuntimeError(f"Unsupported metric: {metric}")
        diffs.append(post_v - base_v)
    return diffs


def _paired_objective_diffs(
    base_classic: List[Dict[str, object]],
    post_classic: List[Dict[str, object]],
    base_agentic: List[Dict[str, object]],
    post_agentic: List[Dict[str, object]],
) -> Dict[str, List[float]]:
    component_diffs = {
        "livecodebench": _align_diffs(base_classic, post_classic, metric="pass", benchmark="livecodebench"),
        "bigcodebench_instruct": _align_diffs(base_classic, post_classic, metric="pass", benchmark="bigcodebench_instruct"),
        "swebench_verified_subset": _align_diffs(base_classic, post_classic, metric="pass", benchmark="swebench_verified_subset"),
        "agentic_success": _align_diffs(base_agentic, post_agentic, metric="pass"),
        # Objective uses (1 - format_error_rate), so improvement equals baseline-format minus post-format.
        "format_quality": [-d for d in _align_diffs(base_classic, post_classic, metric="format")],
    }
    return component_diffs


def _weighted_mean(component_means: Dict[str, float]) -> float:
    return (
        (0.35 * component_means.get("livecodebench", 0.0))
        + (0.25 * component_means.get("bigcodebench_instruct", 0.0))
        + (0.20 * component_means.get("swebench_verified_subset", 0.0))
        + (0.10 * component_means.get("agentic_success", 0.0))
        + (0.10 * component_means.get("format_quality", 0.0))
    )


def _bootstrap_weighted_diff_ci(component_diffs: Dict[str, List[float]], bootstrap_samples: int, seed: int) -> Tuple[float, float, float]:
    required = [
        "livecodebench",
        "bigcodebench_instruct",
        "swebench_verified_subset",
        "agentic_success",
        "format_quality",
    ]
    for key in required:
        if not component_diffs.get(key):
            raise RuntimeError(f"Paired scientific gate requires shared cases for '{key}'.")

    observed_means = {name: (sum(vals) / len(vals)) for name, vals in component_diffs.items()}
    observed = _weighted_mean(observed_means)
    if bootstrap_samples <= 0:
        return observed, observed, observed

    rng = random.Random(seed)
    reps: List[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled_means = {}
        for name, values in component_diffs.items():
            n = len(values)
            sample = [values[rng.randrange(n)] for _ in range(n)]
            sampled_means[name] = sum(sample) / max(1, n)
        reps.append(_weighted_mean(sampled_means))
    reps.sort()
    lo = reps[int(0.025 * (len(reps) - 1))]
    hi = reps[int(0.975 * (len(reps) - 1))]
    return observed, lo, hi


def _sign_flip_pvalue(component_diffs: Dict[str, List[float]], observed_diff: float, samples: int, seed: int) -> float:
    if samples <= 0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(int(samples)):
        means = {}
        for name, values in component_diffs.items():
            signed = [v if rng.random() < 0.5 else -v for v in values]
            means[name] = sum(signed) / max(1, len(signed))
        null_stat = _weighted_mean(means)
        if null_stat >= observed_diff:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def _validate_core_artifacts(manifest_dir: str):
    workspace_root = os.path.abspath(os.path.join(manifest_dir, os.pardir, os.pardir))
    required = [
        os.path.join(manifest_dir, "run_manifest.txt"),
        os.path.join(manifest_dir, "baseline_eval_classic.json"),
        os.path.join(manifest_dir, "baseline_eval_agentic.json"),
        os.path.join(manifest_dir, "posttrain_eval_classic.json"),
        os.path.join(manifest_dir, "posttrain_eval_agentic.json"),
        os.path.join(manifest_dir, "baseline_eval_classic_cases.jsonl"),
        os.path.join(manifest_dir, "baseline_eval_agentic_cases.jsonl"),
        os.path.join(manifest_dir, "posttrain_eval_classic_cases.jsonl"),
        os.path.join(manifest_dir, "posttrain_eval_agentic_cases.jsonl"),
        os.path.join(workspace_root, "sota_slm_coding_dataset", "dataset_manifest.json"),
        os.path.join(workspace_root, "swe_supervised_dataset", "dataset_manifest.json"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"Scientific artifact validation failed. Missing: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Run strict scientific acceptance gates for a completed run.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--require-relative-improvement", type=float, default=0.15)
    parser.add_argument("--require-relative-improvement-lb", type=float, default=0.05)
    parser.add_argument("--max-format-error-rate", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--null-samples", type=int, default=4000)
    parser.add_argument("--significance-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-path", default="")
    args = parser.parse_args()

    manifest_dir = args.manifest_dir
    _validate_core_artifacts(manifest_dir)

    baseline_classic = _load_json(os.path.join(manifest_dir, "baseline_eval_classic.json"))
    baseline_agentic = _load_json(os.path.join(manifest_dir, "baseline_eval_agentic.json"))
    post_classic = _load_json(os.path.join(manifest_dir, "posttrain_eval_classic.json"))
    post_agentic = _load_json(os.path.join(manifest_dir, "posttrain_eval_agentic.json"))

    base_classic_rows = _load_jsonl(os.path.join(manifest_dir, "baseline_eval_classic_cases.jsonl"))
    base_agentic_rows = _load_jsonl(os.path.join(manifest_dir, "baseline_eval_agentic_cases.jsonl"))
    post_classic_rows = _load_jsonl(os.path.join(manifest_dir, "posttrain_eval_classic_cases.jsonl"))
    post_agentic_rows = _load_jsonl(os.path.join(manifest_dir, "posttrain_eval_agentic_cases.jsonl"))

    baseline_score_ci, baseline_fmt_ci = _primary_score_ci(baseline_classic, baseline_agentic)
    post_score_ci, post_fmt_ci = _primary_score_ci(post_classic, post_agentic)
    baseline_score = baseline_score_ci[0]
    post_score = post_score_ci[0]
    post_format = post_fmt_ci[0]

    rel_improvement = (post_score - baseline_score) / max(1e-6, baseline_score)
    rel_improvement_lb = (post_score_ci[1] - baseline_score_ci[2]) / max(1e-6, baseline_score_ci[2])

    component_diffs = _paired_objective_diffs(
        base_classic_rows,
        post_classic_rows,
        base_agentic_rows,
        post_agentic_rows,
    )
    paired_diff_ci = _bootstrap_weighted_diff_ci(
        component_diffs=component_diffs,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    p_value = _sign_flip_pvalue(
        component_diffs=component_diffs,
        observed_diff=paired_diff_ci[0],
        samples=args.null_samples,
        seed=args.seed + 17,
    )

    # Per-benchmark no-regression checks with CI overlap tolerance.
    no_regression_checks = []
    for bench in ["livecodebench", "bigcodebench_instruct", "swebench_verified_subset", "mbpp", "humaneval"]:
        b_pass = _mean_ci_from_report(baseline_classic, bench, "pass_at_1_ci")
        p_pass = _mean_ci_from_report(post_classic, bench, "pass_at_1_ci")
        b_fmt = _mean_ci_from_report(baseline_classic, bench, "format_error_rate_ci")
        p_fmt = _mean_ci_from_report(post_classic, bench, "format_error_rate_ci")
        pass_ok = p_pass[0] + 1e-6 >= b_pass[0] - 0.01
        pass_ci_ok = p_pass[1] + 1e-6 >= b_pass[1] - 0.02
        fmt_ok = p_fmt[0] <= b_fmt[0] + 0.01
        no_regression_checks.append(
            {
                "benchmark": bench,
                "pass_at_1_baseline": b_pass,
                "pass_at_1_post": p_pass,
                "format_error_baseline": b_fmt,
                "format_error_post": p_fmt,
                "pass_ok": bool(pass_ok and pass_ci_ok),
                "format_ok": bool(fmt_ok),
            }
        )

    failures = []
    if post_format >= float(args.max_format_error_rate):
        failures.append(
            f"format_error_rate gate failed: post={post_format:.4f} >= {args.max_format_error_rate:.4f}"
        )
    if rel_improvement < float(args.require_relative_improvement):
        failures.append(
            f"relative primary score improvement gate failed: {rel_improvement:.4f} < {args.require_relative_improvement:.4f}"
        )
    if rel_improvement_lb < float(args.require_relative_improvement_lb):
        failures.append(
            f"lower-bound relative improvement gate failed: {rel_improvement_lb:.4f} < {args.require_relative_improvement_lb:.4f}"
        )
    if p_value > float(args.significance_alpha):
        failures.append(
            f"paired significance gate failed: p={p_value:.6f} > alpha={args.significance_alpha:.4f}"
        )
    for check in no_regression_checks:
        if not (check["pass_ok"] and check["format_ok"]):
            failures.append(
                f"no-regression gate failed for {check['benchmark']} "
                f"(pass_ok={check['pass_ok']}, format_ok={check['format_ok']})"
            )

    report = {
        "schema_version": "scientific_gate_v1",
        "manifest_dir": manifest_dir,
        "primary_score_ci_baseline": baseline_score_ci,
        "primary_score_ci_post": post_score_ci,
        "relative_improvement": rel_improvement,
        "relative_improvement_lower_bound": rel_improvement_lb,
        "post_format_error_rate_ci": post_fmt_ci,
        "paired_primary_diff_ci": paired_diff_ci,
        "paired_sign_flip_pvalue_one_sided": p_value,
        "component_pair_sizes": {k: len(v) for k, v in component_diffs.items()},
        "no_regression_checks": no_regression_checks,
        "failed_gates": failures,
        "passed": len(failures) == 0,
    }

    output_path = args.output_path or os.path.join(manifest_dir, "scientific_acceptance.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[scientific-gate] Wrote report to: {output_path}")
    print(
        "[scientific-gate] Primary objective summary: "
        f"baseline={baseline_score:.6f}, post={post_score:.6f}, "
        f"rel_improvement={rel_improvement * 100:.2f}%, "
        f"rel_improvement_lb={rel_improvement_lb * 100:.2f}%, "
        f"paired_diff={paired_diff_ci[0]:.6f}, p={p_value:.6f}"
    )
    if failures:
        raise SystemExit("[scientific-gate] FAILED: " + " | ".join(failures))


if __name__ == "__main__":
    main()
