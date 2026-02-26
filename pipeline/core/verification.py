import ast
import math
import re
import textwrap
from typing import Dict, List, Tuple

from sandbox import run_code_in_sandbox_detailed


_ASSERT_RE = re.compile(r"^\s*assert\b", flags=re.MULTILINE)
_TEST_FN_RE = re.compile(r"^\s*def\s+test_", flags=re.MULTILINE)
_BOUNDARY_TERMS = (
    "empty",
    "none",
    "null",
    "zero",
    "negative",
    "single",
    "edge",
    "boundary",
    "large",
    "small",
)
_PROPERTY_TERMS = (
    "invariant",
    "property",
    "idempot",
    "commut",
    "associat",
    "monot",
    "sorted",
)
_RANDOM_TERMS = ("random", "randint", "shuffle", "sample", "seed", "hypothesis")
_ASSERT_COUNT_TAG = "__SLM_ASSERT_COUNTS__"


class _AssertCounterTransformer(ast.NodeTransformer):
    """
    Rewrites `assert expr` into counter updates for fractional execution rewards.
    """

    def visit_Assert(self, node):
        total_inc = ast.AugAssign(
            target=ast.Name(id="__slm_assert_total", ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        pass_inc = ast.AugAssign(
            target=ast.Name(id="__slm_assert_passed", ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        safe_eval = ast.Try(
            body=[
                ast.If(
                    test=node.test,
                    body=[pass_inc],
                    orelse=[],
                )
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="Exception", ctx=ast.Load()),
                    name=None,
                    body=[ast.Pass()],
                )
            ],
            orelse=[],
            finalbody=[],
        )
        return [ast.copy_location(total_inc, node), ast.copy_location(safe_eval, node)]


def assess_test_quality(tests: str) -> Dict[str, float]:
    normalized = (tests or "").strip()
    if not normalized:
        return {
            "nonempty_lines": 0,
            "assert_count": 0,
            "test_fn_count": 0,
            "boundary_hits": 0,
            "property_hits": 0,
            "random_hits": 0,
            "quality_score": 0.0,
        }

    lowered = normalized.lower()
    nonempty_lines = len([line for line in normalized.splitlines() if line.strip()])
    assert_count = len(_ASSERT_RE.findall(normalized))
    test_fn_count = len(_TEST_FN_RE.findall(normalized))
    boundary_hits = sum(1 for token in _BOUNDARY_TERMS if token in lowered)
    property_hits = sum(1 for token in _PROPERTY_TERMS if token in lowered)
    random_hits = sum(1 for token in _RANDOM_TERMS if token in lowered)

    quality_score = (
        (0.9 * assert_count)
        + (1.2 * test_fn_count)
        + (0.4 * boundary_hits)
        + (0.5 * property_hits)
        + (0.5 * random_hits)
        + min(1.5, nonempty_lines / 20.0)
    )

    return {
        "nonempty_lines": nonempty_lines,
        "assert_count": assert_count,
        "test_fn_count": test_fn_count,
        "boundary_hits": boundary_hits,
        "property_hits": property_hits,
        "random_hits": random_hits,
        "quality_score": quality_score,
    }


def is_test_quality_sufficient(
    tests: str,
    min_asserts: int = 2,
    min_nonempty_lines: int = 3,
    min_quality_score: float = 2.5,
) -> bool:
    metrics = assess_test_quality(tests)
    return (
        metrics["assert_count"] >= min_asserts
        and metrics["nonempty_lines"] >= min_nonempty_lines
        and metrics["quality_score"] >= min_quality_score
    )


def _compose_eval_code(code: str, tests: str, seed: int) -> str:
    prelude = (
        "import random as __slm_random\n"
        "try:\n"
        f"    __slm_random.seed({seed})\n"
        "except Exception:\n"
        "    pass\n"
    )
    return f"{code}\n\n# --- Verifier Seed Prelude ---\n{prelude}\n# --- Unit Tests ---\n{tests}"


def _instrument_tests_for_fractional_counts(tests: str) -> str:
    try:
        tree = ast.parse(tests)
    except Exception:
        return ""
    transformed = _AssertCounterTransformer().visit(tree)
    transformed = ast.fix_missing_locations(transformed)
    try:
        return ast.unparse(transformed)
    except Exception:
        return ""


def _compose_fractional_eval_code(code: str, tests: str, seed: int) -> str:
    transformed_tests = _instrument_tests_for_fractional_counts(tests)
    if not transformed_tests:
        return ""

    prelude = (
        "import random as __slm_random\n"
        "try:\n"
        f"    __slm_random.seed({seed})\n"
        "except Exception:\n"
        "    pass\n"
    )
    safe_test_block = textwrap.indent(transformed_tests, "    ")
    return (
        f"{code}\n\n"
        f"{prelude}\n"
        "__slm_assert_passed = 0\n"
        "__slm_assert_total = 0\n"
        "try:\n"
        f"{safe_test_block}\n"
        "except Exception:\n"
        "    pass\n"
        "finally:\n"
        f"    print('{_ASSERT_COUNT_TAG}', __slm_assert_passed, __slm_assert_total)\n"
    )


def _parse_assert_counts(stdout: str) -> Tuple[int, int]:
    pattern = re.compile(rf"{_ASSERT_COUNT_TAG}\s+(\d+)\s+(\d+)")
    match = pattern.search(stdout or "")
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _fractional_assert_execution(code: str, tests: str, timeout: float = 2.0) -> Dict[str, object]:
    eval_code = _compose_fractional_eval_code(code, tests, seed=0)
    if not eval_code:
        return {
            "assert_passed": 0,
            "assert_total": 0,
            "pass_fraction": 0.0,
            "error_type": "fractional_unavailable",
        }

    detail = run_code_in_sandbox_detailed(eval_code, timeout=timeout)
    passed, total = _parse_assert_counts(str(detail.get("stdout", "") or ""))
    if total <= 0:
        return {
            "assert_passed": 0,
            "assert_total": 0,
            "pass_fraction": 0.0,
            "error_type": str(detail.get("error_type", "") or "fractional_unavailable"),
        }

    return {
        "assert_passed": max(0, min(passed, total)),
        "assert_total": total,
        "pass_fraction": max(0.0, min(1.0, passed / max(1, total))),
        "error_type": str(detail.get("error_type", "") or ""),
    }


def _score_to_fraction(score: float, error_type: str) -> float:
    if score >= 1.99:
        return 1.0
    normalized = str(error_type or "").strip().lower()
    if normalized == "assertion":
        return 0.45
    if normalized in {"runtime", "syntax"}:
        return 0.1
    if normalized in {"timeout", "security"}:
        return 0.0
    if score >= 0.5:
        return 0.35
    if score > 0.0:
        return 0.15
    return 0.0


def _fallback_fraction_from_rounds(details: List[Dict[str, object]], quality_assert_count: int) -> Dict[str, object]:
    if not details:
        return {
            "assert_passed": 0,
            "assert_total": max(1, int(quality_assert_count)),
            "pass_fraction": 0.0,
            "fractional_mode": "fallback_round_scores",
        }
    estimates = []
    for detail in details:
        score = float(detail.get("score", 0.0))
        err = str(detail.get("error_type", "") or "")
        estimates.append(_score_to_fraction(score=score, error_type=err))
    pass_fraction = sum(estimates) / max(1, len(estimates))
    assert_total = max(1, int(quality_assert_count))
    assert_passed = int(round(pass_fraction * assert_total))
    return {
        "assert_passed": max(0, min(assert_passed, assert_total)),
        "assert_total": assert_total,
        "pass_fraction": max(0.0, min(1.0, pass_fraction)),
        "fractional_mode": "fallback_round_scores",
    }


def _run_single_seed(code: str, tests: str, seed: int, timeout: float, retry_on_timeout: bool, timeout_retry_factor: float):
    eval_code = _compose_eval_code(code, tests, seed=seed)
    result = run_code_in_sandbox_detailed(eval_code, timeout=timeout)
    if retry_on_timeout and str(result.get("error_type", "") or "") == "timeout":
        extended_timeout = max(timeout * max(1.0, timeout_retry_factor), timeout + 0.25)
        retry = run_code_in_sandbox_detailed(eval_code, timeout=extended_timeout)
        if float(retry.get("score", 0.0)) > float(result.get("score", 0.0)):
            retry["retry_used"] = True
            retry["retry_from_timeout"] = True
            retry["retry_base_timeout"] = timeout
            retry["retry_timeout"] = extended_timeout
            return retry
    return result


def run_test_verifier(
    code: str,
    tests: str,
    timeout: float = 2.0,
    rounds: int = 2,
    require_all_pass: bool = True,
    retry_on_timeout: bool = True,
    timeout_retry_factor: float = 1.5,
) -> Dict[str, object]:
    normalized_tests = (tests or "").strip()
    if not normalized_tests:
        return {
            "score": 0.0,
            "exec_time": 0.0,
            "all_passed": False,
            "round_scores": [],
            "round_details": [],
            "pass_fraction": 0.0,
            "assert_passed": 0,
            "assert_total": 0,
            "error_counts": {},
            "quality": assess_test_quality(""),
        }

    rounds = max(1, int(rounds))
    scores: List[float] = []
    details: List[Dict[str, object]] = []
    max_exec_time = 0.0
    best_success_time = math.inf
    error_counts: Dict[str, int] = {}

    for seed in range(rounds):
        result = _run_single_seed(
            code=code,
            tests=normalized_tests,
            seed=seed,
            timeout=timeout,
            retry_on_timeout=retry_on_timeout,
            timeout_retry_factor=timeout_retry_factor,
        )
        details.append(result)
        score = float(result.get("score", 0.0))
        scores.append(score)
        exec_time = float(result.get("exec_time", 0.0))
        max_exec_time = max(max_exec_time, exec_time)
        if score == 2.0:
            best_success_time = min(best_success_time, exec_time)

        error_type = str(result.get("error_type", "") or "")
        if error_type:
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

    all_passed = all(score == 2.0 for score in scores)
    if all_passed:
        final_score = 2.0
        final_exec_time = best_success_time if best_success_time != math.inf else max_exec_time
    else:
        final_score = min(scores) if require_all_pass else max(scores)
        final_exec_time = max_exec_time

    best_detail = sorted(
        details,
        key=lambda row: (
            -float(row.get("score", 0.0)),
            float(row.get("exec_time", 999.0)),
        ),
    )[0]

    quality = assess_test_quality(normalized_tests)
    fractional = _fractional_assert_execution(code=code, tests=normalized_tests, timeout=timeout)
    pass_fraction = float(fractional.get("pass_fraction", 0.0))
    assert_passed = int(fractional.get("assert_passed", 0))
    assert_total = int(fractional.get("assert_total", 0))
    fractional_mode = "instrumented"

    if assert_total == 0:
        fallback = _fallback_fraction_from_rounds(
            details=details,
            quality_assert_count=int(quality.get("assert_count", 0)),
        )
        pass_fraction = max(pass_fraction, float(fallback["pass_fraction"]))
        assert_total = int(fallback["assert_total"])
        assert_passed = int(fallback["assert_passed"])
        fractional_mode = str(fallback["fractional_mode"])

    if assert_total == 0 and all_passed:
        pass_fraction = 1.0

    return {
        "score": final_score,
        "exec_time": final_exec_time,
        "all_passed": all_passed,
        "round_scores": scores,
        "round_details": details,
        "best_detail": best_detail,
        "pass_fraction": pass_fraction,
        "assert_passed": assert_passed,
        "assert_total": assert_total,
        "fractional_mode": fractional_mode,
        "error_counts": error_counts,
        "quality": quality,
    }
