import math
import re
from typing import Dict, List

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

    # Weighted heuristic: assertions + structural tests + diversity hints.
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


def run_test_verifier(
    code: str,
    tests: str,
    timeout: float = 2.0,
    rounds: int = 2,
    require_all_pass: bool = True,
) -> Dict[str, object]:
    normalized_tests = (tests or "").strip()
    if not normalized_tests:
        return {
            "score": 0.0,
            "exec_time": 0.0,
            "all_passed": False,
            "round_scores": [],
            "round_details": [],
            "quality": assess_test_quality(""),
        }

    rounds = max(1, int(rounds))
    scores: List[float] = []
    details: List[Dict[str, object]] = []
    max_exec_time = 0.0
    best_success_time = math.inf

    for seed in range(rounds):
        eval_code = _compose_eval_code(code, normalized_tests, seed=seed)
        result = run_code_in_sandbox_detailed(eval_code, timeout=timeout)
        details.append(result)
        score = float(result.get("score", 0.0))
        scores.append(score)
        exec_time = float(result.get("exec_time", 0.0))
        max_exec_time = max(max_exec_time, exec_time)
        if score == 2.0:
            best_success_time = min(best_success_time, exec_time)

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

    return {
        "score": final_score,
        "exec_time": final_exec_time,
        "all_passed": all_passed,
        "round_scores": scores,
        "round_details": details,
        "best_detail": best_detail,
        "quality": assess_test_quality(normalized_tests),
    }
