import os
import re
from typing import Dict, List

from prm_tiny import load_tiny_prm, predict_tiny_prm
from verification import run_test_verifier


# ==============================================================================
# SOTA SLM Reward Functions (GRPO)
# ==============================================================================


def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# Global state for AERO (Adaptive Execution Reward Optimization)
AERO_GLOBAL_STEP = 0
AERO_MAX_STEPS = 1500
_PRM_MODEL_PATH = ""
_PRM_TINY_MODEL = None


def configure_process_reward_model(model_path: str = ""):
    """
    Configures optional tiny PRM model used by process_reward_func.
    """
    global _PRM_MODEL_PATH, _PRM_TINY_MODEL
    _PRM_MODEL_PATH = (model_path or os.environ.get("SOTA_PRM_MODEL_PATH", "")).strip()
    _PRM_TINY_MODEL = None
    if not _PRM_MODEL_PATH:
        return
    if not os.path.exists(_PRM_MODEL_PATH):
        print(f"WARNUNG: PRM model path not found: {_PRM_MODEL_PATH}. Falling back to heuristic process reward.")
        return
    try:
        _PRM_TINY_MODEL = load_tiny_prm(_PRM_MODEL_PATH)
        print(f"Loaded tiny PRM model from '{_PRM_MODEL_PATH}'.")
    except Exception as exc:
        _PRM_TINY_MODEL = None
        print(f"WARNUNG: Failed to load PRM model '{_PRM_MODEL_PATH}': {exc}")


def _completion_to_text(completion):
    """
    Normalize completion payloads across TRL/Unsloth versions.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(
            completion.get("content")
            or completion.get("text")
            or completion.get("completion")
            or ""
        )
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(
                first.get("content")
                or first.get("text")
                or first.get("completion")
                or ""
            )
    return str(completion or "")


def get_aero_weights():
    progress = min(1.0, AERO_GLOBAL_STEP / max(1.0, float(AERO_MAX_STEPS)))
    format_weight = max(0.1, 1.0 - progress)
    exec_weight = 1.0 + (1.5 * progress)
    return format_weight, exec_weight


def strict_format_reward_func(prompts: List[str], completions: List[Dict[str, str]], **kwargs) -> List[float]:
    rewards = []
    responses = [_completion_to_text(comp) for comp in completions]

    format_weight, _ = get_aero_weights()
    for resp in responses:
        has_reasoning = "<reasoning>" in resp and "</reasoning>" in resp
        has_answer = "<answer>" in resp and "</answer>" in resp

        if has_answer and has_reasoning:
            rewards.append(1.0 * format_weight)
        elif has_answer:
            rewards.append(0.5 * format_weight)
        else:
            rewards.append(-1.5)
    return rewards


def length_penalty_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    responses = [_completion_to_text(comp) for comp in completions]

    for resp in responses:
        tokens = len(resp.split())
        if tokens < 300:
            rewards.append(0.0)
        elif tokens < 600:
            rewards.append(-0.2)
        elif tokens < 1000:
            rewards.append(-0.5)
        else:
            penalty = max(-2.0, -0.5 - ((tokens - 1000) * 0.001))
            rewards.append(penalty)
    return rewards


def self_verification_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    responses = [_completion_to_text(comp) for comp in completions]

    for resp in responses:
        reasoning = extract_xml_content(resp, "reasoning")
        if not reasoning:
            rewards.append(0.0)
            continue

        assert_count = reasoning.count("assert ")
        test_count = reasoning.count("def test_")
        if assert_count >= 2 or test_count >= 1:
            rewards.append(0.5)
        elif assert_count == 1:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _tiny_prm_score(prompt: str, reasoning: str, extracted_answer: str, tests: str) -> float:
    if _PRM_TINY_MODEL is None:
        return 0.5
    record = (
        f"prompt={prompt}\n"
        f"reasoning={reasoning}\n"
        f"tests={tests}\n"
        f"answer={extracted_answer}"
    )
    try:
        return float(predict_tiny_prm(_PRM_TINY_MODEL, record))
    except Exception:
        return 0.5


def process_reward_func(prompts, completions, answer, **kwargs):
    """
    Lightweight PRM-style reward:
    rewards structured repair reasoning and alignment with observed failure signals.
    """
    rewards = []
    responses = [_completion_to_text(comp) for comp in completions]

    for prompt, resp, expected_tests in zip(prompts, responses, answer):
        reasoning = extract_xml_content(resp, "reasoning")
        extracted_answer = extract_xml_content(resp, "answer")
        expected_tests = str(expected_tests or "")
        score = 0.0
        if reasoning:
            reasoning_l = reasoning.lower()
            words = len(reasoning.split())
            if words >= 30:
                score += 0.20
            if words >= 70:
                score += 0.10
            if words > 450:
                # Overly long traces often indicate verbosity without additional signal.
                score -= 0.15

            if "assert" in reasoning_l or "test" in reasoning_l:
                score += 0.20
            if "edge" in reasoning_l or "boundary" in reasoning_l or "corner" in reasoning_l:
                score += 0.15
            if "complex" in reasoning_l or "o(" in reasoning_l:
                score += 0.10
            if _contains_any(reasoning_l, ["first", "then", "finally", "step", "plan"]):
                score += 0.10

            prompt_l = (prompt or "").lower()
            if "error_type=timeout" in prompt_l and ("optimiz" in reasoning_l or "complex" in reasoning_l):
                score += 0.20
            if "error_type=syntax" in prompt_l and ("syntax" in reasoning_l or "parse" in reasoning_l):
                score += 0.20
            if "error_type=assertion" in prompt_l and ("edge" in reasoning_l or "case" in reasoning_l):
                score += 0.20
            if "error_type=format" in prompt_l and ("format" in reasoning_l or "xml" in reasoning_l or "tag" in reasoning_l):
                score += 0.20
            if "error_type=security" in prompt_l and _contains_any(reasoning_l, ["sanitize", "safe", "sandbox", "restrict"]):
                score += 0.20

            # Encourage process alignment with hidden tests when available.
            if expected_tests and ("assert" in expected_tests):
                if "assert" in reasoning_l or "test" in reasoning_l:
                    score += 0.10
                if _contains_any(reasoning_l, ["regression", "edge", "boundary"]):
                    score += 0.10
        else:
            score -= 0.10

        if not extracted_answer:
            score -= 0.20
        else:
            answer_len = len(extracted_answer.split())
            if answer_len < 4:
                score -= 0.12
            elif answer_len <= 220:
                score += 0.06

        prm_prob = _tiny_prm_score(
            prompt=str(prompt or ""),
            reasoning=str(reasoning or ""),
            extracted_answer=str(extracted_answer or ""),
            tests=expected_tests,
        )
        score += (prm_prob - 0.5) * 0.60

        rewards.append(max(-0.5, min(1.0, score)))
    return rewards


def _error_penalty(error_type: str) -> float:
    normalized = (error_type or "").strip().lower()
    penalties = {
        "format": -1.8,
        "security": -1.6,
        "timeout": -1.3,
        "syntax": -0.9,
        "assertion": -0.4,
        "runtime": -0.3,
    }
    return penalties.get(normalized, -0.25 if normalized else 0.0)


def execution_reward_func(prompts: List[str], completions: List[Dict[str, str]], answer: List[str], **kwargs) -> List[float]:
    """
    Dense execution reward with fractional assertion credit.
    """
    global AERO_GLOBAL_STEP
    AERO_GLOBAL_STEP += 1

    _, exec_weight = get_aero_weights()
    rewards = []
    responses = [_completion_to_text(comp) for comp in completions]

    for resp, expected_tests in zip(responses, answer):
        expected_tests = (expected_tests or "").strip()
        if not expected_tests:
            rewards.append(0.0)
            continue

        extracted_code = extract_xml_content(resp, "answer")
        if not extracted_code:
            rewards.append(-2.0)
            continue

        code_snippet = extracted_code.replace("```python", "").replace("```", "").strip()
        verify = run_test_verifier(
            code=code_snippet,
            tests=expected_tests,
            timeout=2.0,
            rounds=1,
            require_all_pass=True,
        )

        all_passed = bool(verify.get("all_passed", False))
        pass_fraction = float(verify.get("pass_fraction", 0.0))
        exec_time = float(verify.get("exec_time", 0.0))
        best_detail = verify.get("best_detail", {}) if isinstance(verify.get("best_detail"), dict) else {}
        error_type = str(best_detail.get("error_type", "") or "")

        if all_passed:
            score = 2.0
            if exec_time < 0.05:
                score += 0.4
            elif exec_time > 0.8:
                score -= 0.4
        else:
            # Fractional execution reward: proportion of passed assertions.
            score = max(-0.2, 1.4 * pass_fraction)
            score += _error_penalty(error_type)

        score = max(-3.0, min(4.0, score * exec_weight))
        rewards.append(score)

    return rewards


def get_reward_functions(profile: str):
    normalized = (profile or "dense_exec_v1").strip().lower()
    if normalized == "dense_exec_v1":
        return [
            strict_format_reward_func,
            length_penalty_reward_func,
            self_verification_reward_func,
            execution_reward_func,
        ]
    if normalized == "prm_outcome_v1":
        return [
            strict_format_reward_func,
            length_penalty_reward_func,
            process_reward_func,
            self_verification_reward_func,
            execution_reward_func,
        ]

    print(f"Warnung: Unknown reward profile '{profile}'. Falling back to prm_outcome_v1.")
    return [
        strict_format_reward_func,
        length_penalty_reward_func,
        process_reward_func,
        self_verification_reward_func,
        execution_reward_func,
    ]
