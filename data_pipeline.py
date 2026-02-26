import os
import json
import hashlib
import argparse
import re
import ast
import subprocess
from typing import Dict
from datetime import datetime

from datasets import DatasetDict, load_dataset, concatenate_datasets

# ==============================================================================
# SOTA SLM Data Preparation Pipeline
# ==============================================================================


def _extract_tests(example):
    """
    Normalize optional unit tests from heterogeneous dataset schemas.
    """
    candidates = [
        example.get("tests"),
        example.get("test"),
        example.get("test_list"),
        example.get("unit_tests"),
        example.get("public_tests"),
    ]
    for value in candidates:
        if not value:
            continue
        if isinstance(value, list):
            return "\n".join([str(item) for item in value if item])
        return str(value)
    return ""


def _extract_swe_tests(example):
    candidates = [
        example.get("tests"),
        example.get("test"),
        example.get("FAIL_TO_PASS"),
        example.get("PASS_TO_PASS"),
        example.get("unit_tests"),
        example.get("public_tests"),
        example.get("private_tests"),
        example.get("regression_tests"),
    ]
    chunks = []
    for value in candidates:
        if not value:
            continue
        if isinstance(value, list):
            text = "\n".join([str(item) for item in value if item])
        else:
            text = str(value)
        text = text.strip()
        if text:
            chunks.append(text)
    if not chunks:
        return ""
    dedup = []
    seen = set()
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        dedup.append(chunk)
    return "\n\n".join(dedup)


def _extract_affected_files(example):
    keys = [
        "files",
        "file_paths",
        "changed_files",
        "impacted_files",
        "relevant_files",
    ]
    for key in keys:
        value = example.get(key)
        if not value:
            continue
        if isinstance(value, list):
            vals = [str(v).strip() for v in value if str(v).strip()]
            if vals:
                return vals
        else:
            text = str(value).strip()
            if text:
                return [text]
    return []


def _truncate_text(text, max_chars):
    text = _text(text).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _prompt_prefix_from_text(formatted_text):
    """
    Keep only prompt context up to the assistant turn to avoid RL label leakage.
    """
    split_token = "<|im_start|>assistant\n"
    if split_token in formatted_text:
        return formatted_text.split(split_token, 1)[0] + split_token
    return formatted_text


def _build_record(example, formatted_text):
    return {
        "text": formatted_text,
        "prompt": _prompt_prefix_from_text(formatted_text),
        "tests": _extract_tests(example),
    }


def _tag_source(dataset, source_name):
    if dataset is None:
        return None
    return dataset.add_column("source", [source_name] * len(dataset))


def _parse_source_weights(source_weights: str) -> Dict[str, float]:
    default = {"code": 0.55, "repo": 0.30, "reasoning": 0.15}
    if not source_weights:
        return default
    parsed = {}
    for part in source_weights.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        try:
            parsed[key] = float(value.strip())
        except ValueError:
            continue
    merged = {**default, **parsed}
    total = sum(max(0.0, v) for v in merged.values())
    if total <= 0:
        return default
    return {k: max(0.0, v) / total for k, v in merged.items()}


_SOURCE_CATEGORY = {
    "nvidia/OpenCodeReasoning": "code",
    "WizardLM/WizardLM_evol_instruct_V2_196k": "code",
    "m-a-p/CodeFeedback-Filtered-Instruction": "code",
    "O1-CODER/OpenO1-SFT": "code",
    "princeton-nlp/SWE-bench_Lite": "repo",
    "princeton-nlp/SWE-agent-trajectories": "repo",
    "bigcode/commitpackft:python": "repo",
    "SWE-bench-Live/SWE-bench-Live:verified": "repo",
    "HuggingFaceH4/Bespoke-Stratos-17k": "reasoning",
    "AI-MO/NuminaMath-CoT": "reasoning",
}


def _cap_sources(source_datasets, max_samples_per_source, seed=3407):
    capped = {}
    for source_name, dataset in source_datasets.items():
        if dataset is None:
            continue
        if max_samples_per_source > 0 and len(dataset) > max_samples_per_source:
            dataset = dataset.shuffle(seed=seed).select(range(max_samples_per_source))
        capped[source_name] = dataset
    return capped


def _mix_sources_by_category(source_datasets, source_weights, seed=3407):
    by_category = {"code": [], "repo": [], "reasoning": []}
    for source_name, dataset in source_datasets.items():
        if dataset is None or len(dataset) == 0:
            continue
        category = _SOURCE_CATEGORY.get(source_name, "code")
        by_category.setdefault(category, []).append(dataset)

    merged_by_category = {}
    for category, datasets_in_cat in by_category.items():
        if not datasets_in_cat:
            continue
        merged = concatenate_datasets(datasets_in_cat).shuffle(seed=seed)
        merged_by_category[category] = merged

    if not merged_by_category:
        raise RuntimeError("No datasets available after category grouping.")

    # Maximize total usable data while preserving desired proportions where possible.
    available_total = sum(len(ds) for ds in merged_by_category.values())
    if available_total <= 0:
        raise RuntimeError("No weighted categories available for source mixing.")

    categories = sorted(merged_by_category.keys())
    desired = {}
    remaining = available_total
    for category in categories:
        weight = max(0.0, source_weights.get(category, 0.0))
        desired_count = int(available_total * weight)
        desired[category] = min(len(merged_by_category[category]), max(0, desired_count))
        remaining -= desired[category]

    # Redistribute leftover budget to categories that still have unused capacity.
    while remaining > 0:
        progressed = False
        for category in categories:
            slack = len(merged_by_category[category]) - desired[category]
            if slack <= 0:
                continue
            desired[category] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    selected = []
    for category in categories:
        ds = merged_by_category[category]
        target = desired.get(category, 0)
        if target <= 0:
            continue
        selected.append(ds.select(range(target)))

    mixed = concatenate_datasets(selected).shuffle(seed=seed)
    print(
        "Applied source-mix weights: "
        + ", ".join([f"{k}={source_weights.get(k, 0.0):.2f}" for k in ["code", "repo", "reasoning"]])
    )
    return mixed


def _deterministic_reasoning_mode(example, ratio=0.7):
    """
    Deterministic replacement for random sampling to keep dataset builds reproducible.
    """
    key_parts = [
        example.get("instruction", ""),
        example.get("prompt", ""),
        example.get("problem_statement", ""),
        example.get("query", ""),
        example.get("instance_id", ""),
    ]
    key = "||".join(str(part) for part in key_parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < ratio


def _text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join([str(item) for item in value if item is not None])
    return str(value)


def _has_nonempty_text(value):
    return bool(_text(value).strip())


def _assistant_block(answer_text, reasoning_text=""):
    answer_text = _text(answer_text).strip()
    reasoning_text = _text(reasoning_text).strip()
    if reasoning_text:
        return (
            f"<|im_start|>assistant\n<reasoning>\n{reasoning_text}\n</reasoning>\n"
            f"<answer>\n{answer_text}\n</answer><|im_end|>"
        )
    return f"<|im_start|>assistant\n<answer>\n{answer_text}\n</answer><|im_end|>"


def format_reasoning_prompt(example):
    """
    Integriert detaillierte Teacher-Rationales in den Datensatz für das SLM.
    Verwendet "Dual-Mode" (zufällig), um Overthinking bei simplen Aufgaben zu verhindern.
    """
    # 70% der Zeit erzwingen wir tiefes Nachdenken, 30% der Zeit trainieren wir "direct output"
    use_detailed_reasoning = _deterministic_reasoning_mode(example, ratio=0.7)
    
    instruction = _text(example.get("instruction")).strip() or _text(example.get("prompt")).strip()
    solution = _text(example.get("solution")).strip()
    rationale = _text(example.get("reasoning_trace")).strip()

    if use_detailed_reasoning and rationale:
        system_prompt = (
            "You are an elite coding assistant. You must thoroughly analyze the problem "
            "step-by-step inside <reasoning> tags before synthesizing the final robust code "
            "inside <answer> tags."
        )
        assistant_block = _assistant_block(solution, rationale)
    else:
        system_prompt = (
            "You are an elite coding assistant. Provide the final robust code directly "
            "inside <answer> tags without extensive preliminary reasoning."
        )
        assistant_block = _assistant_block(solution)

    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{assistant_block}"
    )

    return _build_record(example, formatted_text)


def format_bug_prompt(example):
    """
    Formatierung für verifizierte Bugs (SWE-Bench Stil).
    Lehrt das Modell, aus einer Problembeschreibung konkrete Code-Modifikationen (Patches) abzuleiten.
    """
    system_prompt = (
        "You are a senior software engineer. Analyze the bug report and provide the "
        "necessary code patch to resolve the issue inside <answer> tags. "
        "Think step-by-step in <reasoning> tags first."
    )
    problem = _text(example.get("problem_statement")).strip()
    instruction = f"Issue Description: {problem}"
    solution = _text(example.get("patch")).strip()
    rationale = (
        _text(example.get("analysis")).strip()
        or _text(example.get("hints")).strip()
        or _text(example.get("FAIL_TO_PASS")).strip()
    )
    
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{_assistant_block(solution, rationale)}"
    )
    return _build_record(example, formatted_text)


def format_swe_supervised_bug(example, source_name="swe_bug", max_context_chars=6000):
    issue = _truncate_text(
        _text(example.get("problem_statement", example.get("issue", example.get("prompt", "")))).strip(),
        max_context_chars,
    )
    patch = _text(example.get("patch", example.get("gold_patch", example.get("solution_patch", "")))).strip()
    tests = _extract_swe_tests(example)
    affected_files = _extract_affected_files(example)
    files_block = "\n".join([f"- {p}" for p in affected_files[:20]])
    files_text = f"\nAffected files:\n{files_block}" if files_block else ""

    system_prompt = (
        "You are an expert SWE assistant. Solve repository issues by proposing a precise unified patch "
        "inside <answer> tags after concise reasoning in <reasoning> tags."
    )
    user_prompt = f"Issue:\n{issue}{files_text}"
    reasoning = _text(example.get("analysis", example.get("hints", ""))).strip()
    formatted = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"{_assistant_block(patch, reasoning)}"
    )
    return {
        "text": formatted,
        "prompt": _prompt_prefix_from_text(formatted),
        "tests": tests,
        "issue": issue,
        "affected_files": "\n".join(affected_files[:50]),
        "patch": patch,
        "trajectory": "",
        "source": source_name,
        "swe_granularity": "repo_level",
    }


def format_evol_prompt(example):
    """
    Formatierung für komplexe, schrittweise Evol-Instruct Coding Probleme.
    Fördert massiv die Syntax-Mächtigkeit und Programmier-Tiefe.
    """
    system_prompt = (
        "You are an elite coding assistant. Solve the deeply complex programming "
        "challenge inside <answer> tags. It is highly recommended to think through "
        "edge cases in <reasoning> tags first."
    )
    instruction = _text(example.get("instruction")).strip()
    solution = _text(example.get("output")).strip()

    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{_assistant_block(solution)}"
    )
    return _build_record(example, formatted_text)


def format_trajectory_prompt(example):
    """
    Formatierung für Agentic Trajectories (Multi-Turn).
    Lehrt das Modell den Umgang mit Tools (z.B. view_file, grep) zur Bug-Lokalisation,
    bevor der eigentliche Patch geschrieben wird.
    """
    system_prompt = (
        "You are an autonomous software engineering agent. You are provided with an issue "
        "description and a repository. You must use tools (like shell commands or file editors) "
        "to explore the codebase, understand the bug, and iteratively develop a solution inside <answer> tags."
    )
    
    # SWE-agent trajectories formatieren (vereinfachte Rekonstruktion der History)
    issue = _text(example.get("instance_id")).strip() or "Unknown Issue"
    trajectory = example.get("trajectory", []) # Liste von Actions/Observations
    final_patch = (
        example.get("patch")
        or example.get("gold_patch")
        or example.get("model_patch")
        or example.get("final_patch")
        or ""
    )
    
    history = f"Issue: Resolving {issue}\n"
    reasoning_log = []
    for step in trajectory:
        action = _text(step.get("action")).strip()
        obs = _text(step.get("observation")).strip()
        history += f"\n[Action]: {action}\n[Observation]: {obs}"
        if action:
            reasoning_log.append(f"Action: {action}")
        if obs:
            reasoning_log.append(f"Observation: {obs[:240]}")

    reasoning_text = "\n".join(reasoning_log[:20])
    final_patch = _text(final_patch).strip()
        
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{history}<|im_end|>\n"
        f"{_assistant_block(final_patch, reasoning_text)}"
    )
    return _build_record(example, formatted_text)


def format_swe_supervised_trajectory(example, source_name="swe_trajectory", max_context_chars=6000):
    issue = _truncate_text(
        _text(example.get("problem_statement", example.get("issue", example.get("instance_id", "")))).strip(),
        max_context_chars,
    )
    trajectory = example.get("trajectory", [])
    patch = _text(
        example.get("patch")
        or example.get("gold_patch")
        or example.get("model_patch")
        or example.get("final_patch")
        or ""
    ).strip()
    tests = _extract_swe_tests(example)
    affected_files = _extract_affected_files(example)

    trajectory_lines = []
    if isinstance(trajectory, list):
        for step in trajectory[:60]:
            if not isinstance(step, dict):
                continue
            action = _truncate_text(step.get("action", ""), 280)
            observation = _truncate_text(step.get("observation", ""), 360)
            if action:
                trajectory_lines.append(f"[ACTION] {action}")
            if observation:
                trajectory_lines.append(f"[OBS] {observation}")
    trajectory_text = "\n".join(trajectory_lines)

    files_block = "\n".join([f"- {p}" for p in affected_files[:20]])
    files_text = f"\nAffected files:\n{files_block}" if files_block else ""
    system_prompt = (
        "You are an autonomous SWE agent. Use prior tool traces to craft a robust final patch "
        "inside <answer> tags."
    )
    user_prompt = f"Issue:\n{issue}{files_text}\n\nTrajectory:\n{trajectory_text}"
    formatted = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"{_assistant_block(patch, trajectory_text[:1200])}"
    )
    return {
        "text": formatted,
        "prompt": _prompt_prefix_from_text(formatted),
        "tests": tests,
        "issue": issue,
        "affected_files": "\n".join(affected_files[:50]),
        "patch": patch,
        "trajectory": trajectory_text,
        "source": source_name,
        "swe_granularity": "repo_level",
    }


def format_stratos_prompt(example):
    """
    SOTA Reasoning Distillation.
    """
    system_prompt = (
        "You are an elite reasoning assistant. Solve the problem inside <answer> tags "
        "after deep thought in <reasoning> tags."
    )
    instruction = _text(example.get("prompt", example.get("messages", ""))).strip()
    solution = _text(example.get("response", example.get("generation", ""))).strip()
    reasoning = _text(example.get("reasoning", example.get("analysis", ""))).strip()
    
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{_assistant_block(solution, reasoning)}"
    )
    return _build_record(example, formatted_text)


def format_math_prompt(example):
    """
    Mathematische Kausalität (NuminaMath). Zwingt das Modell, Theoreme aufzustellen.
    """
    system_prompt = (
        "You are an elite mathematical reasoning assistant. Solve the math problem "
        "step-by-step inside <reasoning> tags, then provide the final answer inside <answer> tags."
    )
    instruction = _text(example.get("problem")).strip()
    solution = _text(example.get("solution")).strip()
    reasoning = _text(example.get("reasoning", example.get("scratchpad", ""))).strip()
    
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{_assistant_block(solution, reasoning)}"
    )
    return _build_record(example, formatted_text)


def format_codefeedback_prompt(example):
    """
    Self-Healing (CodeFeedback). Lehrt das Modell, aus Fehlermeldungen zu lernen.
    """
    system_prompt = (
        "You are an autonomous self-healing coder. Analyze the execution error, "
        "explain the flaw inside <reasoning> tags, and provide the fixed code inside <answer> tags."
    )
    instruction = _text(example.get("query")).strip()
    solution = _text(example.get("answer")).strip()
    reasoning = _text(example.get("feedback", example.get("error", ""))).strip()
    
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{_assistant_block(solution, reasoning)}"
    )
    return _build_record(example, formatted_text)


def format_mcts_prompt(example):
    """
    Monte Carlo Tree Search (OpenO1). 
    Lehrt das Modell, in Hypothesen zu denken, Ideen zu evaluieren und (ganz wichtig)
    Entscheidungen zu verwerfen ("Backtracking"), bevor es den endgültigen Code schreibt.
    """
    system_prompt = (
        "You are an advanced reasoning agent capable of Monte Carlo Tree Search (MCTS) exploration. "
        "Before generating the final answer inside <answer> tags, use the <reasoning> tags to:\n"
        "1. Formulate multiple distinct hypotheses or approaches.\n"
        "2. Evaluate each approach logically (pros/cons, edge cases).\n"
        "3. Explicitly verify or reject (backtrack) flawed approaches.\n"
        "4. Select the most robust approach."
    )
    instruction = _text(example.get("instruction", example.get("prompt", ""))).strip()
    solution = _text(example.get("output", example.get("response", ""))).strip()
    reasoning = _text(example.get("reasoning", example.get("trace", ""))).strip()
    
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{_assistant_block(solution, reasoning)}"
    )
    return _build_record(example, formatted_text)


def format_commitpack_prompt(example):
    """
    Multi-File / Long-Context (CommitPackFT).
    Lehrt das Modell, git-ähnliche Diff-Formate über mehrere Dateien 
    und Projektkontexte hinweg (Project-Size) zu verstehen und zu generieren.
    """
    system_prompt = (
        "You are an expert software architect. Analyze the provided codebase context "
        "or issue description, and provide a comprehensive, multi-file git patch "
        "inside <answer> tags. Outline your architectural strategy in <reasoning> tags first."
    )
    
    # In CommitPackFT heißt der Kontext meist 'old_contents' oder die Message 'message'
    instruction = (
        f"Commit Message / Issue: {_text(example.get('message')).strip()}\n\n"
        f"Pre-Commit Context / Code:\n{_text(example.get('old_contents')).strip()}"
    )
    solution = f"Git Patch Diff:\n{_text(example.get('diff')).strip()}"
    reasoning = _text(example.get("analysis", example.get("explanation", ""))).strip()
    
    # Truncate to avoid extreme OOMs during SFT (max ~3000 chars context)
    if len(instruction) > 3000:
         instruction = instruction[:3000] + "\n...[Context truncated for efficiency]"
         
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"{_assistant_block(solution, reasoning)}"
    )
    return _build_record(example, formatted_text)


def _normalize_for_dedup(text):
    lowered = (text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return re.sub(r"[^a-z0-9_<>/| ]", "", lowered)


def _extract_answer_from_text(text):
    text = _text(text)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _is_answer_ast_parseable(text):
    answer = _extract_answer_from_text(text)
    if not answer:
        return False
    answer = answer.replace("```python", "").replace("```", "").strip()
    try:
        ast.parse(answer)
        return True
    except Exception:
        return False


def _drop_prompt_answer_duplicates(dataset):
    """
    Remove exact/near-exact prompt+answer duplicates to reduce memorization and leakage risk.
    """
    seen = set()
    keep_indices = []
    removed = 0

    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", "") or _prompt_prefix_from_text(row.get("text", ""))
        answer = _extract_answer_from_text(row.get("text", ""))
        dedup_key = f"{_normalize_for_dedup(prompt)}||{_normalize_for_dedup(answer)}"
        key = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        keep_indices.append(idx)

    return dataset.select(keep_indices), removed


def _collect_quality_metrics(dataset, pre_dedup_count, dedup_removed):
    total = len(dataset)
    with_tests = 0
    with_prompt = 0
    with_asserts = 0
    answer_parse_ok = 0

    for row in dataset:
        prompt = (row.get("prompt", "") or "").strip()
        tests = (row.get("tests", "") or "").strip()
        if prompt:
            with_prompt += 1
        if tests:
            with_tests += 1
            if "assert " in tests:
                with_asserts += 1
        if row.get("answer_ast_ok", False):
            answer_parse_ok += 1

    test_coverage = (with_tests / total) if total else 0.0
    prompt_coverage = (with_prompt / total) if total else 0.0
    unique_ratio = (total / pre_dedup_count) if pre_dedup_count else 0.0
    dedup_ratio = (dedup_removed / pre_dedup_count) if pre_dedup_count else 0.0
    answer_ast_parse_rate = (answer_parse_ok / total) if total else 0.0
    assert_density_in_tests = (with_asserts / with_tests) if with_tests else 0.0

    return {
        "total": total,
        "with_tests": with_tests,
        "with_prompt": with_prompt,
        "with_asserts": with_asserts,
        "test_coverage": test_coverage,
        "prompt_coverage": prompt_coverage,
        "unique_ratio": unique_ratio,
        "dedup_ratio": dedup_ratio,
        "answer_ast_parse_rate": answer_ast_parse_rate,
        "assert_density_in_tests": assert_density_in_tests,
    }


def _stable_hash_bucket(text, modulo=1000):
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _split_train_val_holdout(dataset, holdout_policy="source_hash_v1", holdout_fraction=0.10, val_fraction=0.05):
    holdout_fraction = max(0.01, min(0.5, holdout_fraction))
    val_fraction = max(0.01, min(0.4, val_fraction))
    holdout_mod = max(1, int(1000 * holdout_fraction))
    val_mod = max(1, int(1000 * val_fraction))

    train_idx = []
    val_idx = []
    holdout_idx = []

    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", "") or ""
        source = row.get("source", "unknown")
        answer = _extract_answer_from_text(row.get("text", ""))
        key = f"{source}||{prompt}||{answer}"

        bucket = _stable_hash_bucket(key, modulo=1000)
        if holdout_policy == "source_hash_v1" and bucket < holdout_mod:
            holdout_idx.append(idx)
            continue
        if bucket < holdout_mod + val_mod:
            val_idx.append(idx)
        else:
            train_idx.append(idx)

    if not train_idx:
        raise RuntimeError("Train split is empty after holdout split.")
    if not val_idx:
        raise RuntimeError("Val split is empty after holdout split.")
    if not holdout_idx:
        raise RuntimeError("Holdout split is empty after holdout split.")

    return {
        "train": dataset.select(train_idx),
        "val_strict": dataset.select(val_idx),
        "holdout_clean": dataset.select(holdout_idx),
    }


def _print_source_report(source_stats):
    print("\n=== Dataset Source Report ===")
    for source_name, count in source_stats.items():
        print(f"- {source_name}: {count}")
    print("=============================\n")


def _safe_git_commit():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _sample_key(row):
    prompt = (row.get("prompt", "") or "").strip()
    answer = _extract_answer_from_text(row.get("text", ""))
    source = str(row.get("source", "") or "unknown").strip().lower()
    key = f"{source}||{prompt}||{answer}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _assert_split_disjoint(splits):
    key_sets = {}
    for split_name, ds in splits.items():
        keys = set()
        for row in ds:
            keys.add(_sample_key(row))
        key_sets[split_name] = keys
    split_names = sorted(key_sets.keys())
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = len(key_sets[left].intersection(key_sets[right]))
            if overlap > 0:
                raise RuntimeError(
                    f"Split contamination detected between '{left}' and '{right}': overlap={overlap}"
                )


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_sota_dataset(
    output_dir="./sota_slm_coding_dataset",
    swe_supervised_output_dir="./swe_supervised_dataset",
    swe_supervised_max_samples=120000,
    require_swe_supervised=True,
    min_total_samples=20000,
    min_test_coverage=0.08,
    min_prompt_coverage=0.99,
    min_unique_ratio=0.75,
    max_missing_sources=8,
    source_weights="code:0.55,repo:0.30,reasoning:0.15",
    max_samples_per_source=25000,
    min_answer_ast_parse_rate=0.98,
    min_assert_density_in_tests=0.65,
    holdout_policy="source_hash_v1",
    holdout_fraction=0.10,
    val_fraction=0.05,
    seed=3407,
):
    """
    Kombiniert Reasoning-Traces und verifizierte Bugs zu einem hochqualitativen SFT-Korpus.
    """
    print("Initiating dataset curation pipeline...")
    source_stats = {}
    missing_sources = []
    swe_supervised_parts = []
    
    # 1. Distilled Rationales (OpenCodeReasoning)
    print("Loading OpenCodeReasoning subset (focused on high diversity)...")
    try:
        ds_reasoning = load_dataset("nvidia/OpenCodeReasoning", split="train[:25000]")
        ds_reasoning = ds_reasoning.filter(
            lambda x: _has_nonempty_text(x.get("instruction"))
            and _has_nonempty_text(x.get("solution"))
        )
        ds_reasoning = ds_reasoning.map(format_reasoning_prompt, remove_columns=ds_reasoning.column_names)
        ds_reasoning = _tag_source(ds_reasoning, "nvidia/OpenCodeReasoning")
        source_stats["nvidia/OpenCodeReasoning"] = len(ds_reasoning)
    except Exception as e:
        print(f"Warnung: Konnte nvidia/OpenCodeReasoning nicht laden. Fehler: {e}")
        ds_reasoning = None
        source_stats["nvidia/OpenCodeReasoning"] = 0
        missing_sources.append("nvidia/OpenCodeReasoning")

    # 1b. Bespoke Stratos (SOTA Reasoning)
    print("Loading Bespoke-Stratos for O1-level reasoning...")
    try:
        ds_stratos = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train[:15000]")
        ds_stratos = ds_stratos.filter(
            lambda x: _has_nonempty_text(x.get("prompt", x.get("messages")))
            and _has_nonempty_text(x.get("response", x.get("generation")))
        )
        ds_stratos = ds_stratos.map(format_stratos_prompt, remove_columns=ds_stratos.column_names)
        ds_stratos = _tag_source(ds_stratos, "HuggingFaceH4/Bespoke-Stratos-17k")
        source_stats["HuggingFaceH4/Bespoke-Stratos-17k"] = len(ds_stratos)
    except Exception as e:
        print(f"Warnung: Konnte Stratos nicht laden. Fehler: {e}")
        ds_stratos = None
        source_stats["HuggingFaceH4/Bespoke-Stratos-17k"] = 0
        missing_sources.append("HuggingFaceH4/Bespoke-Stratos-17k")

    # 2. Verified Bugs (SWE-bench Lite)
    print("Loading Verified Bugs for repository-level alignment...")
    try:
        ds_bugs = load_dataset("princeton-nlp/SWE-bench_Lite", split="train[:15000]")
        ds_swe_supervised_bugs = ds_bugs.map(
            lambda x: format_swe_supervised_bug(
                x,
                source_name="princeton-nlp/SWE-bench_Lite",
            ),
            remove_columns=ds_bugs.column_names,
        )
        swe_supervised_parts.append(ds_swe_supervised_bugs)
        ds_bugs = ds_bugs.filter(
            lambda x: _has_nonempty_text(x.get("problem_statement"))
            and _has_nonempty_text(x.get("patch"))
        )
        ds_bugs = ds_bugs.map(format_bug_prompt, remove_columns=ds_bugs.column_names)
        ds_bugs = _tag_source(ds_bugs, "princeton-nlp/SWE-bench_Lite")
        source_stats["princeton-nlp/SWE-bench_Lite"] = len(ds_bugs)
    except Exception as e:
        print(f"Warnung: Konnte princeton-nlp/SWE-bench_Lite nicht laden. Fehler: {e}")
        ds_bugs = None
        source_stats["princeton-nlp/SWE-bench_Lite"] = 0
        missing_sources.append("princeton-nlp/SWE-bench_Lite")

    # 3. Evol-Instruct (Komplexitäts-Skalierung für $20 Budget)
    print("Loading WizardLM Evol-Instruct for deep structural syntax...")
    try:
        ds_evol = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train[:25000]")
        # Filtere leere Zeilen aus
        ds_evol = ds_evol.filter(
            lambda x: _has_nonempty_text(x.get("instruction"))
            and _has_nonempty_text(x.get("output"))
        )
        ds_evol = ds_evol.map(format_evol_prompt, remove_columns=ds_evol.column_names)
        ds_evol = _tag_source(ds_evol, "WizardLM/WizardLM_evol_instruct_V2_196k")
        source_stats["WizardLM/WizardLM_evol_instruct_V2_196k"] = len(ds_evol)
    except Exception as e:
        print(f"Warnung: Konnte Evol-Instruct nicht laden. Fehler: {e}")
        ds_evol = None
        source_stats["WizardLM/WizardLM_evol_instruct_V2_196k"] = 0
        missing_sources.append("WizardLM/WizardLM_evol_instruct_V2_196k")

    # 4. Agentic Trajectories (SWE-Bench Multi-Turn Mastery)
    print("Loading SWE-agent Trajectories for multi-turn tool use...")
    try:
        ds_traj = load_dataset("princeton-nlp/SWE-agent-trajectories", split="train[:5000]")
        ds_swe_supervised_traj = ds_traj.map(
            lambda x: format_swe_supervised_trajectory(
                x,
                source_name="princeton-nlp/SWE-agent-trajectories",
            ),
            remove_columns=ds_traj.column_names,
        )
        swe_supervised_parts.append(ds_swe_supervised_traj)
        ds_traj = ds_traj.filter(
            lambda x: _has_nonempty_text(
                x.get("patch")
                or x.get("gold_patch")
                or x.get("model_patch")
                or x.get("final_patch")
            )
            and bool(x.get("trajectory"))
        )
        ds_traj = ds_traj.map(format_trajectory_prompt, remove_columns=ds_traj.column_names)
        ds_traj = _tag_source(ds_traj, "princeton-nlp/SWE-agent-trajectories")
        source_stats["princeton-nlp/SWE-agent-trajectories"] = len(ds_traj)
    except Exception as e:
        print(f"Warnung: Konnte SWE-agent Trajectories nicht laden. Fehler: {e}")
        ds_traj = None
        source_stats["princeton-nlp/SWE-agent-trajectories"] = 0
        missing_sources.append("princeton-nlp/SWE-agent-trajectories")

    # 5. NuminaMath (CoT Mathematical Logic)
    print("Loading NuminaMath for hardcore causal logic...")
    try:
        ds_math = load_dataset("AI-MO/NuminaMath-CoT", split="train[:10000]")
        ds_math = ds_math.filter(
            lambda x: _has_nonempty_text(x.get("problem"))
            and _has_nonempty_text(x.get("solution"))
        )
        ds_math = ds_math.map(format_math_prompt, remove_columns=ds_math.column_names)
        ds_math = _tag_source(ds_math, "AI-MO/NuminaMath-CoT")
        source_stats["AI-MO/NuminaMath-CoT"] = len(ds_math)
    except Exception as e:
        print(f"Warnung: Konnte NuminaMath nicht laden. Fehler: {e}")
        ds_math = None
        source_stats["AI-MO/NuminaMath-CoT"] = 0
        missing_sources.append("AI-MO/NuminaMath-CoT")

    # 6. CodeFeedback (Self-Healing)
    print("Loading CodeFeedback for compiler error self-healing...")
    try:
        ds_feedback = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train[:10000]")
        ds_feedback = ds_feedback.filter(
            lambda x: _has_nonempty_text(x.get("query"))
            and _has_nonempty_text(x.get("answer"))
        )
        ds_feedback = ds_feedback.map(format_codefeedback_prompt, remove_columns=ds_feedback.column_names)
        ds_feedback = _tag_source(ds_feedback, "m-a-p/CodeFeedback-Filtered-Instruction")
        source_stats["m-a-p/CodeFeedback-Filtered-Instruction"] = len(ds_feedback)
    except Exception as e:
        print(f"Warnung: Konnte CodeFeedback nicht laden. Fehler: {e}")
        ds_feedback = None
        source_stats["m-a-p/CodeFeedback-Filtered-Instruction"] = 0
        missing_sources.append("m-a-p/CodeFeedback-Filtered-Instruction")

    # 7. MCTS / Deep Exploring (OpenO1-SFT)
    print("Loading OpenO1-SFT for Monte Carlo Tree Search (MCTS) Backtracking logic...")
    try:
        # OpenO1 ist ein hochqualitativer Open-Source Datensatz, der genau diese 
        # "Ich versuche A -> A klappt nicht -> Ich versuche B -> B klappt" Struktur enthält.
        ds_mcts = load_dataset("O1-CODER/OpenO1-SFT", split="train[:8000]")
        ds_mcts = ds_mcts.filter(
            lambda x: _has_nonempty_text(x.get("instruction", x.get("prompt")))
            and _has_nonempty_text(x.get("output", x.get("response")))
        )
        ds_mcts = ds_mcts.map(format_mcts_prompt, remove_columns=ds_mcts.column_names)
        ds_mcts = _tag_source(ds_mcts, "O1-CODER/OpenO1-SFT")
        source_stats["O1-CODER/OpenO1-SFT"] = len(ds_mcts)
    except Exception as e:
        print(f"Warnung: Konnte MCTS Dataset nicht laden. Fehler: {e}")
        ds_mcts = None
        source_stats["O1-CODER/OpenO1-SFT"] = 0
        missing_sources.append("O1-CODER/OpenO1-SFT")

    # 8. Long-Context & Multi-File Healing (CommitPackFT)
    print("Loading CommitPackFT for Long-Context Multi-File Git Diff reasoning...")
    try:
        # Wir laden nur den Python Subset für garantierte Qualität, man kann hier aber 
        # später auch auf 'all' wechseln, wenn das Modell polyglot werden soll.
        ds_commit = load_dataset("bigcode/commitpackft", "python", split="train[:5000]")
        ds_commit = ds_commit.filter(
            lambda x: _has_nonempty_text(x.get("diff"))
            and _has_nonempty_text(x.get("message", x.get("old_contents")))
        )
        ds_commit = ds_commit.map(format_commitpack_prompt, remove_columns=ds_commit.column_names)
        ds_commit = _tag_source(ds_commit, "bigcode/commitpackft:python")
        source_stats["bigcode/commitpackft:python"] = len(ds_commit)
    except Exception as e:
        print(f"Warnung: Konnte CommitPackFT nicht laden. Fehler: {e}")
        ds_commit = None
        source_stats["bigcode/commitpackft:python"] = 0
        missing_sources.append("bigcode/commitpackft:python")

    # 9. SWE-bench-Live Verified (fresh real-world repo issues).
    print("Loading SWE-bench-Live verified for up-to-date repository fixing...")
    try:
        ds_swe_live = load_dataset("SWE-bench-Live/SWE-bench-Live", split="verified[:8000]")
        ds_swe_supervised_live = ds_swe_live.map(
            lambda x: format_swe_supervised_bug(
                x,
                source_name="SWE-bench-Live/SWE-bench-Live:verified",
            ),
            remove_columns=ds_swe_live.column_names,
        )
        swe_supervised_parts.append(ds_swe_supervised_live)
        ds_swe_live = ds_swe_live.filter(
            lambda x: _has_nonempty_text(x.get("problem_statement"))
            and _has_nonempty_text(x.get("patch"))
        )
        ds_swe_live = ds_swe_live.map(format_bug_prompt, remove_columns=ds_swe_live.column_names)
        ds_swe_live = _tag_source(ds_swe_live, "SWE-bench-Live/SWE-bench-Live:verified")
        source_stats["SWE-bench-Live/SWE-bench-Live:verified"] = len(ds_swe_live)
    except Exception as e:
        print(f"Warnung: Konnte SWE-bench-Live verified nicht laden. Fehler: {e}")
        ds_swe_live = None
        source_stats["SWE-bench-Live/SWE-bench-Live:verified"] = 0
        missing_sources.append("SWE-bench-Live/SWE-bench-Live:verified")

    # Aggregation, Reweighting, Dedup & Split
    source_datasets = {
        "nvidia/OpenCodeReasoning": ds_reasoning,
        "HuggingFaceH4/Bespoke-Stratos-17k": ds_stratos,
        "princeton-nlp/SWE-bench_Lite": ds_bugs,
        "WizardLM/WizardLM_evol_instruct_V2_196k": ds_evol,
        "princeton-nlp/SWE-agent-trajectories": ds_traj,
        "AI-MO/NuminaMath-CoT": ds_math,
        "m-a-p/CodeFeedback-Filtered-Instruction": ds_feedback,
        "O1-CODER/OpenO1-SFT": ds_mcts,
        "bigcode/commitpackft:python": ds_commit,
        "SWE-bench-Live/SWE-bench-Live:verified": ds_swe_live,
    }
    source_datasets = {k: v for k, v in source_datasets.items() if v is not None and len(v) > 0}
    if not source_datasets:
        raise RuntimeError("Kritischer Fehler: Keine Datensätze konnten geladen werden.")

    _print_source_report(source_stats)
    if len(missing_sources) > max_missing_sources:
        raise RuntimeError(
            f"Zu viele fehlende Datenquellen ({len(missing_sources)} > {max_missing_sources}): {missing_sources}"
        )

    source_datasets = _cap_sources(
        source_datasets,
        max_samples_per_source=max_samples_per_source,
        seed=seed,
    )
    parsed_weights = _parse_source_weights(source_weights)
    final_ds = _mix_sources_by_category(
        source_datasets,
        source_weights=parsed_weights,
        seed=seed,
    )
    final_ds = final_ds.filter(
        lambda x: _has_nonempty_text(x.get("text"))
        and _has_nonempty_text(x.get("prompt"))
        and "<answer>" in _text(x.get("text"))
        and "</answer>" in _text(x.get("text"))
    )
    final_ds = final_ds.map(
        lambda x: {"answer_ast_ok": _is_answer_ast_parseable(x.get("text", ""))}
    )
    pre_dedup_count = len(final_ds)
    final_ds, dedup_removed = _drop_prompt_answer_duplicates(final_ds)
    metrics = _collect_quality_metrics(final_ds, pre_dedup_count, dedup_removed)

    print(
        "Quality Metrics: "
        f"total={metrics['total']}, "
        f"test_coverage={metrics['test_coverage']:.3f}, "
        f"assert_density_in_tests={metrics['assert_density_in_tests']:.3f}, "
        f"answer_ast_parse_rate={metrics['answer_ast_parse_rate']:.3f}, "
        f"prompt_coverage={metrics['prompt_coverage']:.3f}, "
        f"unique_ratio={metrics['unique_ratio']:.3f}, "
        f"dedup_ratio={metrics['dedup_ratio']:.3f}"
    )

    if metrics["total"] < min_total_samples:
        raise RuntimeError(
            f"Datensatz zu klein: {metrics['total']} < min_total_samples={min_total_samples}"
        )
    if metrics["test_coverage"] < min_test_coverage:
        raise RuntimeError(
            f"Test-Coverage zu niedrig: {metrics['test_coverage']:.3f} < min_test_coverage={min_test_coverage}"
        )
    if metrics["assert_density_in_tests"] < min_assert_density_in_tests:
        raise RuntimeError(
            "Assert-Density in Tests zu niedrig: "
            f"{metrics['assert_density_in_tests']:.3f} < min_assert_density_in_tests={min_assert_density_in_tests}"
        )
    if metrics["answer_ast_parse_rate"] < min_answer_ast_parse_rate:
        raise RuntimeError(
            "Answer-AST-Parse-Rate zu niedrig: "
            f"{metrics['answer_ast_parse_rate']:.3f} < min_answer_ast_parse_rate={min_answer_ast_parse_rate}"
        )
    if metrics["prompt_coverage"] < min_prompt_coverage:
        raise RuntimeError(
            f"Prompt-Coverage zu niedrig: {metrics['prompt_coverage']:.3f} < min_prompt_coverage={min_prompt_coverage}"
        )
    if metrics["unique_ratio"] < min_unique_ratio:
        raise RuntimeError(
            f"Unique-Ratio zu niedrig: {metrics['unique_ratio']:.3f} < min_unique_ratio={min_unique_ratio}"
        )

    # Remove non-parseable answers from final train/eval splits.
    final_ds = final_ds.filter(lambda x: bool(x.get("answer_ast_ok", False)))

    splits = _split_train_val_holdout(
        final_ds,
        holdout_policy=holdout_policy,
        holdout_fraction=holdout_fraction,
        val_fraction=val_fraction,
    )
    _assert_split_disjoint(splits)
    dataset_dict = DatasetDict(
        {
            "train": splits["train"],
            "val_strict": splits["val_strict"],
            "holdout_clean": splits["holdout_clean"],
        }
    )
    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print(
        "Dataset successfully compiled and split: "
        f"train={len(splits['train'])}, "
        f"val_strict={len(splits['val_strict'])}, "
        f"holdout_clean={len(splits['holdout_clean'])}"
    )
    print(f"Saved to: {output_dir}")
    main_manifest = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": _safe_git_commit(),
        "seed": int(seed),
        "schema_version": "sota_data_manifest_v1",
        "output_dir": output_dir,
        "quality_thresholds": {
            "min_total_samples": min_total_samples,
            "min_test_coverage": min_test_coverage,
            "min_prompt_coverage": min_prompt_coverage,
            "min_unique_ratio": min_unique_ratio,
            "min_answer_ast_parse_rate": min_answer_ast_parse_rate,
            "min_assert_density_in_tests": min_assert_density_in_tests,
        },
        "holdout_policy": holdout_policy,
        "holdout_fraction": holdout_fraction,
        "val_fraction": val_fraction,
        "source_weights": parsed_weights,
        "source_stats_loaded": source_stats,
        "missing_sources": missing_sources,
        "metrics": metrics,
        "split_sizes": {
            "train": len(splits["train"]),
            "val_strict": len(splits["val_strict"]),
            "holdout_clean": len(splits["holdout_clean"]),
        },
        "dataset_fingerprints": {
            "train": getattr(splits["train"], "_fingerprint", ""),
            "val_strict": getattr(splits["val_strict"], "_fingerprint", ""),
            "holdout_clean": getattr(splits["holdout_clean"], "_fingerprint", ""),
        },
    }
    _write_json(os.path.join(output_dir, "dataset_manifest.json"), main_manifest)
    print(f"Saved dataset manifest to: {os.path.join(output_dir, 'dataset_manifest.json')}")

    # Build dedicated SWE-supervised dataset (issue->patch + trajectories) for focused SFT/ORPO/RL phases.
    swe_supervised_parts = [ds for ds in swe_supervised_parts if ds is not None and len(ds) > 0]
    if swe_supervised_parts:
        swe_ds = concatenate_datasets(swe_supervised_parts).shuffle(seed=seed)
        swe_ds = swe_ds.filter(
            lambda x: _has_nonempty_text(x.get("prompt"))
            and _has_nonempty_text(x.get("patch"))
            and "<answer>" in _text(x.get("text"))
            and "</answer>" in _text(x.get("text"))
        )
        pre_swe = len(swe_ds)
        swe_ds, swe_dedup_removed = _drop_prompt_answer_duplicates(swe_ds)
        if swe_supervised_max_samples > 0 and len(swe_ds) > swe_supervised_max_samples:
            swe_ds = swe_ds.select(range(int(swe_supervised_max_samples)))

        swe_splits = _split_train_val_holdout(
            swe_ds,
            holdout_policy=holdout_policy,
            holdout_fraction=holdout_fraction,
            val_fraction=val_fraction,
        )
        _assert_split_disjoint(swe_splits)
        swe_dataset_dict = DatasetDict(
            {
                "train": swe_splits["train"],
                "val_strict": swe_splits["val_strict"],
                "holdout_clean": swe_splits["holdout_clean"],
            }
        )
        os.makedirs(os.path.dirname(swe_supervised_output_dir) or ".", exist_ok=True)
        swe_dataset_dict.save_to_disk(swe_supervised_output_dir)
        print(
            "SWE-supervised dataset compiled: "
            f"raw={pre_swe}, dedup_removed={swe_dedup_removed}, "
            f"train={len(swe_splits['train'])}, val_strict={len(swe_splits['val_strict'])}, "
            f"holdout_clean={len(swe_splits['holdout_clean'])}"
        )
        print(f"Saved SWE-supervised dataset to: {swe_supervised_output_dir}")
        swe_manifest = {
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "git_commit": _safe_git_commit(),
            "seed": int(seed),
            "schema_version": "swe_supervised_manifest_v1",
            "output_dir": swe_supervised_output_dir,
            "raw_samples": int(pre_swe),
            "dedup_removed": int(swe_dedup_removed),
            "max_samples": int(swe_supervised_max_samples),
            "holdout_policy": holdout_policy,
            "holdout_fraction": holdout_fraction,
            "val_fraction": val_fraction,
            "split_sizes": {
                "train": len(swe_splits["train"]),
                "val_strict": len(swe_splits["val_strict"]),
                "holdout_clean": len(swe_splits["holdout_clean"]),
            },
            "dataset_fingerprints": {
                "train": getattr(swe_splits["train"], "_fingerprint", ""),
                "val_strict": getattr(swe_splits["val_strict"], "_fingerprint", ""),
                "holdout_clean": getattr(swe_splits["holdout_clean"], "_fingerprint", ""),
            },
        }
        _write_json(os.path.join(swe_supervised_output_dir, "dataset_manifest.json"), swe_manifest)
        print(
            f"Saved SWE-supervised dataset manifest to: {os.path.join(swe_supervised_output_dir, 'dataset_manifest.json')}"
        )
    elif require_swe_supervised:
        raise RuntimeError(
            "SWE-supervised dataset requested but no SWE issue/trajectory sources were available."
        )
    else:
        print("WARNUNG: SWE-supervised dataset could not be built (no SWE sources available).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build curated SOTA SLM coding dataset.")
    parser.add_argument("--output-dir", default="./sota_slm_coding_dataset")
    parser.add_argument("--swe-supervised-output-dir", default="./swe_supervised_dataset")
    parser.add_argument("--swe-supervised-max-samples", type=int, default=120000)
    parser.add_argument("--allow-missing-swe-supervised", action="store_true")
    parser.add_argument("--min-total-samples", type=int, default=20000)
    parser.add_argument("--min-test-coverage", type=float, default=0.08)
    parser.add_argument("--min-prompt-coverage", type=float, default=0.99)
    parser.add_argument("--min-unique-ratio", type=float, default=0.75)
    parser.add_argument("--max-missing-sources", type=int, default=8)
    parser.add_argument("--source-weights", default="code:0.55,repo:0.30,reasoning:0.15")
    parser.add_argument("--max-samples-per-source", type=int, default=25000)
    parser.add_argument("--min-answer-ast-parse-rate", type=float, default=0.98)
    parser.add_argument("--min-assert-density-in-tests", type=float, default=0.65)
    parser.add_argument("--holdout-policy", default="source_hash_v1")
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    build_sota_dataset(
        output_dir=args.output_dir,
        swe_supervised_output_dir=args.swe_supervised_output_dir,
        swe_supervised_max_samples=args.swe_supervised_max_samples,
        require_swe_supervised=not args.allow_missing_swe_supervised,
        min_total_samples=args.min_total_samples,
        min_test_coverage=args.min_test_coverage,
        min_prompt_coverage=args.min_prompt_coverage,
        min_unique_ratio=args.min_unique_ratio,
        max_missing_sources=args.max_missing_sources,
        source_weights=args.source_weights,
        max_samples_per_source=args.max_samples_per_source,
        min_answer_ast_parse_rate=args.min_answer_ast_parse_rate,
        min_assert_density_in_tests=args.min_assert_density_in_tests,
        holdout_policy=args.holdout_policy,
        holdout_fraction=args.holdout_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
