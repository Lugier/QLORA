import os
import hashlib
import argparse
import re
from datasets import load_dataset, concatenate_datasets

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


def _drop_prompt_duplicates(dataset):
    """
    Remove exact/near-exact prompt duplicates to reduce memorization and noisy weighting.
    """
    seen = set()
    keep_indices = []
    removed = 0

    for idx, row in enumerate(dataset):
        prompt = row.get("prompt", "") or _prompt_prefix_from_text(row.get("text", ""))
        key = hashlib.sha256(_normalize_for_dedup(prompt).encode("utf-8")).hexdigest()
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

    for row in dataset:
        prompt = (row.get("prompt", "") or "").strip()
        tests = (row.get("tests", "") or "").strip()
        if prompt:
            with_prompt += 1
        if tests:
            with_tests += 1

    test_coverage = (with_tests / total) if total else 0.0
    prompt_coverage = (with_prompt / total) if total else 0.0
    unique_ratio = (total / pre_dedup_count) if pre_dedup_count else 0.0
    dedup_ratio = (dedup_removed / pre_dedup_count) if pre_dedup_count else 0.0

    return {
        "total": total,
        "with_tests": with_tests,
        "with_prompt": with_prompt,
        "test_coverage": test_coverage,
        "prompt_coverage": prompt_coverage,
        "unique_ratio": unique_ratio,
        "dedup_ratio": dedup_ratio,
    }


def _print_source_report(source_stats):
    print("\n=== Dataset Source Report ===")
    for source_name, count in source_stats.items():
        print(f"- {source_name}: {count}")
    print("=============================\n")


def build_sota_dataset(
    output_dir="./sota_slm_coding_dataset",
    min_total_samples=20000,
    min_test_coverage=0.08,
    min_prompt_coverage=0.99,
    min_unique_ratio=0.75,
    max_missing_sources=6,
):
    """
    Kombiniert Reasoning-Traces und verifizierte Bugs zu einem hochqualitativen SFT-Korpus.
    """
    print("Initiating dataset curation pipeline...")
    source_stats = {}
    missing_sources = []
    
    # 1. Distilled Rationales (OpenCodeReasoning)
    print("Loading OpenCodeReasoning subset (focused on high diversity)...")
    try:
        ds_reasoning = load_dataset("nvidia/OpenCodeReasoning", split="train[:25000]")
        ds_reasoning = ds_reasoning.filter(
            lambda x: _has_nonempty_text(x.get("instruction"))
            and _has_nonempty_text(x.get("solution"))
        )
        ds_reasoning = ds_reasoning.map(format_reasoning_prompt, remove_columns=ds_reasoning.column_names)
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
        ds_bugs = ds_bugs.filter(
            lambda x: _has_nonempty_text(x.get("problem_statement"))
            and _has_nonempty_text(x.get("patch"))
        )
        ds_bugs = ds_bugs.map(format_bug_prompt, remove_columns=ds_bugs.column_names)
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
        source_stats["bigcode/commitpackft:python"] = len(ds_commit)
    except Exception as e:
        print(f"Warnung: Konnte CommitPackFT nicht laden. Fehler: {e}")
        ds_commit = None
        source_stats["bigcode/commitpackft:python"] = 0
        missing_sources.append("bigcode/commitpackft:python")

    # Aggregation & Speicherung
    datasets_to_concat = []
    if ds_reasoning is not None: datasets_to_concat.append(ds_reasoning)
    if ds_stratos is not None: datasets_to_concat.append(ds_stratos)
    if ds_bugs is not None: datasets_to_concat.append(ds_bugs)
    if ds_evol is not None: datasets_to_concat.append(ds_evol)
    if ds_traj is not None: datasets_to_concat.append(ds_traj)
    if ds_math is not None: datasets_to_concat.append(ds_math)
    if ds_feedback is not None: datasets_to_concat.append(ds_feedback)
    if ds_mcts is not None: datasets_to_concat.append(ds_mcts)
    if ds_commit is not None: datasets_to_concat.append(ds_commit)
    
    if not datasets_to_concat:
        raise RuntimeError("Kritischer Fehler: Keine Datensätze konnten geladen werden.")

    _print_source_report(source_stats)
    if len(missing_sources) > max_missing_sources:
        raise RuntimeError(
            f"Zu viele fehlende Datenquellen ({len(missing_sources)} > {max_missing_sources}): {missing_sources}"
        )

    final_ds = concatenate_datasets(datasets_to_concat).shuffle(seed=3407)
    final_ds = final_ds.filter(
        lambda x: _has_nonempty_text(x.get("text"))
        and _has_nonempty_text(x.get("prompt"))
        and "<answer>" in _text(x.get("text"))
        and "</answer>" in _text(x.get("text"))
    )
    pre_dedup_count = len(final_ds)
    final_ds, dedup_removed = _drop_prompt_duplicates(final_ds)
    metrics = _collect_quality_metrics(final_ds, pre_dedup_count, dedup_removed)

    print(
        "Quality Metrics: "
        f"total={metrics['total']}, "
        f"test_coverage={metrics['test_coverage']:.3f}, "
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
    if metrics["prompt_coverage"] < min_prompt_coverage:
        raise RuntimeError(
            f"Prompt-Coverage zu niedrig: {metrics['prompt_coverage']:.3f} < min_prompt_coverage={min_prompt_coverage}"
        )
    if metrics["unique_ratio"] < min_unique_ratio:
        raise RuntimeError(
            f"Unique-Ratio zu niedrig: {metrics['unique_ratio']:.3f} < min_unique_ratio={min_unique_ratio}"
        )
    
    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
    final_ds.save_to_disk(output_dir)
    print(f"Dataset successfully compiled with {len(final_ds)} reasoning-augmented instances.")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build curated SOTA SLM coding dataset.")
    parser.add_argument("--output-dir", default="./sota_slm_coding_dataset")
    parser.add_argument("--min-total-samples", type=int, default=20000)
    parser.add_argument("--min-test-coverage", type=float, default=0.08)
    parser.add_argument("--min-prompt-coverage", type=float, default=0.99)
    parser.add_argument("--min-unique-ratio", type=float, default=0.75)
    parser.add_argument("--max-missing-sources", type=int, default=6)
    args = parser.parse_args()

    build_sota_dataset(
        output_dir=args.output_dir,
        min_total_samples=args.min_total_samples,
        min_test_coverage=args.min_test_coverage,
        min_prompt_coverage=args.min_prompt_coverage,
        min_unique_ratio=args.min_unique_ratio,
        max_missing_sources=args.max_missing_sources,
    )
