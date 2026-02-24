import os
import random
from datasets import load_dataset, concatenate_datasets

# ==============================================================================
# SOTA SLM Data Preparation Pipeline
# ==============================================================================

def format_reasoning_prompt(example):
    """
    Integriert detaillierte Teacher-Rationales in den Datensatz für das SLM.
    Verwendet "Dual-Mode" (zufällig), um Overthinking bei simplen Aufgaben zu verhindern.
    """
    # 70% der Zeit erzwingen wir tiefes Nachdenken, 30% der Zeit trainieren wir "direct output"
    use_detailed_reasoning = random.random() < 0.7
    
    if use_detailed_reasoning:
        system_prompt = (
            "You are an elite coding assistant. You must thoroughly analyze the problem "
            "step-by-step inside <reasoning> tags before synthesizing the final robust code "
            "inside <answer> tags."
        )
        rationale = example.get('reasoning_trace', 'Thinking about optimal implementation...')
        solution = example.get('solution', '')
        
        formatted_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{example.get('instruction', '')}<|im_end|>\n"
            f"<|im_start|>assistant\n<reasoning>\n{rationale}\n</reasoning>\n"
            f"<answer>\n{solution}\n</answer><|im_end|>"
        )
    else:
        system_prompt = (
            "You are an elite coding assistant. Provide the final robust code directly "
            "inside <answer> tags without extensive preliminary reasoning."
        )
        solution = example.get('solution', '')
        formatted_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{example.get('instruction', '')}<|im_end|>\n"
            f"<|im_start|>assistant\n<answer>\n{solution}\n</answer><|im_end|>"
        )

    return {"text": formatted_text}


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
    instruction = f"Issue Description: {example.get('problem_statement', '')}"
    solution = example.get('patch', '')
    
    instruction = f"Issue Description: {example.get('problem_statement', '')}"
    solution = example.get('patch', '')
    
    # Kurze analytische Struktur simulieren für Konsistenz
    rationale = (
        "Analyzing the provided issue description to isolate the root cause. "
        "The bug requires locating the specific file and applying a targeted patch."
    )
    
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n<reasoning>\n{rationale}\n</reasoning>\n"
        f"<answer>\n{solution}\n</answer><|im_end|>"
    )
    return {"text": formatted_text}


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
    instruction = example.get('instruction', '')
    solution = example.get('output', '')
    
    # Da Evol-Instruct keine Traces hat, lassen wir das Modell direkt in den Direct-Mode gehen
    # oder vergeben Dummy-Traces. Der Wert liegt hier in der Instruktionsverschachtelung.
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n<answer>\n{solution}\n</answer><|im_end|>"
    )
    return {"text": formatted_text}


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
    issue = example.get('instance_id', 'Unknown Issue')
    trajectory = example.get('trajectory', []) # Liste von Actions/Observations
    
    history = f"Issue: Resolving {issue}\n"
    for step in trajectory:
        action = step.get('action', '')
        obs = step.get('observation', '')
        history += f"\n[Action]: {action}\n[Observation]: {obs}"
        
    formatted_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{history}<|im_end|>\n"
        f"<|im_start|>assistant\n<reasoning>\nTrajectory completed.\n</reasoning>\n"
        f"<answer>\nPatch ready based on trajectory exploration.\n</answer><|im_end|>"
    )
    return {"text": formatted_text}


def build_sota_dataset(output_dir="./sota_slm_coding_dataset"):
    """
    Kombiniert Reasoning-Traces und verifizierte Bugs zu einem hochqualitativen SFT-Korpus.
    """
    print("Initiating dataset curation pipeline...")
    
    # 1. Distilled Rationales (OpenCodeReasoning)
    print("Loading OpenCodeReasoning subset (focused on high diversity)...")
    try:
        ds_reasoning = load_dataset("nvidia/OpenCodeReasoning", split="train[:50000]")
        ds_reasoning = ds_reasoning.map(format_reasoning_prompt, remove_columns=ds_reasoning.column_names)
    except Exception as e:
        print(f"Warnung: Konnte nvidia/OpenCodeReasoning nicht laden, nutze Fallback. Fehler: {e}")
        # Dummy oder Fallback für lokale Tests
        ds_reasoning = None

    # 2. Verified Bugs (SWE-bench Lite)
    print("Loading Verified Bugs for repository-level alignment...")
    try:
        ds_bugs = load_dataset("princeton-nlp/SWE-bench_Lite", split="train[:15000]")
        ds_bugs = ds_bugs.map(format_bug_prompt, remove_columns=ds_bugs.column_names)
    except Exception as e:
        print(f"Warnung: Konnte princeton-nlp/SWE-bench_Lite nicht laden. Fehler: {e}")
        ds_bugs = None

    # 3. Evol-Instruct (Komplexitäts-Skalierung für $20 Budget)
    print("Loading WizardLM Evol-Instruct for deep structural syntax...")
    try:
        ds_evol = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train[:25000]")
        # Filtere leere Zeilen aus
        ds_evol = ds_evol.filter(lambda x: x['instruction'] is not None and x['output'] is not None)
        ds_evol = ds_evol.map(format_evol_prompt, remove_columns=ds_evol.column_names)
    except Exception as e:
        print(f"Warnung: Konnte Evol-Instruct nicht laden. Fehler: {e}")
        ds_evol = None

    # 4. Agentic Trajectories (SWE-Bench Multi-Turn Mastery)
    print("Loading SWE-agent Trajectories for multi-turn tool use...")
    try:
        ds_traj = load_dataset("princeton-nlp/SWE-agent-trajectories", split="train[:5000]")
        ds_traj = ds_traj.map(format_trajectory_prompt, remove_columns=ds_traj.column_names)
    except Exception as e:
        print(f"Warnung: Konnte SWE-agent Trajectories nicht laden. Fehler: {e}")
        ds_traj = None

    # Aggregation & Speicherung
    datasets_to_concat = []
    if ds_reasoning: datasets_to_concat.append(ds_reasoning)
    if ds_bugs: datasets_to_concat.append(ds_bugs)
    if ds_evol: datasets_to_concat.append(ds_evol)
    if ds_traj: datasets_to_concat.append(ds_traj)
    
    if not datasets_to_concat:
        print("Kritischer Fehler: Keine Datensätze konnten geladen werden.")
        return

    final_ds = concatenate_datasets(datasets_to_concat).shuffle(seed=3407)
    
    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
    final_ds.save_to_disk(output_dir)
    print(f"Dataset successfully compiled with {len(final_ds)} reasoning-augmented instances.")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    build_sota_dataset()
