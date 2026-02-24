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

    # 3. Aggregation & Speicherung
    datasets_to_concat = []
    if ds_reasoning: datasets_to_concat.append(ds_reasoning)
    if ds_bugs: datasets_to_concat.append(ds_bugs)
    
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
