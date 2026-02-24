import re
from typing import List, Dict, Any
from sandbox import run_code_in_sandbox

# ==============================================================================
# SOTA SLM Reward Functions (GRPO)
# ==============================================================================

def extract_xml_content(text: str, tag: str) -> str:
    """Extrahiert sicher Inhalte zwischen XML/HTML Tags wie <answer>...</answer>"""
    # Verwendet non-greedy .*? um das erste korrekte Schließen zu fangen
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# Globaler State für AERO (Adaptive Execution Reward Optimization)
# Wir tracken den Annäherungsfortschritt im Training
AERO_GLOBAL_STEP = 0
AERO_MAX_STEPS = 1500 # Sollte identisch mit max_steps im GRPO Trainer sein

def get_aero_weights():
    """
    Berechnet dynamisch die Gewichtung der Rewards (AERO).
    Am Anfang (Step 0) ist Format-Reward hoch, Execution-Reward niedrig.
    Am Ende (Step 1500) ist Format-Reward 0, Execution-Reward bekommt einen massiven Multiplikator.
    """
    progress = min(1.0, AERO_GLOBAL_STEP / AERO_MAX_STEPS)
    
    # Format-Reward fadet von 1.0 (am Anfang) auf 0.1 (am Ende)
    format_weight = max(0.1, 1.0 - progress)
    
    # Execution-Reward skaliert von 1.0 (am Anfang) auf 2.5 (am Ende)
    exec_weight = 1.0 + (1.5 * progress)
    
    return format_weight, exec_weight


def strict_format_reward_func(prompts: List[str], completions: List[Dict[str, str]], **kwargs) -> List[float]:
    """
    Belohnt Modelle für das strikte Einhalten der kognitiven Trennung
    (<reasoning> gefolgt von <answer>), ist aber tolerant gegenüber Whitespaces
    und zusätzlichem (vorher/nachher) Token-Müll, um Instabilität zu reduzieren.
    """
    rewards = []
    responses = [comp[0]["content"] if isinstance(comp, list) else comp["content"] for comp in completions]
    
    format_weight, _ = get_aero_weights() # Get AERO format weight
    
    for resp in responses:
        has_reasoning = "<reasoning>" in resp and "</reasoning>" in resp
        has_answer = "<answer>" in resp and "</answer>" in resp
        
        if has_answer:
            # Wenn answer vorhanden ist, gibt es einen Baseline-Reward.
            base_reward = 0.5
            # Voller Format-Reward nur, wenn auch reasoning da ist ODER der Prompt direct-mode verlangte
            # (Da Prompts hier schwer 100% dynamisch greifbar sind, belohnen wir schlichtweg saubere Tags).
            if has_reasoning:
                rewards.append(1.0 * format_weight) # Apply AERO scaling
            else:
                rewards.append(base_reward * format_weight) # Apply AERO scaling
        else:
            # Harte Strafe, wenn die Antwort komplett das Format bricht 
            # (und dadurch die Auswertung für execution failed)
            rewards.append(-1.0)
            
    return rewards


def length_penalty_reward_func(prompts: List[str], completions: List[Dict[str, str]], **kwargs) -> List[float]:
    """
    Verhindert "Overthinking", das bei O1/R1-Modellen eine Token-Verschwendung darstellt.
    Bestraft exzessiv lange Lösungen für möglicherweise sehr kurze Probleme.
    """
    rewards = []
    responses = [comp[0]["content"] if isinstance(comp, list) else comp["content"] for comp in completions]
    
    # Toleranzschwelle. Lösungen unter 1500 Zeichen werden nicht bestraft.
    MAX_EFFICIENT_LENGTH = 1500 
    
    for resp in responses:
        length = len(resp)
        if length > MAX_EFFICIENT_LENGTH:
            # Je länger, desto größer der Penalty (linear skaliert)
            penalty = -0.5 * ((length - MAX_EFFICIENT_LENGTH) / 1000)
            # Cap auf max -1.5, damit es formelle und ausführbare Rewards nicht auslöscht, 
            # aber deutlich nach unten zieht.
            rewards.append(max(-1.5, penalty))
        else:
            rewards.append(0.0) # Kein Bonus, nur fehlende Strafe
            
    return rewards


def execution_reward_func(prompts: List[str], completions: List[Dict[str, str]], answer: List[str], **kwargs) -> List[float]:
    """
    Der Kern der Performance-Explosion ("Ground Truth Alignment").
    Extrahiert den Quellcode, kombiniert ihn mit versteckten Unit-Tests
    und führt ihn sicher über den AST-Sandbox-Interpreter aus.
    Erzeugt ein starkes, unbestechliches Gradientensignal zur Code-Korrektheit.
    """
    global AERO_GLOBAL_STEP
    AERO_GLOBAL_STEP += 1 # Inkrement beim Aufruf der Sandbox (pro Batch)
    
    _, exec_weight = get_aero_weights() # Get AERO execution weight
    rewards = []
    responses = [comp[0]["content"] if isinstance(comp, list) else comp["content"] for comp in completions]
    
    for resp, expected_tests in zip(responses, answer):
        extracted_code = extract_xml_content(resp, "answer")
        
        if not extracted_code:
            # Massive Bestrafung für formatlose/leere Ausgaben, damit es nicht "durchrutscht"
            rewards.append(-1.5) 
            continue
        
        # Säubere möglichen Markdown-Syntax (```python ... ```) im Answer-Tag
        code_snippet = extracted_code.replace("```python", "").replace("```", "").strip()
        
        # Konkatenation des generierten Modells (Funktion/Klasse) mit den validierenden Asserts
        full_eval_code = f"{code_snippet}\n\n# --- Unit Tests ---\n{expected_tests}"
        
        # Sicherheitsgarantierte Ausführung
        # run_code_in_sandbox gibt direkt unseren float-Reward und die Exec-Time zurück
        base_score, exec_time = run_code_in_sandbox(full_eval_code, timeout=2.0)
        
        # Gestaffeltes Reward-Shaping (Dense Rewards) basierend auf Ausführungszeit
        final_score = base_score
        
        # Wenn der Code erfolgreich war, belohnen wir Effizienz
        if base_score == 2.0:
            if exec_time < 0.05:
                final_score += 0.5 # Exzellente O(1) oder O(n) Lösung
            elif exec_time > 0.5:
                final_score -= 0.5 # Funktionierender Code, aber ineffizient O(n^2) etc.
                
        # Wenn er überhaupt AST-parsable war, geben wir wenigstens minimalen Base-Float für Gradienten
        if base_score == 0.1: 
            final_score += 0.1
            
        # AERO Multiplikator anwenden (Execution wird im Late-Game massiv gepusht)
        rewards.append(final_score * exec_weight)
        
    return rewards
