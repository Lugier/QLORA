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

def strict_format_reward_func(prompts: List[str], completions: List[Dict[str, str]], **kwargs) -> List[float]:
    """
    Belohnt Modelle für das strikte Einhalten der kognitiven Trennung
    (<reasoning> gefolgt von <answer>), ist aber tolerant gegenüber Whitespaces
    und zusätzlichem (vorher/nachher) Token-Müll, um Instabilität zu reduzieren.
    """
    rewards = []
    responses = [comp[0]["content"] if isinstance(comp, list) else comp["content"] for comp in completions]
    
    for resp in responses:
        has_reasoning = "<reasoning>" in resp and "</reasoning>" in resp
        has_answer = "<answer>" in resp and "</answer>" in resp
        
        if has_answer:
            # Wenn answer vorhanden ist, gibt es einen Baseline-Reward.
            base_reward = 0.5
            # Voller Format-Reward nur, wenn auch reasoning da ist ODER der Prompt direct-mode verlangte
            # (Da Prompts hier schwer 100% dynamisch greifbar sind, belohnen wir schlichtweg saubere Tags).
            if has_reasoning:
                rewards.append(1.0)
            else:
                rewards.append(base_reward)
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
    rewards = []
    responses = [comp[0]["content"] if isinstance(comp, list) else comp["content"] for comp in completions]
    
    # ZIP iteriert über die modellgenerierten Antworten und die versteckten Tests (`answer` = test case snippet)
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
        # run_code_in_sandbox gibt direkt unseren float-Reward zurück:
        # 2.0 (Perfekt), 0.5 (Failed Assert Logik Error), 0.1 (Runtime Error), -0.5 (Timeout), -1.0 (Hack Versuch)
        score = run_code_in_sandbox(full_eval_code, timeout=2.0)
        rewards.append(score)
        
    return rewards
