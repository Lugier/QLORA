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


def length_penalty_reward_func(prompts, completions, answer, **kwargs):
    """
    Bestraft massiv, wenn das Modell versucht, durch endlose (oft repetitive)
    "Gedankengänge" den Reward zu hacken oder Token-Limits zu sprengen.
    Führt zu präzisem, effizientem Denken.
    """
    rewards = []
    responses = [comp[0]["content"] if isinstance(comp, list) else comp["content"] for comp in completions]
    
    for resp in responses:
        tokens = len(resp.split()) # Simple Heuristik (reicht für Penalty)
        
        # Ab 500 wörtern wird es langsam kritisch für ein einfaches SLM
        if tokens < 300:
            rewards.append(0.0) # Perfekt
        elif tokens < 600:
            rewards.append(-0.2) # Okay, aber etwas lang
        elif tokens < 1000:
            rewards.append(-0.5) # Zu geschwätzig
        else:
            # Massive Strafe für Endlos-Loops oder extreme Geschwätzigkeit
            penalty = min(-2.0, -0.5 - ((tokens - 1000) * 0.001))
            rewards.append(penalty)
            
    return rewards

def self_verification_reward_func(prompts, completions, answer, **kwargs):
    """
    Self-Verification (TDD-Alignment):
    Sucht im <reasoning> Block explizit nach proaktiven Testfällen (asserts).
    Belohnt Modelle, die Probleme durch eigenes Test-Driven-Development validieren.
    """
    rewards = []
    responses = [comp[0]["content"] if isinstance(comp, list) else comp["content"] for comp in completions]
    
    for resp in responses:
        reasoning = extract_xml_content(resp, "reasoning")
        
        if not reasoning:
            rewards.append(0.0)
            continue
            
        # Wir zählen, ob das Modell proaktiv Assertions oder Python-Syntax-Checks
        # in seine Denkphase einbaut
        assert_count = reasoning.count("assert ")
        test_count = reasoning.count("def test_")
        
        if assert_count >= 2 or test_count >= 1:
            rewards.append(0.5) # Exzellentes Agentic Behavior (TDD)
        elif assert_count == 1:
            rewards.append(0.2) # Guter Anfang
        else:
            rewards.append(0.0) # Kein Self-Verification
            
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
