import os
from datasets import load_from_disk, Dataset
from vllm import LLM, SamplingParams
from sandbox import run_code_in_sandbox
import re

# ==============================================================================
# Phase 1b: Best-of-N Rejection Sampling (Distillation)
# Generiert N Lösungen pro Prompt, testet sie in der Sandbox und behält nur
# die 100% perfekten (Reasoning + Code) für ein "flawless" SFT-Distillation.
# ==============================================================================

def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def generate_and_filter():
    model_path = "qwen_sft_lora"
    dataset_path = "./sota_slm_coding_dataset"
    output_path = "./sota_best_of_n_dataset"
    
    if not os.path.exists(model_path) or not os.path.exists(dataset_path):
        print("Model or dataset not found. Please run phase1_sft.py first.")
        return

    print("Loading optimized SFT model with vLLM backend...")
    
    # Nutzt vLLM, da wir massiven Durchsatz brauchen
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        tensor_parallel_size=1,
        enforce_eager=True # wichtig für LoRA
    )
    
    print("Loading prompt dataset...")
    # Wir nehmen ein Validation-Set oder subset für Distillation
    full_ds = load_from_disk(dataset_path)
    dataset = full_ds.select(range(min(2000, len(full_ds))))
    
    # N Generationen pro Prompt
    N = 16 
    sampling_params = SamplingParams(
        n=N,
        temperature=0.7, # Hohe Temperatur für Diversität
        top_p=0.9,
        max_tokens=1500,
        stop=["<|im_end|>"]
    )
    
    def _prompt_only(example):
        prompt = example.get("prompt")
        if prompt:
            return prompt
        text = example.get("text", "")
        split_token = "<|im_start|>assistant\n"
        if split_token in text:
            return text.split(split_token, 1)[0] + split_token
        return text

    prompts = [_prompt_only(example) for example in dataset]
    test_cases = [example.get("tests", "") or "" for example in dataset]
    
    print(f"Generating {N} solutions for {len(prompts)} prompts. This might take a while...")
    outputs = llm.generate(prompts, sampling_params)
    
    distilled_data = []
    skipped_without_tests = 0
    
    print("Evaluating generations in secure AST sandbox...")
    for idx, output in enumerate(outputs):
        best_completion = None
        perfect_score_found = False
        
        for completion in output.outputs:
            text = completion.text
            code = extract_xml_content(text, "answer")
            if not code:
                continue
            
            code = code.replace("```python", "").replace("```", "").strip()
            tests = (test_cases[idx] or "").strip()
            if not tests:
                skipped_without_tests += 1
                continue

            eval_code = f"{code}\n\n{test_cases[idx]}"
            
            score, exec_time = run_code_in_sandbox(eval_code, timeout=1.0)
            
            # Nur Antworten, die perfekt ausführen und einen sauberen AST haben
            if score == 2.0:
                best_completion = text
                perfect_score_found = True
                break # Eine perfekte reicht uns
                
        if perfect_score_found:
            distilled_prompt = prompts[idx] + best_completion
            distilled_data.append(
                {
                    "text": distilled_prompt,
                    "prompt": prompts[idx],
                    "tests": test_cases[idx],
                }
            )
            
    print(f"Distillation complete. Kept {len(distilled_data)} perfect trajectories out of {len(prompts)}.")
    if skipped_without_tests:
        print(f"Skipped {skipped_without_tests} candidate generations because no hidden tests were available.")
    
    # Speichern des destillierten, fehlerfreien Datensatzes
    if distilled_data:
        distilled_dataset = Dataset.from_list(distilled_data)
        distilled_dataset.save_to_disk(output_path)
        print(f"Saved flawless dataset to {output_path}. You can now run a brief SFT on this before GRPO.")
    else:
        print("No perfect solutions found. Check your model performance or sandbox constraints.")

if __name__ == "__main__":
    generate_and_filter()
