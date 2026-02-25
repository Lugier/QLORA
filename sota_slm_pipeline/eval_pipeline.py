import os
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from sandbox import run_code_in_sandbox

def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def generate_and_evaluate(model_path="qwen_grpo_final", num_samples=50):
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} not found. Ensure training is complete.")
        return

    print(f"Loading model '{model_path}' via vLLM for rapid evaluation...")
    llm = LLM(
        model=model_path,
        max_model_len=2048,
        tensor_parallel_size=1,
        enforce_eager=True # For LoRA or merged comp
    )

    print("Loading HumanEval/MBPP evaluation split...")
    eval_dataset = load_dataset("mbpp", "sanitized", split=f"test[:{num_samples}]")

    print("Formatting prompts for Agentic Reasoning...")
    system_prompt = (
        "You are an expert python developer. Analyze thoroughly in <reasoning> tags, "
        "then output purely the executable code in <answer> tags."
    )

    prompts = []
    tests = []
    
    for row in eval_dataset:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{row['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)
        test_str = "\n".join(row.get('test_list', []))
        tests.append(test_str)

    sampling_params = SamplingParams(
        temperature=0.0, # Greedy decoding for Pass@1
        max_tokens=1500,
        stop=["<|im_end|>"]
    )

    print(f"Generating answers for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    passed = 0
    failed = 0
    format_errors = 0

    print("Evaluating against secure Sandbox...")
    for idx, output in enumerate(outputs):
        text = output.outputs[0].text
        code = extract_xml_content(text, "answer")
        
        if not code:
            format_errors += 1
            failed += 1
            continue
            
        code = code.replace("```python", "").replace("```", "").strip()
        eval_code = f"{code}\n\n# --- Unit Tests ---\n{tests[idx]}"
        
        score, exec_time = run_code_in_sandbox(eval_code, timeout=2.0)
        
        if score == 2.0:
            passed += 1
        else:
            failed += 1

    pass_at_1 = (passed / len(prompts)) * 100

    print("\n" + "="*50)
    print("📈 EVALUATION REPORT (Pass@1)")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Total Samples: {len(prompts)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed} (Includes Format Errors: {format_errors})")
    print(f"Pass@1 Rate: {pass_at_1:.2f}%")
    print("="*50)

if __name__ == "__main__":
    generate_and_evaluate()
