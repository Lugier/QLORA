import os
from unsloth import FastLanguageModel

# ==============================================================================
# Model Export & Deployment Preparation
# Überführt die trainierten LoRA-Gewichte in das Zielformat.
# ==============================================================================

def export_model():
    model_path = "qwen_grpo_final"
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Trainiertes Modelverzeichnis '{model_path}' nicht gefunden. "
            "Sicherstellen, dass phase2_grpo.py erfolgreich ausgeführt wurde."
        )

    print("Loading optimized weights into memory...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 8192,  # Muss mit Training zusammenpassen
            dtype = None,
            load_in_4bit = False,
        )
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}") from e

    # 1. Export in das GGUF Format
    # Dies ist essentiell für die hochperformante Ausführung auf reinen CPU-Systemen 
    # oder Edge-Geräten (z.B. Macbook Llama.cpp).
    # q8_0 bietet den perfekten Schnittpunkt zwischen Memory (VRAM/RAM) 
    # und Erhalt komplexer Reasoning-Fähigkeit.
    print("Transpiling matrix weights to GGUF (q8_0)...")
    try:
        model.save_pretrained_gguf("model_export_gguf", tokenizer, quantization_method = "q8_0")
        print("[+] GGUF Export Successful in 'model_export_gguf'")
    except Exception as e:
        raise RuntimeError(f"GGUF Export Failed: {e}") from e

    # 2. Verschmelzen (Merge) in ein reguläres, nicht-quantisiertes 16-bit Format
    # Zwingend notwendig für den professionellen Einsatz wie API-Server via vLLM.
    print("Executing weight merge to 16-bit HuggingFace standard format...")
    try:
        model.save_pretrained_merged("model_export_hf", tokenizer, save_method = "merged_16bit")
        print("[+] HuggingFace 16-bit Merge Successful in 'model_export_hf'")
    except Exception as e:
        raise RuntimeError(f"HuggingFace 16-bit Merge Failed: {e}") from e

    print("======================================================")
    print("Deployment artifacts successfully generated.")
    print("Next Steps:")
    print("1. Local Execution (Edge): Use Llama.cpp with the generated GGUF files.")
    print("2. Server Execution (High Throughput): Start vLLM with the merged 16bit HF folder:")
    print("   python3 -m vllm.entrypoints.openai.api_server --model ./model_export_hf --max-model-len 8192")
    print("======================================================")

if __name__ == "__main__":
    export_model()
