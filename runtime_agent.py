import argparse
import math
import os
import re
from typing import Dict, List, Optional, Tuple

from verification import run_test_verifier
from vllm import LLM, SamplingParams


IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}

INCLUDE_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}


def extract_xml_content(text: str, tag: str) -> str:
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _strip_code_fences(text: str) -> str:
    cleaned = text.replace("```python", "").replace("```py", "").replace("```", "")
    return cleaned.strip()


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", (text or "").lower()) if len(tok) > 2]


def _extract_identifiers(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text or "")


def _iter_candidate_files(repo_root: str, max_files: int) -> List[str]:
    files = []
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in INCLUDE_EXTS:
                continue
            path = os.path.join(root, name)
            files.append(path)
            if len(files) >= max_files:
                return files
    return files


def _read_text_safely(path: str, max_chars: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return ""


def _bm25_score(
    query_terms: List[str],
    tf_map: Dict[str, int],
    doc_len: int,
    avg_doc_len: float,
    idf: Dict[str, float],
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    score = 0.0
    norm = k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
    for term in query_terms:
        freq = tf_map.get(term, 0)
        if freq <= 0:
            continue
        denom = freq + norm
        score += idf.get(term, 0.0) * ((freq * (k1 + 1.0)) / max(1e-6, denom))
    return score


def _build_doc_index(
    repo_root: str,
    max_files_scan: int,
    max_chars_per_file: int,
) -> List[Dict[str, object]]:
    docs: List[Dict[str, object]] = []
    for path in _iter_candidate_files(repo_root, max_files=max_files_scan):
        rel_path = os.path.relpath(path, repo_root)
        body = _read_text_safely(path, max_chars=max_chars_per_file)
        if not body.strip():
            continue
        terms = _tokenize(f"{rel_path}\n{body}")
        tf_map: Dict[str, int] = {}
        for term in terms:
            tf_map[term] = tf_map.get(term, 0) + 1
        identifiers = set(_extract_identifiers(f"{rel_path}\n{body}"))
        docs.append(
            {
                "path": rel_path,
                "snippet": body,
                "terms": terms,
                "tf_map": tf_map,
                "doc_len": len(terms),
                "identifiers": identifiers,
            }
        )
    return docs


def _compute_idf(docs: List[Dict[str, object]], query_terms: List[str]) -> Dict[str, float]:
    n_docs = max(1, len(docs))
    idf = {}
    for term in set(query_terms):
        df = 0
        for doc in docs:
            if term in doc["tf_map"]:
                df += 1
        idf[term] = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)
    return idf


def retrieve_repo_context(
    query: str,
    repo_root: Optional[str],
    top_k: int = 8,
    max_files_scan: int = 600,
    max_chars_per_file: int = 2500,
    hints: str = "",
) -> List[Dict[str, object]]:
    if not repo_root:
        return []
    repo_root = os.path.abspath(repo_root)
    if not os.path.isdir(repo_root):
        return []

    merged_query = f"{query}\n{hints}".strip()
    query_terms = _tokenize(merged_query)
    symbol_terms = set(_extract_identifiers(merged_query))
    if not query_terms and not symbol_terms:
        return []

    docs = _build_doc_index(repo_root, max_files_scan=max_files_scan, max_chars_per_file=max_chars_per_file)
    if not docs:
        return []

    avg_doc_len = sum(doc["doc_len"] for doc in docs) / max(1, len(docs))
    idf = _compute_idf(docs, query_terms)

    scored: List[Dict[str, object]] = []
    query_term_set = set(query_terms)
    for doc in docs:
        bm25 = _bm25_score(
            query_terms=query_terms,
            tf_map=doc["tf_map"],
            doc_len=doc["doc_len"],
            avg_doc_len=avg_doc_len,
            idf=idf,
        )
        path_tokens = set(_tokenize(doc["path"]))
        path_overlap = len(path_tokens & query_term_set)
        symbol_hits = len(symbol_terms & doc["identifiers"])
        score = bm25 + (1.5 * path_overlap) + (0.8 * symbol_hits)
        if score <= 0.0:
            continue
        scored.append(
            {
                "path": doc["path"],
                "snippet": doc["snippet"],
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _render_context_blocks(context_blocks: List[Dict[str, object]], max_total_chars: int = 9000) -> str:
    if not context_blocks:
        return "No repository context available."

    parts = []
    total = 0
    for block in context_blocks:
        section = f"[FILE] {block['path']}\n{block['snippet']}\n"
        if total + len(section) > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            section = section[:remaining]
        parts.append(section)
        total += len(section)
        if total >= max_total_chars:
            break
    return "\n".join(parts).strip()


def _build_context_prefix(user_prompt: str, context_text: str) -> str:
    system_prompt = (
        "You are an autonomous software engineer. Use the repository context to solve the task. "
        "Return executable Python code in <answer> tags. Keep reasoning concise."
    )
    user_block = (
        f"Task:\n{user_prompt}\n\n"
        f"Repository Context:\n{context_text}\n\n"
        "Return one candidate in <answer> tags."
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_block}<|im_end|>\n"
    )


def _repair_policy(error_type: str, stderr: str) -> Tuple[str, float]:
    normalized = (error_type or "").lower()
    stderr = (stderr or "").lower()
    if normalized == "format":
        return (
            "Your previous output missed required tags. Output exactly one <answer> block with executable code.",
            0.2,
        )
    if normalized == "syntax":
        return (
            "Fix syntax issues first. Do not change overall approach unless required by syntax constraints.",
            0.2,
        )
    if normalized == "assertion":
        return (
            "Assertions failed. Prioritize edge-case correctness and validate boundary conditions.",
            0.25,
        )
    if normalized == "timeout":
        return (
            "The solution timed out. Improve algorithmic complexity and remove inefficient loops.",
            0.2,
        )
    if normalized == "security":
        return (
            "Avoid forbidden imports and restricted builtins. Use only safe standard-library constructs.",
            0.2,
        )
    if "nameerror" in stderr or "attributeerror" in stderr:
        return ("Resolve missing names/attributes and ensure symbol definitions are complete.", 0.25)
    if "typeerror" in stderr or "valueerror" in stderr:
        return ("Fix type/value handling and input-shape assumptions.", 0.25)
    return ("Repair runtime correctness using test feedback and keep output in <answer> tags.", 0.3)


def _build_repair_prompt(
    context_prefix: str,
    previous_completion: str,
    best_code: str,
    failure: Dict[str, object],
    round_idx: int,
) -> Tuple[str, float]:
    previous_completion = (previous_completion or "").replace("<|im_end|>", "").strip()
    error_type = str(failure.get("error_type", "") or "")
    stderr = str(failure.get("stderr", "") or "")
    policy, temperature = _repair_policy(error_type, stderr)
    failure_text = (
        f"error_type={error_type}\n"
        f"stderr={stderr[:1800]}\n"
        f"stdout={str(failure.get('stdout', '') or '')[:400]}\n"
    )
    repair_block = (
        f"Self-debug round {round_idx}.\n"
        f"{policy}\n"
        f"Failure details:\n{failure_text}\n"
        f"Previous code:\n{best_code}\n\n"
        "Provide corrected code in <answer> tags only."
    )
    prompt = (
        f"{context_prefix}"
        f"<|im_start|>assistant\n{previous_completion}<|im_end|>\n"
        f"<|im_start|>user\n{repair_block}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt, temperature


def _evaluate_completion(
    completion_text: str,
    tests: str,
    timeout: float,
    verifier_rounds: int,
) -> Dict[str, object]:
    code = extract_xml_content(completion_text, "answer")
    if not code:
        return {
            "score": -1.5,
            "exec_time": 0.0,
            "stdout": "",
            "stderr": "Missing <answer> tags.",
            "error_type": "format",
            "code": "",
            "completion_text": completion_text,
            "round_scores": [],
        }

    code = _strip_code_fences(code)
    verify = run_test_verifier(
        code=code,
        tests=tests,
        timeout=timeout,
        rounds=verifier_rounds,
        require_all_pass=True,
    )
    best_detail = verify.get("best_detail", {}) if isinstance(verify.get("best_detail"), dict) else {}
    return {
        "score": float(verify.get("score", 0.0)),
        "exec_time": float(verify.get("exec_time", 0.0)),
        "stdout": str(best_detail.get("stdout", "") or ""),
        "stderr": str(best_detail.get("stderr", "") or ""),
        "error_type": str(best_detail.get("error_type", "") or ""),
        "code": code,
        "completion_text": completion_text,
        "round_scores": list(verify.get("round_scores", [])),
    }


def solve_with_self_debug(
    llm: LLM,
    user_prompt: str,
    tests: str,
    repo_root: Optional[str] = None,
    max_rounds: int = 3,
    n_candidates: int = 8,
    timeout: float = 2.0,
    temperature: float = 0.3,
    verifier_rounds: int = 2,
    early_stop_patience: int = 2,
) -> Dict[str, object]:
    context_blocks = retrieve_repo_context(user_prompt, repo_root=repo_root)
    context_text = _render_context_blocks(context_blocks)
    context_prefix = _build_context_prefix(user_prompt, context_text)
    prompt = context_prefix + "<|im_start|>assistant\n"

    best_overall = None
    history: List[Dict[str, object]] = []
    best_score_seen = -10.0
    stagnation_rounds = 0
    current_temperature = temperature

    for round_idx in range(1, max_rounds + 1):
        sampling_params = SamplingParams(
            n=n_candidates,
            temperature=current_temperature,
            top_p=0.92,
            max_tokens=1200,
            stop=["<|im_end|>"],
        )
        outputs = llm.generate([prompt], sampling_params)
        candidates = outputs[0].outputs if outputs else []
        if not candidates:
            break

        round_results = []
        for candidate in candidates:
            result = _evaluate_completion(
                completion_text=candidate.text,
                tests=tests,
                timeout=timeout,
                verifier_rounds=verifier_rounds,
            )
            round_results.append(result)
            if best_overall is None or result["score"] > best_overall["score"]:
                best_overall = result
            if result["score"] == 2.0:
                result["status"] = "passed"
                result["round"] = round_idx
                result["context_files"] = [x["path"] for x in context_blocks]
                result["history"] = history
                return result

        round_results.sort(
            key=lambda x: (
                float(x["score"]),
                -float(x.get("exec_time", 999.0)),
            ),
            reverse=True,
        )
        best_round = round_results[0]

        if best_round["score"] > best_score_seen:
            best_score_seen = float(best_round["score"])
            stagnation_rounds = 0
        else:
            stagnation_rounds += 1

        history.append(
            {
                "round": round_idx,
                "score": best_round["score"],
                "error_type": best_round.get("error_type", ""),
                "stderr": str(best_round.get("stderr", "") or "")[:300],
                "round_scores": best_round.get("round_scores", []),
            }
        )

        if stagnation_rounds >= max(1, early_stop_patience):
            break

        hint_text = str(best_round.get("stderr", "") or "")
        refreshed = retrieve_repo_context(
            query=user_prompt,
            repo_root=repo_root,
            hints=hint_text,
        )
        if refreshed:
            context_blocks = refreshed
            context_text = _render_context_blocks(context_blocks)
            context_prefix = _build_context_prefix(user_prompt, context_text)

        prompt, current_temperature = _build_repair_prompt(
            context_prefix=context_prefix,
            previous_completion=str(best_round.get("completion_text", "") or ""),
            best_code=str(best_round.get("code", "") or ""),
            failure=best_round,
            round_idx=round_idx,
        )

    if best_overall is None:
        return {
            "status": "failed",
            "score": -2.0,
            "stderr": "No candidates were generated.",
            "code": "",
            "round": 0,
            "context_files": [x["path"] for x in context_blocks],
            "history": history,
        }

    best_overall["status"] = "failed"
    best_overall["context_files"] = [x["path"] for x in context_blocks]
    best_overall["history"] = history
    best_overall["round"] = len(history)
    return best_overall


def _build_llm(model_path: str, max_model_len: int) -> LLM:
    if os.path.isdir(model_path):
        adapter_cfg = os.path.join(model_path, "adapter_config.json")
        hf_cfg = os.path.join(model_path, "config.json")
        if os.path.exists(adapter_cfg) and not os.path.exists(hf_cfg):
            raise RuntimeError(
                f"'{model_path}' is adapter-only and cannot be loaded directly by vLLM. "
                "Use a merged HF checkpoint (for example: qwen_grpo_final)."
            )
    return LLM(
        model=model_path,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


def run_single_task(
    model_path: str,
    prompt: str,
    tests: str,
    repo_root: Optional[str] = None,
    max_rounds: int = 3,
    n_candidates: int = 8,
    timeout: float = 2.0,
    max_model_len: int = 4096,
    verifier_rounds: int = 2,
    early_stop_patience: int = 2,
) -> Dict[str, object]:
    llm = _build_llm(model_path=model_path, max_model_len=max_model_len)
    return solve_with_self_debug(
        llm=llm,
        user_prompt=prompt,
        tests=tests,
        repo_root=repo_root,
        max_rounds=max_rounds,
        n_candidates=n_candidates,
        timeout=timeout,
        verifier_rounds=verifier_rounds,
        early_stop_patience=early_stop_patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic self-debug inference with retrieval and verifier reranking.")
    parser.add_argument("--model-path", default="qwen_grpo_final")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--repo-root", default="")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--verifier-rounds", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    args = parser.parse_args()

    result = run_single_task(
        model_path=args.model_path,
        prompt=args.prompt,
        tests=args.tests,
        repo_root=args.repo_root or None,
        max_rounds=args.max_rounds,
        n_candidates=args.n_candidates,
        timeout=args.timeout,
        max_model_len=args.max_model_len,
        verifier_rounds=args.verifier_rounds,
        early_stop_patience=args.early_stop_patience,
    )
    print(result)
