#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

from prm_tiny import save_tiny_prm, train_tiny_prm


def _load_rows(paths: List[str]) -> List[Dict[str, object]]:
    rows = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _record_text(row: Dict[str, object]) -> str:
    prompt = str(row.get("prompt", "") or "")
    history = row.get("history", [])
    hist_lines = []
    if isinstance(history, list):
        for step in history[:8]:
            if not isinstance(step, dict):
                continue
            err = str(step.get("error_type", "") or "")
            score = str(step.get("score", ""))
            hist_lines.append(f"round_error={err} round_score={score}")
    code = str(row.get("result_code", "") or row.get("best_patch", "") or "")
    error_type = str(row.get("error_type", "") or "")
    tests = str(row.get("tests", "") or "")
    return (
        f"prompt={prompt}\nerror_type={error_type}\n"
        f"history={' | '.join(hist_lines)}\n"
        f"tests={tests}\ncode={code}"
    )


def _label(row: Dict[str, object]) -> int:
    return 1 if bool(row.get("all_passed", False) or row.get("pass_at_1", False)) else 0


def main():
    parser = argparse.ArgumentParser(description="Train tiny hashed PRM model from eval case logs.")
    parser.add_argument("--case-log-paths", required=True, help="Comma-separated JSONL case log paths.")
    parser.add_argument("--output-path", default="./artifacts/prm_tiny_v1.json")
    parser.add_argument("--buckets", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--min-samples", type=int, default=120)
    args = parser.parse_args()

    paths = [p.strip() for p in args.case_log_paths.split(",") if p.strip()]
    rows = _load_rows(paths)
    if len(rows) < args.min_samples:
        raise RuntimeError(f"Too few case-log rows for PRM training: {len(rows)} < {args.min_samples}")

    train_pairs: List[Tuple[str, int]] = []
    for row in rows:
        text = _record_text(row)
        if not text.strip():
            continue
        train_pairs.append((text, _label(row)))

    if len(train_pairs) < args.min_samples:
        raise RuntimeError(
            f"Too few usable PRM training samples: {len(train_pairs)} < {args.min_samples}"
        )

    model = train_tiny_prm(
        train_rows=train_pairs,
        buckets=args.buckets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_tiny_prm(model, args.output_path)
    positives = sum(1 for _t, y in train_pairs if y > 0)
    print(
        f"Saved tiny PRM model to '{args.output_path}' "
        f"(samples={len(train_pairs)}, positives={positives}, buckets={args.buckets})"
    )


if __name__ == "__main__":
    main()
