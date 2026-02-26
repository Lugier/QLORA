import hashlib
import json
import math
import os
import re
from typing import Dict, Iterable, List, Tuple


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", (text or "").lower())


def _hash_feature(token: str, buckets: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, int(buckets))


def _featurize(record: str, buckets: int) -> Dict[int, float]:
    feats: Dict[int, float] = {}
    for tok in _tokens(record):
        idx = _hash_feature(tok, buckets)
        feats[idx] = feats.get(idx, 0.0) + 1.0
    return feats


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def train_tiny_prm(
    train_rows: Iterable[Tuple[str, int]],
    buckets: int = 8192,
    epochs: int = 3,
    learning_rate: float = 0.08,
    l2: float = 1e-6,
) -> Dict[str, object]:
    buckets = max(256, int(buckets))
    epochs = max(1, int(epochs))
    learning_rate = float(learning_rate)
    l2 = float(l2)

    weights = [0.0] * buckets
    bias = 0.0
    cached_rows = list(train_rows)
    if not cached_rows:
        raise RuntimeError("No PRM training rows supplied.")

    for _ in range(epochs):
        for text, label in cached_rows:
            y = 1.0 if int(label) > 0 else 0.0
            feats = _featurize(text, buckets=buckets)
            score = bias
            for idx, val in feats.items():
                score += weights[idx] * val
            pred = _sigmoid(score)
            grad = pred - y
            bias -= learning_rate * grad
            for idx, val in feats.items():
                weights[idx] -= learning_rate * ((grad * val) + (l2 * weights[idx]))

    return {
        "version": 1,
        "buckets": buckets,
        "bias": bias,
        "weights": weights,
    }


def predict_tiny_prm(model: Dict[str, object], text: str) -> float:
    if not model:
        return 0.5
    buckets = int(model.get("buckets", 8192))
    bias = float(model.get("bias", 0.0))
    weights = model.get("weights", [])
    if not isinstance(weights, list) or not weights:
        return 0.5

    feats = _featurize(text, buckets=buckets)
    score = bias
    for idx, val in feats.items():
        if idx < len(weights):
            score += float(weights[idx]) * val
    return _sigmoid(score)


def save_tiny_prm(model: Dict[str, object], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f)


def load_tiny_prm(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError("Invalid PRM model file: expected object.")
    return obj
