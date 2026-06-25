from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


def top1_agreement(selected_a: np.ndarray, selected_b: np.ndarray) -> float:
    return float(np.mean(selected_a[:, 0] == selected_b[:, 0]))


def nestedness(selected_small: np.ndarray, selected_large: np.ndarray) -> float:
    if selected_small.shape[0] != selected_large.shape[0]:
        raise ValueError("route arrays must have the same number of records")
    scores = []
    for small, large in zip(selected_small, selected_large):
        small_set = set(int(x) for x in small)
        large_set = set(int(x) for x in large)
        scores.append(len(small_set.intersection(large_set)) / max(1, len(small_set)))
    return float(np.mean(scores))


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row_ids = np.arange(values.shape[0])[:, None]
    ranks[row_ids, order] = np.arange(values.shape[1], dtype=np.float32)
    return ranks


def spearman_from_logits(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    ranks_a = _ranks_desc(logits_a.astype(np.float32))
    ranks_b = _ranks_desc(logits_b.astype(np.float32))
    ranks_a = ranks_a - ranks_a.mean(axis=1, keepdims=True)
    ranks_b = ranks_b - ranks_b.mean(axis=1, keepdims=True)
    numerator = (ranks_a * ranks_b).sum(axis=1)
    denominator = np.sqrt((ranks_a * ranks_a).sum(axis=1) * (ranks_b * ranks_b).sum(axis=1))
    valid = denominator > 0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def expert_frequency(selected_ids: np.ndarray, n_experts: int) -> dict[str, float]:
    counts = np.bincount(selected_ids.reshape(-1), minlength=n_experts).astype(np.float64)
    total = max(1.0, float(counts.sum()))
    probs = counts / total
    entropy = -float(np.sum([p * math.log(p + 1e-12) for p in probs]))
    return {
        "max_share": float(probs.max()),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(n_experts),
    }


def topk_ids_from_logits(logits: np.ndarray, top_k: int) -> np.ndarray:
    order = np.argsort(-logits.astype(np.float32), axis=1)
    return order[:, :top_k].astype(np.int16)


def coactivation_matrix(selected_ids: np.ndarray, n_experts: int) -> np.ndarray:
    matrix = np.zeros((n_experts, n_experts), dtype=np.float64)
    for row in selected_ids:
        unique = sorted(set(int(x) for x in row))
        for i, left in enumerate(unique):
            for right in unique[i + 1 :]:
                matrix[left, right] += 1.0
                matrix[right, left] += 1.0
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def top_pair_overlap(a: np.ndarray, b: np.ndarray, top_n: int = 10) -> float:
    def top_pairs(matrix: np.ndarray) -> set[tuple[int, int]]:
        pairs = []
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                pairs.append(((i, j), matrix[i, j]))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return set(pair for pair, value in pairs[:top_n] if value > 0)

    left = top_pairs(a)
    right = top_pairs(b)
    if not left and not right:
        return 1.0
    return len(left.intersection(right)) / max(1, len(left.union(right)))


def metric_rows_for_pair(
    metric_type: str,
    selected_a: np.ndarray,
    selected_b: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    n_experts: int,
) -> list[tuple[str, float]]:
    rows = [
        ("top1_agreement", top1_agreement(selected_a, selected_b)),
        ("nestedness", nestedness(selected_a, selected_b)),
        ("spearman", spearman_from_logits(logits_a, logits_b)),
    ]
    if selected_a.shape[1] >= 2 and selected_b.shape[1] >= 2:
        matrix_a = coactivation_matrix(selected_a, n_experts)
        matrix_b = coactivation_matrix(selected_b, n_experts)
        rows.append(("coactivation_cosine", cosine_similarity(matrix_a, matrix_b)))
        rows.append(("top_pair_overlap", top_pair_overlap(matrix_a, matrix_b)))
    return rows
