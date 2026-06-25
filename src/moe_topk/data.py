from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


CATEGORIES = [
    "general_ko",
    "general_en",
    "math_ko",
    "math_en",
    "code",
    "reasoning",
    "translation",
]


TEMPLATES = {
    "general_ko": [
        "hanguk news summary topic {i}: people discuss policy, markets, and daily life.",
        "short korean style memo {i}: school, work, weather, family, and planning.",
        "local context {i}: seoul bus schedule, cafe note, lecture reminder, meeting.",
    ],
    "general_en": [
        "english article {i}: a concise paragraph about science, cities, and culture.",
        "daily note {i}: the team checks progress, risks, dates, and next actions.",
        "plain text {i}: readers compare options and choose a practical plan.",
    ],
    "math_ko": [
        "math korean prompt {i}: prove sum, matrix rank, eigen value, gradient step.",
        "calculation memo {i}: x plus y equals z, variance, norm, and projection.",
        "theorem note {i}: assumption, lemma, proof, counterexample, conclusion.",
    ],
    "math_en": [
        "math problem {i}: compute derivative, eigenvector, expectation, and loss.",
        "linear algebra {i}: matrix A maps vector x and preserves a subspace.",
        "optimization {i}: minimize convex function with gradient descent update.",
    ],
    "code": [
        "def function_{i}(x): return sorted([v * 2 for v in x if v > 0])",
        "class Router{i}: def forward(self, tokens): return topk(tokens)",
        "for step in range({i}): loss.backward(); optimizer.step(); optimizer.zero_grad()",
    ],
    "reasoning": [
        "reasoning case {i}: if A implies B and B excludes C, decide whether A allows C.",
        "logic puzzle {i}: compare three claims, find contradiction, state final answer.",
        "multi step {i}: read evidence, reject weak assumption, then choose action.",
    ],
    "translation": [
        "translate sample {i}: source sentence becomes target sentence with same meaning.",
        "bilingual note {i}: korean phrase maps to english phrase and back consistently.",
        "parallel text {i}: maintain entities, numbers, style, and intent across languages.",
    ],
}


@dataclass(frozen=True)
class Corpus:
    tokens: torch.Tensor
    sample_ids: list[str]
    task_types: list[str]


def _text_for(category: str, index: int, rng: random.Random) -> str:
    template = rng.choice(TEMPLATES[category])
    salt = rng.randint(0, 10_000_000)
    return (template.format(i=index) + f" seedtoken {salt} category {category}. ") * 3


def _encode_bytes(text: str, seq_len: int) -> list[int]:
    raw = list(text.encode("utf-8"))
    needed = seq_len + 1
    if not raw:
        raw = [32]
    while len(raw) < needed:
        raw.extend([32])
        raw.extend(raw[: max(1, needed - len(raw))])
    return raw[:needed]


def build_corpus(samples_per_category: int, seq_len: int, seed: int) -> Corpus:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    sample_ids: list[str] = []
    task_types: list[str] = []
    for category in CATEGORIES:
        for i in range(samples_per_category):
            text = _text_for(category, i, rng)
            rows.append(_encode_bytes(text, seq_len))
            sample_ids.append(f"{category}_{i:05d}")
            task_types.append(category)
    return Corpus(
        tokens=torch.tensor(rows, dtype=torch.long),
        sample_ids=sample_ids,
        task_types=task_types,
    )


def build_jsonl_corpus(path: Path, seq_len: int) -> Corpus:
    rows: list[list[int]] = []
    sample_ids: list[str] = []
    task_types: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", f"jsonl_{line_number:05d}"))
            task_type = str(item["task_type"])
            text = str(item["text"])
            if task_type not in CATEGORIES:
                raise ValueError(f"unknown task_type {task_type!r} at {path}:{line_number}")
            rows.append(_encode_bytes(text, seq_len))
            sample_ids.append(sample_id)
            task_types.append(task_type)
    if not rows:
        raise ValueError(f"no probe rows found in {path}")
    return Corpus(
        tokens=torch.tensor(rows, dtype=torch.long),
        sample_ids=sample_ids,
        task_types=task_types,
    )


def batch_indices(num_items: int, batch_size: int, steps: int, seed: int) -> Iterable[torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    for _ in range(steps):
        yield torch.randint(0, num_items, (batch_size,), generator=generator)
