from __future__ import annotations

from math import comb

import numpy as np


SELECTED_STATS_FIELDS = (
    "num_records",
    "intersection_sum",
    "jaccard_sum",
    "union_tau_sum",
    "union_tau_valid_records",
    "common_tau_sum",
    "common_tau_valid_records",
    "common_pair_count",
    "common_pair_signed_sum",
    "common_exact_order_match_count",
    "full_containment_record_count",
    "ordered_containment_match_count",
)

_POPCOUNT16 = np.unpackbits(
    np.arange(1 << 16, dtype=np.uint16).view(np.uint8)
).reshape(-1, 16).sum(axis=1).astype(np.uint8)


def inverse_ranks_from_logits(logits: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError(f"expected 2D logits, got shape={logits.shape}")
    n_records, n_experts = logits.shape
    if n_experts > np.iinfo(np.uint8).max:
        raise ValueError("uint8 rank storage supports at most 255 experts")
    ranks = np.empty((n_records, n_experts), dtype=np.uint8)
    rank_values = np.arange(1, n_experts + 1, dtype=np.uint8)
    for start in range(0, n_records, chunk_size):
        end = min(start + chunk_size, n_records)
        order = np.argsort(-logits[start:end].astype(np.float32), axis=1)
        row_ids = np.arange(end - start)[:, None]
        ranks[start:end][row_ids, order] = rank_values
    return ranks


def rank_transition_counts(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    record_indices: np.ndarray | None = None,
    chunk_size: int = 8192,
) -> np.ndarray:
    if ranks_a.shape != ranks_b.shape or ranks_a.ndim != 2:
        raise ValueError(f"rank shapes must match and be 2D: {ranks_a.shape} vs {ranks_b.shape}")
    n_experts = ranks_a.shape[1]
    indices = np.arange(ranks_a.shape[0]) if record_indices is None else record_indices
    counts = np.zeros((n_experts, n_experts), dtype=np.int64)
    for start in range(0, indices.size, chunk_size):
        selected = indices[start : start + chunk_size]
        left = ranks_a[selected].astype(np.int16) - 1
        right = ranks_b[selected].astype(np.int16) - 1
        codes = left * n_experts + right
        counts += np.bincount(codes.reshape(-1), minlength=n_experts * n_experts).reshape(
            n_experts, n_experts
        )
    return counts


def topm_overlap_curve(counts: np.ndarray) -> np.ndarray:
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"expected square transition matrix, got shape={counts.shape}")
    n_experts = counts.shape[0]
    n_records = counts.sum() / n_experts
    if n_records <= 0:
        return np.full(n_experts, np.nan, dtype=np.float64)
    curve = np.empty(n_experts, dtype=np.float64)
    for top_m in range(1, n_experts + 1):
        intersection = float(counts[:top_m, :top_m].sum())
        curve[top_m - 1] = intersection / (n_records * top_m)
    return curve


def rank_biased_overlap(overlap_curve: np.ndarray, persistence: float) -> float:
    if not 0.0 < persistence < 1.0:
        raise ValueError("persistence must be in (0, 1)")
    depths = np.arange(overlap_curve.size, dtype=np.float64)
    weighted = (1.0 - persistence) * np.sum((persistence**depths) * overlap_curve)
    return float(weighted + (persistence ** overlap_curve.size) * overlap_curve[-1])


def mean_absolute_rank_displacement(counts: np.ndarray) -> float:
    row, column = np.indices(counts.shape)
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    return float(np.sum(counts * np.abs(row - column)) / total)


def normalized_footrule_similarity(counts: np.ndarray) -> float:
    n_experts = counts.shape[0]
    max_mean_displacement = (n_experts * n_experts // 2) / n_experts
    return 1.0 - mean_absolute_rank_displacement(counts) / max_mean_displacement


def kendall_tau_from_ranks(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    record_indices: np.ndarray | None = None,
    chunk_size: int = 512,
) -> float:
    if ranks_a.shape != ranks_b.shape or ranks_a.ndim != 2:
        raise ValueError(f"rank shapes must match and be 2D: {ranks_a.shape} vs {ranks_b.shape}")
    indices = np.arange(ranks_a.shape[0]) if record_indices is None else record_indices
    n_experts = ranks_a.shape[1]
    n_pairs = n_experts * (n_experts - 1) / 2
    tau_sum = 0.0
    record_count = 0
    for start in range(0, indices.size, chunk_size):
        selected = indices[start : start + chunk_size]
        left = ranks_a[selected]
        right = ranks_b[selected]
        left_order = np.argsort(left, axis=1)
        right_in_left_order = np.take_along_axis(right, left_order, axis=1)
        inversions = np.zeros(selected.size, dtype=np.int32)
        for position in range(n_experts - 1):
            inversions += np.sum(
                right_in_left_order[:, position, None] > right_in_left_order[:, position + 1 :],
                axis=1,
            )
        tau_sum += float(np.sum(1.0 - 2.0 * inversions / n_pairs))
        record_count += selected.size
    return tau_sum / max(1, record_count)


def selected_expert_order_stats(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    cutoff_a: int,
    cutoff_b: int,
    record_indices: np.ndarray | None = None,
    chunk_size: int = 2048,
) -> dict[str, float | np.ndarray]:
    """Accumulate set and order statistics over experts selected by either router."""
    if ranks_a.shape != ranks_b.shape or ranks_a.ndim != 2:
        raise ValueError(f"rank shapes must match and be 2D: {ranks_a.shape} vs {ranks_b.shape}")
    n_experts = ranks_a.shape[1]
    if not 1 <= cutoff_a <= n_experts or not 1 <= cutoff_b <= n_experts:
        raise ValueError(f"cutoffs must be in [1, {n_experts}]")

    indices = np.arange(ranks_a.shape[0]) if record_indices is None else record_indices
    union_pair_left, union_pair_right = np.triu_indices(cutoff_a + cutoff_b, k=1)
    common_pair_left, common_pair_right = np.triu_indices(cutoff_a, k=1)
    stats: dict[str, float | np.ndarray] = {
        field: 0.0 for field in SELECTED_STATS_FIELDS
    }
    stats["intersection_histogram"] = np.zeros(n_experts + 1, dtype=np.int64)

    for start in range(0, indices.size, chunk_size):
        selected = indices[start : start + chunk_size]
        left = ranks_a[selected]
        right = ranks_b[selected]
        selected_ids_a = np.argpartition(left, cutoff_a - 1, axis=1)[:, :cutoff_a]
        selected_ids_b = np.argpartition(right, cutoff_b - 1, axis=1)[:, :cutoff_b]
        right_ranks_of_a = np.take_along_axis(right, selected_ids_a, axis=1)
        common_a = right_ranks_of_a <= cutoff_b
        common_size = common_a.sum(axis=1)
        union_size = cutoff_a + cutoff_b - common_size

        stats["num_records"] += float(selected.size)
        stats["intersection_sum"] += float(common_size.sum())
        stats["jaccard_sum"] += float(np.sum(common_size / union_size))
        stats["intersection_histogram"] += np.bincount(
            common_size, minlength=n_experts + 1
        )

        b_already_in_a = np.any(
            selected_ids_b[:, :, None] == selected_ids_a[:, None, :], axis=2
        )
        union_ids = np.concatenate([selected_ids_a, selected_ids_b], axis=1)
        union_active = np.concatenate(
            [
                np.ones_like(selected_ids_a, dtype=bool),
                ~b_already_in_a,
            ],
            axis=1,
        )
        union_left_ranks = np.take_along_axis(left, union_ids, axis=1)
        union_right_ranks = np.take_along_axis(right, union_ids, axis=1)
        union_signed_agreement = np.where(
            (union_left_ranks[:, union_pair_left] < union_left_ranks[:, union_pair_right])
            == (union_right_ranks[:, union_pair_left] < union_right_ranks[:, union_pair_right]),
            1,
            -1,
        ).astype(np.int8)
        union_pairs = (
            union_active[:, union_pair_left] & union_active[:, union_pair_right]
        )
        union_pair_count = union_pairs.sum(axis=1)
        union_valid = union_pair_count > 0
        union_signed_sum = np.sum(union_signed_agreement * union_pairs, axis=1)
        stats["union_tau_sum"] += float(
            np.sum(union_signed_sum[union_valid] / union_pair_count[union_valid])
        )
        stats["union_tau_valid_records"] += float(union_valid.sum())

        left_ranks_of_a = np.take_along_axis(left, selected_ids_a, axis=1)
        common_signed_agreement = np.where(
            (left_ranks_of_a[:, common_pair_left] < left_ranks_of_a[:, common_pair_right])
            == (right_ranks_of_a[:, common_pair_left] < right_ranks_of_a[:, common_pair_right]),
            1,
            -1,
        ).astype(np.int8)
        common_pairs = common_a[:, common_pair_left] & common_a[:, common_pair_right]
        common_pair_count = common_pairs.sum(axis=1)
        common_valid = common_pair_count > 0
        common_signed_sum = np.sum(common_signed_agreement * common_pairs, axis=1)
        common_exact = common_valid & (common_signed_sum == common_pair_count)
        full_containment = common_size == cutoff_a
        ordered_containment = full_containment & (
            common_signed_sum == common_pair_count
        )
        stats["common_tau_sum"] += float(
            np.sum(common_signed_sum[common_valid] / common_pair_count[common_valid])
        )
        stats["common_tau_valid_records"] += float(common_valid.sum())
        stats["common_pair_count"] += float(common_pair_count.sum())
        stats["common_pair_signed_sum"] += float(common_signed_sum.sum())
        stats["common_exact_order_match_count"] += float(common_exact.sum())
        stats["full_containment_record_count"] += float(full_containment.sum())
        stats["ordered_containment_match_count"] += float(ordered_containment.sum())

    return stats


def merge_selected_expert_order_stats(
    totals: dict[str, float | np.ndarray] | None,
    update: dict[str, float | np.ndarray],
) -> dict[str, float | np.ndarray]:
    if totals is None:
        result: dict[str, float | np.ndarray] = {
            field: float(update[field]) for field in SELECTED_STATS_FIELDS
        }
        result["intersection_histogram"] = np.asarray(
            update["intersection_histogram"], dtype=np.int64
        ).copy()
        return result
    result = {
        field: float(totals[field]) + float(update[field])
        for field in SELECTED_STATS_FIELDS
    }
    result["intersection_histogram"] = np.asarray(
        totals["intersection_histogram"], dtype=np.int64
    ) + np.asarray(update["intersection_histogram"], dtype=np.int64)
    return result


def hypergeometric_intersection_pmf(
    n_experts: int,
    cutoff_a: int,
    cutoff_b: int,
) -> np.ndarray:
    if not 0 <= cutoff_a <= cutoff_b <= n_experts:
        raise ValueError("expected 0 <= cutoff_a <= cutoff_b <= n_experts")
    probabilities = np.zeros(n_experts + 1, dtype=np.float64)
    lower = max(0, cutoff_a + cutoff_b - n_experts)
    denominator = comb(n_experts, cutoff_a)
    for intersection in range(lower, cutoff_a + 1):
        probabilities[intersection] = (
            comb(cutoff_b, intersection)
            * comb(n_experts - cutoff_b, cutoff_a - intersection)
            / denominator
        )
    return probabilities


def hypergeometric_calibrated_overlap(
    intersection_histogram: np.ndarray,
    n_experts: int,
    cutoff_a: int,
    cutoff_b: int,
) -> float:
    histogram = np.asarray(intersection_histogram, dtype=np.float64)
    total = float(histogram.sum())
    if total <= 0:
        return float("nan")
    probabilities = hypergeometric_intersection_pmf(
        n_experts, cutoff_a, cutoff_b
    )
    mid_cdf = np.cumsum(probabilities) - 0.5 * probabilities
    observed_auc = float(np.sum(histogram[: mid_cdf.size] * mid_cdf) / total)
    maximum_auc = float(mid_cdf[cutoff_a])
    return (observed_auc - 0.5) / (maximum_auc - 0.5)


def summarize_selected_expert_order_stats(
    stats: dict[str, float | np.ndarray],
    cutoff_a: int,
    cutoff_b: int,
) -> dict[str, float]:
    num_records = stats["num_records"]
    max_common_pairs = min(cutoff_a, cutoff_b) * (min(cutoff_a, cutoff_b) - 1) / 2
    common_pairs = stats["common_pair_count"]
    common_tau = (
        stats["common_pair_signed_sum"] / common_pairs if common_pairs > 0 else float("nan")
    )
    common_valid_records = stats["common_tau_valid_records"]
    full_containment_records = stats["full_containment_record_count"]
    return {
        "selected_containment_a": stats["intersection_sum"] / (num_records * cutoff_a),
        "selected_containment_b": stats["intersection_sum"] / (num_records * cutoff_b),
        "selected_jaccard": stats["jaccard_sum"] / num_records,
        "hypergeometric_calibrated_overlap": hypergeometric_calibrated_overlap(
            np.asarray(stats["intersection_histogram"]),
            np.asarray(stats["intersection_histogram"]).size - 1,
            cutoff_a,
            cutoff_b,
        ),
        "selected_union_kendall_tau": stats["union_tau_sum"]
        / max(1.0, stats["union_tau_valid_records"]),
        "common_selected_kendall_tau": common_tau,
        "common_pairwise_order_agreement": (
            (common_tau + 1.0) / 2.0 if np.isfinite(common_tau) else float("nan")
        ),
        "common_selected_exact_order_fraction": (
            stats["common_exact_order_match_count"] / common_valid_records
            if common_valid_records > 0
            else float("nan")
        ),
        "full_containment_record_fraction": full_containment_records / num_records,
        "ordered_containment_record_fraction": (
            stats["ordered_containment_match_count"] / num_records
            if cutoff_a >= 2
            else float("nan")
        ),
        "contained_exact_order_fraction": (
            stats["ordered_containment_match_count"] / full_containment_records
            if cutoff_a >= 2 and full_containment_records > 0
            else float("nan")
        ),
        "common_selected_record_mean_tau": (
            stats["common_tau_sum"] / stats["common_tau_valid_records"]
            if stats["common_tau_valid_records"] > 0
            else float("nan")
        ),
        "common_selected_valid_record_fraction": stats["common_tau_valid_records"] / num_records,
        "common_pair_coverage": (
            common_pairs / (num_records * max_common_pairs)
            if max_common_pairs > 0
            else float("nan")
        ),
    }


def selected_bitmasks_from_ranks(ranks: np.ndarray, cutoff: int) -> np.ndarray:
    if ranks.ndim != 2 or ranks.shape[1] > 32:
        raise ValueError("selected bitmasks require a 2D rank array with at most 32 experts")
    if not 1 <= cutoff <= ranks.shape[1]:
        raise ValueError(f"cutoff must be in [1, {ranks.shape[1]}]")
    weights = np.left_shift(
        np.uint32(1), np.arange(ranks.shape[1], dtype=np.uint32)
    )
    return np.sum((ranks <= cutoff).astype(np.uint32) * weights, axis=1, dtype=np.uint32)


def _popcount_uint32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32)
    return _POPCOUNT16[values & np.uint32(0xFFFF)] + _POPCOUNT16[values >> np.uint32(16)]


def empirical_null_ceiling_adjusted_overlap(
    masks_a: np.ndarray,
    masks_b: np.ndarray,
    sample_ids: np.ndarray,
    task_ids: np.ndarray,
    layer_ids: np.ndarray,
    token_positions: np.ndarray,
    cutoff_a: int,
    record_indices: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare aligned blocks against within-task derangement and optimal matchings."""
    from scipy.optimize import linear_sum_assignment

    arrays = (masks_b, sample_ids, task_ids, layer_ids, token_positions)
    if any(np.asarray(array).shape != np.asarray(masks_a).shape for array in arrays):
        raise ValueError("mask and metadata arrays must have identical one-dimensional shapes")
    if cutoff_a < 1:
        raise ValueError("cutoff_a must be positive")
    indices = (
        np.arange(np.asarray(masks_a).size)
        if record_indices is None
        else np.asarray(record_indices)
    )
    total_samples = 0
    observed_sum = 0.0
    null_sum = 0.0
    ceiling_sum = 0.0

    for task_id in sorted(np.unique(task_ids[indices]).tolist()):
        task_indices = indices[task_ids[indices] == task_id]
        samples = np.unique(sample_ids[task_indices])
        coordinate_keys = (
            layer_ids[task_indices].astype(np.int64)
            * (int(np.max(token_positions[task_indices])) + 1)
            + token_positions[task_indices].astype(np.int64)
        )
        coordinates = np.unique(coordinate_keys)
        expected_records = samples.size * coordinates.size
        if task_indices.size != expected_records:
            raise ValueError(
                f"task={task_id} is not a complete sample-coordinate rectangle: "
                f"records={task_indices.size}, expected={expected_records}"
            )
        sample_positions = np.searchsorted(samples, sample_ids[task_indices])
        coordinate_positions = np.searchsorted(coordinates, coordinate_keys)
        linear_positions = sample_positions * coordinates.size + coordinate_positions
        if np.unique(linear_positions).size != task_indices.size:
            raise ValueError(f"task={task_id} contains duplicate sample-coordinate records")
        seen = np.zeros((samples.size, coordinates.size), dtype=bool)
        seen[sample_positions, coordinate_positions] = True
        if not np.all(seen):
            raise ValueError(f"task={task_id} is missing sample-coordinate records")

        blocks_a = np.empty((samples.size, coordinates.size), dtype=np.uint32)
        blocks_b = np.empty_like(blocks_a)
        blocks_a[sample_positions, coordinate_positions] = masks_a[task_indices]
        blocks_b[sample_positions, coordinate_positions] = masks_b[task_indices]
        intersections = _popcount_uint32(
            blocks_a[:, None, :] & blocks_b[None, :, :]
        ).sum(axis=2, dtype=np.int64)
        scores = intersections.astype(np.float64) / (coordinates.size * cutoff_a)
        if samples.size < 2:
            raise ValueError(
                f"task={task_id} needs at least two samples for a derangement null"
            )
        row_ids, column_ids = linear_sum_assignment(-scores)
        observed_sum += float(np.trace(scores))
        null_sum += float((scores.sum() - np.trace(scores)) / (samples.size - 1))
        ceiling_sum += float(scores[row_ids, column_ids].sum())
        total_samples += int(samples.size)

    observed = observed_sum / total_samples
    null = null_sum / total_samples
    ceiling = ceiling_sum / total_samples
    headroom = ceiling - null
    return {
        "empirical_observed_overlap": observed,
        "empirical_null_overlap": null,
        "empirical_ceiling_overlap": ceiling,
        "empirical_alignment_headroom": headroom,
        "frequency_adjusted_overlap": (
            (observed - null) / (1.0 - null) if null < 1.0 else float("nan")
        ),
        "empirical_ceiling_utilization": (
            (observed - null) / headroom if headroom > 0 else float("nan")
        ),
        "empirical_num_samples": float(total_samples),
    }


def displacement_by_source_rank(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_experts = counts.shape[0]
    target_ranks = np.arange(1, n_experts + 1, dtype=np.float64)
    mean_target = np.full(n_experts, np.nan, dtype=np.float64)
    mean_absolute = np.full(n_experts, np.nan, dtype=np.float64)
    for source_index in range(n_experts):
        row = counts[source_index]
        total = float(row.sum())
        if total <= 0:
            continue
        mean_target[source_index] = float(np.sum(row * target_ranks) / total)
        mean_absolute[source_index] = float(
            np.sum(row * np.abs(target_ranks - (source_index + 1))) / total
        )
    return mean_target, mean_absolute
