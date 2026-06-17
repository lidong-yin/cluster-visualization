from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Callable, Optional
from src import faiss_utils


def compute_variance(feats: np.ndarray) -> float:
    """
    Appendix-like variance: mean squared distance to cluster mean.
    """
    if feats.shape[0] < 2:
        return 0.0
    mean_vector = feats.mean(axis=0)
    diffs = feats - mean_vector
    sq_distances = np.sum(diffs**2, axis=1)
    return float(sq_distances.mean())


def group_sizes(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    s = df[group_key].value_counts(dropna=False)
    out = s.rename("size").reset_index()
    # In pandas 2.0+, reset_index preserves the original index name as the column name
    if group_key not in out.columns:
        out = out.rename(columns={out.columns[0]: group_key})
    return out


def group_variances(
    df: pd.DataFrame,
    *,
    group_key: str,
    feats: np.ndarray,
    feats_row_indices: np.ndarray,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Compute variance per group over feature matrix.

    feats are aligned to feats_row_indices which map into df.index.
    """
    # Build label array aligned to feats
    labels = df.loc[feats_row_indices, group_key]
    # Ensure we can group by labels including NaN
    g = pd.DataFrame({"label": labels.values, "pos": np.arange(len(labels))})

    rows: list[dict[str, Any]] = []
    total_groups = int(g["label"].nunique(dropna=False))
    for i, (lb, sub) in enumerate(g.groupby("label", dropna=False)):
        idx = sub["pos"].to_numpy()
        v = compute_variance(feats[idx])
        rows.append({group_key: lb, "size": int(len(idx)), "variance": float(v)})
        if progress_callback and (i % 25 == 0 or i == total_groups - 1):
            progress_callback((i + 1) / max(1, total_groups), f"方差计算: {i+1}/{total_groups}")
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ScatterCandidate:
    label: Any
    min_sim: float


@dataclass(frozen=True)
class ScatterGroup:
    main_label: Any
    main_size: int
    group_min_sim: float
    candidates: list[ScatterCandidate]


def compute_scatter_groups(
    df: pd.DataFrame,
    *,
    group_key: str,
    feats: np.ndarray,
    feats_row_indices: np.ndarray,
    sim_th: float,
    sim_topk: int,
    use_gpu_prefer: bool = True,
    batch_size: int = 2048,
    max_candidates_per_group: int = 10,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> list[ScatterGroup]:
    """
    Scatter analysis:
    - For each node, search topK neighbors over all nodes (cosine).
    - Keep inter-cluster edges with sim > sim_th.
    - For each main cluster, collect candidate clusters and the minimum similarity among
      edges that connect to each candidate cluster.
    - group_min_sim is the minimum similarity among all kept inter-cluster edges for that main cluster.
    """
    if feats.shape[0] != feats_row_indices.shape[0]:
        raise ValueError("feats 与 feats_row_indices 长度不一致。")
    if sim_topk <= 0:
        raise ValueError("sim_topk 必须 > 0")

    if progress_callback:
        progress_callback(0.02, "构建 Faiss 索引 ...")

    feats = faiss_utils.l2_normalize(feats)
    index = faiss_utils.build_index_ip(feats, use_gpu_prefer=use_gpu_prefer)
    index.add(feats)

    if progress_callback:
        progress_callback(0.15, "Faiss 检索 (search) ...")

    def _search_cb(frac: float, text: str) -> None:
        if progress_callback:
            progress_callback(0.15 + 0.10 * float(frac), text)

    sims, nbrs = faiss_utils.search_in_batches(
        index,
        feats,
        k=int(sim_topk) + 1,
        batch_size=batch_size,
        progress_callback=_search_cb if progress_callback else None,
    )
    sims = sims[:, 1:]
    nbrs = nbrs[:, 1:]

    labels = df.loc[feats_row_indices, group_key].to_numpy()

    # main_label -> (cand_label -> min_sim)
    min_sim_map: dict[Any, dict[Any, float]] = {}
    n = int(sims.shape[0])
    for i in range(n):
        lb_i = labels[i]
        for j in range(sims.shape[1]):
            sim = float(sims[i, j])
            if sim <= sim_th:
                break
            nb = int(nbrs[i, j])
            lb_nb = labels[nb]
            if (lb_nb == lb_i) or (pd.isna(lb_nb) and pd.isna(lb_i)):
                continue
            d = min_sim_map.setdefault(lb_i, {})
            prev = d.get(lb_nb)
            if prev is None or sim < prev:
                d[lb_nb] = sim
        if progress_callback and (i % 5000 == 0 or i == n - 1):
            # 25%..95% for scanning phase
            frac = 0.25 + 0.70 * ((i + 1) / max(1, n))
            progress_callback(frac, f"扫描跨簇邻居: {i+1}/{n}")

    groups: list[ScatterGroup] = []
    # Precompute main sizes for display (from full df, not only feats)
    size_map = df[group_key].value_counts(dropna=False).to_dict()

    if progress_callback:
        progress_callback(0.97, "汇总散度组 ...")

    for main_lb, cand_map in min_sim_map.items():
        if not cand_map:
            continue
        # Keep top candidates by min_sim (desc), but group_min_sim uses overall min
        cand_items = [ScatterCandidate(label=k, min_sim=float(v)) for k, v in cand_map.items()]
        group_min_sim = float(min(c.min_sim for c in cand_items))
        cand_items.sort(key=lambda x: x.min_sim, reverse=True)
        cand_items = cand_items[: int(max_candidates_per_group)]
        groups.append(
            ScatterGroup(
                main_label=main_lb,
                main_size=int(size_map.get(main_lb, 0)),
                group_min_sim=group_min_sim,
                candidates=cand_items,
            )
        )

    return groups


def deduplicate_scatter_groups(groups: list[ScatterGroup]) -> list[ScatterGroup]:
    """
    Remove symmetric redundancy: if A->B exists and B->A exists, keep only one.
    Strategy: For each undirected pair (u, v), we only keep the occurrence in the group
    where main_label is smaller (or has some deterministic property).
    
    Actually, to be safe: we iterate all candidates. If (main, cand) pair has been seen
    (as (cand, main)), we remove it from the current group's candidates.
    If a group ends up with 0 candidates, it is removed.
    """
    # Track seen edges as frozenset/tuple of sorted labels
    seen_edges = set()
    
    cleaned_groups = []
    for g in groups:
        valid_cands = []
        for c in g.candidates:
            # Create a canonical edge key
            u, v = str(g.main_label), str(c.label)
            if u > v:
                u, v = v, u
            edge = (u, v)
            
            if edge in seen_edges:
                continue
            
            seen_edges.add(edge)
            valid_cands.append(c)
            
        if valid_cands:
            # Recompute group_min_sim if candidates changed
            new_min = float(min(c.min_sim for c in valid_cands))
            cleaned_groups.append(
                ScatterGroup(
                    main_label=g.main_label,
                    main_size=g.main_size,
                    group_min_sim=new_min,
                    candidates=valid_cands,
                )
            )
            
    return cleaned_groups

