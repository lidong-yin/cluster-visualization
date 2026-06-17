from __future__ import annotations

import os
import time
import uuid
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Literal

# 标注时间, 操作类型, 目标id, 原始簇id, 目标簇id
ANNO_COLS = ["time", "anno_type", "obj_id", "source_cid", "target_cid"]

@dataclass(frozen=True)
class AnnotationConfig:
    obj_id_col: str = "obj_id"
    src_label_col: str = "gt_person_id"


def get_default_annotation_path(data_path: str) -> str:
    if not data_path:
        return "annotations.csv"
    base_dir = os.path.dirname(data_path) or "."
    base_name = os.path.basename(data_path)
    name_no_ext = os.path.splitext(base_name)[0]
    return os.path.join(base_dir, f"{name_no_ext}_annotations.csv")


def _strip_float_suffix(val: object) -> str:
    """Convert string representation: strip trailing '.0' for integer-valued floats.

    E.g., '123.0' -> '123', 'abc' -> 'abc', '1.5' -> '1.5'.
    This handles the common case where pandas stores integer IDs as float64
    and str() produces '123.0' instead of '123'.
    """
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def load_or_create_annotations(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Ensure columns exist
            for c in ANNO_COLS:
                if c not in df.columns:
                    raise ValueError(f"Annotation file is missing required column: {c}")
            # Normalize obj_id on load to match DataFrame's obj_id format
            if "obj_id" in df.columns:
                df["obj_id"] = df["obj_id"].astype(str).map(_strip_float_suffix)
            return df
        except Exception:
            return pd.DataFrame(columns=ANNO_COLS)
    return pd.DataFrame(columns=ANNO_COLS)


def append_annotation_rows(
    path: str,
    rows: list[dict],
) -> pd.DataFrame:
    """
    Append rows to the CSV at path. Creates file if not exists.
    Returns the updated DataFrame.
    """
    if not rows:
        return load_or_create_annotations(path)
        
    new_df = pd.DataFrame(rows)
    # Ensure correct column order/subset
    for c in ANNO_COLS:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[ANNO_COLS]

    if os.path.exists(path):
        # Append mode
        new_df.to_csv(path, mode='a', header=False, index=False)
        # Reload to return full (with normalized obj_id)
        return load_or_create_annotations(path)
    else:
        # New file
        new_df.to_csv(path, index=False)
        return new_df


def create_split_records(
    obj_ids: list[str],
    source_cid: str,
    new_labels: list[str | int],
    anno_type: str = "SPLIT"
) -> list[dict]:
    """
    Generate rows for a split operation.
    new_labels should be aligned with obj_ids.

    Label mapping strategy:
    - If a label is a pure integer (e.g. from auto sub-clustering like 0, 1, 2),
      it is prefixed with split_{source}_{uuid}_{local} to avoid collision with
      existing cluster IDs in the DataFrame.
    - If a label is a non-trivial string (user-specified batch label or move target),
      it is kept as-is since the user chose it deliberately.
    - Labels that look like "0" but were user-typed are indistinguishable from
      auto labels, so they also get UUID prefix. Users should use descriptive names
      for batch labels if they want to avoid prefixing.
    """
    now = time.time()
    records = []

    unique_locals = set(new_labels)
    label_map = {}
    for L in unique_locals:
        if isinstance(L, int) or (isinstance(L, str) and L.isdigit()):
            label_map[L] = f"split_{source_cid}_{uuid.uuid4().hex[:6]}_{L}"
        else:
            label_map[L] = str(L)

    for oid, lbl in zip(obj_ids, new_labels):
        records.append({
            "time": now,
            "anno_type": anno_type,
            "obj_id": oid,
            "source_cid": source_cid,
            "target_cid": label_map.get(lbl, str(lbl))
        })
    return records


def apply_annotations_to_df(
    df: pd.DataFrame,
    anno_df: pd.DataFrame,
    target_col: str,
    *,
    base_col: Optional[str] = None,
    normalize_labels: bool = False,
    obj_id_col: str = "obj_id",
) -> int:
    """
    Apply latest annotation per obj_id to df[target_col].
    - If base_col is provided, initialize target_col from base_col first.
    - Then override with annotations (latest per obj_id).
    Returns number of updated rows.
    """
    if df is None or len(df) == 0:
        return 0
    if anno_df is None or len(anno_df) == 0:
        return 0
    if obj_id_col not in df.columns:
        return 0
    if obj_id_col not in anno_df.columns or "target_cid" not in anno_df.columns:
        return 0

    if target_col not in df.columns:
        df[target_col] = None

    if base_col and base_col in df.columns:
        # Only initialize from base_col if target_col is missing or entirely NaN/None.
        # This prevents wiping results from a previous annotation apply pass.
        if df[target_col].isna().all():
            df[target_col] = df[base_col].astype(str).map(_strip_float_suffix)
        elif df[target_col].dtype == object and (df[target_col].astype(str) == "None").all():
            df[target_col] = df[base_col].astype(str).map(_strip_float_suffix)

    work = anno_df.copy()
    work = work.dropna(subset=[obj_id_col, "target_cid"])
    if "time" in work.columns:
        work = work.sort_values("time")

    work[obj_id_col] = work[obj_id_col].astype(str).map(_strip_float_suffix)
    work["target_cid"] = work["target_cid"].astype(str)

    latest = work.groupby(obj_id_col, as_index=False).last()
    mapping = dict(zip(latest[obj_id_col], latest["target_cid"]))

    obj_series = df[obj_id_col].astype(str).map(_strip_float_suffix)
    mask = obj_series.isin(mapping.keys())
    df.loc[mask, target_col] = obj_series[mask].map(mapping)

    if normalize_labels:
        codes, uniques = pd.factorize(df[target_col])
        # pd.factorize maps NaN to -1; replace with a proper label
        na_mask = codes == -1
        if na_mask.any():
            max_code = int(codes.max()) if len(codes) > 0 else -1
            codes[na_mask] = max_code + 1
        df[target_col] = codes
    return int(mask.sum())
