from __future__ import annotations

import math
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Optional
from src import analysis_utils, data_utils, faiss_utils, ui_utils
from src.state import KEYS, ensure_state, get_df, get_df_rev, get_faiss, get_feature_col, set_faiss, set_feature_col

st.set_page_config(page_title="Analysis - Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")

ui_utils.load_app_style()


def _require_df():
    df = get_df()
    if df is None:
        st.info("请先在 **Home** 页面加载数据。")
        return None
    return df


def _detect_group_keys(df: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    if "gt_person_id" in df.columns:
        keys.append("gt_person_id")
    keys += data_utils.detect_cluster_label_columns(df)
    # fallback: allow any column
    if not keys:
        keys = list(df.columns)
    return keys


def _get_features(df: pd.DataFrame, *, feature_col: str, ok_only: bool, progress_callback=None):
    ok_col = "ok" if "ok" in df.columns else None
    feats, row_idx = data_utils.extract_feature_matrix(
        df,
        feature_col=feature_col,
        ok_col=ok_col,
        ok_only=ok_only,
        progress_callback=progress_callback,
    )
    return feats, row_idx


def _analysis_cache() -> dict:
    cache = st.session_state.get(KEYS.analysis_cache)
    if not isinstance(cache, dict):
        cache = {}
        st.session_state[KEYS.analysis_cache] = cache
    return cache


def _cache_key(prefix: str, *parts: Any) -> str:
    safe = [str(p) for p in parts]
    return prefix + "::" + "::".join(safe)


def _get_group_index_map(df: pd.DataFrame, group_key: str) -> dict[Any, Any]:
    cache = _analysis_cache()
    k = _cache_key("group_index_map", get_df_rev(), group_key)
    if k in cache:
        return cache[k]
    # dropna=False to keep NaN as a group
    m = df.groupby(group_key, dropna=False).groups
    cache[k] = m
    return m


def _set_state(key: str, value: Any) -> None:
    st.session_state[key] = value


def _slice_indices(indices: Any, start: int, end: int):
    # works for list-like / pandas Index
    return indices[start:end]


def _sample_by_indices(
    df: pd.DataFrame,
    indices: Any,
    *,
    n: int,
    random_sample: bool,
    seed: int,
) -> pd.DataFrame:
    total = int(len(indices))
    n = int(max(1, n))
    if total == 0:
        return df.iloc[0:0]
    if total <= n:
        return df.loc[indices]
    if random_sample:
        rng = np.random.default_rng(int(seed))
        # Convert to numpy array for fast choice; avoid copying large arrays if possible
        idx_arr = np.asarray(indices)
        chosen = rng.choice(idx_arr, size=n, replace=False)
        return df.loc[chosen]
    return df.loc[_slice_indices(indices, 0, n)]


def _render_cluster_header(
    *,
    title: str,
    subtitle: str,
    chips: list[str],
) -> None:
    chip_html = "".join([f'<span class="chip">{c}</span>' for c in chips])
    st.markdown(
        f"""
<div class="cluster-header">
  <div>
    <div class="cluster-title">{title}</div>
    <div class="cluster-sub">{subtitle}</div>
  </div>
  <div class="chips">{chip_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _toggle_key(*parts: Any) -> str:
    return _cache_key("ui_toggle", *parts)


def _expanded_key(*parts: Any) -> str:
    return _cache_key("ui_expanded", *parts)


def _page_key(*parts: Any) -> str:
    return _cache_key("ui_page", *parts)


def _get_expand_state(
    *,
    total: int,
    collapsed_n: int,
    cfg: dict,
    key_parts: list[Any],
) -> tuple[bool, int, int]:
    """
    Compute (expanded, page, per_page) from session_state without rendering widgets.
    """
    total = int(total)
    collapsed_n = int(collapsed_n)
    per_page = max(1, int(cfg["img_cols"]) * int(cfg["page_rows"]))
    
    if total <= collapsed_n:
        return False, 1, per_page

    expanded_state_key = _expanded_key(get_df_rev(), *key_parts)
    expanded = bool(st.session_state.get(expanded_state_key, False))
    
    if not expanded:
        return False, 1, per_page

    num_pages = int(np.ceil(total / per_page))
    page_key = _page_key(get_df_rev(), *key_parts)
    page = int(st.session_state.get(page_key, 1))
    page = max(1, min(page, num_pages))
    return True, page, per_page


def _render_expand_trigger(
    *,
    total: int,
    collapsed_n: int,
    cfg: dict,
    key_parts: list[Any],
) -> None:
    """Render the button/pagination controls AFTER the grid."""
    total = int(total)
    collapsed_n = int(collapsed_n)
    if total <= collapsed_n:
        return

    expanded_state_key = _expanded_key(get_df_rev(), *key_parts)
    expanded = bool(st.session_state.get(expanded_state_key, False))
    label = "收起" if expanded else f"查看全部 ({total:,})"
    
    # Button row
    # Use columns to keep button small/aligned left or center
    c1, _ = st.columns([1, 5])
    with c1:
        if st.button(label, key=_toggle_key(get_df_rev(), *key_parts), use_container_width=True):
            st.session_state[expanded_state_key] = not expanded
            st.session_state[_page_key(get_df_rev(), *key_parts)] = 1
            st.rerun()
            
    if expanded:
        per_page = max(1, int(cfg["img_cols"]) * int(cfg["page_rows"]))
        num_pages = int(np.ceil(total / per_page))
        if num_pages > 1:
            cur = int(st.session_state.get(_page_key(get_df_rev(), *key_parts), 1))
            st.slider(
                "页码",
                min_value=1,
                max_value=num_pages,
                value=cur,
                step=1,
                key=_page_key(get_df_rev(), *key_parts),
            )



def _stable_seed(x: Any) -> int:
    # Deterministic small int seed for sampling
    return abs(hash(str(x))) % 1000003


def _expected_faiss_meta(cfg: dict) -> dict[str, Any]:
    return {
        "df_rev": get_df_rev(),
        "feature_col": cfg["feature_col"],
        "ok_only": bool(cfg["ok_only"]),
        "use_gpu_prefer": bool(cfg["use_gpu_prefer"]),
    }


def _ensure_faiss_index(df: pd.DataFrame, cfg: dict, *, progress=None):
    index, _, row_idx = get_faiss()
    meta = st.session_state.get(KEYS.faiss_meta)
    expected = _expected_faiss_meta(cfg)
    if index is not None and row_idx is not None and meta == expected:
        return index, row_idx

    prog = progress or st.progress(0.0, text="构建 Faiss 索引：提取特征 ...")

    def feat_cb(frac: float, text: str) -> None:
        prog.progress(min(0.35, 0.35 * float(frac)), text=text)

    feats, row_idx = _get_features(df, feature_col=cfg["feature_col"], ok_only=cfg["ok_only"], progress_callback=feat_cb)

    # Normalize in-place to support cosine via IP
    prog.progress(0.38, text="构建 Faiss 索引：L2 normalize ...")
    feats = feats.astype(np.float32, copy=False)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    feats /= norms

    prog.progress(0.42, text="构建 Faiss 索引：初始化 index ...")
    index = faiss_utils.build_index_ip(feats, use_gpu_prefer=cfg["use_gpu_prefer"])

    n = int(feats.shape[0])
    add_bs = 50000
    for start in range(0, n, add_bs):
        end = min(start + add_bs, n)
        index.add(feats[start:end])
        prog.progress(0.42 + 0.55 * (end / max(1, n)), text=f"构建 Faiss 索引：add {end:,}/{n:,}")

    set_faiss(index, feats, row_idx)
    st.session_state[KEYS.faiss_meta] = expected
    prog.progress(1.0, text="Faiss 索引就绪")
    return index, row_idx


def _sidebar_controls(df: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.header("分析设置")

    group_keys = _detect_group_keys(df)
    group_key = st.sidebar.selectbox("分组依据 (Grouping Key)", options=group_keys, index=0)

    view_mode = st.sidebar.radio("分析视角 (View Mode)", options=["Size", "Variance", "Scatter"], index=0)
    desc = st.sidebar.checkbox("降序排序 (desc)", value=True)

    st.sidebar.divider()
    st.sidebar.subheader("范围 / 展示")
    default_start = int(st.session_state.get(KEYS.group_range_start, 1))
    default_end = int(st.session_state.get(KEYS.group_range_end, 50))
    start = st.sidebar.number_input("起始组序号 (1-based)", min_value=1, value=default_start, step=1)
    end = st.sidebar.number_input("结束组序号 (inclusive)", min_value=1, value=max(default_end, start), step=1)
    st.session_state[KEYS.group_range_start] = int(start)
    st.session_state[KEYS.group_range_end] = int(end)

    img_cols = st.sidebar.slider("图片列数", min_value=6, max_value=20, value=int(st.session_state.get(KEYS.img_cols, 12)), step=1)
    st.session_state[KEYS.img_cols] = int(img_cols)

    per_group_images = st.sidebar.slider("每簇展示图片数", min_value=10, max_value=200, value=20, step=5)
    random_sample = st.sidebar.checkbox("每簇随机采样展示（固定种子）", value=False)
    page_rows = st.sidebar.slider("展开时每页行数", min_value=5, max_value=30, value=10, step=1)

    st.sidebar.subheader("元数据显示")
    meta_cols = st.sidebar.multiselect(
        "选择图片下方显示的字段",
        options=ui_utils.columns_except(df, exclude=["feature"]),
        default=[],
    )

    st.sidebar.divider()
    st.sidebar.subheader("查询搜索")
    search_mode = st.sidebar.selectbox("搜索模式", options=["cluster_id / label", "obj_id"], index=0)
    search_value = st.sidebar.text_input("搜索值", value="")
    show_topk = st.sidebar.checkbox("obj_id 搜索后显示 TopK 相似样本（需要 Faiss+feature）", value=True)
    sim_topk = st.sidebar.number_input("TopK", min_value=1, max_value=2048, value=20, step=1)

    st.sidebar.divider()
    st.sidebar.subheader("1v1 相似度")
    obj_a = st.sidebar.text_input("obj_id A", value="")
    obj_b = st.sidebar.text_input("obj_id B", value="")

    st.sidebar.divider()
    st.sidebar.subheader("Feature / Faiss")
    feature_candidates = [c for c in df.columns if c.lower() in {"feature", "feat", "embedding", "emb"}]
    if "feature" in df.columns and "feature" not in feature_candidates:
        feature_candidates.insert(0, "feature")
    if not feature_candidates:
        feature_candidates = list(df.columns)
    feature_col = st.sidebar.selectbox(
        "特征列 (feature)",
        options=feature_candidates,
        index=feature_candidates.index(get_feature_col()) if get_feature_col() in feature_candidates else 0,
    )
    if feature_col != get_feature_col():
        set_feature_col(feature_col)

    ok_only = st.sidebar.checkbox("只使用 ok==True 的样本（若存在 ok 列）", value=True)
    use_gpu_prefer = st.sidebar.checkbox("优先使用 GPU Faiss（若可用）", value=True)

    # Scatter params
    sim_th = st.sidebar.number_input("Scatter sim_th", min_value=-1.0, max_value=1.0, value=0.55, step=0.01)
    scatter_topk = st.sidebar.number_input("Scatter sim_topk", min_value=1, max_value=2048, value=100, step=1)
    cand_limit = st.sidebar.number_input("每组显示候选簇数", min_value=1, max_value=20, value=3, step=1)
    dedup_scatter = st.sidebar.checkbox("散度结果去重 (A-B vs B-A)", value=False)

    return {
        "group_key": group_key,
        "view_mode": view_mode,
        "desc": bool(desc),
        "range_start": int(start),
        "range_end": int(end),
        "img_cols": int(img_cols),
        "per_group_images": int(per_group_images),
        "random_sample": bool(random_sample),
        "page_rows": int(page_rows),
        "meta_cols": meta_cols,
        "search_mode": search_mode,
        "search_value": search_value.strip(),
        "show_topk": bool(show_topk),
        "sim_topk": int(sim_topk),
        "obj_a": obj_a.strip(),
        "obj_b": obj_b.strip(),
        "feature_col": feature_col,
        "ok_only": bool(ok_only),
        "use_gpu_prefer": bool(use_gpu_prefer),
        "sim_th": float(sim_th),
        "scatter_topk": int(scatter_topk),
        "cand_limit": int(cand_limit),
        "dedup_scatter": bool(dedup_scatter),
    }


def _locate_by_label(groups_df: pd.DataFrame, group_key: str, value: str) -> Optional[int]:
    if value == "":
        return None
    # Try int first
    try:
        v = int(value)
    except Exception:
        v = value
    matches = groups_df.index[groups_df[group_key] == v].tolist()
    if matches:
        return int(matches[0]) + 1  # 1-based
    # fallback string compare
    matches = groups_df.index[groups_df[group_key].astype(str) == str(value)].tolist()
    if matches:
        return int(matches[0]) + 1
    return None


def _find_obj_row(df: pd.DataFrame, obj_id: str) -> Optional[pd.Series]:
    if not obj_id:
        return None
    if "obj_id" not in df.columns:
        return None
    m = df["obj_id"].astype(str) == str(obj_id)
    if not m.any():
        return None
    return df[m].iloc[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False).reshape(-1)
    b = b.astype(np.float32, copy=False).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _render_1v1(df: pd.DataFrame, cfg: dict) -> None:
    obj_a = cfg["obj_a"]
    obj_b = cfg["obj_b"]
    if not obj_a or not obj_b:
        return
    row_a = _find_obj_row(df, obj_a)
    row_b = _find_obj_row(df, obj_b)
    if row_a is None or row_b is None:
        st.warning("1v1：未找到 obj_id A 或 B。")
        return

    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"**A: {obj_a}**")
        if "img_url" in row_a:
            st.image(row_a["img_url"], use_container_width=True)
    with cols[1]:
        st.markdown(f"**B: {obj_b}**")
        if "img_url" in row_b:
            st.image(row_b["img_url"], use_container_width=True)

    feat_col = cfg["feature_col"]
    if feat_col not in df.columns:
        st.warning(f"1v1：缺少 feature 列 `{feat_col}`，无法计算相似度。")
        return
    fa = row_a.get(feat_col)
    fb = row_b.get(feat_col)
    va = data_utils._parse_feature_vector(fa)  # type: ignore[attr-defined]
    vb = data_utils._parse_feature_vector(fb)  # type: ignore[attr-defined]
    if va is None or vb is None:
        st.warning("1v1：feature 缺失或格式不正确。")
        return
    st.metric("Cosine Similarity", f"{_cosine(va, vb):.3f}")


def _render_obj_topk(df: pd.DataFrame, cfg: dict) -> None:
    if cfg["search_mode"] != "obj_id":
        return
    obj_id = cfg["search_value"]
    if not obj_id:
        return
    row = _find_obj_row(df, obj_id)
    if row is None:
        st.warning("obj_id 搜索：未找到该样本。")
        return

    st.subheader("obj_id 定位")
    # Convert to string to prevent Arrow serialization errors with mixed types (int vs str/url)
    st.write(row.astype(str).to_frame("value"))

    if not cfg["show_topk"]:
        return
    if not faiss_utils.is_faiss_available():
        st.warning("TopK 相似：当前环境未安装 Faiss。")
        return

    feat_col = cfg["feature_col"]
    if feat_col not in df.columns:
        st.warning(f"TopK 相似：缺少 feature 列 `{feat_col}`。")
        return

    ph = st.empty()
    prog = ph.progress(0.0, text="准备 TopK 相似搜索 ...")

    try:
        index, row_idx = _ensure_faiss_index(df, cfg, progress=prog)
    except Exception as e:  # noqa: BLE001
        ph.empty()
        st.error(f"TopK 相似：构建/获取 Faiss 索引失败：{e}")
        return

    # Query vector
    q = data_utils._parse_feature_vector(row.get(feat_col))  # type: ignore[attr-defined]
    if q is None:
        ph.empty()
        st.warning("TopK 相似：该样本 feature 缺失或格式不正确。")
        return
    q = q.astype(np.float32, copy=False).reshape(1, -1)
    q = faiss_utils.l2_normalize(q)

    prog.progress(0.98, text="Faiss search ...")
    sims, nbrs = index.search(q, int(cfg["sim_topk"]) + 1)
    sims = sims[0]
    nbrs = nbrs[0]

    nbr_df_idx = row_idx[nbrs]
    # Remove self if present
    self_mask = np.array([str(x) == str(row.name) for x in nbr_df_idx], dtype=bool)
    nbr_df_idx = nbr_df_idx[~self_mask]
    sims = sims[~self_mask]

    nbr_df_idx = nbr_df_idx[: int(cfg["sim_topk"])]
    sims = sims[: int(cfg["sim_topk"])]

    ph.empty()
    sub = df.loc[nbr_df_idx].copy()
    
    # Force convert ID/URL columns to string to prevent Arrow serialization issues with mixed types
    for c in ["obj_id", "img_url"]:
        if c in sub.columns:
            sub[c] = sub[c].astype(str)
            
    # Format similarity for display
    sub["sim"] = [f"{s:.3f}" for s in sims]
    
    st.subheader("TopK 相似样本（余弦相似度）")
    ui_utils.render_image_grid(
        sub,
        metadata_cols=["sim"] + list(cfg["meta_cols"]),
        ncols=cfg["img_cols"],
        max_images=min(200, int(cfg["sim_topk"])),
    )


def _render_groups_size(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    group_key = cfg["group_key"]
    gdf = analysis_utils.group_sizes(df, group_key)
    gdf = gdf.sort_values("size", ascending=not cfg["desc"]).reset_index(drop=True)
    return gdf


def _render_groups_variance(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not faiss_utils.is_faiss_available():
        st.warning("Variance：未安装 Faiss 也可计算，但需要 feature 可用。")
    feat_col = cfg["feature_col"]
    cache = _analysis_cache()
    ck = _cache_key("variance_groups", get_df_rev(), cfg["group_key"], feat_col, cfg["ok_only"])
    if ck in cache:
        return cache[ck]

    ph = st.empty()
    prog = ph.progress(0.0, text="提取特征 ...")

    def feat_cb(frac: float, text: str) -> None:
        prog.progress(min(0.20, 0.20 * float(frac)), text=text)

    feats, row_idx = _get_features(df, feature_col=feat_col, ok_only=cfg["ok_only"], progress_callback=feat_cb)
    prog.progress(0.22, text="计算簇内方差 ...")

    def var_cb(frac: float, text: str) -> None:
        prog.progress(min(0.98, 0.22 + 0.76 * float(frac)), text=text)

    gdf = analysis_utils.group_variances(
        df,
        group_key=cfg["group_key"],
        feats=feats,
        feats_row_indices=row_idx,
        progress_callback=var_cb,
    )
    gdf = gdf.sort_values("variance", ascending=not cfg["desc"]).reset_index(drop=True)
    prog.progress(1.0, text="方差计算完成（已缓存）")
    ph.empty()
    cache[ck] = gdf
    return gdf


def _render_groups_scatter(df: pd.DataFrame, cfg: dict):
    if not faiss_utils.is_faiss_available():
        st.error("Scatter 需要 Faiss：请安装 faiss-cpu 或 faiss-gpu。")
        return []
    feat_col = cfg["feature_col"]
    cache = _analysis_cache()
    ck = _cache_key(
        "scatter_groups",
        get_df_rev(),
        cfg["group_key"],
        feat_col,
        cfg["ok_only"],
        cfg["sim_th"],
        cfg["scatter_topk"],
        cfg.get("dedup_scatter", False),  # include dedup flag in cache key
    )
    if ck in cache:
        return cache[ck]

    ph = st.empty()
    prog = ph.progress(0.0, text="提取特征 ...")

    def feat_cb(frac: float, text: str) -> None:
        prog.progress(min(0.20, 0.20 * float(frac)), text=text)

    feats, row_idx = _get_features(df, feature_col=feat_col, ok_only=cfg["ok_only"], progress_callback=feat_cb)
    prog.progress(0.22, text="计算散度（含 Faiss 检索） ...")

    def scatter_cb(frac: float, text: str) -> None:
        prog.progress(min(0.98, 0.22 + 0.76 * float(frac)), text=text)

    groups = analysis_utils.compute_scatter_groups(
        df,
        group_key=cfg["group_key"],
        feats=feats,
        feats_row_indices=row_idx,
        sim_th=float(cfg["sim_th"]),
        sim_topk=int(cfg["scatter_topk"]),
        use_gpu_prefer=cfg["use_gpu_prefer"],
        progress_callback=scatter_cb,
    )

    if cfg.get("dedup_scatter"):
        prog.progress(0.99, text="结果去重 ...")
        groups = analysis_utils.deduplicate_scatter_groups(groups)

    groups.sort(key=lambda g: g.group_min_sim, reverse=cfg["desc"])
    prog.progress(1.0, text="散度计算完成（已缓存）")
    ph.empty()
    cache[ck] = groups
    return groups


def _slice_range(items_len: int, start_1: int, end_1: int) -> tuple[int, int]:
    start = max(0, int(start_1) - 1)
    end = max(start, int(end_1) - 1)
    end = min(end, items_len - 1)
    return start, end


def main() -> None:
    ensure_state()
    st.title("分析：簇质量评估")
    st.caption("按大小 / 方差(纯度) / 散度(跨簇相似) 排序查看簇，并支持搜索与相似度分析。")

    df = _require_df()
    if df is None:
        return

    cfg = _sidebar_controls(df)

    # 1v1
    has_1v1 = bool(cfg["obj_a"] and cfg["obj_b"])
    if has_1v1:
        st.subheader("1v1 相似度比较")
        _render_1v1(df, cfg)
        st.divider()
        # Exclusive view: return early
        return

    # obj topk (if requested)
    has_search_obj = (cfg["search_mode"] == "obj_id" and cfg["search_value"])
    if has_search_obj:
        st.subheader(f"搜索结果: {cfg['search_value']}")
        _render_obj_topk(df, cfg)
        st.divider()
        # Exclusive view: return early
        return

    st.divider()

    group_key = cfg["group_key"]
    view_mode = cfg["view_mode"]

    # Build groups list
    if view_mode == "Size":
        groups_df = _render_groups_size(df, cfg)
        metric_col = "size"
    elif view_mode == "Variance":
        groups_df = _render_groups_variance(df, cfg)
        metric_col = "variance"
    else:
        scatter_groups = _render_groups_scatter(df, cfg)
        # cluster/label search in scatter mode will search main_label
        if cfg["search_mode"] == "cluster_id / label" and cfg["search_value"]:
            # locate 1-based
            try:
                v = int(cfg["search_value"])
            except Exception:
                v = cfg["search_value"]
            found = None
            for i, g in enumerate(scatter_groups):
                if g.main_label == v or str(g.main_label) == str(v):
                    found = i + 1
                    break
            if found is not None:
                st.sidebar.success(f"定位到主簇序号: {found}")
                cfg["range_start"] = found
                cfg["range_end"] = found
        start, end = _slice_range(len(scatter_groups), cfg["range_start"], cfg["range_end"])
        st.subheader(f"{view_mode}排序 | 簇标签:`{group_key}` | 总数: {len(scatter_groups):,} | 当前序号 {start+1}～{end+1}")

        idx_map = _get_group_index_map(df, group_key)
        limit = int(cfg["per_group_images"])
        rand = bool(cfg["random_sample"])
        cand_limit = int(cfg.get("cand_limit", 3))

        for i in range(start, end + 1):
            g = scatter_groups[i]
            # Outer container for the whole scatter group
            with st.container(border=True):
                # Group Header
                _render_cluster_header(
                    title=f"#{i+1} Scatter Group",
                    subtitle=f"Main: {g.main_label} | Candidates: {len(g.candidates)}",
                    chips=[f"min_sim={g.group_min_sim:.3f}"]
                )
                
                # 1. Main Cluster
                main_indices = idx_map.get(g.main_label, [])
                main_size = len(main_indices)
                
                st.markdown(f"##### 🟢 Main Cluster: {g.main_label} <span style='color:grey;font-size:0.9em'>(size={main_size:,})</span>", unsafe_allow_html=True)
                
                # Expand controls for Main Cluster
                expanded, page, per_page = _get_expand_state(
                    total=main_size,
                    collapsed_n=limit,
                    cfg=cfg,
                    key_parts=["scatter_main", i, g.main_label]
                )
                
                if not expanded:
                    show = _sample_by_indices(df, main_indices, n=limit, random_sample=rand, seed=_stable_seed(f"main::{g.main_label}"))
                    ui_utils.render_image_grid(show, metadata_cols=list(cfg["meta_cols"]), ncols=cfg["img_cols"], max_images=len(show))
                else:
                    p_start = (page - 1) * per_page
                    p_end = min(p_start + per_page, main_size)
                    pg_indices = _slice_indices(main_indices, p_start, p_end)
                    ui_utils.render_image_grid(df.loc[pg_indices], metadata_cols=list(cfg["meta_cols"]), ncols=cfg["img_cols"], max_images=len(pg_indices))
                
                _render_expand_trigger(total=main_size, collapsed_n=limit, cfg=cfg, key_parts=["scatter_main", i, g.main_label])

                # 2. Candidate Clusters
                candidates_to_show = g.candidates[:cand_limit]
                for c_idx, c in enumerate(candidates_to_show):
                    # Use padding instead of divider for visual separation
                    st.markdown('<div style="margin-top: 1.2rem;"></div>', unsafe_allow_html=True)
                    c_indices = idx_map.get(c.label, [])
                    c_size = len(c_indices)
                    st.markdown(
                        f"##### 🟠 Candidate #{c_idx+1}: {c.label} "
                        f"<span style='color:grey;font-size:0.9em'>(size={c_size:,}, min_sim={c.min_sim:.3f})</span>", 
                        unsafe_allow_html=True
                    )
                    
                    # Expand controls for THIS candidate
                    expanded_c, page_c, per_page_c = _get_expand_state(
                        total=c_size,
                        collapsed_n=limit,
                        cfg=cfg,
                        key_parts=["scatter_cand", i, g.main_label, c.label]
                    )
                    
                    if not expanded_c:
                        c_show = _sample_by_indices(df, c_indices, n=limit, random_sample=rand, seed=_stable_seed(f"cand::{g.main_label}::{c.label}"))
                        ui_utils.render_image_grid(c_show, metadata_cols=list(cfg["meta_cols"]), ncols=cfg["img_cols"], max_images=len(c_show))
                    else:
                        p_start_c = (page_c - 1) * per_page_c
                        p_end_c = min(p_start_c + per_page_c, c_size)
                        pg_indices_c = _slice_indices(c_indices, p_start_c, p_end_c)
                        ui_utils.render_image_grid(df.loc[pg_indices_c], metadata_cols=list(cfg["meta_cols"]), ncols=cfg["img_cols"], max_images=len(pg_indices_c))
                        
                    _render_expand_trigger(total=c_size, collapsed_n=limit, cfg=cfg, key_parts=["scatter_cand", i, g.main_label, c.label])

                if len(g.candidates) > cand_limit:
                    st.caption(f"... 还有 {len(g.candidates) - cand_limit} 个候选簇被隐藏 (可在侧边栏调整显示数量)")
            
            st.divider()
        return

    # Cluster/label search (non-scatter)
    if cfg["search_mode"] == "cluster_id / label" and cfg["search_value"]:
        pos = _locate_by_label(groups_df, group_key, cfg["search_value"])
        if pos is None:
            st.sidebar.warning("未找到对应簇。")
        else:
            st.sidebar.success(f"定位到簇序号: {pos}")
            cfg["range_start"] = pos
            cfg["range_end"] = pos

    start, end = _slice_range(len(groups_df), cfg["range_start"], cfg["range_end"])
    st.subheader(f"{view_mode}排序 | 簇标签:`{group_key}` | 总数: {len(groups_df):,} | 当前序号 {start+1}～{end+1}")

    idx_map = _get_group_index_map(df, group_key)
    limit = int(cfg["per_group_images"])
    rand = bool(cfg["random_sample"])

    for i in range(start, end + 1):
        row = groups_df.iloc[i]
        gid = row[group_key]
        metric_val = row.get(metric_col)
        indices = idx_map.get(gid, [])
        size = int(len(indices))
        
        chips = [f"size={size:,}"]
        if view_mode == "Variance":
            chips.append(f"variance={float(metric_val):.6f}")

        with st.container(border=True):
            _render_cluster_header(
                title=f"#{i+1} {group_key}={gid}",
                subtitle=f"View: {view_mode}",
                chips=chips
            )
            
            limit = int(cfg["per_group_images"])
            rand = bool(cfg["random_sample"])
            
            expanded, page, per_page = _get_expand_state(
                total=size,
                collapsed_n=limit,
                cfg=cfg,
                key_parts=["group", i, gid]
            )
            
            if not expanded:
                show = _sample_by_indices(df, indices, n=limit, random_sample=rand, seed=_stable_seed(gid))
                st.caption(f"预览模式 (上限 {limit}) | 当前展示 {len(show):,}/{size:,}")
                ui_utils.render_image_grid(
                    show,
                    metadata_cols=list(cfg["meta_cols"]),
                    ncols=cfg["img_cols"],
                    max_images=len(show)
                )
            else:
                p_start = (page - 1) * per_page
                p_end = min(p_start + per_page, size)
                pg_indices = _slice_indices(indices, p_start, p_end)
                st.caption(f"全量浏览 | 第 {page} 页 | 展示 {p_start+1}..{p_end} (总计 {size:,})")
                ui_utils.render_image_grid(
                    df.loc[pg_indices],
                    metadata_cols=list(cfg["meta_cols"]),
                    ncols=cfg["img_cols"],
                    max_images=len(pg_indices)
                )
            
            _render_expand_trigger(
                total=size,
                collapsed_n=limit,
                cfg=cfg,
                key_parts=["group", i, gid]
            )
        st.divider()
        # st.divider() is implicit via container borders, but we can add space
        st.write("")

if __name__ == "__main__":
    main()

