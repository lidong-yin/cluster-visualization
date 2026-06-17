from __future__ import annotations

import time
import numpy as np
import pandas as pd
import streamlit as st
from src import clustering_utils, data_utils, faiss_utils, ui_utils
from src.state import KEYS, bump_df_rev, ensure_state, get_df, get_feature_col, set_feature_col

st.set_page_config(page_title="Clustering - Face Clustering Analyzer", layout="wide", initial_sidebar_state="expanded")

ui_utils.load_app_style()


def _require_df():
    df = get_df()
    if df is None:
        st.info("请先在 **Home** 页面加载数据。")
        return None
    return df


def _render_sidebar(df) -> dict:
    st.sidebar.header("聚类设置")

    feature_candidates = [c for c in df.columns if c.lower() in {"feature", "feat", "embedding", "emb"}]
    if "feature" in df.columns and "feature" not in feature_candidates:
        feature_candidates.insert(0, "feature")
    if not feature_candidates and "feature" in df.columns:
        feature_candidates = ["feature"]
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

    method = st.sidebar.selectbox("聚类方法", options=["HAC", "Infomap", "KMeans"], index=0)

    params: dict = {"method": method, "feature_col": feature_col, "ok_only": ok_only}

    if method == "HAC":
        params["cluster_th"] = st.sidebar.number_input("聚类阈值 (cos sim)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        params["topk"] = st.sidebar.number_input("TopK 邻居", min_value=10, max_value=1024, value=256, step=1)
        init_label_col = st.sidebar.selectbox(
            "初始化标签列（可选）",
            options=["(none)"] + list(df.columns),
            index=0,
        )
        params["init_label_col"] = None if init_label_col == "(none)" else init_label_col
        params["use_gpu_prefer"] = st.sidebar.checkbox("优先使用 GPU Faiss（若可用）", value=True)

    elif method == "Infomap":
        params["cluster_th"] = st.sidebar.number_input("聚类阈值 (cos sim)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
        params["topk"] = st.sidebar.number_input("TopK 邻居", min_value=10, max_value=1024, value=256, step=1)
        params["sparsification"] = st.sidebar.checkbox("稀疏化", value=True)
        params["node_link"] = st.sidebar.number_input("稀疏化点数(node_link)", min_value=1, max_value=256, value=20, step=1)
        params["use_gpu_prefer"] = st.sidebar.checkbox("优先使用 GPU Faiss（若可用）", value=True)

    else:  # KMeans
        params["n_clusters"] = st.sidebar.number_input("聚类数 (K)", min_value=2, max_value=1000000, value=1000, step=1)
        params["metric"] = st.sidebar.selectbox("距离/相似度", options=["cosine", "l2"], index=0)
        params["minibatch"] = st.sidebar.checkbox("MiniBatchKMeans（推荐大数据）", value=True)
        params["batch_size"] = st.sidebar.number_input("batch_size", min_value=128, max_value=200000, value=2048, step=128)
        params["max_iter"] = st.sidebar.number_input("最大迭代次数", min_value=1, max_value=1000, value=20, step=1)
        params["use_gpu_prefer"] = st.sidebar.checkbox("优先使用 Faiss GPU/CPU (更快)", value=True)

    st.sidebar.divider()
    out_col = st.sidebar.text_input("输出标签列名", value=f"cluster_id_{method.lower()}")
    save_path = st.sidebar.text_input("保存数据路径（pkl/parquet/csv）", value="")

    params["out_col"] = out_col.strip()
    params["save_path"] = save_path.strip()
    params["do_save"] = st.sidebar.checkbox("聚类后保存到路径", value=False)
    return params


def _run_clustering(df, params: dict):
    ph = st.empty()
    prog = ph.progress(0.0, text="提取特征 ...")

    def feat_cb(frac: float, text: str) -> None:
        prog.progress(min(0.35, 0.35 * float(frac)), text=text)

    ok_col = "ok" if "ok" in df.columns else None
    feats, row_idx = data_utils.extract_feature_matrix(
        df,
        feature_col=params["feature_col"],
        ok_col=ok_col,
        ok_only=bool(params["ok_only"]),
        progress_callback=feat_cb,
    )
    st.info(f"用于聚类的样本数: {feats.shape[0]:,} | dim={feats.shape[1]}")

    method = params["method"]
    t0 = time.time()
    prog.progress(0.38, text=f"运行聚类算法: {method} ...")

    def cluster_cb(frac: float, text: str) -> None:
        prog.progress(min(0.98, 0.38 + 0.60 * float(frac)), text=text)

    if method == "HAC":
        init_labels = None
        if params.get("init_label_col"):
            init_labels = df.loc[row_idx, params["init_label_col"]].to_numpy()
        res = clustering_utils.hac_cluster(
            feats,
            cluster_th=float(params["cluster_th"]),
            topk=int(params["topk"]),
            init_labels=init_labels,
            use_gpu_prefer=bool(params["use_gpu_prefer"]),
            progress_callback=cluster_cb,
        )
    elif method == "Infomap":
        res = clustering_utils.infomap_cluster(
            feats,
            cluster_th=float(params["cluster_th"]),
            topk=int(params["topk"]),
            sparsification=bool(params["sparsification"]),
            node_link=int(params["node_link"]),
            use_gpu_prefer=bool(params["use_gpu_prefer"]),
            progress_callback=cluster_cb,
        )
    else:
        res = clustering_utils.kmeans_cluster(
            feats,
            n_clusters=int(params["n_clusters"]),
            metric=str(params["metric"]),
            minibatch=bool(params["minibatch"]),
            batch_size=int(params["batch_size"]),
            max_iter=int(params["max_iter"]),
            use_gpu_prefer=bool(params.get("use_gpu_prefer", True)),
            progress_callback=cluster_cb,
        )
        prog.progress(0.95, text="KMeans 完成，写回结果 ...")
    dt = time.time() - t0

    out_col = params["out_col"]
    if not out_col:
        raise ValueError("输出标签列名为空。请在侧边栏填写 out_col。")

    # Write back labels to df (align by row_idx)
    # Use object dtype to preserve integer labels without float coercion from NaN
    df[out_col] = pd.NA
    df[out_col] = df[out_col].astype(object)
    df.loc[row_idx, out_col] = res.labels.astype(int).tolist()
    bump_df_rev()

    st.success(f"聚类完成：{method} | clusters={len(np.unique(res.labels)):,} | time={dt:.1f}s | out_col=`{out_col}`")
    with st.expander("运行参数与元信息", expanded=False):
        st.json({"params": params, "meta": res.meta})

    if params.get("do_save"):
        sp = params.get("save_path", "")
        if not sp:
            st.warning("已勾选保存，但保存路径为空；跳过保存。")
        else:
            prog.progress(0.99, text="保存数据 ...")
            data_utils.save_dataframe(df, sp)
            st.success(f"已保存到: `{sp}`")
    prog.progress(1.0, text="完成")
    ph.empty()


def main() -> None:
    ensure_state()
    st.title("聚类：算法执行")
    st.caption("选择聚类方法与参数，生成新的标签列（cluster_id*）并可选保存到文件。")

    df = _require_df()
    if df is None:
        return

    # Quick availability hints
    if not faiss_utils.is_faiss_available():
        st.warning("当前环境未检测到 Faiss：HAC/Infomap 的 TopK 检索会不可用（将报错）。KMeans 仍可用。")

    params = _render_sidebar(df)

    st.subheader("运行聚类")
    if st.button("开始聚类", type="primary"):
        try:
            _run_clustering(df, params)
        except Exception as e:  # noqa: BLE001
            st.error(f"聚类失败：{e}")

    st.subheader("数据预览")
    st.dataframe(df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()

